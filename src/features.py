"""Feature engineering for the MA demand model and the Henry Hub price model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Weather normals
# --------------------------------------------------------------------------
def monthly_degree_days(weather_daily: pd.DataFrame) -> pd.DataFrame:
    """Sum daily HDD/CDD to calendar months; drop incomplete trailing months."""
    m = weather_daily[["hdd", "cdd"]].resample("MS").sum()
    counts = weather_daily["hdd"].resample("MS").count()
    full = counts >= m.index.days_in_month * 0.95
    return m[full]


def monthly_hdd_normals(hdd_monthly: pd.Series, years: int = 30) -> pd.DataFrame:
    """Per calendar month: normal (mean), cold (P90) and warm (P10) HDD.

    Uses the most recent `years` complete years, like a utility 'normal' year.
    """
    recent = hdd_monthly[hdd_monthly.index.year >= hdd_monthly.index.year.max() - years]
    g = recent.groupby(recent.index.month)
    return pd.DataFrame({
        "normal": g.mean(),
        "cold_p90": g.quantile(0.90),
        "warm_p10": g.quantile(0.10),
    }).rename_axis("month_of_year")


# --------------------------------------------------------------------------
# Demand (monthly, Massachusetts)
# --------------------------------------------------------------------------
def build_demand_frame(consumption: pd.DataFrame, weather_daily: pd.DataFrame,
                       target: str = "total") -> pd.DataFrame:
    """Monthly modeling table: target volume + weather + calendar features."""
    dd = monthly_degree_days(weather_daily)
    df = consumption[[target]].rename(columns={target: "y"}).join(dd, how="inner")
    df["days"] = df.index.days_in_month
    df["hdd_per_day"] = df["hdd"] / df["days"]
    df["month"] = df.index.month
    df["t"] = np.arange(len(df)) / 12.0  # trend in years
    for k in (1, 2):
        df[f"sin{k}"] = np.sin(2 * np.pi * k * df["month"] / 12)
        df[f"cos{k}"] = np.cos(2 * np.pi * k * df["month"] / 12)
    # Lags of the target (only usable for horizon-aware models, see models.py)
    df["y_lag12"] = df["y"].shift(12)
    return df.dropna(subset=["y", "hdd"])


# --------------------------------------------------------------------------
# Price (weekly, Henry Hub)
# --------------------------------------------------------------------------
def _prior_year_weekly_norm(s: pd.Series, years: int = 5) -> pd.Series:
    """For each week, mean of the same ISO week over the previous `years` years.

    Uses only earlier years, so there is no look-ahead.
    """
    woy = s.index.isocalendar().week.astype(int).clip(upper=52).values
    yr = s.index.year.values
    tbl = pd.DataFrame({"v": s.values, "woy": woy, "yr": yr}, index=s.index)
    piv = tbl.pivot_table(index="yr", columns="woy", values="v", aggfunc="mean")
    norm = piv.shift(1).rolling(years, min_periods=years).mean()
    vals = [norm.at[y, w] if (y in norm.index and w in norm.columns) else np.nan
            for y, w in zip(yr, woy)]
    return pd.Series(vals, index=s.index)


def build_price_frame(henry_hub_daily: pd.Series, storage: pd.Series,
                      lng_monthly: pd.Series, weather_us_daily: pd.DataFrame,
                      horizons: tuple[int, ...] = (1, 4)) -> pd.DataFrame:
    """Weekly modeling table (weeks ending Friday).

    Every feature is as-of the end of week t and respects publication lags.
    Targets: y_h{h} = log(P[t+h]) - log(P[t]) for each horizon h.
    """
    W = "W-FRI"
    price = henry_hub_daily.resample(W).mean().rename("price")
    n_days = henry_hub_daily.resample(W).count()
    price = price[n_days >= 3]  # drop a partial trailing week
    df = price.to_frame()
    df["log_p"] = np.log(df["price"])

    # Price dynamics (known at end of week t)
    df["ret_1"] = df["log_p"].diff(1)
    df["ret_4"] = df["log_p"].diff(4)
    df["ret_13"] = df["log_p"].diff(13)
    df["vol_8"] = df["ret_1"].rolling(8).std()

    # Storage: surplus vs prior 5-year same-week average, lagged for release
    st = storage.resample(W).last()
    st_norm = _prior_year_weekly_norm(st)
    inj = st.diff()
    inj_norm = _prior_year_weekly_norm(inj)
    stor = pd.DataFrame({
        "storage_bcf": st,
        "storage_surplus_pct": (st / st_norm - 1) * 100,
        "injection_vs_norm_bcf": inj - inj_norm,
    }).shift(config.STORAGE_LAG_WEEKS, freq=W)  # move to the week it is known
    df = df.join(stor)

    # Weather: weekly gas-weighted degree days and anomaly vs prior years
    wk = weather_us_daily[["hdd", "cdd"]].resample(W).sum()
    wk["dd_total"] = wk["hdd"] + wk["cdd"]
    wk["dd_anom"] = wk["dd_total"] - _prior_year_weekly_norm(wk["dd_total"])
    df = df.join(wk[["hdd", "cdd", "dd_anom"]])
    df["dd_anom_4"] = df["dd_anom"].rolling(4).sum()

    # LNG exports: monthly, lagged for publication, YoY growth, carried weekly
    lng = lng_monthly.copy()
    lng.index = lng.index.to_period("M").to_timestamp()
    lng_feat = pd.DataFrame({"lng_yoy_pct": lng.pct_change(12) * 100,
                             "lng_bcfd": lng / lng.index.days_in_month / 1000})
    lng_feat.index = lng_feat.index + pd.DateOffset(months=config.LNG_LAG_MONTHS + 1)
    # Carry each value forward from the date it became available
    lng_weekly = (lng_feat.reindex(lng_feat.index.union(df.index))
                  .ffill().reindex(df.index))
    df = df.join(lng_weekly)

    # Seasonality
    woy = df.index.isocalendar().week.astype(int)
    df["sin_w"] = np.sin(2 * np.pi * woy / 52)
    df["cos_w"] = np.cos(2 * np.pi * woy / 52)

    for h in horizons:
        df[f"y_h{h}"] = df["log_p"].shift(-h) - df["log_p"]
    return df


PRICE_FEATURES = [
    "ret_1", "ret_4", "ret_13", "vol_8",
    "storage_surplus_pct", "injection_vs_norm_bcf",
    "dd_anom", "dd_anom_4",
    "lng_yoy_pct", "lng_bcfd",
    "sin_w", "cos_w",
]
