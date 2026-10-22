"""Rolling-origin backtest, scenario forecasts and prediction intervals for
the monthly demand models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .demand_models import ALL_MODELS

WINTER = [11, 12, 1, 2, 3]


def _normals_from(train: pd.DataFrame, years: int = 30) -> pd.DataFrame:
    recent = train[train.index >= train.index[-1] - pd.DateOffset(years=years)]
    g = recent.groupby(recent.index.month)
    return pd.DataFrame({
        "hdd": g["hdd"].mean(), "cdd": g["cdd"].mean(),
        "hdd_p90": g["hdd"].quantile(0.9), "cdd_p90": g["cdd"].quantile(0.9),
        "hdd_p10": g["hdd"].quantile(0.1), "cdd_p10": g["cdd"].quantile(0.1),
    })


def future_frame(history: pd.DataFrame, months: pd.DatetimeIndex,
                 weather: pd.DataFrame | None = None,
                 scenario: str = "normal") -> pd.DataFrame:
    """Exogenous rows for `months`, using `history` for lags and the trend.

    scenario: "actual" (use weather in `weather`), "normal", "cold" or "warm".
    """
    f = pd.DataFrame(index=months)
    f["days"] = f.index.days_in_month
    f["month"] = f.index.month
    t0 = history["t"].iloc[-1]
    f["t"] = t0 + np.arange(1, len(f) + 1) / 12.0
    for k in (1, 2):
        f[f"sin{k}"] = np.sin(2 * np.pi * k * f["month"] / 12)
        f[f"cos{k}"] = np.cos(2 * np.pi * k * f["month"] / 12)
    f["y_lag12"] = [history["y"].get(d - pd.DateOffset(years=1), np.nan) for d in f.index]

    if scenario == "actual":
        f[["hdd", "cdd"]] = weather.loc[months, ["hdd", "cdd"]].values
    else:
        n = _normals_from(history)
        suffix = {"normal": "", "cold": "_p90", "warm": "_p10"}[scenario]
        f["hdd"] = f["month"].map(n[f"hdd{suffix}"])
        f["cdd"] = f["month"].map(n["cdd"])  # scenarios vary heating weather only
    f["hdd_per_day"] = f["hdd"] / f["days"]
    return f


def rolling_backtest(df: pd.DataFrame, models=ALL_MODELS, first_origin: str = "2015-12-01",
                     step: int = 3, horizon: int = 12, weather: str = "normal",
                     verbose: bool = True) -> pd.DataFrame:
    """Expanding-window backtest.

    weather="normal": future HDD/CDD are the 30-year normals known at the
        origin (an honest ex-ante forecast).
    weather="actual": future HDD/CDD are what actually happened, isolating
        model skill from weather-forecast error (ex-post, as in weather
        normalization studies).
    """
    origins = df.index[(df.index >= pd.Timestamp(first_origin))][:-1][::step]
    rows = []
    for i, origin in enumerate(origins):
        train = df.loc[:origin]
        months = df.index[(df.index > origin)][:horizon]
        if len(months) == 0:
            continue
        fut = future_frame(train, months, weather=df, scenario=weather)
        actual = df.loc[months, "y"].values
        for M in models:
            try:
                pred = M().fit(train).predict(fut)
            except Exception as e:  # keep the backtest running, record failure
                if verbose:
                    print(f"  {M.name} failed at {origin:%Y-%m}: {e}")
                pred = np.full(len(months), np.nan)
            for h, (d, a, p) in enumerate(zip(months, actual, pred), start=1):
                rows.append({"origin": origin, "date": d, "h": h, "model": M.name,
                             "y": a, "yhat": p})
        if verbose and (i % 8 == 0 or i == len(origins) - 1):
            print(f"  origin {i + 1}/{len(origins)} ({origin:%Y-%m}) done")
    out = pd.DataFrame(rows)
    out["err"] = out["yhat"] - out["y"]
    out["ape"] = (out["err"].abs() / out["y"]) * 100
    return out


def score(bt: pd.DataFrame) -> pd.DataFrame:
    """Overall, winter and short-horizon accuracy per model."""
    def agg(g):
        w = g[g["date"].dt.month.isin(WINTER)]
        s = g[g["h"] <= 3]
        return pd.Series({
            "MAPE %": g["ape"].mean(),
            "Winter MAPE %": w["ape"].mean(),
            "1-3 mo MAPE %": s["ape"].mean(),
            "RMSE (MMcf)": np.sqrt((g["err"] ** 2).mean()),
            "Bias %": (g["err"] / g["y"]).mean() * 100,
            "n": len(g),
        })
    res = bt.dropna(subset=["yhat"]).groupby("model").apply(agg, include_groups=False)
    base = res.loc["Seasonal naive", "MAPE %"] if "Seasonal naive" in res.index else np.nan
    res["Skill vs naive %"] = (1 - res["MAPE %"] / base) * 100
    return res.sort_values("MAPE %")


def mape_by_horizon(bt: pd.DataFrame) -> pd.DataFrame:
    return bt.dropna(subset=["yhat"]).pivot_table(index="h", columns="model",
                                                  values="ape", aggfunc="mean")


def interval_quantiles(bt: pd.DataFrame, model: str,
                       levels=(0.05, 0.95)) -> pd.DataFrame:
    """Empirical % error quantiles by horizon (split-conformal style)."""
    g = bt[(bt["model"] == model)].dropna(subset=["yhat"])
    rel = (g["y"] / g["yhat"] - 1)
    return rel.groupby(g["h"]).quantile(list(levels)).unstack()


def scenario_forecast(df: pd.DataFrame, model_cls, horizon: int = 12,
                      bt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit on all data, forecast the next `horizon` months under normal,
    cold (P90 HDD) and warm (P10 HDD) weather, with backtest-based
    90% intervals around the normal-weather path."""
    months = pd.date_range(df.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    model = model_cls().fit(df)
    out = pd.DataFrame(index=months)
    for sc in ("normal", "cold", "warm"):
        fut = future_frame(df, months, scenario=sc)
        out[sc] = model.predict(fut)
        if sc == "normal":
            out["hdd_normal"] = fut["hdd"].values
    if bt is not None:
        q = interval_quantiles(bt, model_cls.name)
        h = np.arange(1, horizon + 1)
        out["lo90"] = out["normal"] * (1 + q.iloc[:, 0].reindex(h).ffill().values)
        out["hi90"] = out["normal"] * (1 + q.iloc[:, 1].reindex(h).ffill().values)
    out.index.name = "month"
    return out


def weather_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Weather-normalized history: actual + b * (normal HDD - actual HDD).

    The standard regulatory adjustment: what demand would have been in a
    normal-weather month. Separates weather from structural change
    (efficiency, electrification, customer growth).
    """
    import statsmodels.api as sm

    # Weather terms only (no seasonal dummies, which would soak up the
    # weather effect and understate the HDD coefficient).
    X = pd.DataFrame({"days": df["days"], "hdd": df["hdd"], "cdd": df["cdd"],
                      "t_days": df["t"] * df["days"]})
    b = sm.OLS(df["y"], X).fit().params["hdd"]
    n = _normals_from(df)
    out = df[["y", "hdd"]].copy()
    out["hdd_normal"] = out.index.month.map(n["hdd"])
    out["y_wn"] = out["y"] + b * (out["hdd_normal"] - out["hdd"])
    out.attrs["mmcf_per_hdd"] = b
    return out
