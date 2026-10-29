"""Forward-looking forecast data shared by the Streamlit app and the dashboard site.

Price: recent weekly Henry Hub spot, the best 4-week model's 1- and 4-week-ahead forecasts with an
80% range taken from that model's own walk-forward errors, then the NYMEX futures curve by month.
Demand: recent monthly Massachusetts utility demand and the 12-month scenario forecast.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKET = "NYMEX futures (market)"


def _read(p: Path, name: str, **kw) -> pd.DataFrame | None:
    f = p / name
    return pd.read_csv(f, **kw) if f.exists() else None


def best_price_model(processed: Path, h: int = 4) -> str:
    sc = pd.read_csv(processed / f"price_scores_h{h}.csv", index_col="model")
    return sc.drop(index=["Random walk", MARKET], errors="ignore")["RMSE log-return"].idxmin()


def price_outlook(processed: Path, history_weeks: int = 56, level: float = 0.80) -> dict:
    """History, model forecast path (with range) and futures curve for the forward price chart."""
    processed = Path(processed)
    model = best_price_model(processed, 4)
    pf = pd.read_csv(processed / "price_frame.csv", parse_dates=["period"]).set_index("period")
    hist = pf["price"].dropna().iloc[-history_weeks:]
    last_date, last_price = hist.index[-1], float(hist.iloc[-1])

    lat = pd.read_csv(processed / "price_latest_forecasts.csv", parse_dates=["as_of"])
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    path = [{"date": last_date, "fcst": last_price, "lo": last_price, "hi": last_price, "h": 0}]
    for h in (1, 4):
        row = lat[(lat["h"] == h) & (lat["model"] == model)]
        if row.empty:
            continue
        f = float(row["price_fcst"].iloc[0])
        wf = _read(processed, f"price_wf_h{h}.csv")
        e = np.log(wf.loc[wf["model"] == model, "price_actual"] / wf.loc[wf["model"] == model, "price_fcst"]).dropna()
        path.append({"date": last_date + pd.Timedelta(weeks=h), "fcst": f,
                     "lo": f * float(np.exp(e.quantile(lo_q))), "hi": f * float(np.exp(e.quantile(hi_q))), "h": h})

    strip = _read(processed, "nymex_strip_latest.csv")
    futures = pd.DataFrame(columns=["date", "contract", "price"])
    as_of = None
    if strip is not None and len(strip):
        as_of = str(strip["as_of"].iloc[0])
        futures = pd.DataFrame({
            "date": pd.to_datetime(strip["contract"] + "-15"),          # plotted mid-delivery-month
            "contract": strip["contract"], "price": strip["settle"].astype(float)})
    return {"model": model, "history": hist.rename("price").reset_index().rename(columns={"period": "date"}),
            "path": pd.DataFrame(path), "futures": futures, "futures_as_of": as_of,
            "last_date": last_date, "last_price": last_price, "level": level}


def demand_outlook(processed: Path, history_months: int = 24) -> dict:
    """Recent actual demand and the 12-month forecast (normal/cold/warm weather, 90% interval), in Bcf."""
    processed = Path(processed)
    dfd = pd.read_csv(processed / "demand_frame.csv", index_col=0, parse_dates=True)
    fc = pd.read_csv(processed / "demand_forecast.csv", index_col=0, parse_dates=True)
    meta = json.loads((processed / "meta.json").read_text())
    hist = (dfd["y"].dropna().iloc[-history_months:] / 1000).rename("actual")
    cols = [c for c in ["normal", "cold", "warm", "lo90", "hi90"] if c in fc]
    return {"history": hist, "forecast": fc[cols] / 1000, "model": meta.get("best_demand_model", ""),
            "last_actual": hist.index[-1]}
