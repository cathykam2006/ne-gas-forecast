"""Generate SYNTHETIC raw files with the same schema as `src.data.pull_all`.

For testing the pipeline offline only. These numbers are simulated, not
real EIA or weather data, and must never be presented as results.

    python tests/make_fixture.py /path/to/fixture_dir
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)


def weather(start="2001-01-01", end="2026-10-15", amp=22, mean=52):
    d = pd.date_range(start, end, freq="D")
    doy = d.dayofyear.values
    t = mean - amp * np.cos(2 * np.pi * (doy - 20) / 365.25)
    noise = np.zeros(len(d))
    for i in range(1, len(d)):  # autocorrelated weather anomalies
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 5)
    tm = t + noise
    df = pd.DataFrame({"tmean": tm}, index=pd.Index(d, name="date"))
    df["hdd"] = (65 - df["tmean"]).clip(lower=0)
    df["cdd"] = (df["tmean"] - 65).clip(lower=0)
    return df


def main(root: Path):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    wma = weather()
    wus = weather(amp=18, mean=58)
    wma.to_csv(raw / "weather_ma.csv")
    wus.to_csv(raw / "weather_us.csv")

    # Monthly MA consumption (lags EIA by ~3 months)
    m = wma[["hdd", "cdd"]].resample("MS").sum().loc[:"2026-07-01"]
    days = m.index.days_in_month.values
    trend = np.linspace(1.0, 0.92, len(m))  # efficiency / electrification drift
    res = (90 * days + 22 * m["hdd"].values) * trend + rng.normal(0, 600, len(m))
    com = (60 * days + 13 * m["hdd"].values) * trend + rng.normal(0, 400, len(m))
    ind = 140 * days + rng.normal(0, 200, len(m))
    elec = 250 * days + 40 * m["cdd"].values + rng.normal(0, 800, len(m))
    cons = pd.DataFrame({"residential": res, "commercial": com, "industrial": ind,
                         "electric_power": elec}, index=pd.Index(m.index, name="month"))
    cons.insert(0, "total", cons.sum(axis=1))
    cons.to_csv(raw / "ma_consumption.csv")

    # Daily Henry Hub: mean-reverting log price with weather and storage effects
    d = pd.bdate_range("2001-01-02", "2026-10-16")
    lp = np.zeros(len(d))
    lp[0] = np.log(3.0)
    for i in range(1, len(d)):
        lp[i] = lp[i - 1] + 0.004 * (np.log(3.5) - lp[i - 1]) + rng.normal(0, 0.03)
    pd.DataFrame({"value": np.exp(lp)}, index=pd.Index(d, name="period")).to_csv(raw / "henry_hub.csv")

    # Weekly storage (Fridays), seasonal
    w = pd.date_range("2001-01-05", "2026-10-09", freq="W-FRI")
    woy = w.isocalendar().week.values.astype(float)
    st = 2600 + 1100 * np.sin(2 * np.pi * (woy - 20) / 52) + np.cumsum(rng.normal(0, 15, len(w)))
    pd.DataFrame({"value": st}, index=pd.Index(w, name="period")).to_csv(raw / "storage_l48.csv")
    pd.DataFrame({"value": st * 0.3}, index=pd.Index(w, name="period")).to_csv(raw / "storage_east.csv")

    # Monthly LNG exports, MMcf (near zero until 2016, then ramping)
    mm = pd.date_range("2001-01-01", "2026-07-01", freq="MS")
    ramp = np.clip((mm.year + mm.month / 12 - 2016) / 8, 0, 1)
    lng = 3000 + 450000 * ramp + rng.normal(0, 8000, len(mm)).clip(-2000)
    pd.DataFrame({"value": np.abs(lng)}, index=pd.Index(mm, name="period")).to_csv(raw / "lng_exports.csv")
    # Risk series (same file layout as data.pull_risk)
    dd = pd.date_range("2001-01-01", "2026-10-16", freq="D")
    gpr = np.exp(np.log(100) + np.cumsum(rng.normal(0, 0.02, len(dd))) * 0.3 + rng.normal(0, 0.3, len(dd)))
    epu = np.exp(np.log(120) + rng.normal(0, 0.4, len(dd)))
    brent = np.exp(np.log(70) + np.cumsum(rng.normal(0, 0.02, len(dd))) * 0.5)
    for name, v, idx in (("gpr", gpr, dd), ("epu_daily", epu, dd), ("brent", brent, dd)):
        pd.DataFrame({"value": v}, index=pd.Index(idx, name="period")).to_csv(raw / f"risk_{name}.csv")
    eu = 6 + 4 * np.abs(np.sin(np.arange(len(mm)) / 30)) + rng.normal(0, 1, len(mm))
    tpu = np.exp(np.log(100) + rng.normal(0, 0.5, len(mm)))
    pd.DataFrame({"value": eu}, index=pd.Index(mm, name="period")).to_csv(raw / "risk_eu_gas.csv")
    pd.DataFrame({"value": tpu}, index=pd.Index(mm, name="period")).to_csv(raw / "risk_trade_epu.csv")
    print(f"Synthetic fixture written to {raw}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
