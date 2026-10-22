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
    # Other New England states: scaled copies with their own weather noise
    for st, scale in (("ct", 0.55), ("ri", 0.2), ("nh", 0.15), ("me", 0.08), ("vt", 0.05)):
        w = weather(amp=23, mean=48)
        w.to_csv(raw / f"weather_{st}.csv")
        mm_ = w[["hdd", "cdd"]].resample("MS").sum().loc[:"2026-07-01"]
        d_ = mm_.index.days_in_month.values
        sc = pd.DataFrame({
            "residential": scale * (90 * d_ + 22 * mm_["hdd"].values) + rng.normal(0, 300 * scale, len(mm_)),
            "commercial": scale * (60 * d_ + 13 * mm_["hdd"].values) + rng.normal(0, 200 * scale, len(mm_)),
            "industrial": scale * 140 * d_ + rng.normal(0, 100 * scale, len(mm_)),
        }, index=pd.Index(mm_.index, name="month"))
        sc.insert(0, "total", sc.sum(axis=1))
        sc.to_csv(raw / f"{st}_consumption.csv")
    # EIA-176 company file (same layout as src.companies.download)
    cos = [("MA", "17601391MA", "BOSTON GAS CO DBA NATIONAL GRID", 59), ("MA", "17615111MA", "NSTAR GAS COMPANY", 19),
           ("MA", "17680531MA", "EVERSOURCE GAS CO OF MA", 23), ("MA", "17601093MA", "BERKSHIRE GAS COMPANY THE", 3),
           ("CT", "17602792CT", "YANKEE GAS SVC CO", 16), ("CT", "17619018CT", "CONNECTICUT NAT GAS CORP", 16),
           ("CT", "17619865CT", "SOUTHERN CONNECTICUT GAS COMPANY", 15), ("RI", "17681346RI", "THE NARRAGANSETT ELECCO DBA RIENERGY", 18),
           ("NH", "17675803NH", "LIBERTY UTILITIES DBA ENERGY NORTH", 5), ("NH", "17600223NH", "NORTHERN UTILITIES INC", 2),
           ("ME", "17617891ME", "NORTHERN UTILITIES DBA UNITIL CORP", 1.5), ("ME", "17676620ME", "SUMMIT NATURAL GAS OF MAINE INC", 0.4),
           ("VT", "17618610VT", "VERMONT GAS SYSTEMS INC", 3.5)]
    rows = []
    for st_, cid, nm, bcf in cos:
        for y in range(2010, 2025):
            g = 1 + 0.01 * (y - 2010) + rng.normal(0, 0.03)
            vals = {"res_volume": bcf * 1e6 * g, "com_volume": bcf * 0.9e6 * g, "ind_volume": bcf * 0.3e6 * g,
                    "res_sales_customers": bcf * 14e3, "res_sales_volume": bcf * 0.98e6 * g,
                    "res_sales_revenue": bcf * 0.98e6 * g * (14 + 0.5 * (y - 2010)) * (1 + rng.normal(0, 0.05)),
                    "com_sales_customers": bcf * 1.5e3, "com_sales_volume": bcf * 0.5e6 * g,
                    "com_sales_revenue": bcf * 0.5e6 * g * (11 + 0.4 * (y - 2010)),
                    "res_transport_customers": bcf * 100, "com_transport_customers": bcf * 150,
                    "citygate_sales_volume": bcf * 1.5e6 * g, "citygate_transport_volume": bcf * 0.7e6 * g,
                    "lng_withdrawals": bcf * 0.03e6, "total_supply": bcf * 2.3e6 * g,
                    "unaccounted_for": bcf * 0.02e6 * rng.normal(0, 1), "heat_content_btu_cf": 1030}
            rows += [{"state": st_, "company_id": cid, "eia_name": nm, "item": k, "year": y, "value": v}
                     for k, v in vals.items()]
    pd.DataFrame(rows).to_csv(raw / "ne_companies_eia176.csv", index=False)
    print(f"Synthetic fixture written to {raw}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
