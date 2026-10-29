"""Data loader for the Massachusetts GAF page: DPU-filed GAF and LDAF by company and rate category
(monthly, $/therm), Henry Hub monthly average, and the NYMEX strip for the coming peak period."""
import json, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
P = REPO / "data" / "processed"

t = pd.read_csv(P / "ma_gaf_ldaf_monthly.csv", parse_dates=["month"])
r4 = lambda v: None if pd.isna(v) else round(float(v), 4)
t = t.dropna(subset=["gaf_therm"])
companies, categories = sorted(t["company"].unique()), sorted(t["category"].unique())
ci, ki = {c: i for i, c in enumerate(companies)}, {c: i for i, c in enumerate(categories)}
# compact rows: [month, company index, category index, GAF, LDAF] in $/therm
rows = [[r.month.strftime("%Y-%m"), ci[r.company], ki[r.category], r4(r.gaf_therm), r4(r.ldaf_therm)]
        for r in t.itertuples()]
hh = pd.read_csv(P / "henry_hub_monthly_therm.csv", parse_dates=["period"]).dropna()
nm = json.loads((P / "nymex_meta.json").read_text()) if (P / "nymex_meta.json").exists() else {}
strip = pd.read_csv(P / "nymex_strip_latest.csv") if (P / "nymex_strip_latest.csv").exists() else pd.DataFrame()
out = {
    "companies": companies, "categories": categories,
    "rows": rows,
    "hh": [{"m": r.period.strftime("%Y-%m"), "v": r4(r.henry_hub_therm)} for r in hh.itertuples()],
    "strip": [{"m": r.contract, "therm": r4(r.settle_therm)} for r in strip.itertuples()],
    "strip_as_of": nm.get("nymex_strip_as_of"),
    "latest_month": t["month"].max().strftime("%Y-%m"),
}
json.dump(out, sys.stdout, separators=(",", ":"))
