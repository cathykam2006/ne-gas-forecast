"""Data loader for the Henry Hub page: reads ../data/processed (written by src/pipeline.py and
src/nymex.py) and prints one compact JSON document. Observable Framework runs it at build time."""
import json, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src import config  # noqa: E402

P = str(REPO / "data" / "processed") + "/"
out = {}
meta = json.load(open(P + "meta.json")); nm = json.load(open(P + "nymex_meta.json"))
lat = pd.read_csv(P + "price_latest_forecasts.csv", parse_dates=["as_of"])
r3 = lambda v: None if pd.isna(v) else round(float(v), 4)
for h in (1, 4):
    sc = pd.read_csv(P + f"price_scores_h{h}.csv", index_col="model")
    best = sc.drop(index=["Random walk", "NYMEX futures (market)"], errors="ignore")["RMSE log-return"].idxmin()
    wf = pd.read_csv(P + f"price_wf_h{h}.csv", parse_dates=["date"])
    g = wf[wf.model == best].set_index("date"); rw = wf[wf.model == "Random walk"].set_index("date")
    shift = pd.Timedelta(weeks=h)
    rows = [{"t": (d + shift).strftime("%Y-%m-%d"), "made": d.strftime("%Y-%m-%d"),
             "actual": r3(rw.at[d, "price_actual"]), "fcst": r3(g.at[d, "price_fcst"])}
            for d in g.index if d in rw.index]
    lf = lat[(lat.h == h) & (lat.model == best)].iloc[0]
    out[f"h{h}"] = {
        "best": best, "rows": rows,
        "latest": {"as_of": lf.as_of.strftime("%Y-%m-%d"), "price_now": r3(lf.price_now),
                   "fcst": r3(lf.price_fcst), "target": (lf.as_of + shift).strftime("%Y-%m-%d")},
        "scores": [{"model": m, "rmse": r3(s["RMSE $/MMBtu"]), "vs_rw": r3(s["RMSE vs RW %"]),
                    "hit": r3(s["Direction hit %"]), "p": r3(s["DM p-value"])} for m, s in sc.iterrows()],
    }
nx = pd.read_csv(P + "nymex_daily.csv", index_col=0, parse_dates=True)
nx = nx.loc[nx.index.max() - pd.DateOffset(years=2):]
out["front"] = [{"d": d.strftime("%Y-%m-%d"), "front": r3(r.nymex_front), "spot": r3(r.henry_hub_spot)}
                for d, r in nx.iterrows()]
st = pd.read_csv(P + "nymex_strip_latest.csv")
out["strip"] = [{"m": r.contract, "p": r3(r.settle)} for r in st.itertuples()]
out["strip_as_of"] = nm["nymex_strip_as_of"]; out["front_last"] = nm["nymex_front_last"]
pf = pd.read_csv(P + "price_frame.csv", parse_dates=["period"]).set_index("period")
s = pf["storage_surplus_pct"].dropna().iloc[-156:]
out["storage"] = [{"d": d.strftime("%Y-%m-%d"), "v": round(float(v), 2)} for d, v in s.items()]
sh = pd.read_csv(P + "price_shap_h4.csv", index_col="period").abs().mean().sort_values(ascending=False)
out["shap"] = [{"f": k, "v": round(float(v) * 100, 3)} for k, v in sh.items()]
out["events"] = [{"d": d, "label": l} for d, l in config.EVENTS]
out["periods"] = [{"a": a, "b": b, "label": l} for a, b, l in config.EVENT_PERIODS]
out["notes"] = config.EVENT_NOTES
out["price_data_through"] = meta["price_data_through"]
json.dump(out, sys.stdout, separators=(",", ":"))
