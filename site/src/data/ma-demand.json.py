"""Data loader for the Massachusetts demand page: recent monthly utility demand and the 12-month
forecast under normal, cold and warm weather (Bcf), plus backtest accuracy."""
import json, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src import outlook  # noqa: E402

P = REPO / "data" / "processed"
do = outlook.demand_outlook(P)
r2 = lambda v: None if pd.isna(v) else round(float(v), 3)
sc = pd.read_csv(P / "demand_scores.csv", index_col=0)
meta = json.loads((P / "meta.json").read_text())
fc = do["forecast"]
out = {
    "model": do["model"], "last_actual": do["last_actual"].strftime("%Y-%m"),
    "history": [{"m": d.strftime("%Y-%m"), "v": r2(v)} for d, v in do["history"].items()],
    "forecast": [{"m": d.strftime("%Y-%m"), **{c: r2(r[c]) for c in fc.columns}} for d, r in fc.iterrows()],
    "mape": r2(sc.at[do["model"], "MAPE %"]) if do["model"] in sc.index and "MAPE %" in sc else None,
    "naive_mape": r2(sc.at["Seasonal naive", "MAPE %"]) if "Seasonal naive" in sc.index and "MAPE %" in sc else None,
    "mmcf_per_hdd": meta.get("mmcf_per_hdd"),
}
json.dump(out, sys.stdout, separators=(",", ":"))
