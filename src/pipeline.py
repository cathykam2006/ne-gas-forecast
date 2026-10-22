"""End-to-end run: raw data -> features -> backtests -> forecasts -> app files.

    python -m src.pipeline            # uses data already in data/raw/
    python -m src.pipeline --pull     # pull fresh data first (needs EIA_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from . import config, data, features
from . import demand_backtest as dbt
from . import demand_models as dm
from . import price_models as pm


def run(pull: bool = False, quick: bool = False) -> dict:
    if pull:
        data.pull_all()
    raw = data.load_raw()
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Demand ----------------
    print("\n[Demand] building features")
    dfd = features.build_demand_frame(raw["ma_consumption"], raw["weather_ma"])
    dfd.to_csv(out / "demand_frame.csv")

    step = 6 if quick else 3
    print("[Demand] backtest with normal weather (ex-ante)")
    bt = dbt.rolling_backtest(dfd, step=step, weather="normal")
    print("[Demand] backtest with actual weather (ex-post)")
    bt_actual = dbt.rolling_backtest(dfd, step=step, weather="actual", verbose=False)
    bt.to_csv(out / "demand_backtest.csv", index=False)
    bt_actual.to_csv(out / "demand_backtest_actualwx.csv", index=False)

    scores = dbt.score(bt)
    scores_actual = dbt.score(bt_actual)
    scores.to_csv(out / "demand_scores.csv")
    scores_actual.to_csv(out / "demand_scores_actualwx.csv")
    dbt.mape_by_horizon(bt).to_csv(out / "demand_mape_by_h.csv")
    print(scores.round(2).to_string())

    best_name = scores.index[0] if scores.index[0] != "Seasonal naive" else scores.index[1]
    best_cls = next(M for M in dm.ALL_MODELS if M.name == best_name)
    print(f"[Demand] best model: {best_name}; scenario forecast")
    fc = dbt.scenario_forecast(dfd, best_cls, bt=bt)
    fc.to_csv(out / "demand_forecast.csv")

    wn = dbt.weather_normalize(dfd)
    wn.to_csv(out / "demand_weather_normalized.csv")
    normals = features.monthly_hdd_normals(dfd["hdd"])
    normals.to_csv(out / "hdd_normals.csv")

    # ---------------- Price ----------------
    print("\n[Price] building features")
    dfp = features.build_price_frame(raw["henry_hub"], raw["storage_l48"],
                                     raw["lng_exports"], raw["weather_us"],
                                     risk=raw.get("risk"))
    dfp.to_csv(out / "price_frame.csv")
    risk_cols = features.risk_features_in(dfp)
    print(f"[Price] geopolitical/trade-risk features available: {risk_cols or 'none'}")

    price_scores, latest, ablation = {}, [], []
    for h in (1, 4):
        print(f"[Price] walk-forward h={h}")
        wf = pm.walk_forward(dfp, h=h, refit_every=26 if quick else 13)
        wf.to_csv(out / f"price_wf_h{h}.csv", index=False)
        sc = pm.score_price(wf, h=h)
        sc.to_csv(out / f"price_scores_h{h}.csv")
        price_scores[h] = sc
        print(sc.round(4).to_string())
        if risk_cols:
            ablation.append(pm.compare_models(wf, pm.RISK_RIDGE, "Ridge (all features)", h))
            ablation.append(pm.compare_models(wf, pm.RISK_LGBM, "LightGBM (all features)", h))
        for m in pm.model_zoo(risk_cols):
            f = pm.latest_forecast(dfp, m, h)
            f["model"] = m.name
            latest.append(f)
    pd.DataFrame(latest).to_csv(out / "price_latest_forecasts.csv", index=False)
    if ablation:
        ab = pd.DataFrame(ablation)
        ab.to_csv(out / "price_risk_ablation.csv", index=False)
        print("\n[Price] Do geopolitical/trade-risk features help? (negative = better)")
        print(ab.round(4).to_string(index=False))

    # SHAP on the full-history LightGBM (with risk features when available), h=4
    cols = features.PRICE_FEATURES + risk_cols
    lg = pm.LGBMPrice(cols)
    tr = dfp.dropna(subset=cols + ["y_h4"])
    lg.fit(tr, tr["y_h4"])
    lg.shap(dfp.dropna(subset=cols)).to_csv(out / "price_shap_h4.csv")

    meta = {
        "run_at": datetime.now().isoformat(timespec="minutes"),
        "demand_data_through": f"{dfd.index[-1]:%Y-%m}",
        "price_data_through": f"{dfp['price'].dropna().index[-1]:%Y-%m-%d}",
        "best_demand_model": best_name,
        "mmcf_per_hdd": round(float(wn.attrs["mmcf_per_hdd"]), 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDone:", meta)
    return {"demand_scores": scores, "price_scores": price_scores, "meta": meta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="pull fresh data first")
    ap.add_argument("--quick", action="store_true", help="fewer backtest origins")
    a = ap.parse_args()
    run(pull=a.pull, quick=a.quick)
