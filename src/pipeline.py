"""End-to-end run: raw data -> features -> backtests -> forecasts -> app files.

    python -m src.pipeline            # uses data already in data/raw/
    python -m src.pipeline --pull     # pull fresh data first (needs EIA_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from . import companies, config, data, features, nymex, region, tariffs
from . import demand_backtest as dbt
from . import demand_models as dm
from . import price_models as pm


def run(pull: bool = False, quick: bool = False) -> dict:
    if pull:
        data.pull_all()
        nymex.pull()                      # never raises: a failed download keeps the old files
    raw = data.load_raw()
    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Demand ----------------
    print("\n[Demand] building features")
    dfd = features.build_demand_frame(raw["ma_consumption"], raw["weather_ma"],
                                     target=config.DEMAND_TARGET)
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

    # ---------------- New England states ----------------
    if len(raw.get("states", {})) > 1:
        print("\n[New England] state forecasts")
        rs, rf, rh = region.run_region(raw["states"], ma_result=(dfd, scores, fc), step=step)
        rs.to_csv(out / "region_scores.csv", index=False)
        rf.to_csv(out / "region_forecasts.csv", index=False)
        rh.to_csv(out / "region_history.csv", index=False)
        best = rs[rs["model"] != "Seasonal naive"].sort_values("MAPE %").groupby("state").first()
        print(best[["model", "MAPE %", "Skill vs naive %"]].round(2).to_string())

        print("\n[New England] company breakdown (EIA Form 176)")
        companies.run(region_fc=rf, fetch_data=pull)

    # ---------------- MA GAF / LDAF ----------------
    wb, page = config.RAW_DIR / tariffs.WORKBOOK_FILE, config.RAW_DIR / tariffs.PAGE_FILE
    if pull or not (wb.exists() or page.exists()):
        print("\n[MA tariffs] downloading DPU GAF/LDAF files")
        tariffs.download()
    if wb.exists() or page.exists():
        print("[MA tariffs] building GAF/LDAF tables")
        tariffs.run(fetch=False)
        tariffs.henry_hub_per_therm(raw["henry_hub"]).to_csv(out / "henry_hub_monthly_therm.csv")

    # ---------------- Price ----------------
    print("\n[Price] building features")
    dfp = features.build_price_frame(raw["henry_hub"], raw["storage_l48"],
                                     raw["lng_exports"], raw["weather_us"])
    dfp.to_csv(out / "price_frame.csv")

    price_scores, latest = {}, []
    front = nymex.load_front()
    for h in (1, 4):
        print(f"[Price] walk-forward h={h}")
        wf = pm.walk_forward(dfp, h=h, refit_every=26 if quick else 13)
        wf = nymex.add_market_benchmark(wf, front, h)   # the futures market's own forecast
        wf.to_csv(out / f"price_wf_h{h}.csv", index=False)
        sc = pm.score_price(wf, h=h)
        sc.to_csv(out / f"price_scores_h{h}.csv")
        if nymex.MARKET in set(wf["model"]):      # each model against the futures market (DM test)
            vm = [pm.compare_models(wf, m, nymex.MARKET, h) for m in sc.index
                  if m not in ("Random walk", nymex.MARKET)]
            pd.DataFrame(vm).to_csv(out / f"price_vs_market_h{h}.csv", index=False)
        price_scores[h] = sc
        print(sc.round(4).to_string())
        for m in pm.model_zoo():
            f = pm.latest_forecast(dfp, m, h)
            f["model"] = m.name
            latest.append(f)
    last = dfp["price"].dropna()
    latest += nymex.latest_market_forecast(front, float(last.iloc[-1]), last.index[-1])
    pd.DataFrame(latest).to_csv(out / "price_latest_forecasts.csv", index=False)

    print("\n[NYMEX] futures history and forward strip")
    nymex_meta = nymex.build(out)

    # SHAP on the full-history LightGBM, h=4
    cols = features.PRICE_FEATURES
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
        **nymex_meta,
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
