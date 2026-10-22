"""New England: state-level demand forecasts and the company breakdown.

Each state gets the same tested approach as Massachusetts (utility load
target, weather normals, rolling-origin backtest); the best model per state
produces the 12-month scenario forecast. Company forecasts split each state's
monthly forecast by the company's recent share of utility deliveries
(EIA-176), see `company_split`.
"""
from __future__ import annotations

import pandas as pd

from . import config, features
from . import demand_backtest as dbt
from . import demand_models as dm

# Prophet is slow and underperformed on Massachusetts, so the other states use
# the remaining models.
STATE_MODELS = [dm.SeasonalNaive, dm.HDDRegression, dm.SARIMAXModel,
                dm.LGBMModel, dm.HybridModel]


def state_forecast(cons: pd.DataFrame, weather: pd.DataFrame, models=STATE_MODELS,
                   step: int = 3, bt: pd.DataFrame | None = None):
    """Backtest, choose the best model, and forecast one state."""
    dfd = features.build_demand_frame(cons, weather, target=config.DEMAND_TARGET)
    if bt is None:
        bt = dbt.rolling_backtest(dfd, models=models, step=step, verbose=False)
    sc = dbt.score(bt)
    challengers = [m for m in sc.index if m != "Seasonal naive"]
    best = challengers[0]
    best_cls = next(M for M in dm.ALL_MODELS if M.name == best)
    fc = dbt.scenario_forecast(dfd, best_cls, bt=bt)
    fc["model"] = best
    return dfd, sc, fc


def run_region(states: dict, ma_result: tuple | None = None, step: int = 3,
               verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forecast every state present.

    Returns (scores, forecasts, history) in long format with a `state` column.
    `ma_result` = (dfd, scores, forecast) already computed for Massachusetts.
    """
    scores, fcs, hist = [], [], []
    for state, (cons, wx) in states.items():
        if state == "MA" and ma_result is not None:
            dfd, sc, fc = ma_result
            fc = fc.copy()
            fc["model"] = sc.index[0] if sc.index[0] != "Seasonal naive" else sc.index[1]
        else:
            if verbose:
                print(f"  {config.STATES[state]['name']} ...")
            dfd, sc, fc = state_forecast(cons, wx, step=step)
        scores.append(sc.reset_index().assign(state=state))
        fcs.append(fc.reset_index().assign(state=state))
        hist.append(dfd[["y", "hdd"]].reset_index().rename(columns={dfd.index.name or "index": "month"})
                    .assign(state=state))
    return pd.concat(scores), pd.concat(fcs), pd.concat(hist)
