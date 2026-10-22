"""Weekly Henry Hub models and walk-forward evaluation.

Target: log change in the weekly average Henry Hub spot price h weeks ahead.
The benchmark is a random walk (no change), which is famously hard to beat
for commodity prices; every result is reported relative to it, with a
Diebold-Mariano test so a small edge isn't mistaken for skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import PRICE_FEATURES, risk_features_in

FUNDAMENTALS = ["storage_surplus_pct", "injection_vs_norm_bcf", "dd_anom",
                "dd_anom_4", "lng_yoy_pct", "lng_bcfd", "sin_w", "cos_w"]
MOMENTUM = ["ret_1", "ret_4", "ret_13", "vol_8"]


class RandomWalk:
    name = "Random walk"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RidgeModel:
    def __init__(self, cols, name):
        self.cols, self.name = cols, name

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 3, 20)))
        self.m.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])


class LGBMPrice:
    def __init__(self, cols=PRICE_FEATURES, name="LightGBM (all features)"):
        self.cols, self.name = list(cols), name

    def fit(self, X, y):
        import lightgbm as lgb

        self.m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02,
                                   num_leaves=7, min_child_samples=30,
                                   subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.8, reg_lambda=5.0,
                                   random_state=config.RANDOM_STATE, verbose=-1)
        self.m.fit(X[self.cols], y)
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols])

    def shap(self, X):
        """Per-row SHAP contributions (LightGBM's native TreeSHAP)."""
        c = self.m.predict(X[self.cols], pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.cols, index=X.index)


RISK_RIDGE = "Ridge (all + geopolitical/trade risk)"
RISK_LGBM = "LightGBM (all + geopolitical/trade risk)"


def model_zoo(risk_cols: list[str] | None = None):
    zoo = [
        RandomWalk(),
        RidgeModel(MOMENTUM, "Ridge (momentum only)"),
        RidgeModel(FUNDAMENTALS, "Ridge (fundamentals)"),
        RidgeModel(PRICE_FEATURES, "Ridge (all features)"),
        LGBMPrice(),
    ]
    if risk_cols:
        zoo += [RidgeModel(PRICE_FEATURES + risk_cols, RISK_RIDGE),
                LGBMPrice(PRICE_FEATURES + risk_cols, RISK_LGBM)]
    return zoo


def walk_forward(df: pd.DataFrame, h: int = 1, first_test: str = "2015-01-01",
                 refit_every: int = 13, min_train: int = 260,
                 models_fn=model_zoo, use_risk: bool = True) -> pd.DataFrame:
    """Expanding-window walk-forward test.

    At each refit date the models are trained only on rows whose target is
    already known (target date <= refit date), then predict the next
    `refit_every` weeks out-of-sample.
    """
    target = f"y_h{h}"
    risk_cols = risk_features_in(df) if use_risk else []
    # Every model is scored on exactly the same weeks, so comparisons are fair.
    data = df.dropna(subset=PRICE_FEATURES + risk_cols + ["log_p"]).copy()
    test_idx = data.index[data.index >= pd.Timestamp(first_test)]
    test_idx = test_idx[data.loc[test_idx, target].notna()]
    rows = []
    for start in range(0, len(test_idx), refit_every):
        block = test_idx[start:start + refit_every]
        cutoff = block[0]
        # only targets realized by the cutoff: drop the last h rows
        train = data[data.index < cutoff].dropna(subset=[target]).iloc[:-h or None]
        if len(train) < min_train:
            continue
        for m in models_fn(risk_cols):
            m.fit(train, train[target])
            pred = m.predict(data.loc[block])
            for d, p in zip(block, pred):
                rows.append({"date": d, "model": m.name, "pred": p,
                             "y": data.at[d, target], "price": data.at[d, "price"]})
    out = pd.DataFrame(rows)
    out["price_fcst"] = out["price"] * np.exp(out["pred"])
    out["price_actual"] = out["price"] * np.exp(out["y"])
    return out


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test on squared errors with Newey-West variance (lag h-1).

    Negative statistic = model beats benchmark. Returns (stat, two-sided p).
    """
    d = e_model ** 2 - e_bench ** 2
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    stat = dbar / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return float(stat), float(p)


def score_price(wf: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    rw = wf[wf["model"] == "Random walk"].set_index("date")
    res = []
    for name, g in wf.groupby("model"):
        g = g.set_index("date").loc[rw.index]
        err = g["pred"] - g["y"]
        perr = g["price_fcst"] - g["price_actual"]
        row = {
            "model": name,
            "RMSE log-return": np.sqrt((err ** 2).mean()),
            "RMSE $/MMBtu": np.sqrt((perr ** 2).mean()),
            "MAE $/MMBtu": perr.abs().mean(),
        }
        if name != "Random walk":
            nz = g["y"] != 0
            row["Direction hit %"] = (np.sign(g["pred"][nz]) == np.sign(g["y"][nz])).mean() * 100
            stat, p = diebold_mariano(err.values, (rw["pred"] - rw["y"]).values, h)
            row["DM stat"], row["DM p-value"] = stat, p
        res.append(row)
    out = pd.DataFrame(res).set_index("model")
    out["RMSE vs RW %"] = (out["RMSE log-return"] / out.at["Random walk", "RMSE log-return"] - 1) * 100
    return out.sort_values("RMSE log-return")


def compare_models(wf: pd.DataFrame, challenger: str, base: str, h: int) -> dict:
    """Diebold-Mariano test of one model against another (not the random walk).

    Used for the ablation question: does adding geopolitical/trade-risk
    features improve on the same model without them?
    """
    a = wf[wf["model"] == challenger].set_index("date")
    b = wf[wf["model"] == base].set_index("date").loc[a.index]
    ea, eb = (a["pred"] - a["y"]).values, (b["pred"] - b["y"]).values
    stat, p = diebold_mariano(ea, eb, h)
    return {"challenger": challenger, "base": base, "h": h,
            "RMSE change %": (np.sqrt(np.mean(ea ** 2)) / np.sqrt(np.mean(eb ** 2)) - 1) * 100,
            "DM stat": stat, "DM p-value": p}


def latest_forecast(df: pd.DataFrame, model, h: int) -> dict:
    """Fit on everything known now and forecast h weeks ahead."""
    target = f"y_h{h}"
    data = df.dropna(subset=list(dict.fromkeys(PRICE_FEATURES + getattr(model, "cols", []) + ["log_p"])))
    train = data.dropna(subset=[target])
    model.fit(train, train[target])
    x_now = data.iloc[[-1]]
    r = float(model.predict(x_now)[0])
    p0 = float(x_now["price"].iloc[0])
    return {"as_of": x_now.index[0], "price_now": p0, "h": h,
            "pred_logret": r, "price_fcst": p0 * np.exp(r)}
