"""Monthly Massachusetts gas demand models.

Every model exposes the same interface:
    model.fit(train: DataFrame) -> self
    model.predict(future: DataFrame) -> ndarray
`future` holds the exogenous columns (hdd, cdd, days, calendar, t, y_lag12)
for the months to forecast; the target column `y` is never read from it.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd
import statsmodels.api as sm

from . import config

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)



class SeasonalNaive:
    """Same month last year. The benchmark every model must beat."""
    name = "Seasonal naive"

    def fit(self, train):
        return self

    def predict(self, future):
        # y_lag12 for a month up to 12 months ahead is already observed.
        return future["y_lag12"].values


class HDDRegression:
    """Utility-style weather regression on the most recent 10 years:

        y = days * (base + seasonal terms) + b * HDD + c * CDD

    `b` is the weather sensitivity (MMcf per heating degree day). Seasonal
    terms scale with days in the month and capture non-weather patterns
    (e.g. slower response in shoulder months). The 10-year window lets the
    heating slope and base load follow structural change such as customer
    growth, efficiency and EIA reclassifications between sectors, which a
    single all-history fit cannot (backtested in notebook 03).
    """
    name = "HDD regression"
    window_years = 10
    cols = ["days", "hdd", "cdd", "s1", "c1", "s2", "c2"]

    def _X(self, df):
        X = df[["days", "hdd", "cdd"]].copy()
        for k in (1, 2):
            X[f"s{k}"] = df[f"sin{k}"] * df["days"]
            X[f"c{k}"] = df[f"cos{k}"] * df["days"]
        return X[self.cols]

    def fit(self, train):
        tr = train[train.index > train.index[-1] - pd.DateOffset(years=self.window_years)]
        self.res = sm.OLS(tr["y"], self._X(tr)).fit()
        return self

    def predict(self, future):
        return self.res.predict(self._X(future)).values


class SARIMAXModel:
    """Seasonal ARIMA (regression with SARIMA errors) on volume, with HDD,
    CDD and days-in-month as exogenous regressors."""
    name = "SARIMAX"
    exog = ["hdd", "cdd", "days"]

    def fit(self, train):
        self.train_end = train.index[-1]
        self.model = sm.tsa.statespace.SARIMAX(
            train["y"], exog=train[self.exog],
            order=(1, 0, 0), seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=False, maxiter=200)
        return self

    def predict(self, future):
        fc = self.model.get_forecast(steps=len(future), exog=future[self.exog])
        return fc.predicted_mean.values


class ProphetModel:
    """Prophet with yearly seasonality and HDD/CDD regressors, fit on the
    most recent 10 years (on full history its trend extrapolation overshoots
    badly; see notebook 03)."""
    name = "Prophet"
    window_years = 10

    def fit(self, train):
        from prophet import Prophet

        train = train[train.index > train.index[-1] - pd.DateOffset(years=self.window_years)]

        self.m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                         daily_seasonality=False, seasonality_mode="additive")
        self.m.add_regressor("hdd")
        self.m.add_regressor("cdd")
        d = pd.DataFrame({"ds": train.index, "y": train["y"].values,
                          "hdd": train["hdd"].values, "cdd": train["cdd"].values})
        self.m.fit(d)
        return self

    def predict(self, future):
        d = pd.DataFrame({"ds": future.index, "hdd": future["hdd"].values,
                          "cdd": future["cdd"].values})
        return self.m.predict(d)["yhat"].values


class LGBMModel:
    """Gradient boosting on weather, calendar and same-month-last-year volume."""
    name = "LightGBM"
    cols = ["hdd", "cdd", "days", "month", "t", "y_lag12"]

    def fit(self, train):
        import lightgbm as lgb

        self.m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                                   num_leaves=8, min_child_samples=8,
                                   subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.9,
                                   random_state=config.RANDOM_STATE, verbose=-1)
        tr = train.dropna(subset=self.cols)
        self.m.fit(tr[self.cols], tr["y"])
        return self

    def predict(self, future):
        return self.m.predict(future[self.cols])


class HybridModel:
    """HDD regression for the physics, LightGBM on its residuals.

    The regression extrapolates trend and weather sensibly; the boosted
    trees pick up nonlinear effects (e.g. saturation on extreme cold days,
    shifts in electric-power burn) the linear model misses.
    """
    name = "Hybrid (HDD reg + LightGBM)"
    resid_cols = ["hdd", "cdd", "month", "t"]

    def fit(self, train):
        import lightgbm as lgb

        self.base = HDDRegression().fit(train)
        resid = train["y"] - self.base.predict(train)
        self.m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03,
                                   num_leaves=6, min_child_samples=10,
                                   random_state=config.RANDOM_STATE, verbose=-1)
        self.m.fit(train[self.resid_cols], resid)
        return self

    def predict(self, future):
        return self.base.predict(future) + self.m.predict(future[self.resid_cols])


ALL_MODELS = [SeasonalNaive, HDDRegression, SARIMAXModel, ProphetModel,
              LGBMModel, HybridModel]
