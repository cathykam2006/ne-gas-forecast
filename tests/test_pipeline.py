"""Leakage and sanity tests, run on the synthetic fixture.

    pytest -q tests/
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent


@pytest.fixture(scope="session")
def raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    subprocess.run([sys.executable, str(HERE / "make_fixture.py"), str(root)], check=True)
    os.environ["GAS_DATA_DIR"] = str(root)
    from src import config, data
    importlib.reload(config)
    importlib.reload(data)
    return data.load_raw()


def test_storage_is_lagged_one_week(raw):
    from src import features
    df = features.build_price_frame(raw["henry_hub"], raw["storage_l48"],
                                    raw["lng_exports"], raw["weather_us"])
    st = raw["storage_l48"].resample("W-FRI").last()
    d = df.index[600]
    assert np.isclose(df.at[d, "storage_bcf"], st.loc[d - pd.Timedelta(weeks=1)])


def test_lng_respects_publication_lag(raw):
    from src import features
    df = features.build_price_frame(raw["henry_hub"], raw["storage_l48"],
                                    raw["lng_exports"], raw["weather_us"])
    lng = raw["lng_exports"]
    d = pd.Timestamp("2020-03-06")  # early March: latest known month is December
    expected = lng.loc["2019-12-01"] / 31 / 1000
    assert np.isclose(df.at[d, "lng_bcfd"], expected)


def test_targets_are_future_changes(raw):
    from src import features
    df = features.build_price_frame(raw["henry_hub"], raw["storage_l48"],
                                    raw["lng_exports"], raw["weather_us"])
    i = 500
    assert np.isclose(df["y_h4"].iloc[i], df["log_p"].iloc[i + 4] - df["log_p"].iloc[i])


def test_demand_backtest_only_uses_past(raw):
    from src import demand_backtest as dbt, demand_models as dm, features
    dfd = features.build_demand_frame(raw["ma_consumption"], raw["weather_ma"])
    seen = {}

    class Spy(dm.HDDRegression):
        name = "spy"

        def fit(self, train):
            seen.setdefault("max_train", []).append(train.index.max())
            return super().fit(train)

    bt = dbt.rolling_backtest(dfd, models=[Spy], first_origin="2020-01-01",
                              step=12, verbose=False)
    for origin, mx in zip(sorted(bt["origin"].unique()), seen["max_train"]):
        assert mx <= origin
    assert (bt["date"] > bt["origin"]).all()


def test_scenarios_are_ordered(raw):
    from src import demand_backtest as dbt, demand_models as dm, features
    dfd = features.build_demand_frame(raw["ma_consumption"], raw["weather_ma"])
    fc = dbt.scenario_forecast(dfd, dm.HDDRegression)
    winter = fc[fc.index.month.isin([12, 1, 2])]
    assert (winter["cold"] > winter["normal"]).all()
    assert (winter["warm"] < winter["normal"]).all()


def test_risk_features_respect_publication_lags(raw):
    from src import features
    df = features.build_price_frame(raw["henry_hub"], raw["storage_l48"], raw["lng_exports"],
                                    raw["weather_us"], risk=raw["risk"])
    assert set(features.risk_features_in(df)) == set(features.RISK_FEATURES_ALL)
    # EU-Henry Hub spread for Dec 2019 can't be used before March 2020
    d_before, d_after = pd.Timestamp("2020-02-28"), pd.Timestamp("2020-03-06")
    hh = raw["henry_hub"].resample("MS").mean()
    eu = raw["risk"]["eu_gas"]
    dec = eu.loc["2019-12-01"] - hh.loc["2019-12-01"]
    assert np.isclose(df.at[d_after, "eu_hh_spread"], dec)
    assert not np.isclose(df.at[d_before, "eu_hh_spread"], dec)
    # Geopolitical risk index is lagged one week
    g = np.log(raw["risk"]["gpr"].resample("W-FRI").mean()).rolling(4, min_periods=2).mean()
    d = df.index[700]
    assert np.isclose(df.at[d, "gpr_4w"], g.loc[d - pd.Timedelta(weeks=1)])


def test_price_models_run_with_and_without_risk(raw):
    from src import features, price_models as pm
    df = features.build_price_frame(raw["henry_hub"], raw["storage_l48"], raw["lng_exports"],
                                    raw["weather_us"], risk=raw["risk"])
    wf = pm.walk_forward(df, h=1, first_test="2024-01-01", refit_every=52)
    assert {pm.RISK_RIDGE, pm.RISK_LGBM} <= set(wf["model"])
    assert wf.groupby("model").size().nunique() == 1  # same weeks for every model
    wf0 = pm.walk_forward(df, h=1, first_test="2024-01-01", refit_every=52, use_risk=False)
    assert pm.RISK_RIDGE not in set(wf0["model"])
