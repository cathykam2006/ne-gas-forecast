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



# ---------------------------------------------------------------- MA tariffs
def test_rate_class_mapping():
    from src.tariffs import classify, canonical_company
    assert classify("Residential R-3, R-4") == ["Residential heating"]
    assert classify("R-1, R-2") == ["Residential non-heating"]
    assert set(classify("All Residential")) == {"Residential heating", "Residential non-heating"}
    assert classify("G-41B/E, G-51B/E") == ["Small C&I"]
    assert classify("G-41, 42, 43 & G-51, 52, 53") == ["Small C&I", "Medium C&I", "Large C&I"]
    assert classify("Large C&I (G43/G53/T-1)") == ["Large C&I"]
    assert classify("R-2, R-4") == []          # low-income only
    assert canonical_company("KEYSPAN - BOSTON GAS") == "National Grid – Boston Gas"
    assert canonical_company("NSTAR Gas d/b/a Eversource Energy") == "NSTAR Gas (Eversource)"
    assert canonical_company("Eversource Gas Company of Massachusetts (formerly Bay State Gas "
                             "d/b/a Columbia Gas of Massachusetts)").startswith("Eversource Gas of MA")


def test_page_parser_and_monthly_asof(tmp_path):
    from src import tariffs
    html = """<h2>Peak/Winter 2025-2026 Rates (Effective 11/1/25 through 11/30/25)</h2>
    <h3>National Grid - Boston Gas</h3>
    <table><tr><th>Rate Class</th><th>Billing Unit</th><th>GAF</th><th>LDAF</th></tr>
    <tr><th>All Residential</th><td>$/Therm</td><td>0.9564</td><td>0.5572</td></tr>
    <tr><th>G-41B/E, G-51B/E</th><td>$/Therm</td><td>0.9564</td><td>(0.0100)</td></tr></table>
    <h2>Peak/Winter 2025-2026 Rates Revision 3 (Effective 2/1/26 through 2/28/26)</h2>
    <h3>National Grid - Boston Gas</h3>
    <table><tr><th>Rate Class</th><th>Billing Unit</th><th>GAF</th><th>LDAF</th></tr>
    <tr><th>All Residential</th><td>$/Therm</td><td>1.2329</td><td>0.5572</td></tr></table>"""
    f = tmp_path / "page.html"
    f.write_text(html)
    t = tariffs.build_rate_table(None, f)
    assert len(t) == 3 and t["ldaf"].min() == -0.01          # parentheses = credit
    m = tariffs.monthly_factors(t)
    r = m[m["category"] == "Residential heating"].set_index("month")["gaf_therm"]
    assert r.loc["2025-12-01"] == 0.9564 and r.loc["2026-01-01"] == 0.9564   # carried forward
    assert r.loc["2026-02-01"] == 1.2329


def test_ccf_rates_converted_to_therms():
    from src.tariffs import _to_therm, THERMS_PER_CCF
    assert _to_therm(1.037, "$/Ccf") == 1.037 / THERMS_PER_CCF
    assert np.isnan(_to_therm(3.2, "$/MDCQ"))


def test_region_forecasts_every_state(raw):
    from src import region
    assert len(raw["states"]) == 6
    sc, fc, hist = region.run_region({k: raw["states"][k] for k in ("VT", "RI")}, step=12,
                                     verbose=False)
    assert set(fc["state"]) == {"VT", "RI"} and fc.groupby("state").size().eq(12).all()
