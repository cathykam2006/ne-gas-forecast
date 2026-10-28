"""NYMEX futures: contracts, incremental updates, validation, strip, benchmark, freshness.

Downloads are replaced by frames shaped like yfinance output, so these run offline.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src import nymex as N


def yf_frame(series_by_ticker: dict, layout: str = "price_first") -> pd.DataFrame:
    """A yfinance-like frame: columns ('Close', ticker) (group_by='column') or (ticker, 'Close')."""
    parts = {}
    for t, s in series_by_ticker.items():
        for field in ("Open", "Close"):
            key = (field, t) if layout == "price_first" else (t, field)
            parts[key] = s
    df = pd.DataFrame(parts)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_contract_tickers_and_roll():
    assert N.contract_ticker(2026, 12) == "NGZ26.NYM" and N.contract_ticker(2027, 1) == "NGF27.NYM"
    assert N.last_trade_date(2026, 11) == pd.Timestamp("2026-10-28")     # 3 business days before Nov 1
    on_expiry = N.live_contracts(pd.Timestamp("2026-10-28"), 3)
    after = N.live_contracts(pd.Timestamp("2026-10-29"), 3)
    assert on_expiry[0] == ("2026-11", "NGX26.NYM") and after[0] == ("2026-12", "NGZ26.NYM")
    assert len(N.live_contracts(pd.Timestamp("2026-10-27"))) == N.STRIP_MONTHS


def test_close_handles_both_yfinance_layouts():
    s = pd.Series([3.1, 3.2], index=pd.to_datetime(["2026-10-26", "2026-10-27"]))
    for layout in ("price_first", "ticker_first"):
        out = N._close(yf_frame({"NGZ26.NYM": s, "NGF27.NYM": s + 1}, layout), "NGF27.NYM")
        assert list(out) == [4.1, 4.2]
    assert N._close(pd.DataFrame(), "X").empty
    assert list(N._close(pd.DataFrame({"Close": s}), "NG=F")) == [3.1, 3.2]


def test_bad_feed_is_rejected_and_outliers_dropped():
    good = pd.Series([3.0, 3.1, 0.0, 3.2] + [3.3] * 10)
    assert len(N.clean_prices(good)) == 13                               # the 0.0 is dropped
    with pytest.raises(ValueError):
        N.clean_prices(pd.Series([3000.0] * 10 + [3.0]))                 # e.g. a feed in cents


def test_front_updates_incrementally_and_keeps_history_on_failure(tmp_path):
    f = tmp_path / "front.csv"
    old = pd.Series(3.0, index=pd.bdate_range("2001-01-02", "2026-10-16"), name="close")
    old.index.name = "date"
    old.to_frame().to_csv(f)
    starts = []

    def fetch(start):
        starts.append(start)
        return pd.Series(3.5, index=pd.bdate_range("2026-10-06", "2026-10-27"))

    out = N.update_front(f, fetch=fetch)
    assert starts == ["2026-10-06"]                                      # only the last 10 days
    assert out.index.min() == pd.Timestamp("2001-01-02") and out.index.is_unique
    assert out["2020-06-01"] == 3.0 and out["2026-10-27"] == 3.5
    before = f.read_text()
    N.update_front(f, fetch=lambda start: pd.Series(dtype=float))         # download failed
    assert f.read_text() == before


def test_strip_drops_expired_and_stale_and_keeps_history(tmp_path):
    f = tmp_path / "strip.csv"
    as_of = pd.Timestamp("2026-10-27")
    days = pd.bdate_range("2026-10-20", "2026-10-27")

    def fetch(tickers):
        data = {}
        for i, t in enumerate(tickers):
            if t == "NGJ27.NYM":                                         # no recent trade: stale
                data[t] = pd.Series(3.0, index=pd.bdate_range("2026-09-01", "2026-09-10"))
            else:
                data[t] = pd.Series(np.linspace(3.0, 3.2, len(days)) + i * 0.1, index=days)
        return yf_frame(data)

    out = N.update_strip(f, as_of, fetch)
    contracts = list(out["contract"])
    assert contracts[0] == "2026-11" and "2027-04" not in contracts      # Apr 2027 (NGJ27) was stale
    assert (out["as_of"] == pd.Timestamp("2026-10-27")).all()
    assert out.loc[0, "settle"] == pytest.approx(3.2)
    # a later day appends a new strip; the earlier one is kept for the week-on-week change
    N.update_strip(f, pd.Timestamp("2026-11-03"),
                   lambda t: yf_frame({x: pd.Series(4.0, index=pd.bdate_range("2026-10-28", "2026-11-03"))
                                       for x in t}))
    hist = pd.read_csv(f, parse_dates=["as_of"])
    assert set(hist["as_of"].dt.strftime("%Y-%m-%d")) == {"2026-10-27", "2026-11-03"}
    assert "2026-11" not in set(hist[hist["as_of"] == "2026-11-03"]["contract"])   # Nov expired Oct 28
    latest, _ = N.strip_tables(hist)
    dec = latest.set_index("contract").loc["2026-12"]
    assert dec["settle"] == 4.0 and dec["change"] == pytest.approx(4.0 - 3.3) and dec["settle_therm"] == 0.4


def test_peak_period_strip_is_nov_to_apr():
    months = pd.period_range("2026-11", "2028-10", freq="M").strftime("%Y-%m")
    latest = pd.DataFrame({"contract": months, "settle": np.arange(len(months), dtype=float) + 3})
    ws = N.winter_strip(latest, pd.Timestamp("2026-10-27"))
    assert ws["label"] == "Nov 2026–Apr 2027" and ws["months"] == 6
    assert ws["mmbtu"] == pytest.approx(np.mean([3, 4, 5, 6, 7, 8])) and ws["therm"] == pytest.approx(ws["mmbtu"] / 10)
    mid = N.winter_strip(latest[latest["contract"] >= "2027-02"], pd.Timestamp("2027-01-28"))
    assert mid["label"] == "Nov 2026–Apr 2027 (remaining 3 months)"


def wf_fixture():
    weeks = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    price = pd.Series(3.0 + 0.5 * np.sin(np.arange(60) / 5), index=weeks)
    y = np.log(price.shift(-4) / price).fillna(0)
    rows = [{"date": d, "model": m, "pred": 0.0, "y": y[d], "price": price[d]}
            for d in weeks for m in ("Random walk", "Ridge (momentum only)")]
    wf = pd.DataFrame(rows)
    wf["price_fcst"] = wf["price"]
    wf["price_actual"] = wf["price"] * np.exp(wf["y"])
    return wf, weeks, price


def test_market_benchmark_uses_futures_known_at_the_time():
    wf, weeks, price = wf_fixture()
    days = pd.bdate_range("2019-12-30", weeks[-1])
    front = pd.Series(3.3, index=days)
    out = N.add_market_benchmark(wf, front, 4)
    mk = out[out["model"] == N.MARKET].set_index("date")
    assert len(mk) == len(weeks)
    assert np.allclose(mk["pred"], np.log(3.3 / price.values)) and np.allclose(mk["price_fcst"], 3.3)
    from src import price_models as pm
    sc = pm.score_price(out, h=4)
    assert N.MARKET in sc.index and not np.isnan(sc.at[N.MARKET, "DM p-value"])
    # futures missing for some scored weeks: no benchmark rather than an unfair one
    gap = front.drop(front.loc["2020-06-01":"2020-06-30"].index)
    assert N.MARKET not in set(N.add_market_benchmark(wf, gap, 4)["model"])
    assert N.add_market_benchmark(wf, None, 4) is wf


def test_latest_market_forecast():
    front = pd.Series([3.0, 3.1, 3.2, 3.3, 3.4], index=pd.bdate_range("2026-10-19", periods=5))
    rows = N.latest_market_forecast(front, 3.0, pd.Timestamp("2026-10-23"))
    assert [r["h"] for r in rows] == [1, 4] and rows[0]["price_fcst"] == pytest.approx(3.2)
    assert rows[0]["model"] == N.MARKET and rows[1]["pred_logret"] == pytest.approx(np.log(3.2 / 3.0))


def test_build_and_freshness(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(N.config, "RAW_DIR", raw)
    days = pd.bdate_range("2026-01-02", "2026-10-27")
    pd.Series(3.2, index=pd.Index(days, name="date"), name="close").to_frame().to_csv(raw / N.FRONT_FILE)
    pd.DataFrame({"value": 3.0}, index=pd.Index(days, name="period")).to_csv(raw / "henry_hub.csv")
    contracts = pd.period_range("2026-12", periods=24, freq="M").strftime("%Y-%m")
    pd.DataFrame({"as_of": "2026-10-27", "contract": contracts, "ticker": "x", "settle": 3.5}).to_csv(
        raw / N.STRIP_FILE, index=False)
    meta = N.build(out)
    assert meta["nymex_front_through"] == "2026-10-27" and meta["nymex_strip_as_of"] == "2026-10-27"
    assert meta["nymex_winter_strip"]["label"] == "Nov 2026–Apr 2027 (remaining 5 months)"
    assert json.loads((out / "nymex_meta.json").read_text()) == meta
    daily = pd.read_csv(out / "nymex_daily.csv", index_col=0)
    assert list(daily.columns) == ["nymex_front", "henry_hub_spot"]
    assert N.check_fresh(today="2026-10-30") == []
    stale = N.check_fresh(today="2026-11-06")                            # 8 business days later
    assert len(stale) == 2 and "2026-10-27" in stale[0]


def test_first_run_without_history_downloads_everything(tmp_path):
    f = tmp_path / "front.csv"
    starts = []
    out = N.update_front(f, fetch=lambda start: starts.append(start) or
                         pd.Series(3.1, index=pd.bdate_range("2001-01-02", "2026-10-27")))
    assert starts == ["2001-01-01"] and f.exists()
    saved = pd.read_csv(f, index_col=0, parse_dates=True)["close"]
    assert saved.index.min() == pd.Timestamp("2001-01-02") and saved.index.max() == pd.Timestamp("2026-10-27")
    assert len(out) == len(saved)
