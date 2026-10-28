"""NYMEX Henry Hub natural gas futures: front-month history and the forward strip.

EIA stopped publishing NYMEX futures prices after April 5, 2024, so they come from Yahoo Finance
(the same CME contracts; the daily close is approximately the settlement price):

* NG=F: the continuous front-month contract, daily, 2001 onward. Used for the chart next to Henry
  Hub spot, and as the "market" forecast the price models are scored against.
* NG<month code><yy>.NYM: each monthly contract, for the forward strip (the market's price for
  each coming delivery month, which drives the commodity cost in GAF filings).

Updates are incremental and checked before saving: prices must be plausible, expired contracts
are dropped, stale bars are rejected, and a failed download never overwrites good history.

    python -m src.nymex            # pull, then rebuild the NYMEX files the app reads
    python -m src.nymex --check    # exit 1 if the data is stale (used by the daily job)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

import numpy as np
import pandas as pd

from . import config

FRONT_FILE = "nymex_front.csv"           # raw: date, close ($/MMBtu)
STRIP_FILE = "nymex_strip.csv"           # raw: as_of, contract (YYYY-MM), ticker, settle ($/MMBtu)
MONTH_CODES = "FGHJKMNQUVXZ"             # CME month codes, January .. December
PRICE_RANGE = (0.5, 50.0)                # plausible $/MMBtu; anything outside is rejected
STRIP_MONTHS = 24                        # two winters ahead
STALE_BUSINESS_DAYS = 4                  # the daily job fails (and GitHub emails) beyond this


# --------------------------------------------------------------------------
# contracts
# --------------------------------------------------------------------------
def contract_ticker(year: int, month: int) -> str:
    return f"NG{MONTH_CODES[month - 1]}{year % 100:02d}.NYM"


def last_trade_date(year: int, month: int) -> pd.Timestamp:
    """NYMEX Henry Hub futures stop trading 3 business days before the delivery month
    (exchange holidays are ignored here, so this can be a day early, never late)."""
    first = pd.Timestamp(year, month, 1)
    return first - pd.offsets.BDay(3)


def live_contracts(as_of: pd.Timestamp, months: int = STRIP_MONTHS) -> list[tuple[str, str]]:
    """(contract 'YYYY-MM', ticker) for contracts still trading on as_of, nearest first."""
    out, m = [], pd.Timestamp(as_of.year, as_of.month, 1)
    while len(out) < months:
        m = m + pd.DateOffset(months=1)
        if last_trade_date(m.year, m.month) >= as_of.normalize():
            out.append((f"{m:%Y-%m}", contract_ticker(m.year, m.month)))
    return out


# --------------------------------------------------------------------------
# download (Yahoo Finance through yfinance; only the pipeline needs it, not the app)
# --------------------------------------------------------------------------
def _download(tickers, retries: int = 3, **kw) -> pd.DataFrame:
    import yfinance as yf

    last = None
    for attempt in range(retries):
        try:
            df = yf.download(tickers, progress=False, auto_adjust=False, threads=False, **kw)
            if df is not None and len(df):
                return df
        except Exception as e:              # Yahoo rate limits and network hiccups
            last = e
        time.sleep(5 * (attempt + 1))
    if last:
        print(f"  NYMEX download failed: {last}")
    return pd.DataFrame()


def _close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Close prices for one ticker from a yfinance frame (columns ('Close', ticker), (ticker, 'Close')
    or plain 'Close')."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        if ("Close", ticker) in cols:
            s = df[("Close", ticker)]
        elif (ticker, "Close") in cols:
            s = df[(ticker, "Close")]
        else:
            return pd.Series(dtype=float)
    elif "Close" in cols:
        s = df["Close"]
    else:
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def clean_prices(s: pd.Series) -> pd.Series:
    """Drop implausible prices; raise if most of the download is implausible (bad feed)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    ok = s.between(*PRICE_RANGE)
    if len(s) and ok.mean() < 0.9:
        raise ValueError(f"NYMEX feed looks wrong: {int((~ok).sum())} of {len(s)} prices outside "
                         f"{PRICE_RANGE[0]}-{PRICE_RANGE[1]} $/MMBtu")
    return s[ok]


def update_front(path=None, overlap_days: int = 10, fetch=None) -> pd.Series:
    """Refresh the front-month history incrementally; keep the old file if the download fails."""
    path = path or config.RAW_DIR / FRONT_FILE
    fetch = fetch or (lambda start: _close(_download("NG=F", start=start), "NG=F"))
    old = pd.read_csv(path, index_col=0, parse_dates=True)["close"] if path.exists() else pd.Series(dtype=float)
    start = (old.index.max() - pd.Timedelta(days=overlap_days)).date().isoformat() if len(old) else config.START_DATE
    new = clean_prices(fetch(start))
    if new.empty:
        print("  NYMEX front month: no new data (kept the existing file)")
        return old
    keep = old[old.index < new.index.min()] if len(old) else old.iloc[0:0]
    merged = (pd.concat([keep, new]) if len(keep) else new).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")].rename("close")
    merged.index.name = "date"
    merged.to_frame().to_csv(path)
    print(f"  NYMEX front month: {len(new)} days updated, through {merged.index.max():%Y-%m-%d}")
    return merged


def update_strip(path=None, as_of: pd.Timestamp | None = None, fetch=None) -> pd.DataFrame:
    """Append today's forward strip (one row per live contract) to the strip history."""
    path = path or config.RAW_DIR / STRIP_FILE
    as_of = (as_of or pd.Timestamp.today()).normalize()
    contracts = live_contracts(as_of)
    tickers = [t for _, t in contracts]
    raw = fetch(tickers) if fetch else _download(tickers, period="10d", group_by="column")
    rows = []
    for contract, t in contracts:
        s = clean_prices(_close(raw, t)) if not raw.empty else pd.Series(dtype=float)
        if s.empty:
            continue
        last_day = s.index.max()
        if (as_of - last_day).days > 7:          # no recent settlement: not a live quote
            continue
        rows.append({"as_of": last_day, "contract": contract, "ticker": t, "settle": float(s.iloc[-1])})
    new = pd.DataFrame(rows, columns=["as_of", "contract", "ticker", "settle"])
    old = pd.read_csv(path, parse_dates=["as_of"]) if path.exists() else new.iloc[0:0]
    if new.empty:
        print("  NYMEX strip: no quotes returned (kept the existing file)")
        return old
    # the strip "as of" date is the most common settlement date among the contracts
    new["as_of"] = new["as_of"].mode().iloc[0]
    out = pd.concat([old, new]).drop_duplicates(["as_of", "contract"], keep="last")
    out = out.sort_values(["as_of", "contract"]).reset_index(drop=True)
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  NYMEX strip: {len(new)} contracts as of {new['as_of'].iloc[0]:%Y-%m-%d} "
          f"({new['contract'].iloc[0]} to {new['contract'].iloc[-1]})")
    return out


def pull() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("Pulling NYMEX Henry Hub futures (Yahoo Finance) ...")
    for step in (update_front, update_strip):
        try:
            step()
        except Exception as e:                  # never let a bad feed break the refresh
            print(f"  {step.__name__} skipped: {e}")


# --------------------------------------------------------------------------
# derived files the app reads
# --------------------------------------------------------------------------
def load_front() -> pd.Series | None:
    p = config.RAW_DIR / FRONT_FILE
    return pd.read_csv(p, index_col=0, parse_dates=True)["close"] if p.exists() else None


def load_strip() -> pd.DataFrame | None:
    p = config.RAW_DIR / STRIP_FILE
    return pd.read_csv(p, parse_dates=["as_of"]) if p.exists() else None


def weekly_front(front: pd.Series) -> pd.Series:
    """Weekly average of the front-month close, weeks ending Friday (as the Henry Hub price)."""
    w = front.resample("W-FRI").mean()
    n = front.resample("W-FRI").count()
    return w[n >= 3].rename("nymex_front")


def strip_tables(strip: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest strip, and the one about a week earlier for the change column."""
    dates = sorted(strip["as_of"].unique())
    latest = strip[strip["as_of"] == dates[-1]].copy()
    prior_date = next((d for d in reversed(dates[:-1]) if (dates[-1] - d).days >= 5), None)
    prior = strip[strip["as_of"] == prior_date] if prior_date is not None else strip.iloc[0:0]
    latest = latest.merge(prior[["contract", "settle"]].rename(columns={"settle": "settle_prior"}),
                          on="contract", how="left")
    latest["change"] = latest["settle"] - latest["settle_prior"]
    latest["settle_therm"] = latest["settle"] / 10
    latest["prior_as_of"] = prior_date
    return latest, prior


def winter_strip(latest: pd.DataFrame, as_of: pd.Timestamp) -> dict | None:
    """Average strip for the Massachusetts peak GAF period (November-April), in $/MMBtu and $/therm:
    the coming peak period, or the months still to come once it has started."""
    y = as_of.year - 1 if as_of.month <= 4 else as_of.year
    months = [f"{y}-11", f"{y}-12"] + [f"{y + 1}-{m:02d}" for m in range(1, 5)]
    w = latest[latest["contract"].isin(months)]
    if w.empty:
        return None
    label = f"Nov {y}–Apr {y + 1}" + ("" if len(w) == 6 else f" (remaining {len(w)} months)")
    return {"label": label, "months": len(w), "mmbtu": float(w["settle"].mean()),
            "therm": float(w["settle"].mean()) / 10}


def build(out=None) -> dict:
    """Write the NYMEX files the app reads; returns metadata (dates) for meta.json."""
    out = out or config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    front = load_front()
    if front is not None and len(front):
        hh = pd.read_csv(config.RAW_DIR / "henry_hub.csv", index_col=0, parse_dates=True)["value"] \
            if (config.RAW_DIR / "henry_hub.csv").exists() else pd.Series(dtype=float)
        daily = pd.DataFrame({"nymex_front": front, "henry_hub_spot": hh}).loc[front.index.min():]
        daily.index.name = "date"
        daily.to_csv(out / "nymex_daily.csv")
        monthly = front.resample("MS").mean().rename("nymex_front_therm") / 10
        monthly.index.name = "period"
        monthly.to_frame().to_csv(out / "nymex_monthly_therm.csv")
        meta["nymex_front_through"] = f"{front.index.max():%Y-%m-%d}"
        meta["nymex_front_last"] = round(float(front.iloc[-1]), 3)
    strip = load_strip()
    if strip is not None and len(strip):
        latest, _ = strip_tables(strip)
        latest.to_csv(out / "nymex_strip_latest.csv", index=False, date_format="%Y-%m-%d")
        meta["nymex_strip_as_of"] = f"{latest['as_of'].iloc[0]:%Y-%m-%d}"
        ws = winter_strip(latest, latest["as_of"].iloc[0])
        if ws:
            meta["nymex_winter_strip"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ws.items()}
    if meta:
        (out / "nymex_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# --------------------------------------------------------------------------
# market benchmark for the price backtest
# --------------------------------------------------------------------------
MARKET = "NYMEX futures (market)"


def add_market_benchmark(wf: pd.DataFrame, front: pd.Series | None, h: int) -> pd.DataFrame:
    """Score the futures market's own forecast alongside the models.

    At the end of each week t the market's price for the coming month is the front-month futures
    price, so its forecast of the log change is log(F_t / P_t). It uses only prices known at t.
    Added only when futures cover every week the models are scored on, so comparisons stay fair.
    """
    if front is None or front.empty or wf.empty:
        return wf
    fw = weekly_front(front)
    rw = wf[wf["model"] == "Random walk"][["date", "y", "price"]].copy()
    f = fw.reindex(pd.to_datetime(rw["date"]))
    if f.isna().any():
        print(f"  [Price] NYMEX benchmark skipped for h={h}: futures missing for {int(f.isna().sum())} weeks")
        return wf
    rw["model"] = MARKET
    rw["pred"] = np.log(f.values / rw["price"].values)
    rw["price_fcst"] = f.values
    rw["price_actual"] = rw["price"] * np.exp(rw["y"])
    return pd.concat([wf, rw[wf.columns]], ignore_index=True)


def latest_market_forecast(front: pd.Series | None, price_now: float, as_of) -> list[dict]:
    if front is None or front.empty:
        return []
    wk = weekly_front(front)
    f = float(wk.iloc[-1]) if len(wk) else float(front.iloc[-1])
    return [{"as_of": as_of, "price_now": price_now, "h": h, "pred_logret": float(np.log(f / price_now)),
             "price_fcst": f, "model": MARKET} for h in (1, 4)]


# --------------------------------------------------------------------------
# freshness
# --------------------------------------------------------------------------
def business_days_since(d, today=None) -> int:
    today = pd.Timestamp(today or date.today()).normalize()
    return max(0, len(pd.bdate_range(pd.Timestamp(d).normalize(), today)) - 1)


def check_fresh(today=None) -> list[str]:
    """Problems that should fail the daily job (and email the repo owner)."""
    problems = []
    front = load_front()
    if front is None or front.empty:
        problems.append("no NYMEX front-month data")
    elif business_days_since(front.index.max(), today) > STALE_BUSINESS_DAYS:
        problems.append(f"NYMEX front month last updated {front.index.max():%Y-%m-%d}")
    strip = load_strip()
    if strip is None or strip.empty:
        problems.append("no NYMEX strip")
    elif business_days_since(strip["as_of"].max(), today) > STALE_BUSINESS_DAYS:
        problems.append(f"NYMEX strip last updated {strip['as_of'].max():%Y-%m-%d}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true", help="only rebuild the app files")
    ap.add_argument("--check", action="store_true", help="exit 1 if the data is stale")
    a = ap.parse_args()
    if not a.no_pull:
        pull()
    print(build())
    if a.check:
        bad = check_fresh()
        if bad:
            print("STALE:", "; ".join(bad))
            sys.exit(1)
