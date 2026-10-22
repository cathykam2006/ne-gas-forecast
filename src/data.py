"""Data acquisition: EIA API v2 and Open-Meteo historical weather.

Usage:
    export EIA_API_KEY=your_key      # free: https://www.eia.gov/opendata/register.php
    python -m src.data               # pulls everything into data/raw/
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from . import config

EIA_BASE = "https://api.eia.gov/v2"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PAGE_SIZE = 5000


# --------------------------------------------------------------------------
# EIA
# --------------------------------------------------------------------------
def _get_api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("EIA_API_KEY")
    if not key:
        try:  # Streamlit Cloud secrets, if running inside the app
            import streamlit as st

            key = st.secrets.get("EIA_API_KEY")
        except Exception:
            key = None
    if not key:
        raise RuntimeError(
            "No EIA API key found. Set the EIA_API_KEY environment variable "
            "(free key: https://www.eia.gov/opendata/register.php)."
        )
    return key


def _request(url: str, params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=120)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt == retries - 1:
                raise
            wait = 20 * (attempt + 1)
            print(f"  network timeout ({type(e).__name__}), retrying in {wait}s ...")
            time.sleep(wait)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 and attempt < retries - 1:
            # Open-Meteo's free tier has a per-minute limit; wait it out.
            print("  rate limit reached, waiting 65 seconds ...")
            time.sleep(65)
            continue
        if resp.status_code in (500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(10 * (attempt + 1))
            continue
        raise RuntimeError(f"Request failed ({resp.status_code}): {resp.text[:300]}")
    raise RuntimeError("Request failed after retries")


def fetch_eia_series(series_id: str, start: str = config.START_DATE,
                     api_key: str | None = None) -> pd.Series:
    """Fetch one EIA series by its APIv1-style ID via the v2 `seriesid` route.

    Returns a float Series indexed by period (Timestamp), sorted ascending.
    """
    key = _get_api_key(api_key)
    url = f"{EIA_BASE}/seriesid/{series_id}"
    rows, offset = [], 0
    while True:
        payload = _request(url, {"api_key": key, "offset": offset, "length": PAGE_SIZE})
        resp = payload.get("response", {})
        batch = resp.get("data", [])
        rows.extend(batch)
        total = int(resp.get("total", len(rows)) or 0)
        offset += len(batch)
        if not batch or offset >= total:
            break
    if not rows:
        raise RuntimeError(f"EIA returned no rows for {series_id}")

    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.to_datetime(df["period"]), name=series_id)
    s = s[~s.index.duplicated()].sort_index()
    return s[s.index >= pd.Timestamp(start)].dropna()


def fetch_state_consumption(state: str, api_key: str | None = None) -> pd.DataFrame:
    """Monthly deliveries by end use (MMcf) for one state.

    Electric-power deliveries are missing or 'not applicable' for some small
    states; that column is optional (the demand target excludes it anyway).
    """
    cols = {}
    for name, sid in config.consumption_series(state).items():
        try:
            cols[name] = fetch_eia_series(sid, api_key=api_key)
        except RuntimeError:
            if name in ("electric_power", "total"):
                print(f"  {state} {name}: no data, skipped")
                continue
            raise
    df = pd.DataFrame(cols)
    df.index = df.index.to_period("M").to_timestamp()
    df.index.name = "month"
    return df


def fetch_ma_consumption(api_key: str | None = None) -> pd.DataFrame:
    return fetch_state_consumption("MA", api_key)


def state_files(state: str):
    st = state.lower()
    return config.RAW_DIR / f"{st}_consumption.csv", config.RAW_DIR / f"weather_{st}.csv"


def pull_states(api_key: str | None = None, only_missing: bool = False) -> None:
    """Consumption and weighted degree days for every New England state."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for state, meta in config.STATES.items():
        cons_f, wx_f = state_files(state)
        if only_missing and cons_f.exists() and wx_f.exists():
            print(f"  {state}: already present, skipped")
            continue
        print(f"Pulling {meta['name']} consumption and weather ...")
        fetch_state_consumption(state, api_key).to_csv(cons_f)
        update_weather(meta["stations"], wx_f)


def fetch_market_data(api_key: str | None = None) -> dict[str, pd.Series]:
    return {name: fetch_eia_series(sid, api_key=api_key)
            for name, sid in config.MARKET_SERIES.items()}


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------
def fetch_station_temps(lat: float, lon: float, start: str = config.START_DATE,
                        end: str | None = None) -> pd.DataFrame:
    """Daily max/min temperature (deg F) from the Open-Meteo ERA5 archive."""
    end = end or (date.today() - timedelta(days=3)).isoformat()
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_min",
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    daily = _request(OPEN_METEO_ARCHIVE, params)["daily"]
    df = pd.DataFrame(daily).rename(columns={"temperature_2m_max": "tmax",
                                             "temperature_2m_min": "tmin"})
    df["date"] = pd.to_datetime(df.pop("time"))
    return df.set_index("date")


def weighted_degree_days(stations: dict, start: str = config.START_DATE,
                         end: str | None = None) -> pd.DataFrame:
    """Weighted daily mean temperature, HDD and CDD across a set of stations.

    Uses the NOAA convention: daily mean = (Tmax + Tmin) / 2.
    """
    frames, weights = [], []
    for name, st in stations.items():
        t = fetch_station_temps(st["lat"], st["lon"], start, end)
        frames.append(((t["tmax"] + t["tmin"]) / 2).rename(name))
        weights.append(st["weight"])
        time.sleep(5)  # be polite to the free API (long ranges count as many calls)
    temps = pd.concat(frames, axis=1)
    w = np.array(weights) / np.sum(weights)
    out = pd.DataFrame(index=temps.index)
    out["tmean"] = temps.mul(w, axis=1).sum(axis=1, min_count=1)
    out["hdd"] = (config.HDD_BASE_F - out["tmean"]).clip(lower=0)
    out["cdd"] = (out["tmean"] - config.CDD_BASE_F).clip(lower=0)
    return out.dropna()


def update_weather(stations: dict, path, overlap_days: int = 60) -> None:
    """Refresh a weather file incrementally.

    If the file exists, only the last `overlap_days` plus new days are
    downloaded (ERA5 revises the most recent days slightly), instead of 25
    years for every station. A weekly refresh is then a few small requests.
    """
    from pathlib import Path

    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        start = (old.index.max() - pd.Timedelta(days=overlap_days)).date().isoformat()
        new = weighted_degree_days(stations, start=start)
        merged = pd.concat([old[old.index < new.index.min()], new])
    else:
        merged = weighted_degree_days(stations)
    merged.index.name = "date"
    merged.to_csv(path)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def pull_all(api_key: str | None = None) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    pull_states(api_key)

    print("Pulling market series ...")
    for name, s in fetch_market_data(api_key).items():
        s.rename("value").to_frame().rename_axis("period").to_csv(
            config.RAW_DIR / f"{name}.csv")

    print("Pulling US weather proxy ...")
    update_weather(config.US_WEATHER_STATIONS, config.RAW_DIR / "weather_us.csv")
    print(f"Done. Raw files in {config.RAW_DIR}")


def load_raw() -> dict[str, pd.DataFrame | pd.Series]:
    """Load previously pulled raw files."""
    r = config.RAW_DIR
    out: dict = {
        "ma_consumption": pd.read_csv(r / "ma_consumption.csv", index_col=0, parse_dates=True),
        "weather_ma": pd.read_csv(r / "weather_ma.csv", index_col=0, parse_dates=True),
        "weather_us": pd.read_csv(r / "weather_us.csv", index_col=0, parse_dates=True),
    }
    for name in config.MARKET_SERIES:
        out[name] = pd.read_csv(r / f"{name}.csv", index_col=0, parse_dates=True)["value"]
    states = {}
    for state in config.STATES:
        cons_f, wx_f = state_files(state)
        if cons_f.exists() and wx_f.exists():
            states[state] = (pd.read_csv(cons_f, index_col=0, parse_dates=True),
                             pd.read_csv(wx_f, index_col=0, parse_dates=True))
    out["states"] = states
    return out


if __name__ == "__main__":
    import sys

    if "--new-england" in sys.argv:   # add the other five states to an existing pull
        pull_states(only_missing=True)
    else:
        pull_all()
