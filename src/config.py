"""Project configuration: data series, locations and paths.

All series are public. EIA series are pulled through the EIA API v2
`seriesid` route using their APIv1-style IDs (e.g. NG.N3010MA2.M).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# DATA_DIR can be overridden (used by the test suite).
DATA_DIR = Path(os.environ.get("GAS_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2001-01-01"

# --- EIA series -----------------------------------------------------------
# Massachusetts natural gas deliveries by end use, monthly, MMcf.
MA_CONSUMPTION_SERIES = {
    "total": "NG.N3060MA2.M",        # volumes delivered to consumers
    "residential": "NG.N3010MA2.M",
    "commercial": "NG.N3020MA2.M",
    "industrial": "NG.N3035MA2.M",
    "electric_power": "NG.N3045MA2.M",
}

# National market drivers for the Henry Hub model.
MARKET_SERIES = {
    "henry_hub": "NG.RNGWHHD.D",                # daily spot, $/MMBtu
    "storage_l48": "NG.NW2_EPG0_SWO_R48_BCF.W",  # weekly working gas, Bcf
    "storage_east": "NG.NW2_EPG0_SWO_R31_BCF.W", # weekly working gas, East region, Bcf
    "lng_exports": "NG.N9133US2.M",             # monthly LNG exports, MMcf
}

# --- Weather --------------------------------------------------------------
# Massachusetts demand weather: three stations weighted roughly by the
# population each one represents (eastern MA dominates gas load).
MA_WEATHER_STATIONS = {
    "boston": {"lat": 42.36, "lon": -71.06, "weight": 0.70},
    "worcester": {"lat": 42.27, "lon": -71.80, "weight": 0.18},
    "springfield": {"lat": 42.10, "lon": -72.59, "weight": 0.12},
}

# National weather proxy for Henry Hub: large heating/cooling load centers.
US_WEATHER_STATIONS = {
    "new_york": {"lat": 40.71, "lon": -74.01, "weight": 0.25},
    "chicago": {"lat": 41.88, "lon": -87.63, "weight": 0.25},
    "houston": {"lat": 29.76, "lon": -95.37, "weight": 0.20},
    "atlanta": {"lat": 33.75, "lon": -84.39, "weight": 0.15},
    "philadelphia": {"lat": 39.95, "lon": -75.17, "weight": 0.15},
}

HDD_BASE_F = 65.0
CDD_BASE_F = 65.0

# --- Publication lags (avoid look-ahead leakage) --------------------------
# Weekly storage for the week ending Friday t is published the following
# Thursday, so a forecast made at the end of week t only knows week t-1.
STORAGE_LAG_WEEKS = 1
# Monthly LNG export volumes appear roughly two months after the month ends.
LNG_LAG_MONTHS = 2

RANDOM_STATE = 42
