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

# --- New England states ----------------------------------------------------
# EIA monthly deliveries by end use (MMcf), series N30xx<ST>2, and weather
# stations weighted roughly toward where each state's gas load sits.
_SECTORS = {"total": "N3060", "residential": "N3010", "commercial": "N3020",
            "industrial": "N3035", "electric_power": "N3045"}


def consumption_series(state: str) -> dict[str, str]:
    return {k: f"NG.{code}{state}2.M" for k, code in _SECTORS.items()}


STATES = {
    "MA": {"name": "Massachusetts", "stations": {
        "boston": {"lat": 42.36, "lon": -71.06, "weight": 0.70},
        "worcester": {"lat": 42.27, "lon": -71.80, "weight": 0.18},
        "springfield": {"lat": 42.10, "lon": -72.59, "weight": 0.12}}},
    "CT": {"name": "Connecticut", "stations": {
        "hartford": {"lat": 41.76, "lon": -72.68, "weight": 0.55},
        "new_haven": {"lat": 41.31, "lon": -72.92, "weight": 0.45}}},
    "RI": {"name": "Rhode Island", "stations": {
        "providence": {"lat": 41.82, "lon": -71.41, "weight": 1.0}}},
    "NH": {"name": "New Hampshire", "stations": {
        "manchester": {"lat": 42.99, "lon": -71.46, "weight": 0.6},
        "portsmouth": {"lat": 43.07, "lon": -70.76, "weight": 0.4}}},
    "ME": {"name": "Maine", "stations": {
        "portland_me": {"lat": 43.66, "lon": -70.26, "weight": 0.6},
        "bangor": {"lat": 44.80, "lon": -68.77, "weight": 0.4}}},
    "VT": {"name": "Vermont", "stations": {
        "burlington": {"lat": 44.48, "lon": -73.21, "weight": 1.0}}},
}

# Backward-compatible names for Massachusetts
MA_CONSUMPTION_SERIES = consumption_series("MA")
MA_WEATHER_STATIONS = STATES["MA"]["stations"]

# National market drivers for the Henry Hub model.
MARKET_SERIES = {
    "henry_hub": "NG.RNGWHHD.D",                # daily spot, $/MMBtu
    "storage_l48": "NG.NW2_EPG0_SWO_R48_BCF.W",  # weekly working gas, Bcf
    "storage_east": "NG.NW2_EPG0_SWO_R31_BCF.W", # weekly working gas, East region, Bcf
    "lng_exports": "NG.N9133US2.M",             # monthly LNG exports, MMcf
}

# --- Weather --------------------------------------------------------------
# National weather proxy for Henry Hub: large heating/cooling load centers.
US_WEATHER_STATIONS = {
    "new_york": {"lat": 40.71, "lon": -74.01, "weight": 0.25},
    "chicago": {"lat": 41.88, "lon": -87.63, "weight": 0.25},
    "houston": {"lat": 29.76, "lon": -95.37, "weight": 0.20},
    "atlanta": {"lat": 33.75, "lon": -84.39, "weight": 0.15},
    "philadelphia": {"lat": 39.95, "lon": -75.17, "weight": 0.15},
}

# Market events annotated on charts (context, not model inputs)
EVENTS = [
    ("2020-03-11", "COVID-19 pandemic declared"),
    ("2021-02-13", "Winter Storm Uri"),
    ("2022-02-24", "Russia invades Ukraine"),
    ("2022-06-08", "Freeport LNG outage"),
    ("2025-02-10", "China 15% tariff on US LNG"),
]

# --- Demand model settings (chosen by backtest; see notebook 03) ----------
# "ldc" = residential + commercial + industrial: the load gas utilities serve.
# Power-plant burn is excluded: it follows electricity markets, not heating
# weather, and most plants are not utility customers.
DEMAND_TARGET = "ldc"
NORMAL_YEARS = 15      # years of weather history behind "normal" HDD (recent winters are warmer)
TRAIN_YEARS = None     # None = all history; an int = only the most recent N years

HDD_BASE_F = 65.0
CDD_BASE_F = 65.0

# --- Publication lags (avoid look-ahead leakage) --------------------------
# Weekly storage for the week ending Friday t is published the following
# Thursday, so a forecast made at the end of week t only knows week t-1.
STORAGE_LAG_WEEKS = 1
# Monthly LNG export volumes appear roughly two months after the month ends.
LNG_LAG_MONTHS = 2

RANDOM_STATE = 42
