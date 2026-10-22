"""New England gas utilities: supply, demand, customers and prices by company.

Source: EIA Form 176, the annual report every gas utility files for each
state it serves, read from EIA's Natural Gas Query System (public JSON
behind https://www.eia.gov/naturalgas/ngqs/). Volumes are Mcf, revenues $.

What each company gets:
* Demand: deliveries to residential, commercial and industrial customers
  (sales + transportation), customer counts.
* Supply: gas received at the citygate (for sales and for transportation
  customers), LNG withdrawals, total supply; storage injections, losses and
  unaccounted-for gas.
* Price: average residential and commercial sales price, revenue / volume,
  in $/therm using the company's own reported heat content.
* Forecast: each state's monthly utility-load forecast split by the
  company's share of that state's utility deliveries.

    python -m src.companies        # download, then the pipeline uses it
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from . import config

NGQS = "https://www.eia.gov/naturalgas/ngqs/data/report/RPC/data/{y1}/{y2}/ACI/ID_Name/{items}"
RAW_FILE = "ne_companies_eia176.csv"
FIRST_YEAR = 2010

STATE_NAMES = {"Connecticut": "CT", "Maine": "ME", "Massachusetts": "MA",
               "New Hampshire": "NH", "Rhode Island": "RI", "Vermont": "VT"}

ITEMS = {
    # demand: sales + transportation volumes by sector
    "101TVL": "res_volume", "102TVL": "com_volume", "103TVL": "ind_volume",
    # sales customers, volumes and revenue
    "1010CT": "res_sales_customers", "1010VL": "res_sales_volume", "1010CS": "res_sales_revenue",
    "1020CT": "com_sales_customers", "1020VL": "com_sales_volume", "1020CS": "com_sales_revenue",
    "1030CT": "ind_sales_customers", "1030VL": "ind_sales_volume", "1030CS": "ind_sales_revenue",
    # transportation (customer choice) customers
    "1110CT": "res_transport_customers", "1120CT": "com_transport_customers",
    "1130CT": "ind_transport_customers",
    # supply side
    "0410VL": "citygate_sales_volume", "0420VL": "citygate_transport_volume",
    "0220VL": "lng_withdrawals", "0700VL": "total_supply",
    "1310VL": "storage_injections_ug", "1320VL": "storage_injections_lng",
    "1700VL": "losses", "2000VL": "unaccounted_for",
    "0900VL": "heat_content_btu_cf",
}

# Successor names for utilities that were renamed or merged.
CANONICAL = {
    "17601391MA": "National Grid – Boston Gas",
    "17601248MA": "National Grid – Boston Gas",          # Colonial Gas, merged into Boston Gas
    "17615111MA": "Eversource – NSTAR Gas",
    "17680531MA": "Eversource Gas of MA",
    "17616558MA": "Eversource Gas of MA",                # Columbia Gas of Massachusetts
    "17601093MA": "Berkshire Gas",
    "17604115MA": "Liberty – Fall River/N. Attleboro",
    "17625015MA": "Liberty – Blackstone",
    "17604315MA": "Unitil – Fitchburg Gas",
    "17605948MA": "Holyoke Gas & Electric (municipal)",
    "17614749MA": "Westfield Gas & Electric (municipal)",
    "17609453MA": "Middleborough Gas & Electric (municipal)",
    "17609923MA": "Wakefield Gas & Light (municipal)",
    "17602792CT": "Eversource – Yankee Gas",
    "17619018CT": "Avangrid – Connecticut Natural Gas",
    "17619865CT": "Avangrid – Southern Connecticut Gas",
    "17610359CT": "Norwich Public Utilities (municipal)",
    "17681346RI": "Rhode Island Energy",
    "17698114RI": "Rhode Island Energy",                 # Narragansett Electric, gas division
    "17600223NH": "Unitil – Northern Utilities (NH)",
    "17675803NH": "Liberty – EnergyNorth",
    "17617891ME": "Unitil – Northern Utilities (ME)",
    "17676620ME": "Summit Natural Gas of Maine",
    "17695252ME": "Bangor Natural Gas",
    "17698117ME": "Maine Natural Gas",
    "17618610VT": "Vermont Gas Systems",
}

MIN_RES_CUSTOMERS = 1000   # filters out pipelines, LNG and CNG suppliers


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def _parse_company(label: str) -> tuple[str, str]:
    """'17601391MA (BOSTON GAS CO DBA NATIONAL GRID)' -> (id, name)."""
    m = re.match(r"\s*(\d{8}[A-Z]{2})\s*\((.*)\)\s*$", str(label))
    return (m.group(1), m.group(2).strip()) if m else ("", str(label).strip())


def fetch(first_year: int = FIRST_YEAR, last_year: int | None = None) -> pd.DataFrame:
    """Long table: state, company_id, eia_name, item, year, value."""
    last_year = last_year or pd.Timestamp.today().year - 1
    headers = {"User-Agent": "Mozilla/5.0 (research; ne-gas-forecast)"}
    rows = []
    y1 = first_year
    while y1 <= last_year:          # the service is queried in 5-year chunks
        y2 = min(y1 + 4, last_year)
        url = NGQS.format(y1=y1, y2=y2, items="/".join(ITEMS))
        r = requests.get(url, headers=headers, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"EIA NGQS request failed ({r.status_code}) for {y1}-{y2}")
        for d in r.json().get("data", []):
            state = STATE_NAMES.get(str(d.get("a", "")).strip())
            if not state:
                continue
            cid, name = _parse_company(d.get("b", ""))
            if not cid:            # state totals are recomputed from companies
                continue
            item = _item_key(d.get("line"), d.get("atype"))
            for k, v in d.items():
                if re.fullmatch(r"y\d{4}", k) and v is not None:
                    rows.append({"state": state, "company_id": cid, "eia_name": name,
                                 "item": item, "year": int(k[1:]), "value": float(v)})
        y1 = y2 + 1
    df = pd.DataFrame(rows).drop_duplicates(["company_id", "item", "year"])
    if df.empty:
        raise RuntimeError("EIA NGQS returned no New England rows")
    return df


def _item_key(line, atype) -> str | None:
    """Map NGQS (line, atype) back to our item names; lines drop leading zeros."""
    line = str(line)
    for code, name in ITEMS.items():
        if code[-2:] == atype and code[:-2].lstrip("0") == line.lstrip("0"):
            return name
    return None


def download() -> pd.DataFrame:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch()
    df = df.dropna(subset=["item"])
    df.to_csv(config.RAW_DIR / RAW_FILE, index=False)
    print(f"  {df['company_id'].nunique()} New England reporters, "
          f"{df['year'].min()}–{df['year'].max()}")
    return df


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def company_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per utility, state and year with demand, supply and prices."""
    w = (raw.pivot_table(index=["state", "company_id", "year"], columns="item",
                         values="value", aggfunc="sum").reset_index())
    for col in ITEMS.values():
        if col not in w:
            w[col] = np.nan
    w["company"] = w["company_id"].map(CANONICAL)
    names = raw.drop_duplicates("company_id").set_index("company_id")["eia_name"]
    w["company"] = w["company"].fillna(w["company_id"].map(names).str.title())
    # keep distribution utilities only
    big = w.groupby("company_id")["res_sales_customers"].max()
    w = w[w["company_id"].isin(big[big >= MIN_RES_CUSTOMERS].index)]
    # merge predecessor filings into the successor company
    sum_cols = [c for c in ITEMS.values() if c != "heat_content_btu_cf"]
    agg = {c: "sum" for c in sum_cols}
    agg["heat_content_btu_cf"] = "mean"
    t = w.groupby(["state", "company", "year"]).agg(agg).reset_index()
    t[sum_cols] = t[sum_cols].replace(0, np.nan)

    t["ldc_volume_mcf"] = t[["res_volume", "com_volume", "ind_volume"]].sum(axis=1, min_count=1)
    t["res_customers"] = t[["res_sales_customers", "res_transport_customers"]].sum(axis=1, min_count=1)
    t["total_customers"] = t[["res_sales_customers", "res_transport_customers",
                              "com_sales_customers", "com_transport_customers",
                              "ind_sales_customers", "ind_transport_customers"]].sum(axis=1, min_count=1)
    t["citygate_receipts_mcf"] = t[["citygate_sales_volume", "citygate_transport_volume"]].sum(
        axis=1, min_count=1)
    t["transport_share_pct"] = t["citygate_transport_volume"] / t["citygate_receipts_mcf"] * 100
    t["unaccounted_pct"] = t["unaccounted_for"] / t["total_supply"] * 100

    therms_per_mcf = (t["heat_content_btu_cf"].where(t["heat_content_btu_cf"].between(900, 1200))
                      .fillna(1037) / 100)
    for sec in ("res", "com"):
        price_mcf = t[f"{sec}_sales_revenue"] / t[f"{sec}_sales_volume"]
        t[f"{sec}_price_per_mcf"] = price_mcf
        t[f"{sec}_price_per_therm"] = price_mcf / therms_per_mcf
    t["use_per_res_customer_therms"] = (t["res_volume"] * therms_per_mcf) / t["res_customers"]
    return t.sort_values(["state", "company", "year"]).reset_index(drop=True)


def shares(t: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """Each company's share of its state's utility deliveries.

    share: average over the latest `years` reported years.
    share_trend_pts: change in share over the last five years (points).
    """
    t = t.copy()
    t["state_total"] = t.groupby(["state", "year"])["ldc_volume_mcf"].transform("sum")
    t["share_y"] = t["ldc_volume_mcf"] / t["state_total"]
    last = t["year"].max()
    recent = t[t["year"] > last - years]
    sh = recent.groupby(["state", "company"])["share_y"].mean().rename("share").reset_index()
    sh["share"] = sh["share"] / sh.groupby("state")["share"].transform("sum")
    then = t[t["year"] == last - 5].set_index(["state", "company"])["share_y"]
    now = t[t["year"] == last].set_index(["state", "company"])["share_y"]
    sh = sh.join((now - then).rename("share_trend_pts") * 100, on=["state", "company"])
    return sh


def company_forecasts(region_fc: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    """Split each state's monthly forecast (normal/cold/warm, interval) by share."""
    f = region_fc.merge(sh[["state", "company", "share"]], on="state", how="inner")
    for col in ("normal", "cold", "warm", "lo90", "hi90"):
        if col in f:
            f[col] = f[col] * f["share"]
    return f[[c for c in ("state", "company", "month", "normal", "cold", "warm", "lo90", "hi90",
                          "share") if c in f]]


def run(region_fc: pd.DataFrame | None = None, fetch_data: bool = False) -> pd.DataFrame | None:
    raw_path = config.RAW_DIR / RAW_FILE
    if fetch_data or not raw_path.exists():
        try:
            download()
        except Exception as e:
            print(f"  company download failed: {e}")
            return None
    raw = pd.read_csv(raw_path)
    t = company_table(raw)
    out = config.PROCESSED_DIR
    t.to_csv(out / "company_annual.csv", index=False)
    sh = shares(t)
    sh.to_csv(out / "company_shares.csv", index=False)
    prices = t[["state", "company", "year", "res_price_per_therm", "com_price_per_therm",
                "res_customers", "use_per_res_customer_therms"]].rename(
        columns={"res_price_per_therm": "residential_per_therm",
                 "com_price_per_therm": "commercial_per_therm"})
    prices.to_csv(out / "company_prices.csv", index=False)
    if region_fc is not None:
        company_forecasts(region_fc, sh).to_csv(out / "company_forecasts.csv", index=False)
    print(f"  {t['company'].nunique()} utilities across {t['state'].nunique()} states, "
          f"{t['year'].min()}–{t['year'].max()}")
    return t


if __name__ == "__main__":
    download()
