"""Massachusetts gas supply and delivery adjustment factors, $/therm.

* GAF  (Gas Adjustment Factor): recovers the cost of buying, transporting
  and storing gas, passed through dollar-for-dollar. Set for the peak
  (Nov-Apr) and off-peak (May-Oct) seasons, with mid-season revisions.
* LDAF (Local Distribution Adjustment Factor): recovers distribution-related
  program costs (energy efficiency, environmental response, residential
  assistance, GSEP pipe replacement and others).

Source: Massachusetts DPU, "Information on Gas Supply and Delivery Charges"
(current rates page) and its "Historical GAF/LDAF rates and data" workbook.
Both are public. Base distribution rates and customer charges are set in
each company's rate case and are not part of these files.

    python -m src.tariffs        # download both files and build the tables
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

PAGE_URL = "https://www.mass.gov/info-details/cost-of-gas-adjustment-factor-rates-and-information"
WORKBOOK_URL = "https://www.mass.gov/doc/historical-gafldaf-rates-and-data/download"
PAGE_FILE = "gaf_rates_page.html"
WORKBOOK_FILE = "gaf_history.xlsx"

THERMS_PER_CCF = 1.037   # 100 cubic feet at ~1,037 Btu/cf (EIA average heat content)

CATEGORIES = ["Residential heating", "Residential non-heating",
              "Small C&I", "Medium C&I", "Large C&I"]

# Order matters: the first matching rule wins.
_COMPANY_RULES = [
    ("NSTAR", "NSTAR Gas (Eversource)"),
    ("BAY ?STATE|COLUMBIA|EVERSOURCE GAS COMPANY", "Eversource Gas of MA (fmr. Bay State/Columbia)"),
    ("BERKSHIRE", "Berkshire Gas"),
    ("BLACKSTONE", "Liberty – Blackstone"),
    ("FITCHBURG|UNITIL", "Unitil – Fitchburg Gas"),
    ("BOSTON", "National Grid – Boston Gas"),
    ("ESSEX", "National Grid – Essex Gas (merged)"),
    ("COLONIAL.*CAPE", "National Grid – Colonial Gas (Cape div.)"),
    ("COLONIAL", "National Grid – Colonial Gas"),
    ("NORTH ATTLEBORO$", "Liberty – North Attleboro div. (merged)"),
    ("FALL RIVER|NEW ENGLAND (NATURAL )?GAS", "Liberty – Fall River/N. Attleboro"),
]


def canonical_company(name: str) -> str | None:
    up = re.sub(r"\s+", " ", str(name)).upper().strip()
    for pattern, canon in _COMPANY_RULES:
        if re.search(pattern, up):
            return canon
    return None


def classify(rate_class: str) -> list[str]:
    """Map a tariff rate-class label to standard categories (may be several).

    R-1/R-2 = residential non-heating, R-3/R-4 = residential heating
    (R-2/R-4 are the low-income versions). G-4x/G-5x = C&I with high/low
    winter use; the second digit sets size: 0-1 small, 2 medium, 3 large.
    """
    s = str(rate_class).upper()
    if "RESIDENTIAL" in s and not re.search(r"R-?\s?\d", s.replace("RESIDENTIAL", "")) \
            or "R1 - R4" in s:
        return ["Residential heating", "Residential non-heating"]
    r = {int(d) for d in re.findall(r"R-?\.?\s?(\d)", s)}
    if r:
        heat, nonheat = bool(r & {3, 4}), bool(r & {1, 2})
        if heat and not nonheat:
            return ["Residential heating"]
        if nonheat and not heat:
            return ["Residential non-heating"]
        if r == {1, 3}:          # a typo in a few sheets for "R-1, R-2"
            return ["Residential non-heating"]
        return []                # e.g. "R-2, R-4": low-income only
    if re.fullmatch(r"\s*(C&I|C&I AND OTHER)\s*", s):
        return ["Small C&I", "Medium C&I", "Large C&I"]
    if "SMALL C&I" in s:
        return ["Small C&I"]
    if "MEDIUM C&I" in s:
        return ["Medium C&I"]
    if "LARGE C&I" in s:
        return ["Large C&I"]
    cats = set()
    # explicit codes like G-41, G-52B; shorthand "G-41, 42, 43" handled too
    for m in re.finditer(r"G-?\s?([45])(\d)", s):
        size = int(m.group(2))
        cats.add({0: "Small C&I", 1: "Small C&I", 2: "Medium C&I", 3: "Large C&I"}.get(size, ""))
    for m in re.finditer(r",\s*([45])([0-3])\b", s):
        cats.add({0: "Small C&I", 1: "Small C&I", 2: "Medium C&I", 3: "Large C&I"}[int(m.group(2))])
    cats.discard("")
    return [c for c in CATEGORIES if c in cats]


def _num(v) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).replace("\t", "").replace("$", "").replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip()
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -x if neg else x


def _to_therm(value: float, unit: str) -> float:
    u = str(unit).upper()
    if "THERM" in u and "MDCQ" not in u:
        return value
    if "CCF" in u:
        return value / THERMS_PER_CCF
    return np.nan     # $/MDCQ demand charges are not per-therm rates


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def parse_workbook(path: Path) -> pd.DataFrame:
    """One sheet per effective date; company heading, then a rate table."""
    xls = pd.ExcelFile(path)
    rows = []
    for sheet in xls.sheet_names:
        m = re.search(r"(\d{1,2})-(\d{1,2})-(\d{2})\s*$", sheet)
        if not m:          # table of contents, pre-2008 combined sheet
            continue
        eff = pd.Timestamp(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2)))
        d = pd.read_excel(xls, sheet, header=None).astype(object)
        col0 = d[0].map(lambda v: str(v).strip() if pd.notna(v) else "")
        company, cols = None, None
        for i in range(len(d)):
            label = col0[i]
            if label.lower().startswith("rate class"):
                company = canonical_company(col0[i - 1]) if i > 0 else None
                hdr = [str(v).strip().upper() for v in d.iloc[i]]
                cols = {"unit": next((j for j, h in enumerate(hdr) if "UNIT" in h), 1),
                        "gaf": next((j for j, h in enumerate(hdr) if h == "GAF"), 2),
                        "ldaf": next((j for j, h in enumerate(hdr) if h == "LDAF"), 3)}
                continue
            if not label:
                cols = None
                continue
            if cols is None or company is None:
                continue
            unit = d.iat[i, cols["unit"]]
            if not (isinstance(unit, str) and "$" in unit):
                continue
            rows.append({"effective": eff, "company": company, "rate_class": label,
                         "unit": unit.strip(),
                         "gaf": _num(d.iat[i, cols["gaf"]]),
                         "ldaf": _num(d.iat[i, cols["ldaf"]]) if cols["ldaf"] < d.shape[1] else np.nan,
                         "source": "DPU historical workbook"})
    return pd.DataFrame(rows)


def parse_page(path: Path) -> pd.DataFrame:
    """Current-rates page: <h2> period (with effective dates), <h3> company,
    then GAF/LDAF tables."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="ignore"), "html.parser")
    rows, eff, company = [], None, None
    for el in soup.find_all(["h2", "h3", "table"]):
        text = el.get_text(" ", strip=True)
        if el.name == "h2":
            m = re.search(r"Effective\s+(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
            eff = None
            if m:
                y = int(m.group(3))
                eff = pd.Timestamp(y + 2000 if y < 100 else y, int(m.group(1)), int(m.group(2)))
            company = None
        elif el.name == "h3":
            company = canonical_company(text)
        elif eff is not None and company is not None:
            trs = el.find_all("tr")
            if not trs:
                continue
            hdr = [c.get_text(" ", strip=True).upper() for c in trs[0].find_all(["th", "td"])]
            if "GAF" not in hdr:
                continue
            ig, il = hdr.index("GAF"), (hdr.index("LDAF") if "LDAF" in hdr else None)
            iu = next((j for j, h in enumerate(hdr) if "UNIT" in h), 1)
            for tr in trs[1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if len(cells) <= ig:
                    continue
                rows.append({"effective": eff, "company": company, "rate_class": cells[0],
                             "unit": cells[iu], "gaf": _num(cells[ig]),
                             "ldaf": _num(cells[il]) if il is not None and il < len(cells) else np.nan,
                             "source": "DPU current rates page"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_rate_table(workbook: Path | None, page: Path | None) -> pd.DataFrame:
    """All filed GAF/LDAF rows in $/therm, tagged with standard categories.

    Where the page and the workbook both cover an effective date, the page
    (the newer source) wins.
    """
    parts = []
    if page is not None and Path(page).exists():
        parts.append(parse_page(page))
    if workbook is not None and Path(workbook).exists():
        wb = parse_workbook(workbook)
        if parts:
            wb = wb[~wb["effective"].isin(set(parts[0]["effective"]))]
        parts.append(wb)
    if not parts:
        raise FileNotFoundError("No GAF/LDAF source files found")
    t = pd.concat(parts, ignore_index=True)
    t["gaf_therm"] = [_to_therm(v, u) for v, u in zip(t["gaf"], t["unit"])]
    t["ldaf_therm"] = [_to_therm(v, u) for v, u in zip(t["ldaf"], t["unit"])]
    t = t.dropna(subset=["gaf_therm"], how="all")
    t["categories"] = t["rate_class"].map(classify)
    return t.sort_values(["effective", "company"]).reset_index(drop=True)


def monthly_factors(table: pd.DataFrame, end: str | None = None) -> pd.DataFrame:
    """Month-by-month GAF and LDAF per company and category ($/therm).

    Each revision lists every company, so the rate in force in a month is
    the latest revision on or before the first of that month. Where several
    tariff classes fall in one category (e.g. G-41 and G-51), they are
    averaged.
    """
    ex = table.explode("categories").dropna(subset=["categories"])
    per_rev = (ex.groupby(["effective", "company", "categories"])[["gaf_therm", "ldaf_therm"]]
               .mean().reset_index().rename(columns={"categories": "category"}))
    revs = np.sort(per_rev["effective"].unique())
    last = pd.Timestamp(end) if end else pd.Timestamp(revs[-1])
    months = pd.date_range(pd.Timestamp(revs[0]), last, freq="MS")
    in_force = pd.Series(revs[np.searchsorted(revs, months.values, side="right") - 1], index=months)
    out = (pd.DataFrame({"month": months, "effective": in_force.values})
           .merge(per_rev, on="effective", how="left"))
    out["total_therm"] = out["gaf_therm"] + out["ldaf_therm"].fillna(0)
    return out.sort_values(["company", "category", "month"]).reset_index(drop=True)


def henry_hub_per_therm(henry_hub_daily: pd.Series) -> pd.Series:
    """Monthly average Henry Hub spot in $/therm (1 MMBtu = 10 therms)."""
    return (henry_hub_daily.resample("MS").mean() / 10).rename("henry_hub_therm")


def download(raw_dir: Path = config.RAW_DIR) -> tuple[Path, Path]:
    """Fetch both DPU files (public). Saves into data/raw/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (research; ne-gas-forecast)"}
    out = []
    for url, name in ((WORKBOOK_URL, WORKBOOK_FILE), (PAGE_URL, PAGE_FILE)):
        p = raw_dir / name
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
            print(f"  saved {name}")
        except Exception as e:
            print(f"  could not download {name} ({e}); save it by hand to {p}")
        out.append(p)
    return out[0], out[1]


def run(raw_dir: Path = config.RAW_DIR, out_dir: Path = config.PROCESSED_DIR,
        fetch: bool = True) -> pd.DataFrame:
    wb, page = (download(raw_dir) if fetch else
                (raw_dir / WORKBOOK_FILE, raw_dir / PAGE_FILE))
    table = build_rate_table(wb, page)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.drop(columns="categories").to_csv(out_dir / "ma_gaf_ldaf_filed.csv", index=False)
    monthly = monthly_factors(table)
    monthly.to_csv(out_dir / "ma_gaf_ldaf_monthly.csv", index=False)
    print(f"  {len(table)} filed rows, {table['effective'].nunique()} revisions, "
          f"{table['effective'].min():%Y-%m} to {table['effective'].max():%Y-%m}")
    return monthly


if __name__ == "__main__":
    import sys

    run(fetch="--no-download" not in sys.argv)
