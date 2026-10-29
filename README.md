# New England Natural Gas: Demand & Price Forecasting

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ne-gas-forecast-zmi45pfx6pdyncuxxazxcq.streamlit.app/)

Forecasts **New England utility gas demand by state and company**, breaks down **Massachusetts gas prices per therm into GAF and LDAF** for every utility since 2008, and forecasts **Massachusetts monthly natural gas demand** 1–12 months ahead under normal, cold and warm weather, and the **Henry Hub spot price** 1 and 4 weeks ahead, using only public data. Built to the standards a utility or energy-trading team would expect: weather normalization, rolling-origin backtests, honest benchmarks, and explicit controls for look-ahead leakage.

**App tabs:** Gas companies (map, treemap, leaderboard and deep dive for every New England utility) · New England by state · Massachusetts demand · MA price per therm (GAF/LDAF) · Henry Hub price · Method

**Live app:** [https://ne-gas-forecast-zmi45pfx6pdyncuxxazxcq.streamlit.app/](https://ne-gas-forecast-zmi45pfx6pdyncuxxazxcq.streamlit.app/) · **Dashboard site (Henry Hub, Massachusetts GAF):** [https://cathykam2006.github.io/ne-gas-forecast/](https://cathykam2006.github.io/ne-gas-forecast/) · **Author:** Cathy Kam

---

## Why it matters

Gas utilities plan supply portfolios, storage and pipeline capacity around winter demand under a range of weather outcomes, and regulators review those plans and the costs passed to customers. Two practical questions drive this project:

1. **How much gas will Massachusetts need next winter, and how much more if it's cold?**
2. **Do supply/demand fundamentals tell us anything about near-term Henry Hub prices that a "no change" forecast doesn't?**

## Results

Backtests on data through July 2026 (demand) and October 2026 (price):

| Forecast | Best model | Out-of-sample accuracy | vs. benchmark |
|---|---|---|---|
| MA utility gas demand, 1–12 months, normal weather | SARIMAX | 8.1% MAPE | 18% better than seasonal naive |
| Same, with actual weather (model skill only) | LightGBM / SARIMAX | 5.9% MAPE | 40% better than seasonal naive |
| Henry Hub, 1 week ahead | Ridge (fundamentals) | 0.9% lower RMSE than random walk | statistically significant (DM p = 0.03) |
| Henry Hub, 4 weeks ahead | Ridge (momentum) | 6.4% lower RMSE than random walk | 55% direction hit rate; not significant (p = 0.31) |

**Key findings**
- **Weather sensitivity:** each heating degree day adds about 30 MMcf of utility gas demand.
- **Winter outlook:** about 183 Bcf of utility demand in a normal winter; a cold (P90) winter adds about 9%.
- **Structural trend:** weather-normalized demand has been roughly flat (−0.3% a year) over the last decade.
- **Weather dominates forecast error:** with weather known, error falls from 8.1% to 5.9%, so better weather forecasts would help more than more complex models.
- **Prices:** fundamentals give a small but statistically real edge at one week; nothing reliably beats a random walk at four weeks, consistent with an efficient market.
- **What didn't work:** news-based geopolitical and trade-policy risk indices (tested separately) did not improve the price forecast, and Prophet underperformed the naive benchmark on demand because its trend extrapolation overshoots.

## Approach

### Data (all public)
| Data | Source | Frequency |
|---|---|---|
| MA deliveries to consumers by end use (MMcf) | EIA API v2 (`N3060MA2`, `N3010MA2`, `N3020MA2`, `N3035MA2`, `N3045MA2`) | Monthly |
| Henry Hub spot (<span>$</span>/MMBtu) | EIA API v2 (`RNGWHHD`) | Daily |
| NYMEX Henry Hub futures (national benchmark, delivered at Henry Hub, Louisiana; not a New England price): front month and the futures price for each of the next 24 delivery months (<span>$</span>/MMBtu) | CME NYMEX contracts via Yahoo Finance (`NG=F`, `NG<month><yy>.NYM`); EIA stopped publishing futures after April 5, 2024 | Daily (weekdays) |
| Working gas in storage, Lower 48 / East (Bcf) | EIA Weekly Natural Gas Storage Report | Weekly |
| U.S. LNG exports (MMcf) | EIA API v2 (`N9133US2`) | Monthly |
| Daily max/min temperature | Open-Meteo ERA5 archive (no key) | Daily |

Massachusetts heating degree days weight Boston (70%), Worcester (18%) and Springfield (12%). The price model uses a five-city national weather proxy.

### Demand target and design choices (tested on the data)
- **Target = utility (LDC) load: residential + commercial + industrial.** Power plants burn about a third of Massachusetts gas, but that use follows electricity markets and runs *opposite* to heating weather (correlation with HDD of −0.5, vs. +0.96 for LDC load). Including it made every weather model worse.
- **15-year weather normals, not 30.** Massachusetts averaged ~12% fewer heating degree days in the 2020s than in the 2000s, so 30-year normals bias forecasts upward. 15-year normals backtested best.
- **10-year window for the regression models.** EIA sector data have structural breaks (e.g. a large shift into the commercial category around 2013), and fitting all history at once made the HDD regression over-forecast summers by 20–30%.

### Demand models
Seasonal naive (benchmark) · **HDD regression** (utility standard, interpretable) · **SARIMAX** with HDD/CDD · **Prophet** with HDD/CDD regressors · **LightGBM** · **Hybrid**: HDD regression + LightGBM on its residuals.

Evaluated with an expanding-window rolling-origin backtest (origins every quarter since Dec 2015, 12-month horizon), run twice:
- **Normal weather (ex-ante):** future HDD set to 15-year normals known at the origin, i.e. a real forecast.
- **Actual weather (ex-post):** future HDD set to what happened, isolating model skill from weather uncertainty.

Forecasts are produced for **normal, cold (P90 HDD) and warm (P10 HDD)** winters, with 90% intervals from each model's own backtest errors by horizon. Demand history is also **weather-normalized** to separate warm winters from structural change such as efficiency and electrification.

### Price models
Target: log change in weekly average Henry Hub price, 1 and 4 weeks ahead.
Random walk (benchmark) · Ridge on momentum · Ridge on fundamentals · Ridge on all features · LightGBM. Features: storage surplus vs. prior 5-year average, injections vs. normal, national degree-day anomaly, LNG export level and growth, price momentum and volatility, seasonality.

Walk-forward evaluation with quarterly refits. Every challenger is compared to the random walk with a **Diebold–Mariano test** (Newey–West variance, Harvey–Leybourne–Newbold correction), so a small edge isn't mistaken for skill. LightGBM's native **TreeSHAP** shows which drivers move the forecast.

**The market benchmark.** The NYMEX front-month price at the end of each week is the futures market's own forecast. It's scored on the same weeks as the models, and each model is tested against it with a Diebold–Mariano test. Beating a random walk is the minimum; beating the futures market is the real bar.

### NYMEX futures
`src/nymex.py` keeps the front-month history (2001 onward) and a dated daily history of the futures price for each delivery month (the "forward strip"). Updates are incremental and checked before anything is saved:
- prices outside <span>$</span>0.50–<span>$</span>50/MMBtu are rejected, and a feed that is mostly implausible fails;
- contracts are dropped once they stop trading (three business days before the delivery month);
- quotes without a settlement in the past week are rejected;
- a failed download never overwrites good history.

The Henry Hub tab shows the front-month price against spot, and a table and chart of the futures price for each of the next 24 delivery months, with the change over the past week.

### New England by state and company
The Massachusetts method (utility-load target, 15-year normals, rolling-origin backtest) runs for Connecticut, Rhode Island, New Hampshire, Maine and Vermont (each state's history ends at its last fully reported month, since EIA publishes or withholds small-state sectors late), each with its own weather stations; each state keeps whichever model backtests best. For each of the ~25 New England gas utilities (EIA Form 176 annual company reports, 2010 onward, read from EIA's Natural Gas Query System):
- **Demand:** residential, commercial and industrial deliveries (sales + transportation), customer counts, use per home; a 12-month forecast that splits the state forecast by the company's share of state utility deliveries (latest three years).
- **Supply:** gas received at the citygate for sales customers and for transportation (customer-choice) customers, LNG withdrawals, total supply, and unaccounted-for gas.
- **Price:** average residential and commercial price per therm (sales revenue ÷ volume, using each company's reported heat content).
Predecessor filings are merged into successors (Colonial → Boston Gas, Columbia → Eversource Gas of MA, Narragansett → Rhode Island Energy); pipelines, LNG and CNG suppliers are excluded.

### Massachusetts price per therm: GAF and LDAF
- **GAF** (Gas Adjustment Factor) recovers gas supply, interstate transportation and storage costs dollar-for-dollar; set per peak/off-peak season with mid-season revisions.
- **LDAF** (Local Distribution Adjustment Factor) recovers distribution program costs (energy efficiency from 2025/26, GSEP pipe replacement, environmental response, residential assistance and others).

`src/tariffs.py` parses the DPU's current-rates page and historical workbook (6,700+ filed rates, 170 revisions, May 2008 onward) into monthly <span>$</span>/therm factors for each company and rate category (residential heating/non-heating, small/medium/large C&I), maps predecessor names (Bay State/Columbia → Eversource Gas of MA; KeySpan → National Grid; New England Gas → Liberty), converts <span>$</span>/Ccf rates to <span>$</span>/therm, and compares GAF with the Henry Hub price. Base distribution rates and customer charges are set in rate cases and are not included.

### Leakage controls (unit-tested in `tests/`)
- Storage enters with a one-week lag (the report for week *t* is released the following Thursday).
- LNG exports are used only after a two-month publication delay.
- Seasonal "normals" use prior years only.
- Models train only on targets already realized at each refit date; the backtest never sees the future.

## Repository

```
├── app.py                      Streamlit dashboard
├── src/
│   ├── config.py               series IDs, stations, publication lags
│   ├── data.py                 EIA API v2 + Open-Meteo pulls
│   ├── features.py             degree days, normals, storage/LNG features
│   ├── demand_models.py        six demand models, common interface
│   ├── demand_backtest.py      rolling-origin backtest, scenarios, weather normalization
│   ├── price_models.py         price models, walk-forward, Diebold–Mariano
│   ├── region.py               New England state forecasts
│   ├── companies.py            EIA-176 company supply, demand, prices; company forecasts
│   ├── tariffs.py              MA GAF/LDAF parser (DPU page + workbook)
│   ├── nymex.py                NYMEX futures: front month, prices by delivery month, market benchmark
│   ├── outlook.py              forward-looking price and demand forecasts shared by the app and site
│   ├── pipeline.py             end-to-end run → data/processed/
│   └── plotting.py             shared chart style
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_demand_forecasting.ipynb
│   ├── 04_price_forecasting.ipynb
│   └── 05_ma_price_per_therm.ipynb
├── site/                       Observable Framework dashboard (static, GitHub Pages)
│   ├── src/index.md            Henry Hub price outlook
│   ├── src/massachusetts-demand.md  Massachusetts demand outlook
│   ├── src/massachusetts-gaf.md  Massachusetts GAF and LDAF by company
│   └── src/data/*.json.py      data loaders: read data/processed at build time
├── tests/                      leakage and sanity tests (synthetic fixture)
└── .github/workflows/
    ├── refresh.yml             weekly: all data, models and forecasts (Fridays)
    ├── nymex-daily.yml         weekdays after settlement: NYMEX futures only
    ├── site.yml                rebuilds and publishes the dashboard site after each data update
    └── keep-awake.yml          every 6 hours: keeps the Streamlit app from sleeping
```

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-pipeline.txt

export EIA_API_KEY=your_key        # free: https://www.eia.gov/opendata/register.php
python -m src.pipeline --pull      # pulls EIA, weather and DPU data, runs everything (~15-25 min)
pytest -q tests/                   # leakage tests
streamlit run app.py
```

Then work through the notebooks in order.

## Deploy

1. Commit `data/processed/` (the app reads only these files; no API key needed to view).
2. On Streamlit Community Cloud, point a new app at `app.py`. `requirements.txt` holds only the app's light dependencies.
3. Optional: add `EIA_API_KEY` as a GitHub Actions secret to enable the weekly refresh workflow, which reruns the pipeline every Friday after the storage report and commits new results.
4. The NYMEX daily update needs no key. It runs each weekday evening after settlement and commits only the NYMEX files. If the data goes more than four business days without an update, the run fails and GitHub emails you. The app reloads its data whenever these files change, with no restart.

### Dashboard site (Observable Framework)

`site/` is a static version of the Henry Hub, Massachusetts demand and Massachusetts GAF tabs built with [Observable Framework](https://observablehq.com/framework/). Its Python data loaders read `data/processed/`, so it shows the same numbers as the Streamlit app. The `Publish site` workflow rebuilds it and deploys it to GitHub Pages after every NYMEX daily update and weekly refresh, and after any push that changes the site or the data. It loads instantly and never sleeps.

One-time setup: in the repository, **Settings → Pages → Build and deployment → Source: GitHub Actions**, then **Actions → Publish site → Run workflow**.

To work on it locally (Node 20+): `cd site && npm install && npm run dev`, then open http://127.0.0.1:3000.

## Limitations and next steps
- EIA publishes state deliveries monthly with a ~2-month lag; daily utility sendout data would allow day-ahead and design-day modeling.
- The price model uses realized weather; adding NOAA 6–10 day forecasts would capture what traders actually react to.
- New England basis (Algonquin citygate) is the price that matters most locally, but daily history is not freely available.

---
*Created by Cathy Kam. Uses public data only.*
