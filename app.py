"""Streamlit app: New England gas demand & Henry Hub price forecasts.

Reads the files written by `python -m src.pipeline` into data/processed/.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.plotting import (GRID, INK, INK_2, MODEL_COLORS, NEUTRAL, SCENARIO_COLORS,
                          SERIES, SURFACE)

st.set_page_config(page_title="New England Gas Forecast", layout="wide")
P = config.PROCESSED_DIR


# ------------------------------------------------------------------ data
@st.cache_data
def load():
    if not (P / "meta.json").exists():
        return None
    r = lambda f, **k: pd.read_csv(P / f, **k)
    return {
        "meta": json.loads((P / "meta.json").read_text()),
        "dfd": r("demand_frame.csv", index_col=0, parse_dates=True),
        "fc": r("demand_forecast.csv", index_col=0, parse_dates=True),
        "scores": r("demand_scores.csv", index_col=0),
        "scores_x": r("demand_scores_actualwx.csv", index_col=0),
        "mape_h": r("demand_mape_by_h.csv", index_col=0),
        "wn": r("demand_weather_normalized.csv", index_col=0, parse_dates=True),
        "bt": r("demand_backtest.csv", parse_dates=["origin", "date"]),
        "dfp": r("price_frame.csv", index_col=0, parse_dates=True),
        "ps1": r("price_scores_h1.csv", index_col=0),
        "ps4": r("price_scores_h4.csv", index_col=0),
        "wf4": r("price_wf_h4.csv", parse_dates=["date"]),
        "wf1": r("price_wf_h1.csv", parse_dates=["date"]),
        "latest": r("price_latest_forecasts.csv", parse_dates=["as_of"]),
        "shap": r("price_shap_h4.csv", index_col=0, parse_dates=True),
        "ablation": (r("price_risk_ablation.csv") if (P / "price_risk_ablation.csv").exists()
                     else None),
    }


def style(fig, title, ytitle, height=400):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", font=dict(color=INK_2)),
        font=dict(color=INK_2),
    )
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    return fig


def add_events(fig, start):
    """Dotted markers for geopolitical and market events (context only)."""
    for d, label in config.EVENTS:
        d = pd.Timestamp(d)
        if d >= start:
            fig.add_vline(x=d, line=dict(color=INK_2, width=1, dash="dot"))
            fig.add_annotation(x=d, y=1, yref="paper", text=label, showarrow=False,
                               textangle=-90, xanchor="right", yanchor="top",
                               font=dict(size=10, color=INK_2))


D = load()
st.title("New England Natural Gas: Demand & Price Forecasting")
st.caption("Massachusetts monthly gas demand and weekly Henry Hub spot price · public EIA and "
           "ERA5 weather data · Created by Cathy Kam")

if D is None:
    st.warning("No processed data found. Run `python -m src.pipeline --pull` with "
               "`EIA_API_KEY` set, then commit `data/processed/`.")
    st.stop()

m = D["meta"]
tab_d, tab_p, tab_m = st.tabs(["Massachusetts demand", "Henry Hub price", "Method"])

# ------------------------------------------------------------------ demand
with tab_d:
    fc, dfd, sc = D["fc"], D["dfd"], D["scores"]
    best = m["best_demand_model"]
    winter = fc[fc.index.month.isin([11, 12, 1, 2, 3])]
    naive = sc.at["Seasonal naive", "MAPE %"] if "Seasonal naive" in sc.index else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next-winter demand, normal weather", f"{winter['normal'].sum()/1000:,.0f} Bcf",
              help="Sum of forecast Nov-Mar months within the 12-month horizon")
    c2.metric("Cold-winter upside", f"{(winter['cold'].sum()/winter['normal'].sum()-1)*100:+.1f}%")
    c3.metric(f"Backtest MAPE · {best}", f"{sc.at[best, 'MAPE %']:.1f}%",
              f"{sc.at[best, 'MAPE %'] - naive:+.1f} pts vs naive", delta_color="inverse")
    c4.metric("Weather sensitivity", f"{m['mmcf_per_hdd']:,.1f} MMcf/HDD")

    hist = dfd["y"].iloc[-36:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist / 1000, name="Actual",
                             line=dict(color=NEUTRAL, width=2)))
    if "lo90" in fc:
        fig.add_trace(go.Scatter(x=list(fc.index) + list(fc.index[::-1]),
                                 y=list(fc["hi90"] / 1000) + list(fc["lo90"][::-1] / 1000),
                                 fill="toself", fillcolor="rgba(42,120,214,0.15)",
                                 line=dict(width=0), name="90% interval", hoverinfo="skip"))
    for s_, dash in (("cold", "dash"), ("normal", "solid"), ("warm", "dash")):
        fig.add_trace(go.Scatter(x=fc.index, y=fc[s_] / 1000, name=f"{s_.title()} weather",
                                 line=dict(color=SCENARIO_COLORS[s_], width=2.5 if s_ == "normal" else 1.8,
                                           dash=dash)))
    st.plotly_chart(style(fig, f"12-month forecast · {best}", "Bcf per month"),
                    width="stretch")
    st.caption(f"Data through {m['demand_data_through']}. Cold/warm = 90th/10th percentile "
               "monthly HDD over the last 30 years. Interval from backtest errors at each horizon.")

    a, b = st.columns(2)
    with a:
        st.subheader("Backtest scorecard")
        view = st.radio("Future weather in backtest", ["Normal (ex-ante)", "Actual (ex-post)"],
                        horizontal=True)
        tbl = sc if view.startswith("Normal") else D["scores_x"]
        st.dataframe(tbl.drop(columns=["n"]).style.format("{:.2f}"), width="stretch")
        st.caption("Rolling-origin backtest, 12-month horizon, origins every quarter since Dec 2015.")
    with b:
        mh = D["mape_h"]
        fig = go.Figure()
        for col in mh.columns:
            fig.add_trace(go.Scatter(x=mh.index, y=mh[col], name=col, mode="lines+markers",
                                     line=dict(color=MODEL_COLORS.get(col, NEUTRAL), width=2),
                                     marker=dict(size=7)))
        fig.update_xaxes(title="Months ahead")
        fig = style(fig, "Error by horizon", "MAPE (%)", 460)
        fig.update_layout(legend=dict(orientation="h", y=-0.22, yanchor="top", x=0),
                          margin=dict(b=90))
        st.plotly_chart(fig, width="stretch")

    wn = D["wn"]
    roll = wn[["y", "y_wn"]].rolling(12).sum().dropna() / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll.index, y=roll["y"], name="Actual", line=dict(color=NEUTRAL, width=1.5)))
    fig.add_trace(go.Scatter(x=roll.index, y=roll["y_wn"], name="Weather-normalized",
                             line=dict(color=SERIES[0], width=2.5)))
    st.plotly_chart(style(fig, "Rolling 12-month demand, actual vs. weather-normalized", "Bcf"),
                    width="stretch")
    st.caption("Weather-normalized = actual + sensitivity × (normal HDD − actual HDD). "
               "Its trend shows structural change: efficiency, electrification, customer growth.")

# ------------------------------------------------------------------ price
with tab_p:
    lat = D["latest"]
    h_choice = st.radio("Horizon", ["1 week", "4 weeks"], horizontal=True, index=1)
    h = 1 if h_choice.startswith("1") else 4
    ps = D["ps1"] if h == 1 else D["ps4"]
    wf = D["wf1"] if h == 1 else D["wf4"]
    challengers = ps.drop(index="Random walk")
    best_p = challengers["RMSE log-return"].idxmin()

    cur = lat[(lat["h"] == h) & (lat["model"] == best_p)].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Henry Hub, week ending {cur['as_of']:%b %d, %Y}", f"${cur['price_now']:.2f}")
    c2.metric(f"{h}-week forecast · {best_p}", f"${cur['price_fcst']:.2f}",
              f"{(cur['price_fcst']/cur['price_now']-1)*100:+.1f}%")
    p_ = ps.at[best_p, "DM p-value"]
    rel = ps.at[best_p, "RMSE vs RW %"]
    if rel < 0 and p_ < 0.05:
        verdict = f"beats random walk (p={p_:.3f})"
    elif rel < 0:
        verdict = f"edge not significant (p={p_:.2f})"
    else:
        verdict = "random walk is better"
    c3.metric("RMSE vs. random walk (lower is better)", f"{rel:+.1f}%", verdict,
              delta_color="off")

    g = wf[wf["model"] == best_p].set_index("date")
    rw = wf[wf["model"] == "Random walk"].set_index("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rw.index, y=rw["price_actual"], name=f"Actual ({h} wk later)",
                             line=dict(color=NEUTRAL, width=1.5)))
    fig.add_trace(go.Scatter(x=g.index, y=g["price_fcst"], name=f"{best_p} forecast",
                             line=dict(color=MODEL_COLORS.get(best_p, SERIES[0]), width=1.5)))
    add_events(fig, g.index.min())
    st.plotly_chart(style(fig, f"Out-of-sample {h}-week-ahead forecasts", "$/MMBtu"),
                    width="stretch")

    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("Walk-forward scorecard")
        st.dataframe(ps.astype(float).map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"), width="stretch")
        st.caption("Negative DM stat with p < 0.05 = statistically better than a random walk "
                   "(Diebold-Mariano, Newey-West variance, HLN correction).")
    with b:
        imp = D["shap"].abs().mean().sort_values()
        fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h",
                               marker=dict(color=SERIES[0])))
        fig.update_layout(hovermode="closest")
        fig = style(fig, "What drives the 4-week model (mean |SHAP|)", "", 420)
        fig.update_xaxes(title="pts of 4-week % change", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")

    dfp = D["dfp"]
    st.subheader("Geopolitical and trade-policy risk")
    ab = D["ablation"]
    if ab is not None:
        r_ = ab[ab["h"] == h].copy()
        for _, row in r_.iterrows():
            better = row["RMSE change %"] < 0
            sig = row["DM p-value"] < 0.05
            msg = (f"**{row['challenger']}** vs. the same model without risk features, "
                   f"{h}-week horizon: RMSE {row['RMSE change %']:+.1f}% "
                   f"(DM p = {row['DM p-value']:.2f}). ")
            msg += ("Risk features significantly improve the forecast." if better and sig else
                    "Small improvement, not statistically significant." if better else
                    "Risk features make it significantly worse (overfitting)." if sig else
                    "Risk features do not improve the forecast.")
            st.markdown("- " + msg)
        st.caption("War, sanctions and tariffs are mostly unpredictable shocks. These features test "
                   "whether measured risk levels (news-based indices, global prices) carry "
                   "forecastable information once the shock has started.")
    rc = [c for c in ("gpr_4w", "epu_4w") if c in dfp]
    if rc:
        recent = dfp[rc].dropna().iloc[-520:]
        names = {"gpr_4w": "Geopolitical risk (Caldara-Iacoviello)",
                 "epu_4w": "Economic policy uncertainty (Baker-Bloom-Davis)"}
        fig = go.Figure()
        for i, c in enumerate(rc):
            fig.add_trace(go.Scatter(x=recent.index, y=np.exp(recent[c]), name=names[c],
                                     line=dict(color=SERIES[[6, 1][i]], width=1.8)))
        add_events(fig, recent.index.min())
        st.plotly_chart(style(fig, "Risk indices, 4-week average (100 ≈ long-run average)", "Index"),
                        width="stretch")
    if "eu_hh_spread" in dfp:
        sp = dfp["eu_hh_spread"].dropna()
        fig = go.Figure(go.Scatter(x=sp.index, y=sp, name="Europe minus Henry Hub",
                                   line=dict(color=SERIES[0], width=1.8)))
        add_events(fig, sp.index.min())
        st.plotly_chart(style(fig, "Europe–US gas price spread (pull on US LNG exports)", "$/MMBtu"),
                        width="stretch")

    s = dfp["storage_surplus_pct"].dropna().iloc[-260:]
    fig = go.Figure(go.Bar(x=s.index, y=s.values,
                           marker=dict(color=np.where(s.values >= 0, SERIES[0], SERIES[1]))))
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, "Lower 48 storage vs. prior 5-year average (as known each week)", "%"),
                    width="stretch")

# ------------------------------------------------------------------ method
with tab_m:
    st.markdown(f"""
### Data
* **Demand:** EIA monthly natural gas deliveries to consumers in Massachusetts (MMcf), by end use.
* **Price:** EIA daily Henry Hub spot, averaged to weeks ending Friday.
* **Drivers:** EIA weekly working gas in storage (Lower 48), EIA monthly LNG exports.
* **Geopolitical / trade risk:** Caldara-Iacoviello Geopolitical Risk index (daily), Baker-Bloom-Davis
  Economic Policy Uncertainty (daily) and trade-policy uncertainty (monthly), Brent crude and the
  IMF European gas price (FRED). All lagged to when they were published.
* **Weather:** ERA5 daily max/min temperature via Open-Meteo. MA degree days weight Boston,
  Worcester and Springfield; the price model uses a five-city national proxy. HDD/CDD base 65°F.

### Demand models
Seasonal naive · HDD regression · SARIMAX (HDD/CDD exogenous) · Prophet (HDD/CDD regressors) ·
LightGBM · Hybrid (HDD regression + LightGBM on residuals). Expanding-window rolling-origin
backtest, 12-month horizon, run once with **normal weather** (realistic) and once with **actual
weather** (model skill only).

### Price models
Random walk · Ridge (momentum) · Ridge (fundamentals) · Ridge (all) · LightGBM. Target is the
log change in weekly average price 1 or 4 weeks ahead. Walk-forward, refit every 13 weeks.

### Leakage controls
* Storage lagged one week (published the Thursday after the report week).
* LNG exports used only after their ~2-month publication delay.
* Seasonal "normals" computed from prior years only.
* Models trained only on targets already realized at each refit date.
* These rules are covered by unit tests in `tests/`.

*Pipeline last run {m['run_at']}. Demand data through {m['demand_data_through']}; price data
through {m['price_data_through']}.*
""")
