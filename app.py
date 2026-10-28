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
from src.nymex import MARKET
from src.plotting import (GRID, INK, INK_2, MODEL_COLORS, NEUTRAL, SCENARIO_COLORS,
                          SERIES, SURFACE)

st.set_page_config(page_title="New England Gas Forecast", layout="wide")
P = config.PROCESSED_DIR


# ------------------------------------------------------------------ data
def data_stamp() -> float:
    """Latest change to any processed file. The cache below is keyed on it, so a refresh that
    commits new data shows up on the next page load instead of after an app restart."""
    return max((f.stat().st_mtime for f in P.glob("*") if f.is_file()), default=0.0)


@st.cache_data(max_entries=2)
def load(stamp: float):
    if not (P / "meta.json").exists():
        return None
    r = lambda f, **k: pd.read_csv(P / f, **k)
    opt = lambda f, **k: pd.read_csv(P / f, **k) if (P / f).exists() else None
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
        # optional extensions (present when the pipeline produced them)
        "region_fc": opt("region_forecasts.csv", parse_dates=["month"]),
        "region_sc": opt("region_scores.csv"),
        "region_hist": opt("region_history.csv", parse_dates=["month"]),
        "tariff": opt("ma_gaf_ldaf_monthly.csv", parse_dates=["month", "effective"]),
        "hh_therm": opt("henry_hub_monthly_therm.csv", index_col=0, parse_dates=True),
        "co_fc": opt("company_forecasts.csv", parse_dates=["month"]),
        "co_shares": opt("company_shares.csv"),
        "co_price": opt("company_prices.csv"),
        "co_annual": opt("company_annual.csv"),
        # NYMEX futures (updated every weekday by .github/workflows/nymex-daily.yml)
        "nx_daily": opt("nymex_daily.csv", index_col=0, parse_dates=True),
        "vm1": opt("price_vs_market_h1.csv", index_col="challenger"),
        "vm4": opt("price_vs_market_h4.csv", index_col="challenger"),
        "nx_strip": opt("nymex_strip_latest.csv", parse_dates=["as_of", "prior_as_of"]),
        "nx_meta": json.loads((P / "nymex_meta.json").read_text()) if (P / "nymex_meta.json").exists() else {},
    }


LEGEND_WIDTH = 620   # px a legend row is assumed to have (narrow screens); wider screens just get spare room
LEGEND_ROW = 22      # px per legend row


def legend_rows(fig) -> int:
    """How many rows a horizontal legend needs at LEGEND_WIDTH (over- rather than under-estimated)."""
    names = [t.name for t in fig.data if t.name and t.showlegend is not False]
    rows, used = (1 if names else 0), 0
    for n in names:
        w = 8 * len(n) + 50                                          # text + swatch + gap
        if used and used + w > LEGEND_WIDTH:
            rows, used = rows + 1, 0
        used += w
    return rows


def legend_below(fig):
    """Pin the legend to the bottom edge of the chart and reserve exactly enough room for it,
    so it never covers the title (above) or the x-axis dates (below the plot)."""
    rows = legend_rows(fig)
    axis = 30 + (22 if fig.layout.xaxis.title.text else 0)      # tick labels (+ axis title)
    fig.update_layout(
        legend=dict(orientation="h", x=0, xanchor="left", yref="container", y=0, yanchor="bottom",
                    font=dict(color=INK_2)),
        margin=dict(b=axis + rows * LEGEND_ROW + 8))
    return fig


def style(fig, title, ytitle, height=400):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=height,
        margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified",
        font=dict(color=INK_2),
    )
    fig.update_yaxes(title=ytitle, gridcolor=GRID, zeroline=False)
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    # Legend below the plot: above it, a legend that wraps onto 2+ lines covers the title.
    return legend_below(fig)


def add_events(fig, start):
    """Dotted markers for major market events (context only, not model inputs)."""
    for d, label in config.EVENTS:
        d = pd.Timestamp(d)
        if d >= start:
            fig.add_vline(x=d, line=dict(color=INK_2, width=1, dash="dot"))
            # background box keeps the label readable where it crosses a price line
            fig.add_annotation(x=d, y=1, yref="paper", text=label, showarrow=False,
                               textangle=-90, xanchor="right", yanchor="top",
                               bgcolor=SURFACE, opacity=0.95, borderpad=1,
                               font=dict(size=10, color=INK_2))


def scenario_chart(hist: pd.Series, fc: pd.DataFrame, title: str):
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
    return style(fig, title, "Bcf per month")


WINTER_M = [11, 12, 1, 2, 3]


D = load(data_stamp())
st.title("New England Natural Gas: Demand & Price Forecasting")
st.caption("New England utility gas demand by state and company, Massachusetts GAF/LDAF per therm, "
           "and Henry Hub spot and NYMEX futures prices · public EIA, CME NYMEX, Massachusetts DPU and "
           "ERA5 weather data · "
           "Created by Cathy Kam")

if D is None:
    st.warning("No processed data found. Run `python -m src.pipeline --pull` with "
               "`EIA_API_KEY` set, then commit `data/processed/`.")
    st.stop()

m = D["meta"]
tab_co, tab_ne, tab_d, tab_t, tab_p, tab_m = st.tabs(
    ["Gas companies", "New England by state", "Massachusetts demand", "MA price per therm",
     "Henry Hub price", "Method"])

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
    st.plotly_chart(style(fig, f"Massachusetts utility gas demand, 12-month forecast · {best}", "Bcf per month"),
                    width="stretch")
    st.caption(f"Data through {m['demand_data_through']}. Cold/warm = 90th/10th percentile "
               "monthly HDD over the last 15 years. Interval from backtest errors at each horizon.")

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
# ------------------------------------------------------------------ Gas companies
# Names used in the DPU GAF/LDAF files vs. the EIA company names
TARIFF_NAME = {
    "National Grid – Boston Gas": ["National Grid – Boston Gas", "National Grid – Colonial Gas"],
    "Eversource – NSTAR Gas": ["NSTAR Gas (Eversource)"],
    "Eversource Gas of MA": ["Eversource Gas of MA (fmr. Bay State/Columbia)"],
    "Berkshire Gas": ["Berkshire Gas"],
    "Liberty – Fall River/N. Attleboro": ["Liberty – Fall River/N. Attleboro"],
    "Liberty – Blackstone": ["Liberty – Blackstone"],
    "Unitil – Fitchburg Gas": ["Unitil – Fitchburg Gas"],
}
STATE_ORDER = ["MA", "CT", "RI", "NH", "ME", "VT"]
STATE_COLOR = {s: SERIES[i] for i, s in enumerate(STATE_ORDER)}

with tab_co:
    ca, sh, cf = D["co_annual"], D["co_shares"], D["co_fc"]
    if ca is None:
        st.info("Company data appears after `python -m src.companies` and a pipeline run.")
    else:
        names = {k: v["name"] for k, v in config.STATES.items()}
        last_y = int(ca["year"].max())
        lat = ca[ca["year"] == last_y].copy()
        lat["deliveries_bcf"] = lat["ldc_volume_mcf"] / 1e6
        if cf is not None:
            w = (cf[cf["month"].dt.month.isin(WINTER_M)].groupby(["state", "company"])[["normal", "cold"]]
                 .sum() / 1000).reset_index()
            lat = lat.merge(w, on=["state", "company"], how="left")
        if sh is not None:
            lat = lat.merge(sh[["state", "company", "share", "share_trend_pts"]],
                            on=["state", "company"], how="left")
        lat["label"] = lat["company"] + " (" + lat["state"] + ")"

        st.markdown(f"#### {len(lat)} gas utilities across New England · EIA Form 176, {last_y}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers served", f"{lat['total_customers'].sum() / 1e6:,.2f} M")
        c2.metric(f"Utility deliveries, {last_y}", f"{lat['deliveries_bcf'].sum():,.0f} Bcf")
        if "normal" in lat:
            c3.metric("Next-winter demand (normal)", f"{lat['normal'].sum():,.0f} Bcf",
                      f"{(lat['cold'].sum() / lat['normal'].sum() - 1) * 100:+.1f}% if cold",
                      delta_color="off")
        wavg = (lat["res_price_per_therm"] * lat["res_sales_volume"]).sum() / lat.loc[
            lat["res_price_per_therm"].notna(), "res_sales_volume"].sum()
        c4.metric(f"Avg residential price, {last_y}", f"${wavg:.2f}/therm")

        metric_opts = {
            "Residential price ($/therm)": ("res_price_per_therm", "mean"),
            "Customers": ("total_customers", "sum"),
            f"Deliveries {last_y} (Bcf)": ("deliveries_bcf", "sum"),
            "Next-winter demand, normal (Bcf)": ("normal", "sum"),
        }
        metric = st.radio("Map and chart metric", [k for k, v in metric_opts.items() if v[0] in lat],
                          horizontal=True)
        col, how = metric_opts[metric]

        m1, m2 = st.columns([1, 1.25])
        with m1:
            if how == "mean":   # volume-weighted state average price
                g = lat.dropna(subset=[col]).groupby("state").apply(
                    lambda d: (d[col] * d["res_sales_volume"]).sum() / d["res_sales_volume"].sum(),
                    include_groups=False)
            else:
                g = lat.groupby("state")[col].sum()
            tops = (lat.sort_values("deliveries_bcf", ascending=False).groupby("state")["company"]
                    .apply(lambda x: "<br>".join(x.head(4))))
            fig = go.Figure(go.Choropleth(
                locations=g.index, z=g.values, locationmode="USA-states",
                colorscale=[[0, "#dbe8f8"], [1, "#184f95"]], marker_line_color=SURFACE,
                marker_line_width=1.5, colorbar=dict(title="", thickness=10, len=0.6),
                text=[f"<b>{names.get(s, s)}</b><br>{tops.get(s, '')}" for s in g.index],
                hovertemplate="%{text}<br>" + metric + ": %{z:,.2f}<extra></extra>"))
            fig.update_geos(fitbounds="locations", visible=False, bgcolor=SURFACE)
            fig.update_layout(height=470, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor=SURFACE,
                              title=dict(text=f"{metric} by state", x=0, font=dict(size=15, color=INK)))
            st.plotly_chart(fig, width="stretch")
        with m2:
            tm = lat[lat["deliveries_bcf"] > 0]
            fig = go.Figure(go.Treemap(
                labels=[names.get(s, s) for s in STATE_ORDER if s in set(tm["state"])] + tm["label"].tolist(),
                parents=[""] * tm["state"].nunique() + [names.get(s, s) for s in tm["state"]],
                values=[tm.loc[tm["state"] == s, "deliveries_bcf"].sum() for s in STATE_ORDER
                        if s in set(tm["state"])] + tm["deliveries_bcf"].tolist(),
                branchvalues="total",
                marker=dict(colors=[STATE_COLOR[s] for s in STATE_ORDER if s in set(tm["state"])]
                            + [STATE_COLOR.get(s, NEUTRAL) for s in tm["state"]],
                            line=dict(color=SURFACE, width=2)),
                texttemplate="<b>%{label}</b><br>%{value:,.1f} Bcf<br>%{percentParent:.0%} of state",
                hovertemplate="%{label}<br>%{value:,.1f} Bcf · %{percentParent:.1%} of state"
                              "<extra></extra>"))
            fig.update_layout(height=470, margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor=SURFACE,
                              title=dict(text=f"Who serves each state · utility deliveries {last_y}",
                                         x=0, font=dict(size=15, color=INK)))
            st.plotly_chart(fig, width="stretch")

        # ranking chart for the chosen metric
        r = lat.dropna(subset=[col]).sort_values(col)
        fig = go.Figure(go.Bar(x=r[col], y=r["label"], orientation="h",
                               marker=dict(color=[STATE_COLOR.get(s, NEUTRAL) for s in r["state"]]),
                               hovertemplate="%{y}: %{x:,.2f}<extra></extra>"))
        fig.update_layout(hovermode="closest")
        fig = style(fig, f"{metric} by company (bar color = state)", "", max(380, 24 * len(r)))
        fig.update_xaxes(showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### All companies")
        tbl = lat.set_index("label")[[c for c in [
            "share", "share_trend_pts", "normal", "cold", "deliveries_bcf", "total_customers",
            "res_price_per_therm", "com_price_per_therm", "use_per_res_customer_therms",
            "transport_share_pct", "unaccounted_pct"] if c in lat]].sort_values("deliveries_bcf",
                                                                                 ascending=False)
        tbl.columns = [{"share": "Share of state", "share_trend_pts": "Share chg 5y (pts)",
                        "normal": "Winter Bcf (normal)", "cold": "Winter Bcf (cold)",
                        "deliveries_bcf": f"Deliveries {last_y} (Bcf)",
                        "total_customers": "Customers", "res_price_per_therm": "Res $/therm",
                        "com_price_per_therm": "Com $/therm",
                        "use_per_res_customer_therms": "Therms/home",
                        "transport_share_pct": "Choice share %", "unaccounted_pct": "UAF %"}[c]
                       for c in tbl.columns]
        tbl.index.name = "Company (state)"
        fmt = {"Share of state": "{:.1%}", "Share chg 5y (pts)": "{:+.1f}", "Winter Bcf (normal)": "{:,.2f}",
               "Winter Bcf (cold)": "{:,.2f}", f"Deliveries {last_y} (Bcf)": "{:,.2f}",
               "Customers": "{:,.0f}", "Res $/therm": "${:.2f}", "Com $/therm": "${:.2f}",
               "Therms/home": "{:,.0f}", "Choice share %": "{:.0f}", "UAF %": "{:+.1f}"}
        st.dataframe(tbl.style.format(fmt, na_rep="–"), width="stretch", height=420)
        st.caption("Share of state = average share of state utility deliveries, latest 3 years; winter = "
                   "Nov–Mar forecast from the state model split by share. Prices = sales revenue ÷ sales "
                   "volume (supply, delivery and fixed charges together). Choice share = gas delivered for "
                   "customers who buy supply from a marketer. UAF = unaccounted-for gas as % of supply.")

        # ---------------- company deep dive ----------------
        st.divider()
        st.markdown("### Company deep dive")
        opts = lat.sort_values(["state", "deliveries_bcf"], ascending=[True, False])["label"].tolist()
        default = next((i for i, o in enumerate(opts) if o.startswith("National Grid – Boston Gas")), 0)
        pick = st.selectbox("Company", opts, index=default)
        row = lat[lat["label"] == pick].iloc[0]
        co, stt = row["company"], row["state"]
        hist = ca[(ca["company"] == co) & (ca["state"] == stt)].set_index("year")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Share of state utility load", f"{row.get('share', np.nan):.1%}"
                  if pd.notna(row.get("share")) else "–",
                  f"{row.get('share_trend_pts', 0):+.1f} pts in 5 yrs" if pd.notna(row.get("share_trend_pts"))
                  else None, delta_color="off")
        k2.metric(f"Customers ({last_y})", f"{row['total_customers']:,.0f}")
        k3.metric(f"Residential price ({last_y})", f"${row['res_price_per_therm']:.2f}/therm"
                  if pd.notna(row["res_price_per_therm"]) else "–")
        if "normal" in row and pd.notna(row["normal"]):
            k4.metric("Next-winter demand", f"{row['normal']:,.1f} Bcf",
                      f"{row['cold']:,.1f} Bcf if cold", delta_color="off")

        d1, d2 = st.columns(2)
        with d1:
            if cf is not None:
                f = cf[(cf["company"] == co) & (cf["state"] == stt)].set_index("month")
                fig = go.Figure()
                if "lo90" in f:
                    fig.add_trace(go.Scatter(x=list(f.index) + list(f.index[::-1]),
                                             y=list(f["hi90"] / 1000) + list(f["lo90"][::-1] / 1000),
                                             fill="toself", fillcolor="rgba(42,120,214,0.15)",
                                             line=dict(width=0), name="90% interval", hoverinfo="skip"))
                for s_, dash in (("cold", "dash"), ("normal", "solid"), ("warm", "dash")):
                    fig.add_trace(go.Scatter(x=f.index, y=f[s_] / 1000, name=f"{s_.title()} weather",
                                             line=dict(color=SCENARIO_COLORS[s_], dash=dash,
                                                       width=2.5 if s_ == "normal" else 1.8)))
                fig = style(fig, "12-month demand forecast", "Bcf per month", 420)
                st.plotly_chart(fig, width="stretch")
        with d2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["res_price_per_therm"], name="Residential",
                                     mode="lines+markers", line=dict(color=SERIES[0], width=2)))
            fig.add_trace(go.Scatter(x=hist.index, y=hist["com_price_per_therm"], name="Commercial",
                                     mode="lines+markers", line=dict(color=SERIES[1], width=2)))
            hh = D["hh_therm"]
            if hh is not None:
                hy = hh["henry_hub_therm"].groupby(hh.index.year).mean().reindex(hist.index)
                fig.add_trace(go.Scatter(x=hy.index, y=hy, name="Henry Hub (wholesale)",
                                         line=dict(color=NEUTRAL, dash="dot", width=2)))
            fig = style(fig, "Average price per therm", "$/therm", 420)
            st.plotly_chart(fig, width="stretch")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=hist.index, y=hist["citygate_sales_volume"] / 1e6,
                             name="Supply: bought for sales customers", marker=dict(color=SERIES[0])))
        fig.add_trace(go.Bar(x=hist.index, y=hist["citygate_transport_volume"] / 1e6,
                             name="Supply: delivered for choice customers", marker=dict(color=SERIES[2])))
        fig.add_trace(go.Bar(x=hist.index, y=hist["lng_withdrawals"] / 1e6, name="Supply: LNG withdrawals",
                             marker=dict(color=SERIES[3])))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["ldc_volume_mcf"] / 1e6, name="Demand: deliveries",
                                 mode="lines+markers", line=dict(color=INK, width=2.5)))
        fig.update_layout(barmode="stack")
        st.plotly_chart(style(fig, "Annual supply vs. customer deliveries", "Bcf per year", 420),
                        width="stretch")

        tf = D["tariff"]
        if stt == "MA" and tf is not None and co in TARIFF_NAME:
            t = tf[(tf["company"].isin(TARIFF_NAME[co])) & (tf["category"] == "Residential heating")]
            if len(t):
                t = t.groupby("month")[["gaf_therm", "ldaf_therm"]].mean().iloc[-48:]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=t.index, y=t["gaf_therm"], name="GAF (supply)",
                                     marker=dict(color=SERIES[0])))
                fig.add_trace(go.Bar(x=t.index, y=t["ldaf_therm"], name="LDAF (delivery adjustments)",
                                     marker=dict(color=SERIES[1])))
                fig.update_layout(barmode="relative", bargap=0.15)
                st.plotly_chart(style(fig, "Residential heating GAF + LDAF in force each month (DPU)",
                                      "$/therm", 400), width="stretch")
                st.caption("Massachusetts only. Full breakdown by rate class in the **MA price per "
                           "therm** tab.")

# ------------------------------------------------------------------ New England
with tab_ne:
    rf, rs, rh = D["region_fc"], D["region_sc"], D["region_hist"]
    if rf is None:
        st.info("State forecasts appear here after `python -m src.data --new-england` "
                "and a pipeline run.")
    else:
        names = {k: v["name"] for k, v in config.STATES.items()}
        win = rf[rf["month"].dt.month.isin(WINTER_M)]
        summ = (win.groupby("state")[["normal", "cold", "warm"]].sum() / 1000)
        best_rows = (rs[rs["model"] != "Seasonal naive"].sort_values("MAPE %")
                     .groupby("state").first())
        summ["Best model"] = best_rows["model"]
        summ["Backtest MAPE %"] = best_rows["MAPE %"]
        summ["Skill vs naive %"] = best_rows["Skill vs naive %"]
        summ.index = [names.get(i, i) for i in summ.index]
        c1, c2, c3 = st.columns(3)
        all_ne = {"MA", "CT", "RI", "NH", "ME", "VT"}
        missing = sorted(all_ne - set(rf["state"]))
        label = "New England utility demand, next winter (normal)" + (
            f", excl. {', '.join(missing)}" if missing else "")
        c1.metric(label, f"{summ['normal'].sum():,.0f} Bcf")
        c2.metric("Cold-winter upside", f"{(summ['cold'].sum() / summ['normal'].sum() - 1) * 100:+.1f}%")
        c3.metric("States modeled", f"{len(summ)}")
        st.dataframe(summ.rename(columns={"normal": "Winter Bcf (normal)", "cold": "Winter Bcf (cold)",
                                          "warm": "Winter Bcf (warm)"})
                     .style.format({"Winter Bcf (normal)": "{:,.1f}", "Winter Bcf (cold)": "{:,.1f}",
                                    "Winter Bcf (warm)": "{:,.1f}", "Backtest MAPE %": "{:.1f}",
                                    "Skill vs naive %": "{:.0f}"}), width="stretch")
        st.caption("Utility (LDC) load = residential + commercial + industrial. Each state's model was "
                   "chosen by its own rolling-origin backtest. Winter = Nov–Mar within the 12-month horizon.")

        state = st.selectbox("State", list(names), format_func=lambda k: names[k])
        fs = rf[rf["state"] == state].set_index("month")
        hs = rh[rh["state"] == state].set_index("month")["y"].iloc[-36:]
        st.plotly_chart(scenario_chart(hs, fs, f"{names[state]} utility gas demand · {fs['model'].iloc[0]}"),
                        width="stretch")

        st.caption("See the **Gas companies** tab for each utility's share of this forecast, "
                   "supply, customers and prices.")

# ------------------------------------------------------------------ MA tariffs
with tab_t:
    tf, hh = D["tariff"], D["hh_therm"]
    if tf is None:
        st.info("Run `python -m src.tariffs` to download the DPU GAF/LDAF files.")
    else:
        cats = [c for c in ["Residential heating", "Residential non-heating", "Small C&I",
                            "Medium C&I", "Large C&I"] if c in set(tf["category"])]
        cat = st.selectbox("Rate category", cats)
        t = tf[tf["category"] == cat]
        latest_m = t["month"].max()
        cur = t[t["month"] == latest_m].dropna(subset=["gaf_therm"]).sort_values("total_therm")
        hh_now = hh["henry_hub_therm"].dropna()
        st.markdown(f"**Rates in effect {latest_m:%B %Y}** · {cat} · $/therm")
        tbl = cur.set_index("company")[["gaf_therm", "ldaf_therm", "total_therm"]]
        tbl.columns = ["GAF (supply)", "LDAF (delivery adjustments)", "GAF + LDAF"]
        st.dataframe(tbl.style.format("{:.4f}"), width="stretch")
        if len(hh_now):
            st.caption(f"Henry Hub, {hh_now.index[-1]:%B %Y} average: ${hh_now.iloc[-1]:.3f}/therm. "
                       "GAF and LDAF are per-therm adjustments only; customer charges and base "
                       "distribution rates (set in each company's rate case) are billed on top.")

        companies = sorted(cur["company"])
        co = st.selectbox("Company", companies,
                          index=companies.index("National Grid – Boston Gas")
                          if "National Grid – Boston Gas" in companies else 0)
        months = st.slider("Months shown", 12, 120, 48, step=12)
        tc = t[t["company"] == co].set_index("month").iloc[-months:]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tc.index, y=tc["gaf_therm"], name="GAF (supply)",
                             marker=dict(color=SERIES[0])))
        fig.add_trace(go.Bar(x=tc.index, y=tc["ldaf_therm"], name="LDAF (delivery adjustments)",
                             marker=dict(color=SERIES[1])))
        h = hh["henry_hub_therm"].reindex(tc.index)
        fig.add_trace(go.Scatter(x=h.index, y=h, name="Henry Hub (wholesale)",
                                 line=dict(color=INK, width=2)))
        fig.update_layout(barmode="relative", bargap=0.15)
        st.plotly_chart(style(fig, f"{co}: price per therm by component · {cat}", "$/therm", 440),
                        width="stretch")
        st.caption("Bars stack GAF and LDAF in force each month (a negative LDAF is a credit). "
                   "The line is the wholesale Henry Hub price for comparison.")

        recent = t[t["month"] > latest_m - pd.DateOffset(years=months // 12)]
        active = recent.groupby("company")["gaf_therm"].count()
        active = active[active > 0].index
        fig = go.Figure()
        for i, c in enumerate(sorted(active)):
            d = recent[recent["company"] == c]
            fig.add_trace(go.Scatter(x=d["month"], y=d["gaf_therm"], name=c, line_shape="hv",
                                     line=dict(color=SERIES[i % len(SERIES)], width=1.8)))
        h = hh["henry_hub_therm"].loc[recent["month"].min():]
        fig.add_trace(go.Scatter(x=h.index, y=h, name="Henry Hub", line=dict(color=NEUTRAL, width=2,
                                                                                  dash="dot")))
        st.plotly_chart(style(fig, f"GAF by company vs. Henry Hub · {cat}", "$/therm", 460),
                        width="stretch")
        prem = (recent.groupby("company")["gaf_therm"].mean()
                - hh["henry_hub_therm"].loc[recent["month"].min():].mean())
        st.caption(f"Average GAF premium over Henry Hub in this window: "
                   f"{prem.min():.2f}–{prem.max():.2f} $/therm across companies. The premium covers "
                   "pipeline capacity into New England, storage, LNG peaking supply and prior-period "
                   "reconciliations. Source: Massachusetts DPU.")

with tab_p:
    lat = D["latest"]
    h_choice = st.radio("Horizon", ["1 week", "4 weeks"], horizontal=True, index=1)
    h = 1 if h_choice.startswith("1") else 4
    ps = D["ps1"] if h == 1 else D["ps4"]
    wf = D["wf1"] if h == 1 else D["wf4"]
    challengers = ps.drop(index=["Random walk", MARKET], errors="ignore")
    best_p = challengers["RMSE log-return"].idxmin()

    cur = lat[(lat["h"] == h) & (lat["model"] == best_p)].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
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
    if MARKET in ps.index:
        vs_mkt = (ps.at[best_p, "RMSE log-return"] / ps.at[MARKET, "RMSE log-return"] - 1) * 100
        vm = D["vm1"] if h == 1 else D["vm4"]
        pm_ = float(vm.at[best_p, "DM p-value"]) if vm is not None and best_p in vm.index else float("nan")
        if np.isnan(pm_):
            mv = "beats the market" if vs_mkt < 0 else "market is better"
        elif pm_ < 0.05:
            mv = f"{'beats the market' if vs_mkt < 0 else 'market is better'} (p={pm_:.3f})"
        else:
            mv = f"difference not significant (p={pm_:.2f})"
        c4.metric("RMSE vs. NYMEX futures (lower is better)", f"{vs_mkt:+.1f}%", mv,
                  delta_color="off",
                  help="The futures market's forecast: the front-month NYMEX price at the end of each week, "
                       "scored on the same weeks as the models.")

    g = wf[wf["model"] == best_p].set_index("date")
    rw = wf[wf["model"] == "Random walk"].set_index("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rw.index, y=rw["price_actual"], name=f"Actual ({h} wk later)",
                             line=dict(color=NEUTRAL, width=1.5)))
    fig.add_trace(go.Scatter(x=g.index, y=g["price_fcst"], name=f"{best_p} forecast",
                             line=dict(color=MODEL_COLORS.get(best_p, SERIES[0]), width=1.5)))
    mk = wf[wf["model"] == MARKET].set_index("date")
    if len(mk):
        fig.add_trace(go.Scatter(x=mk.index, y=mk["price_fcst"], name="NYMEX futures (market)",
                                 line=dict(color=MODEL_COLORS.get(MARKET, INK_2), width=1.2, dash="dot"),
                                 visible="legendonly"))
    add_events(fig, g.index.min())
    st.plotly_chart(style(fig, f"Out-of-sample {h}-week-ahead forecasts", "$/MMBtu"),
                    width="stretch")

    a, b = st.columns([1.2, 1])
    with a:
        st.subheader("Walk-forward scorecard")
        st.dataframe(ps.astype(float).map(lambda v: "–" if pd.isna(v) else f"{v:.3f}"), width="stretch")
        st.caption("Negative DM stat with p < 0.05 = statistically better than a random walk "
                   "(Diebold-Mariano, Newey-West variance, HLN correction). “NYMEX futures (market)” is the "
                   "front-month futures price used as a forecast: the bar a model has to clear to add value "
                   "over the market.")
    with b:
        imp = D["shap"].abs().mean().sort_values()
        fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h",
                               marker=dict(color=SERIES[0])))
        fig.update_layout(hovermode="closest")
        fig = style(fig, "What drives the 4-week model (mean |SHAP|)", "", 420)
        fig.update_xaxes(title="pts of 4-week % change", showgrid=True, gridcolor=GRID)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, width="stretch")

    nx, strip, nm = D["nx_daily"], D["nx_strip"], D["nx_meta"]
    st.subheader("NYMEX Henry Hub futures")
    if nx is None or strip is None:
        st.info("NYMEX futures appear after the next data refresh (Actions → “NYMEX daily update”).")
    else:
        asof = pd.Timestamp(nm.get("nymex_strip_as_of", strip["as_of"].max()))
        lag = len(pd.bdate_range(asof, pd.Timestamp.today().normalize())) - 1
        (st.warning if lag > 4 else st.caption)(
            f"Settlements as of {asof:%A, %B %d, %Y}" + (f" ({lag} business days old: the daily update may "
            "have failed; check the repository's Actions tab)" if lag > 4 else ", updated every weekday "
            "evening") + ". Source: CME NYMEX Henry Hub contracts via Yahoo Finance (daily close ≈ settlement).")
        a, b = st.columns([1.15, 1])
        with a:
            d2 = nx.loc[nx.index.max() - pd.DateOffset(years=2):]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d2.index, y=d2["henry_hub_spot"], name="Henry Hub spot (EIA)",
                                     line=dict(color=NEUTRAL, width=1.4), connectgaps=True))
            fig.add_trace(go.Scatter(x=d2.index, y=d2["nymex_front"], name="NYMEX front month",
                                     line=dict(color=SERIES[0], width=1.8)))
            st.plotly_chart(style(fig, "Front-month futures vs. spot, last 2 years", "$/MMBtu", 400),
                            width="stretch")
        with b:
            sx = strip.sort_values("contract")
            labels = pd.to_datetime(sx["contract"]).dt.strftime("%b %y")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=labels, y=sx["settle"], name=f"{asof:%b %d}",
                                 marker=dict(color=SERIES[0])))
            if sx["settle_prior"].notna().any():
                fig.add_trace(go.Scatter(x=labels, y=sx["settle_prior"], mode="lines+markers",
                                         name=f"{pd.Timestamp(sx['prior_as_of'].dropna().iloc[0]):%b %d}",
                                         line=dict(color=INK_2, width=1.5, dash="dot")))
            st.plotly_chart(style(fig, "Forward strip: price for each delivery month", "$/MMBtu", 400),
                            width="stretch")
        tbl = sx.assign(Month=labels)[["Month", "settle", "settle_therm", "change"]]
        tbl.columns = ["Delivery month", "$/MMBtu", "$/therm", "Change vs. a week earlier ($/MMBtu)"]
        with st.expander("Strip table"):
            st.dataframe(tbl.set_index("Delivery month").style.format("{:+.3f}", subset=[tbl.columns[-1]])
                         .format("{:.3f}", subset=["$/MMBtu"]).format("{:.4f}", subset=["$/therm"]),
                         width="stretch")
            st.download_button("Download strip (CSV)", sx.to_csv(index=False).encode(),
                               file_name=f"nymex_strip_{asof:%Y%m%d}.csv")

    dfp = D["dfp"]
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
* **Demand:** EIA monthly Massachusetts gas deliveries (MMcf). The target is utility (LDC) load:
  residential + commercial + industrial. Power-plant burn is excluded because it follows electricity
  markets rather than heating weather.
* **Price:** EIA daily Henry Hub spot, averaged to weeks ending Friday.
* **NYMEX futures:** Henry Hub front-month and monthly contracts (CME NYMEX, via Yahoo Finance; EIA stopped
  publishing futures in April 2024). Updated every weekday after settlement; downloads are checked
  (plausible prices, expired contracts dropped, stale quotes rejected) and never overwrite good history.
  The front-month price is also scored as the market's forecast, the benchmark a model must beat.
* **Drivers:** EIA weekly working gas in storage (Lower 48), EIA monthly LNG exports.
* **New England:** the same demand method run for CT, RI, NH, ME and VT (each state's history ends at its last month with all sectors reported), each with its own weather
  stations; company forecasts split each state's forecast by EIA Form 176 delivery shares.
* **MA price per therm:** GAF and LDAF by company and rate class from the Massachusetts DPU
  (current-rates page and historical workbook, 2008 onward). Ccf rates converted at 1.037 therms/Ccf.
* **Weather:** ERA5 daily max/min temperature via Open-Meteo. MA degree days weight Boston,
  Worcester and Springfield; the price model uses a five-city national proxy. HDD/CDD base 65°F.

### Demand models
Seasonal naive · HDD regression · SARIMAX (HDD/CDD exogenous) · Prophet (HDD/CDD regressors) ·
LightGBM · Hybrid (HDD regression + LightGBM on residuals). Expanding-window rolling-origin
backtest, 12-month horizon, run once with **normal weather** (15-year normals, realistic) and once with **actual
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
through {m['price_data_through']}; NYMEX through {D['nx_meta'].get('nymex_front_through', 'n/a')}.*
""")
