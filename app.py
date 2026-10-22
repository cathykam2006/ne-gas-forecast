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
    """Dotted markers for major market events (context only, not model inputs)."""
    for d, label in config.EVENTS:
        d = pd.Timestamp(d)
        if d >= start:
            fig.add_vline(x=d, line=dict(color=INK_2, width=1, dash="dot"))
            fig.add_annotation(x=d, y=1, yref="paper", text=label, showarrow=False,
                               textangle=-90, xanchor="right", yanchor="top",
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


D = load()
st.title("New England Natural Gas: Demand & Price Forecasting")
st.caption("New England utility gas demand by state and company, Massachusetts GAF/LDAF per therm, "
           "and the Henry Hub spot price · public EIA, Massachusetts DPU and ERA5 weather data · "
           "Created by Cathy Kam")

if D is None:
    st.warning("No processed data found. Run `python -m src.pipeline --pull` with "
               "`EIA_API_KEY` set, then commit `data/processed/`.")
    st.stop()

m = D["meta"]
tab_d, tab_ne, tab_t, tab_p, tab_m = st.tabs(
    ["Massachusetts demand", "New England by state & company", "MA price per therm",
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

        st.subheader(f"{names[state]} by gas company")
        cf, sh, ca = D["co_fc"], D["co_shares"], D["co_annual"]
        if cf is None or ca is None:
            st.info("Company breakdown appears after the pipeline downloads EIA Form 176 company "
                    "data (`python -m src.companies`).")
        else:
            cas = ca[ca["state"] == state]
            last_y = int(cas["year"].max())
            lat = cas[cas["year"] == last_y].set_index("company")
            shs = sh[sh["state"] == state].set_index("company")
            cfs = cf[cf["state"] == state]
            wins = (cfs[cfs["month"].dt.month.isin(WINTER_M)].groupby("company")[["normal", "cold"]]
                    .sum() / 1000)
            tbl = pd.DataFrame({
                "Share of state utility load": shs["share"],
                "Share change, 5 yrs (pts)": shs["share_trend_pts"],
                "Winter Bcf, normal": wins["normal"],
                "Winter Bcf, cold": wins["cold"],
                f"Customers ({last_y})": lat["total_customers"],
                f"Residential $/therm ({last_y})": lat["res_price_per_therm"],
                f"Commercial $/therm ({last_y})": lat["com_price_per_therm"],
                "Therms per home": lat["use_per_res_customer_therms"],
                "Transport (choice) share of supply %": lat["transport_share_pct"],
                "Unaccounted-for gas %": lat["unaccounted_pct"],
            }).sort_values("Share of state utility load", ascending=False)
            st.dataframe(tbl.style.format({
                "Share of state utility load": "{:.1%}", "Share change, 5 yrs (pts)": "{:+.1f}",
                "Winter Bcf, normal": "{:,.2f}", "Winter Bcf, cold": "{:,.2f}",
                f"Customers ({last_y})": "{:,.0f}", f"Residential $/therm ({last_y})": "${:.2f}",
                f"Commercial $/therm ({last_y})": "${:.2f}", "Therms per home": "{:,.0f}",
                "Transport (choice) share of supply %": "{:.0f}", "Unaccounted-for gas %": "{:+.1f}",
            }, na_rep="–"), width="stretch")
            st.caption("EIA Form 176 annual company reports. Shares are the latest three years' average "
                       "of residential + commercial + industrial deliveries; company forecasts apply "
                       "them to the state's monthly forecast. Prices are sales revenue ÷ sales volume "
                       "(supply, delivery and fixed charges together), converted with each company's "
                       "reported heat content.")

            order = tbl.index.tolist()
            colors = {c: SERIES[i % len(SERIES)] for i, c in enumerate(order)}
            a1, a2 = st.columns(2)
            with a1:
                fig = go.Figure()
                for c in order:
                    d = cfs[cfs["company"] == c]
                    fig.add_trace(go.Scatter(x=d["month"], y=d["normal"] / 1000, name=c,
                                             line=dict(color=colors[c], width=2)))
                fig = style(fig, "12-month demand forecast by company (normal weather)",
                            "Bcf per month", 480)
                fig.update_layout(legend=dict(orientation="h", y=-0.18, yanchor="top", x=0),
                                  margin=dict(b=110))
                st.plotly_chart(fig, width="stretch")
            with a2:
                fig = go.Figure()
                for c in order:
                    d = cas[cas["company"] == c]
                    fig.add_trace(go.Scatter(x=d["year"], y=d["res_price_per_therm"], name=c,
                                             mode="lines+markers", line=dict(color=colors[c], width=2),
                                             marker=dict(size=6)))
                fig = style(fig, "Average residential price by company", "$/therm", 480)
                fig.update_layout(legend=dict(orientation="h", y=-0.18, yanchor="top", x=0),
                                  margin=dict(b=110))
                st.plotly_chart(fig, width="stretch")

            co_sel = st.selectbox("Company detail", order)
            d = cas[cas["company"] == co_sel].set_index("year")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d.index, y=d["citygate_sales_volume"] / 1e6,
                                 name="Supply: citygate, sales customers", marker=dict(color=SERIES[0])))
            fig.add_trace(go.Bar(x=d.index, y=d["citygate_transport_volume"] / 1e6,
                                 name="Supply: citygate, transport customers", marker=dict(color=SERIES[2])))
            fig.add_trace(go.Bar(x=d.index, y=d["lng_withdrawals"] / 1e6,
                                 name="Supply: LNG withdrawals", marker=dict(color=SERIES[3])))
            fig.add_trace(go.Scatter(x=d.index, y=d["ldc_volume_mcf"] / 1e6, name="Demand: deliveries",
                                     mode="lines+markers", line=dict(color=INK, width=2.5)))
            fig.update_layout(barmode="stack")
            st.plotly_chart(style(fig, f"{co_sel}: annual supply vs. customer deliveries", "Bcf per year",
                                  440), width="stretch")
            st.caption("Supply = gas received at the citygate (bought for sales customers, or "
                       "delivered for customers who buy from marketers) plus LNG withdrawals. "
                       "The gap to deliveries is storage, company use, losses and unaccounted-for gas.")

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
through {m['price_data_through']}.*
""")
