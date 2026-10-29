---
title: Henry Hub price outlook
---

<div class="eyebrow">New England gas forecast · Henry Hub</div>

# Henry Hub Price Outlook

<p class="lede">Weekly forecasts of the U.S. benchmark natural gas price at Henry Hub, Louisiana, checked against what the price actually did, next to what the NYMEX futures market expects. This is the national wholesale price, not a New England or utility rate; see <a href="./massachusetts-gaf">Massachusetts GAF</a> for what each company charges.</p>

```js
const D = FileAttachment("data/henry-hub.json").json();
```

```js
const parse = d3.utcParse("%Y-%m-%d");
const fmtD = d3.utcFormat("%b %-d, %Y"), fmtMD = d3.utcFormat("%b %-d");
const usd = (v, n = 2) => "$" + v.toFixed(n);
const FEATURES = {vol_8: "Price volatility (8 wk)", ret_4: "Price change, last 4 wk", ret_13: "Price change, last 13 wk",
  ret_1: "Price change, last week", lng_bcfd: "LNG exports (Bcf/d)", dd_anom: "Weather vs. normal (degree days)",
  dd_anom_4: "Weather vs. normal, 4 wk", storage_surplus_pct: "Storage vs. 5-yr average", injection_vs_norm_bcf: "Storage injection vs. normal",
  lng_yoy_pct: "LNG exports, year over year", sin_w: "Season (sine)", cos_w: "Season (cosine)"};
const SHORT_F = {vol_8: "Volatility, 8 wk", ret_4: "Change, 4 wk", ret_13: "Change, 13 wk", ret_1: "Change, 1 wk",
  lng_bcfd: "LNG exports", dd_anom: "Weather anomaly", dd_anom_4: "Weather, 4 wk", storage_surplus_pct: "Storage vs. norm",
  injection_vs_norm_bcf: "Injection vs. norm", lng_yoy_pct: "LNG exports, y/y", sin_w: "Season (sin)", cos_w: "Season (cos)"};
const SHORT_E = {"Winter Storm Fern": "Storm Fern", "Golden Pass LNG first cargo": "Golden Pass LNG", "Record-hot July": "Hot July",
  "China 15% tariff on US LNG": "China LNG tariff", "Russia invades Ukraine": "Ukraine war", "Freeport LNG outage": "Freeport outage",
  "Winter Storm Uri": "Storm Uri", "COVID-19 pandemic declared": "COVID-19"};
```

```js
{
  const L = D.h4.latest, sc = D.h4.scores.find((s) => s.model === D.h4.best);
  const chg = (L.fcst / L.price_now - 1) * 100;
  const dir = Math.abs(chg) < 1 ? "" : chg > 0 ? "up" : "down";
  const frontMonth = d3.utcFormat("%b '%y")(parse(D.strip[0].m + "-01"));
  display(html`<div class="kpis">
    <div class="kpi"><div class="label">Henry Hub spot, week ending ${fmtMD(parse(L.as_of))}</div>
      <div class="value">${usd(L.price_now)}<small>/MMBtu</small></div><div class="sub">Weekly average of daily spot prices</div></div>
    <div class="kpi"><div class="label">4-week forecast, week of ${fmtMD(parse(L.target))}</div>
      <div class="value">${usd(L.fcst)}<small>/MMBtu</small></div>
      <span class="chip ${dir}">${chg >= 0 ? "+" : ""}${chg.toFixed(1)}% vs. now</span></div>
    <div class="kpi"><div class="label">Forecast error vs. “no change”</div>
      <div class="value">${sc.vs_rw.toFixed(1)}%</div>
      <span class="chip">${sc.p < 0.05 ? "significant" : "not significant"} (p = ${sc.p.toFixed(2)})</span></div>
    <div class="kpi"><div class="label">NYMEX front month (${frontMonth}), ${fmtMD(parse(D.strip_as_of))}</div>
      <div class="value">${usd(D.front_last)}<small>/MMBtu</small></div><div class="sub">Market price for next month’s delivery</div></div>
  </div>`);
}
```

## Where the price is headed

<div class="controls">

```js
const ahead = view(Inputs.radio(["Next 12 months", "Next 24 months"], {value: "Next 12 months", label: "Look ahead"}));
```

</div>

```js
const O = D.outlook;
const oHist = O.history.map((r) => ({d: parse(r.d), v: r.v}));
const oPath = O.path.map((r) => ({...r, d: parse(r.d)}));
const oLast = parse(O.last_date);
const oEnd = d3.utcMonth.offset(oLast, ahead === "Next 12 months" ? 12 : 24);
const oStart = d3.utcMonth.offset(oLast, -12);
const oFut = O.futures.map((r) => ({...r, d: parse(r.d)})).filter((r) => r.d <= oEnd);
const oShown = oHist.filter((r) => r.d >= oStart);
const p4 = oPath.at(-1);
```

<div class="card">
<div class="legend"><span><i class="sw" style="border-color:var(--ink)"></i>Actual (weekly average)</span><span><i class="sw" style="border-color:var(--accent);border-top-style:dashed;border-top-width:3px"></i>Model forecast · ${O.model}</span><span><i class="sw box" style="background:var(--fc-band)"></i>Likely range (${Math.round(O.level * 100)}%)</span><span><i class="sw" style="border-color:var(--futures);border-top-style:dotted;border-top-width:3px"></i>NYMEX futures (market forecast)</span></div>

```js
display(resize((width) => {
  const narrow = width < 560;
  const q = (p) => d3.quantile(oShown, p, (r) => r.v);
  const yMax = Math.max(d3.max(oPath, (r) => r.hi), d3.max(oFut, (r) => r.v) ?? 0, q(0.9)) * 1.25;
  const yMin = Math.min(d3.min(oPath, (r) => r.lo), d3.min(oFut, (r) => r.v) ?? 1e9, q(0.05)) * 0.8;
  const peak = d3.greatest(oShown, (r) => r.v);
  const clip = (v) => Math.min(v, yMax);
  const today = new Date();
  return Plot.plot({
    width, height: narrow ? 320 : 400, marginLeft: 46, marginTop: 30, marginRight: 12, style: {fontSize: "12px"},
    x: {type: "utc", domain: [oStart, d3.utcDay.offset(oEnd, 10)], label: null,
        ticks: d3.utcMonth.every(narrow ? 6 : ahead === "Next 12 months" ? 2 : 3),
        tickFormat: (d) => narrow ? d3.utcFormat("%b '%y")(d) : d3.utcFormat(d.getUTCMonth() === 0 ? "%b %Y" : "%b")(d)},
    y: {domain: [yMin, yMax], grid: true, label: "$/MMBtu", labelArrow: "none", tickFormat: (d) => "$" + d.toFixed(d % 1 ? 1 : 0), clamp: true},
    marks: [
      Plot.rectX([{}], {x1: oLast, x2: oEnd, y1: yMin, y2: yMax, fill: "var(--fc-wash)"}),
      Plot.ruleX([oLast], {stroke: "var(--muted)", strokeDasharray: "2,3"}),
      narrow ? null : Plot.text([`Latest actual · ${fmtMD(oLast)}`], {x: oLast, frameAnchor: "top", dy: -20, dx: -6, textAnchor: "end", fill: "var(--muted)", fontSize: 11}),
      Plot.text(["FORECAST →"], {x: oLast, frameAnchor: "top", dy: -20, dx: 6, textAnchor: "start", fill: "var(--accent)", fontWeight: 700, fontSize: 11}),
      today > oLast && today < oEnd ? Plot.ruleX([today], {stroke: "var(--rule)"}) : null,
      today > oLast && today < oEnd ? Plot.text(["Today"], {x: today, frameAnchor: "bottom", dy: -4, dx: 3, textAnchor: "start", fill: "var(--muted)", fontSize: 10}) : null,
      Plot.areaY(oPath, {x: "d", y1: "lo", y2: (r) => clip(r.hi), fill: "var(--fc-band)", curve: "linear"}),
      Plot.lineY(oShown, {x: "d", y: (r) => clip(r.v), stroke: "var(--ink)", strokeWidth: 2}),
      peak.v > yMax ? Plot.text([peak], {x: "d", y: yMax, text: (r) => `↑ $${r.v.toFixed(2)} week of ${fmtMD(r.d)} (off scale)`,
        textAnchor: "start", dx: 6, dy: 8, fill: "var(--muted)", fontSize: 11}) : null,
      Plot.lineY(oPath, {x: "d", y: "f", stroke: "var(--accent)", strokeWidth: 3.2, strokeDasharray: "7,5"}),
      Plot.dot(oPath.filter((r) => r.h > 0), {x: "d", y: "f", r: 5.5, fill: "var(--surface)", stroke: "var(--accent)", strokeWidth: 2.5}),
      Plot.text([p4], {x: "d", y: "hi", text: (r) => `Model: $${r.f.toFixed(2)} in ${r.h} wk`, dy: -12, textAnchor: "middle",
        fill: "var(--accent)", fontWeight: 700, fontSize: 12, stroke: "var(--surface)", strokeWidth: 4, paintOrder: "stroke"}),
      Plot.lineY(oFut, {x: "d", y: "v", stroke: "var(--futures)", strokeWidth: 2, strokeDasharray: "2,4", curve: "catmull-rom"}),
      Plot.dot(oFut, {x: "d", y: "v", symbol: "diamond", r: 4.5, fill: "var(--futures)"}),
      oFut.length ? Plot.text([oFut[0]], {x: "d", y: "v", text: () => "NYMEX futures", dy: 18, dx: 6, textAnchor: "start",
        fill: "var(--futures)", fontWeight: 600, fontSize: 11}) : null,
      Plot.tip([...oShown.map((r) => ({...r, kind: "Actual"})), ...oPath.filter((r) => r.h > 0).map((r) => ({d: r.d, v: r.f, lo: r.lo, hi: r.hi, h: r.h, kind: "Model"})),
                ...oFut.map((r) => ({...r, kind: "Futures"}))],
        Plot.pointerX({x: "d", y: (r) => clip(r.v), title: (r) => r.kind === "Actual" ? `Week ending ${fmtD(r.d)}\nActual: ${usd(r.v)}`
          : r.kind === "Model" ? `${fmtD(r.d)}: ${r.h}-week forecast\n${usd(r.v)} (likely ${usd(r.lo)}–${usd(r.hi)})`
          : `${d3.utcFormat("%B %Y")(r.d)} futures contract\n${usd(r.v, 3)}/MMBtu`}))
    ].filter(Boolean)
  });
}));
```

</div>

<p class="fine">Solid line: actual weekly price through ${fmtD(oLast)}. Dashed blue: the ${O.model} model’s 1- and 4-week forecasts, with the range that held ${Math.round(O.level * 100)}% of the time in its walk-forward backtest. Dotted green: NYMEX futures by delivery month${O.futures_as_of ? `, as of ${fmtMD(parse(O.futures_as_of))}` : ""}, the price the market will lock in today for each future month.</p>

## Forecast track record

<p class="fine">How the model's past forecasts compared with what actually happened.</p>


<div class="controls">

```js
const horizon = view(Inputs.radio(new Map([["4 weeks ahead", 4], ["1 week ahead", 1]]), {value: 4, label: "Horizon"}));
```

```js
const period = view(Inputs.radio(["Latest 12 months", "Since 2020"], {value: "Latest 12 months", label: "Period"}));
```

</div>

```js
const H = D["h" + horizon];
const recent = period === "Latest 12 months";
const rowsAll = H.rows.map((r) => ({...r, t: parse(r.t)}));
const fcEnd = rowsAll.at(-1).t;
const fcStart = recent ? d3.utcMonth.offset(fcEnd, -12) : rowsAll[0].t;
const fcRows = rowsAll.filter((r) => r.t >= fcStart);
const periods = D.periods.map((p) => ({...p, a: parse(p.a), b: parse(p.b)}))
  .filter((p) => p.b >= fcStart && p.a <= fcEnd && (p.b - p.a) / 864e5 >= 7);
const pointEvents = D.events.map((e) => ({...e, t: parse(e.d)}))
  .filter((e) => e.t >= fcStart && e.t <= fcEnd && !periods.some((p) => p.label === e.label));
const eventLabels = [...pointEvents.map((e) => ({t: e.t, label: e.label})), ...periods.map((p) => ({t: p.a, label: p.label}))]
  .sort((a, b) => a.t - b.t).map((d, i) => ({...d, row: i % 2}));
```

<p class="fine">${fmtD(fcStart)} – ${fmtD(fcEnd)}. Each point is the week the forecast was for, so the blue line shows what the model predicted ${horizon} week${horizon > 1 ? "s" : ""} earlier. Events are context, not model inputs.</p>

<div class="card">
<div class="legend"><span><i class="sw" style="border-color:var(--actual)"></i>Actual price (weekly average)</span><span><i class="sw" style="border-color:var(--accent)"></i>${H.best} forecast, made ${horizon} week${horizon > 1 ? "s" : ""} earlier</span>${eventLabels.length ? html`<span><i class="sw" style="border-color:var(--event);border-top-style:dashed"></i>Market event</span>` : ""}</div>

```js
display(resize((width) => {
  const narrow = width < 560, tight = narrow || !recent;
  const yMax = d3.max(fcRows, (r) => Math.max(r.actual, r.fcst)) * 1.12;
  const labels = eventLabels.map((d) => ({...d, text: tight ? (SHORT_E[d.label] ?? d.label) : d.label}));
  return Plot.plot({
    width, height: narrow ? 300 : 380, marginLeft: 44, marginTop: 34, marginRight: 8,
    style: {fontSize: "12px"},
    x: {type: "utc", domain: [fcStart, fcEnd], label: null,
        ticks: recent ? d3.utcMonth.every(narrow ? 2 : 1) : d3.utcYear.every(1),
        tickFormat: recent ? (d) => d3.utcFormat(d.getUTCMonth() === 0 ? "%b %Y" : "%b")(d) : "%Y"},
    y: {domain: [0, yMax], grid: true, label: "$/MMBtu", labelArrow: "none", tickFormat: (d) => "$" + d},
    marks: [
      Plot.rectX(periods, {x1: "a", x2: "b", y1: 0, y2: yMax, fill: "var(--event-wash)"}),
      Plot.ruleX(pointEvents, {x: "t", stroke: "var(--event)", strokeDasharray: "3,3", strokeWidth: 1.2}),
      ...[0, 1].map((row) => Plot.text(labels.filter((d) => d.row === row), {x: "t", y: yMax, dy: -24 + row * 14, text: "text",
        textAnchor: "start", dx: 4, fill: "var(--event)", fontWeight: 600, fontSize: narrow ? 10.5 : 11.5, lineAnchor: "top"})),
      Plot.lineY(fcRows, {x: "t", y: "actual", stroke: "var(--actual)", strokeWidth: recent ? 2 : 1.4, curve: "monotone-x"}),
      Plot.lineY(fcRows, {x: "t", y: "fcst", stroke: "var(--accent)", strokeWidth: recent ? 2.2 : 1.4, curve: "monotone-x"}),
      recent ? Plot.dot(fcRows, {x: "t", y: "actual", r: 2.4, fill: "var(--actual)"}) : null,
      recent ? Plot.dot(fcRows, {x: "t", y: "fcst", r: 2.4, fill: "var(--accent)"}) : null,
      Plot.dot(fcRows.slice(-1), {x: "t", y: "actual", r: 4.5, fill: "var(--surface)", stroke: "var(--actual)", strokeWidth: 2}),
      Plot.ruleX(fcRows, Plot.pointerX({x: "t", stroke: "var(--rule)"})),
      Plot.tip(fcRows, Plot.pointerX({x: "t", y: "actual", title: (r) =>
        `Week ending ${fmtD(r.t)}\nActual: ${usd(r.actual)}\nForecast: ${usd(r.fcst)} (made ${fmtMD(parse(r.made))})`}))
    ].filter(Boolean)
  });
}));
```

</div>

```js
display(html`<ul class="events">${[...new Set(eventLabels.map((l) => l.label))].filter((l) => D.notes[l])
  .map((l) => html`<li><b>${l}</b><span>${D.notes[l]}</span></li>`)}</ul>`);
```

## What the futures market expects

<p class="fine">CME NYMEX Henry Hub contracts, daily close. Settlements as of ${fmtD(parse(D.strip_as_of))}.</p>

```js
const front = D.front.map((r) => ({...r, d: parse(r.d)}));
const strip = D.strip.map((r) => {
  const d = parse(r.m + "-01"), m = d.getUTCMonth();
  return {...r, d, peak: m >= 10 || m <= 3, lab: d3.utcFormat("%b %y")(d)};
});
const asOf = parse(D.strip_as_of);
const peakYear = asOf.getUTCMonth() <= 3 ? asOf.getUTCFullYear() - 1 : asOf.getUTCFullYear();
const peakRows = strip.filter((r) => r.d >= Date.UTC(peakYear, 10, 1) && r.d <= Date.UTC(peakYear + 1, 3, 1));
const peakAvg = d3.mean(peakRows, (r) => r.p);
```

<div class="grid grid-cols-2">
<div class="card">
<h3>Front-month futures vs. spot, last 2 years</h3>
<div class="legend"><span><i class="sw" style="border-color:var(--accent)"></i>NYMEX front month</span><span><i class="sw" style="border-color:var(--actual)"></i>Henry Hub spot (EIA)</span></div>

```js
display(resize((width) => Plot.plot({
  width, height: 280, marginLeft: 40, marginRight: 8, style: {fontSize: "12px"},
  x: {type: "utc", ticks: width < 420 ? 3 : 5, label: null},
  y: {type: "log", grid: true, label: "$/MMBtu (log scale)", labelArrow: "none", ticks: [2, 3, 5, 10, 20, 30], tickFormat: (d) => "$" + d},
  marks: [
    Plot.lineY(front.filter((r) => r.spot != null), {x: "d", y: "spot", stroke: "var(--actual)", strokeWidth: 1.1, strokeOpacity: 0.85}),
    Plot.lineY(front.filter((r) => r.front != null), {x: "d", y: "front", stroke: "var(--accent)", strokeWidth: 1.8}),
    Plot.tip(front, Plot.pointerX({x: "d", y: "front", title: (r) =>
      `${fmtD(r.d)}\nFront month: ${r.front != null ? usd(r.front, 3) : "–"}\nSpot: ${r.spot != null ? usd(r.spot) : "–"}`}))
  ]
})));
```

<p class="fine">The front month is the nearest contract to expiry; it switches to the next month a day or two before expiry, which can show as a step.</p>
</div>
<div class="card">
<h3>Futures price by delivery month</h3>
<div class="legend"><span><i class="sw box" style="background:var(--peak)"></i>Peak GAF period (Nov–Apr)</span><span><i class="sw box" style="background:var(--rule)"></i>Off-peak months</span></div>

```js
display(resize((width) => Plot.plot({
  width, height: 280, marginLeft: 40, marginRight: 4, marginBottom: 34, style: {fontSize: "11px"},
  x: {type: "band", domain: strip.map((r) => r.lab), padding: 0.18, label: null,
      tickFormat: (d, i) => (i % (width < 480 ? 6 : 3) === 0 ? d.replace(" ", " '") : "")},
  y: {grid: true, label: "$/MMBtu", labelArrow: "none", tickFormat: (d) => "$" + d},
  marks: [
    Plot.barY(strip, {x: "lab", y: "p", fill: (r) => (r.peak ? "var(--peak)" : "var(--rule)"), rx: 1}),
    Plot.ruleY([0], {stroke: "var(--muted)"}),
    Plot.tip(strip, Plot.pointerX({x: "lab", y: "p", title: (r) =>
      `${d3.utcFormat("%B %Y")(r.d)}\n${usd(r.p, 3)}/MMBtu · ${usd(r.p / 10, 4)}/therm${r.peak ? "\nPeak GAF period" : ""}`}))
  ]
})));
```

${peakRows.length ? html`<p class="callout"><b>Coming peak GAF period (${d3.utcFormat("%b")(peakRows[0].d)} ${peakYear}–Apr ${peakYear + 1}):</b> futures average <span class="num">${usd(peakAvg, 3)}</span>/MMBtu, or <span class="num">${usd(peakAvg / 10, 4)}</span>/therm, before pipeline, storage and delivery costs into New England.</p>` : ""}

</div>
</div>

## What moves the forecast

<p class="fine">Storage against its five-year norm is the clearest fundamental signal; the machine-learning model (LightGBM) leans mostly on recent price momentum and volatility.</p>

<div class="grid grid-cols-2">
<div class="card">
<h3>Lower 48 storage vs. prior 5-year average</h3>
<div class="legend"><span><i class="sw box" style="background:var(--pos)"></i>Above average (looser market)</span><span><i class="sw box" style="background:var(--neg)"></i>Below average (tighter)</span></div>

```js
const storage = D.storage.map((r) => ({...r, d: parse(r.d)}));
display(resize((width) => Plot.plot({
  width, height: 260, marginLeft: 40, marginRight: 8, style: {fontSize: "12px"},
  x: {type: "utc", ticks: width < 420 ? 3 : 4, label: null},
  y: {grid: true, label: "% vs. 5-yr average", labelArrow: "none", tickFormat: (d) => (d > 0 ? "+" : "") + d + "%"},
  marks: [
    Plot.rectY(storage, {x1: (d) => d3.utcDay.offset(d.d, -3), x2: (d) => d3.utcDay.offset(d.d, 3), y: "v",
      fill: (d) => (d.v >= 0 ? "var(--pos)" : "var(--neg)")}),
    Plot.ruleY([0], {stroke: "var(--muted)"}),
    Plot.tip(storage, Plot.pointerX({x: "d", y: "v", title: (r) =>
      `Week ending ${fmtD(r.d)}\n${r.v > 0 ? "+" : ""}${r.v.toFixed(1)}% vs. 5-year average`}))
  ]
})));
```

</div>
<div class="card">
<h3>Average influence on the 4-week forecast</h3>
<div class="legend"><span>Mean |SHAP|, percentage points of 4-week price change (LightGBM)</span></div>

```js
display(resize((width) => {
  const rows = D.shap.map((r) => ({f: (width < 400 ? SHORT_F : FEATURES)[r.f] ?? r.f, v: r.v}));
  return Plot.plot({
    width, height: 290, marginLeft: width < 400 ? 118 : 190, marginRight: 36, style: {fontSize: "12px"},
    x: {grid: true, label: null, ticks: 4},
    y: {domain: rows.map((r) => r.f), label: null, tickSize: 0},
    marks: [
      Plot.barX(rows, {y: "f", x: "v", fill: (d, i) => (i < 3 ? "var(--accent)" : "var(--rule)"), rx: 1}),
      Plot.text(rows, {y: "f", x: "v", text: (d) => d.v.toFixed(1), dx: 5, textAnchor: "start", fill: "var(--muted)", fontSize: 11})
    ]
  });
}));
```

</div>
</div>

## Walk-forward scorecard

<p class="fine">Every model is refit each week using only data available at the time, then scored on the following weeks. Lower error is better; a p-value under 0.05 means the model beats “no change” by more than chance. Showing the ${horizon}-week horizon chosen above.</p>

```js
display(html`<div class="tablewrap"><table class="plain">
  <thead><tr><th>Model</th><th>Avg. error ($/MMBtu)</th><th>Error vs. “no change”</th><th>Direction hit</th><th>p-value</th></tr></thead>
  <tbody>${H.scores.slice().sort((a, b) => a.rmse - b.rmse).map((s) => html`<tr class=${s.model === H.best ? "hl" : ""}>
    <td>${s.model === "Random walk" ? "No change (random walk)" : s.model}</td><td>${s.rmse.toFixed(2)}</td>
    <td>${s.model === "Random walk" ? "–" : (s.vs_rw > 0 ? "+" : "") + s.vs_rw.toFixed(1) + "%"}</td>
    <td>${s.hit == null ? "–" : s.hit.toFixed(0) + "%"}</td><td>${s.p == null ? "–" : s.p.toFixed(2)}</td></tr>`)}</tbody>
</table></div>
<p class="fine">Direction hit = share of weeks the model called the up/down move correctly. Data through ${fmtD(parse(D.price_data_through))} (spot) and ${fmtD(parse(D.strip_as_of))} (futures).</p>`);
```
