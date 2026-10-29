---
title: Massachusetts demand
---

<div class="eyebrow">New England gas forecast · Massachusetts utility demand</div>

# Massachusetts Gas Demand Outlook

<p class="lede">Monthly natural gas delivered by Massachusetts utilities to homes, businesses and industry (power plants excluded), with a 12-month forecast under normal, cold and warm weather. EIA publishes the actual figures about two months late, so the forecast starts from the latest published month.</p>

```js
const M = FileAttachment("data/ma-demand.json").json();
```

```js
const pm = d3.utcParse("%Y-%m");
const fmtMon = d3.utcFormat("%b %Y"), fmtMonth = d3.utcFormat("%B %Y");
const hist = M.history.map((r) => ({d: pm(r.m), v: r.v}));
const fc = M.forecast.map((r) => ({...r, d: pm(r.m)}));
const last = pm(M.last_actual);
const lastV = hist.at(-1).v;
const now = new Date(), thisMonth = new Date(Date.UTC(now.getFullYear(), now.getMonth(), 1));
const cur = fc.find((r) => +r.d === +thisMonth);
const winter = fc.filter((r) => [10, 11, 0, 1, 2].includes(r.d.getUTCMonth()));
const wN = d3.sum(winter, (r) => r.normal), wC = d3.sum(winter, (r) => r.cold), wW = d3.sum(winter, (r) => r.warm);
const peak = d3.greatest(fc, (r) => r.normal);
```

```js
display(html`<div class="kpis">
  <div class="kpi"><div class="label">This month (${fmtMon(thisMonth)}), normal weather</div>
    <div class="value">${cur ? cur.normal.toFixed(1) : "–"}<small>Bcf</small></div>
    <div class="sub">${cur ? `Likely ${cur.lo90.toFixed(1)}–${cur.hi90.toFixed(1)} Bcf` : "Outside the forecast window"}</div></div>
  <div class="kpi"><div class="label">Coming winter (Nov–Mar), normal weather</div>
    <div class="value">${wN.toFixed(0)}<small>Bcf</small></div>
    <div class="sub">Peak month: ${fmtMon(peak.d)}, ${peak.normal.toFixed(1)} Bcf</div></div>
  <div class="kpi"><div class="label">If the winter is cold / warm</div>
    <div class="value">${((wC / wN - 1) * 100) >= 0 ? "+" : ""}${((wC / wN - 1) * 100).toFixed(1)}%</div>
    <span class="chip">warm: ${((wW / wN - 1) * 100).toFixed(1)}%</span></div>
  <div class="kpi"><div class="label">Backtest error, ${M.model}</div>
    <div class="value">${M.mape != null ? M.mape.toFixed(1) + "%" : "–"}</div>
    <div class="sub">${M.naive_mape != null ? `vs. ${M.naive_mape.toFixed(1)}% for “same month last year”` : "Mean absolute % error"}</div></div>
</div>`);
```

## Recent demand and the next 12 months

<div class="card">
<div class="legend"><span><i class="sw" style="border-color:var(--ink)"></i>Actual</span><span><i class="sw" style="border-color:var(--accent);border-top-style:dashed;border-top-width:3px"></i>Forecast, normal weather</span><span><i class="sw" style="border-color:var(--cold);border-top-style:dotted"></i>Cold winter</span><span><i class="sw" style="border-color:var(--warm);border-top-style:dotted"></i>Warm winter</span><span><i class="sw box" style="background:var(--fc-band)"></i>Likely range (90%)</span></div>

```js
display(resize((width) => {
  const narrow = width < 560;
  const end = d3.utcDay.offset(fc.at(-1).d, 20);
  const yMax = d3.max([...hist.map((r) => r.v), ...fc.map((r) => r.hi90)]) * 1.08;
  const normalPath = [{d: last, normal: lastV}, ...fc];
  return Plot.plot({
    width, height: narrow ? 340 : 420, marginLeft: 44, marginTop: 30, marginRight: 12, style: {fontSize: "12px"},
    x: {type: "utc", domain: [d3.utcDay.offset(hist[0].d, -15), end], label: null,
        ticks: d3.utcMonth.every(narrow ? 6 : 3), tickFormat: (d) => d3.utcFormat(d.getUTCMonth() === 0 || narrow ? "%b %Y" : "%b")(d)},
    y: {domain: [0, yMax], grid: true, label: "Bcf per month", labelArrow: "none"},
    marks: [
      Plot.rectX([{}], {x1: last, x2: end, y1: 0, y2: yMax, fill: "var(--fc-wash)"}),
      Plot.ruleX([last], {stroke: "var(--muted)", strokeDasharray: "2,3"}),
      Plot.text([`Latest actual · ${fmtMon(last)}`], {x: last, frameAnchor: "top", dy: -20, dx: -6, textAnchor: "end", fill: "var(--muted)", fontSize: 11}),
      Plot.text(["FORECAST →"], {x: last, frameAnchor: "top", dy: -20, dx: 6, textAnchor: "start", fill: "var(--accent)", fontWeight: 700, fontSize: 11}),
      Plot.areaY(fc, {x: "d", y1: "lo90", y2: "hi90", fill: "var(--fc-band)", curve: "monotone-x"}),
      Plot.lineY(hist, {x: "d", y: "v", stroke: "var(--ink)", strokeWidth: 2, curve: "monotone-x"}),
      Plot.dot(hist, {x: "d", y: "v", r: 2.2, fill: "var(--ink)"}),
      Plot.lineY(fc, {x: "d", y: "cold", stroke: "var(--cold)", strokeWidth: 1.6, strokeDasharray: "2,3", curve: "monotone-x"}),
      Plot.lineY(fc, {x: "d", y: "warm", stroke: "var(--warm)", strokeWidth: 1.6, strokeDasharray: "2,3", curve: "monotone-x"}),
      Plot.lineY(normalPath, {x: "d", y: "normal", stroke: "var(--accent)", strokeWidth: 3.2, strokeDasharray: "7,5", curve: "monotone-x"}),
      Plot.dot(fc, {x: "d", y: "normal", r: 4.5, fill: "var(--surface)", stroke: "var(--accent)", strokeWidth: 2}),
      cur ? Plot.dot([cur], {x: "d", y: "normal", r: 10, stroke: "var(--accent)", strokeWidth: 2}) : null,
      cur ? Plot.text([cur], {x: "d", y: "normal", text: (r) => `This month (${d3.utcFormat("%b")(r.d)}): ${r.normal.toFixed(1)} Bcf`,
        dx: -14, dy: -18, textAnchor: "end", fill: "var(--accent)", fontWeight: 700, fontSize: 12, stroke: "var(--surface)", strokeWidth: 4, paintOrder: "stroke"}) : null,
      Plot.text([peak], {x: "d", y: "hi90", text: (r) => `Peak ${d3.utcFormat("%b")(r.d)}: ${r.normal.toFixed(1)} Bcf`, dy: -10,
        textAnchor: "middle", fill: "var(--muted)", fontSize: 11, stroke: "var(--surface)", strokeWidth: 4, paintOrder: "stroke"}),
      Plot.tip([...hist.map((r) => ({...r, kind: "Actual"})), ...fc.map((r) => ({...r, v: r.normal, kind: "Forecast"}))],
        Plot.pointerX({x: "d", y: "v", title: (r) => r.kind === "Actual" ? `${fmtMonth(r.d)}\nActual: ${r.v.toFixed(1)} Bcf`
          : `${fmtMonth(r.d)} forecast\nNormal weather: ${r.normal.toFixed(1)} Bcf\nCold: ${r.cold.toFixed(1)} · Warm: ${r.warm.toFixed(1)}\nLikely range: ${r.lo90.toFixed(1)}–${r.hi90.toFixed(1)} Bcf`}))
    ].filter(Boolean)
  });
}));
```

</div>

<p class="fine">Actual data through ${fmtMonth(last)}. The dashed blue line is the ${M.model} forecast with normal weather; the dotted lines are a cold or warm winter (90th and 10th percentile monthly heating degree days over the last 15 years); the band is the likely range from backtest errors at each horizon. Weather sensitivity: about ${M.mmcf_per_hdd?.toFixed(1)} MMcf of extra demand per heating degree day.</p>

<p class="fine">Source: EIA natural gas deliveries to Massachusetts residential, commercial and industrial customers; weather from ERA5 via Open-Meteo.</p>
