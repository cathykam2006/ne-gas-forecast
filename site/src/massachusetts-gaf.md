---
title: Massachusetts GAF
---

<div class="eyebrow">New England gas forecast · Massachusetts DPU filings</div>

# Massachusetts Gas Adjustment Factors

<p class="lede">The gas adjustment factor (GAF) passes each utility’s cost of buying and delivering gas supply through to customers; the local distribution adjustment factor (LDAF) recovers other reconciling costs. Both are filed with the Massachusetts Department of Public Utilities and billed per therm, on top of base distribution rates and customer charges set in each company’s rate case.</p>

```js
const M = FileAttachment("data/ma-gaf.json").json();
```

```js
const parseM = d3.utcParse("%Y-%m");
const fmtMonth = d3.utcFormat("%B %Y"), fmtMon = d3.utcFormat("%b %Y");
const usd4 = (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toFixed(4);
const all = M.rows.map(([m, c, k, gaf, ldaf]) => ({d: parseM(m), co: M.companies[c], cat: M.categories[k], gaf, ldaf: ldaf ?? 0, total: gaf + (ldaf ?? 0)}));
const hh = M.hh.map((r) => ({d: parseM(r.m), v: r.v}));
const hhByMonth = new Map(hh.map((r) => [+r.d, r.v]));
const latest = parseM(M.latest_month);
const order = ["Residential heating", "Residential non-heating", "Small C&I", "Medium C&I", "Large C&I"].filter((c) => M.categories.includes(c));
```

<div class="controls">

```js
const cat = view(Inputs.select(order, {value: "Residential heating", label: "Rate category"}));
```

</div>

```js
const inCat = all.filter((r) => r.cat === cat);
const current = inCat.filter((r) => +r.d === +latest).sort((a, b) => a.total - b.total);
const hhNow = hh.at(-1);
const strip = M.strip.map((r) => ({...r, d: parseM(r.m)}));
const asOf = M.strip_as_of ? d3.utcParse("%Y-%m-%d")(M.strip_as_of) : null;
const py = asOf && (asOf.getUTCMonth() <= 3 ? asOf.getUTCFullYear() - 1 : asOf.getUTCFullYear());
const peakStrip = asOf ? strip.filter((r) => r.d >= Date.UTC(py, 10, 1) && r.d <= Date.UTC(py + 1, 3, 1)) : [];
const peakAvg = d3.mean(peakStrip, (r) => r.therm);
```

```js
{
  const lo = current[0], hi = current.at(-1);
  display(html`<div class="kpis">
    <div class="kpi"><div class="label">Lowest GAF + LDAF, ${fmtMon(latest)}</div>
      <div class="value">${usd4(lo.total)}<small>/therm</small></div><div class="sub">${lo.co}</div></div>
    <div class="kpi"><div class="label">Highest GAF + LDAF, ${fmtMon(latest)}</div>
      <div class="value">${usd4(hi.total)}<small>/therm</small></div><div class="sub">${hi.co}</div></div>
    <div class="kpi"><div class="label">Henry Hub, ${fmtMon(hhNow.d)} average</div>
      <div class="value">${usd4(hhNow.v)}<small>/therm</small></div><div class="sub">Wholesale commodity benchmark</div></div>
    <div class="kpi"><div class="label">NYMEX futures, coming peak period</div>
      <div class="value">${peakStrip.length ? usd4(peakAvg) : "–"}<small>/therm</small></div>
      <div class="sub">${peakStrip.length ? `${d3.utcFormat("%b")(peakStrip[0].d)} ${py}–Apr ${py + 1} average, as of ${d3.utcFormat("%b %-d")(asOf)}` : "Not available"}</div></div>
  </div>`);
}
```

## Rates in effect, ${fmtMonth(latest)}

<p class="fine">${cat}, $/therm, sorted from lowest to highest. The bar shows GAF + LDAF.</p>

```js
{
  const max = d3.max(current, (r) => r.total);
  display(html`<div class="tablewrap"><table class="plain">
    <thead><tr><th>Company</th><th>GAF (supply)</th><th>LDAF (other adjustments)</th><th>GAF + LDAF</th></tr></thead>
    <tbody>${current.map((r) => html`<tr><td>${r.co}</td><td>${usd4(r.gaf)}</td><td>${usd4(r.ldaf)}</td>
      <td>${usd4(r.total)}<span class="bar" style=${`width:${Math.max(2, (r.total / max) * 90)}px`}></span></td></tr>`)}
    <tr><td style="color:var(--muted)">Henry Hub wholesale (${fmtMon(hhNow.d)})</td><td style="color:var(--muted)">${usd4(hhNow.v)}</td><td></td><td></td></tr>
    </tbody></table></div>`);
}
```

## One company over time

<div class="controls">

```js
const company = view(Inputs.select(current.map((r) => r.co).sort(), {
  value: current.some((r) => r.co === "National Grid – Boston Gas") ? "National Grid – Boston Gas" : current[0].co, label: "Company"}));
```

```js
const span = view(Inputs.radio(["Latest 12 months", "Last 5 years", "Since 2008"], {value: "Latest 12 months", label: "Period"}));
```

</div>

```js
const spanStart = span === "Latest 12 months" ? d3.utcMonth.offset(latest, -11)
  : span === "Last 5 years" ? d3.utcMonth.offset(latest, -59) : d3.min(inCat, (r) => r.d);
const co = inCat.filter((r) => r.co === company && r.d >= spanStart).sort((a, b) => a.d - b.d);
const hhSpan = hh.filter((r) => r.d >= spanStart && r.d <= latest);
// peak GAF periods (Nov–Apr) inside the window, for shading
const peaks = d3.utcYears(d3.utcYear.offset(spanStart, -1), d3.utcYear.offset(latest, 1))
  .map((y) => ({a: new Date(Date.UTC(y.getUTCFullYear(), 10, 1)), b: new Date(Date.UTC(y.getUTCFullYear() + 1, 4, 1))}))
  .filter((p) => p.b > spanStart && p.a <= d3.utcMonth.offset(latest, 1))
  .map((p) => ({a: p.a < spanStart ? spanStart : p.a, b: p.b > d3.utcMonth.offset(latest, 1) ? d3.utcMonth.offset(latest, 1) : p.b}));
const monthly = span === "Latest 12 months";
```

<div class="card">
<h3>${company}: GAF and LDAF in force each month · ${cat}</h3>
<div class="legend"><span><i class="sw box" style="background:var(--accent)"></i>GAF (supply)</span><span><i class="sw box" style="background:var(--ldaf)"></i>LDAF (negative = credit)</span><span><i class="sw" style="border-color:var(--actual)"></i>Henry Hub wholesale</span><span><i class="sw box" style="background:var(--event-wash)"></i>Peak period (Nov–Apr)</span></div>

```js
display(resize((width) => {
  const stacked = co.flatMap((r) => [{d: r.d, part: "GAF (supply)", v: r.gaf}, {d: r.d, part: "LDAF", v: r.ldaf}]);
  const end = d3.utcMonth.offset(latest, 1);
  return Plot.plot({
    width, height: 340, marginLeft: 48, marginRight: 8, style: {fontSize: "12px"},
    x: {type: "utc", domain: [spanStart, end], label: null,
        ticks: monthly ? d3.utcMonth.range(spanStart, end, width < 560 ? 2 : 1) : span === "Last 5 years" ? d3.utcYear.every(1) : d3.utcYear.every(width < 560 ? 4 : 2),
        tickFormat: monthly ? (d) => d3.utcFormat(d.getUTCMonth() === 0 ? "%b %Y" : "%b")(d) : "%Y"},
    y: {grid: true, label: "$/therm", labelArrow: "none", tickFormat: (d) => "$" + d.toFixed(2)},
    marks: [
      Plot.rectX(peaks, {x1: "a", x2: "b", y1: d3.min(co, (r) => Math.min(0, r.ldaf)) ?? 0, y2: d3.max(co, (r) => r.gaf + Math.max(0, r.ldaf)) * 1.05, fill: "var(--event-wash)"}),
      Plot.rectY(stacked, Plot.stackY({x1: "d", x2: (r) => d3.utcDay.offset(d3.utcMonth.offset(r.d, 1), monthly ? -3 : 0), y: "v",
        fill: (r) => (r.part === "LDAF" ? "var(--ldaf)" : "var(--accent)"), insetLeft: monthly ? 2 : 0})),
      Plot.ruleY([0], {stroke: "var(--muted)"}),
      Plot.lineY(hhSpan, {x: (r) => d3.utcDay.offset(r.d, 14), y: "v", stroke: "var(--actual)", strokeWidth: 2, curve: "monotone-x"}),
      Plot.tip(co, Plot.pointerX({x: (r) => d3.utcDay.offset(r.d, 14), y: "gaf", title: (r) =>
        `${fmtMonth(r.d)}\nGAF: ${usd4(r.gaf)}\nLDAF: ${usd4(r.ldaf)}\nGAF + LDAF: ${usd4(r.total)}` +
        (hhByMonth.has(+r.d) ? `\nHenry Hub: ${usd4(hhByMonth.get(+r.d))}` : "")}))
    ]
  });
}));
```

</div>

## All companies compared

```js
const active = new Set(current.map((r) => r.co));
const lines = inCat.filter((r) => active.has(r.co) && r.d >= spanStart);
const lineEnds = lines.filter((r) => +r.d === +latest).map((r) => ({...r, d: d3.utcMonth.offset(latest, 1)}));
const prem = d3.rollups(lines, (v) => d3.mean(v, (r) => r.gaf), (r) => r.co)
  .map(([c, g]) => g - d3.mean(hhSpan, (r) => r.v));
```

<p class="fine">GAF by company, ${span.toLowerCase()}. ${company} is highlighted; change it above. Over this window the average GAF sits $${d3.min(prem).toFixed(2)}–$${d3.max(prem).toFixed(2)}/therm above Henry Hub across companies: that premium covers pipeline capacity into New England, storage, LNG peaking supply and prior-period reconciliations.</p>

<div class="card">
<div class="legend"><span><i class="sw" style="border-color:var(--accent)"></i>${company}</span><span><i class="sw" style="border-color:var(--rule)"></i>Other companies</span><span><i class="sw" style="border-color:var(--actual);border-top-style:dotted"></i>Henry Hub wholesale</span></div>

```js
display(resize((width) => {
  const end = d3.utcMonth.offset(latest, 1);
  const steps = [...lines, ...lineEnds];
  const others = steps.filter((r) => r.co !== company), mine = steps.filter((r) => r.co === company);
  return Plot.plot({
    width, height: 360, marginLeft: 48, marginRight: 8, style: {fontSize: "12px"},
    x: {type: "utc", domain: [spanStart, end], label: null,
        ticks: monthly ? d3.utcMonth.range(spanStart, end, width < 560 ? 2 : 1) : span === "Last 5 years" ? d3.utcYear.every(1) : d3.utcYear.every(width < 560 ? 4 : 2),
        tickFormat: monthly ? (d) => d3.utcFormat(d.getUTCMonth() === 0 ? "%b %Y" : "%b")(d) : "%Y"},
    y: {grid: true, label: "GAF, $/therm", labelArrow: "none", tickFormat: (d) => "$" + d.toFixed(2)},
    marks: [
      Plot.lineY(others, {x: "d", y: "gaf", z: "co", curve: "step-after", stroke: "var(--rule)", strokeWidth: 1.4}),
      Plot.lineY(hhSpan, {x: "d", y: "v", curve: "monotone-x", stroke: "var(--actual)", strokeWidth: 1.6, strokeDasharray: "2,3"}),
      Plot.lineY(mine, {x: "d", y: "gaf", curve: "step-after", stroke: "var(--accent)", strokeWidth: 2.4}),
      Plot.tip(lines, Plot.pointer({x: "d", y: "gaf", title: (r) => `${r.co}\n${fmtMonth(r.d)}\nGAF: ${usd4(r.gaf)}`}))
    ]
  });
}));
```

</div>

<p class="fine">Source: Massachusetts DPU current gas adjustment factors and historical GAF/LDAF workbook, May 2008 onward. Predecessor companies are mapped to today’s names (Bay State/Columbia → Eversource Gas of MA; KeySpan → National Grid; New England Gas → Liberty). Rates filed in $/Ccf are converted to $/therm.</p>
