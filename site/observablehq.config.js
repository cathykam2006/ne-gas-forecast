// Observable Framework site for ne-gas-forecast. Pages read data/processed through the
// Python data loaders in src/data/, so every pipeline run refreshes the site on the next build.
export default {
  title: "New England Gas Forecast",
  root: "src",
  style: "style.css",
  pages: [
    {name: "Henry Hub price outlook", path: "/"},
    {name: "Massachusetts demand", path: "/massachusetts-demand"},
    {name: "Massachusetts GAF", path: "/massachusetts-gaf"}
  ],
  toc: false,
  pager: false,
  search: false,
  head: `<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=Public+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">`,
  footer: `Data: EIA (Henry Hub spot, storage, LNG exports), CME NYMEX via Yahoo Finance (futures), Massachusetts DPU (GAF/LDAF), NOAA (weather).
Source code: <a href="https://github.com/cathykam2006/ne-gas-forecast">github.com/cathykam2006/ne-gas-forecast</a>.`
};
