/* IMC Prosperity dashboard — frontend logic. */

const DARK_LAYOUT = {
  paper_bgcolor: "#161b22",
  plot_bgcolor: "#0e1117",
  font: { color: "#e6edf3", size: 11 },
  margin: { l: 55, r: 20, t: 10, b: 40 },
  xaxis: { gridcolor: "#222831", zerolinecolor: "#222831" },
  yaxis: { gridcolor: "#222831", zerolinecolor: "#222831" },
  legend: { orientation: "h", y: -0.18 },
};

const CONFIG = { displaylogo: false, responsive: true };

let sources = null;     // /api/sources result
let current = null;     // { prices, trades, has_submission }

const els = {
  status: document.getElementById("status"),
  source: () => document.querySelector("input[name=source]:checked").value,
  roundSel: document.getElementById("round-select"),
  daySel: document.getElementById("day-select"),
  btSel: document.getElementById("backtest-select"),
  productSel: document.getElementById("product-select"),
  roundGrp: document.getElementById("round-controls"),
  btGrp: document.getElementById("backtest-controls"),
  loadBtn: document.getElementById("load-btn"),
  summary: document.getElementById("summary-strip"),
};

function setStatus(msg, cls = "") {
  els.status.textContent = msg;
  els.status.className = "status " + cls;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  const json = await res.json();
  if (!res.ok) throw new Error(json.error || res.statusText);
  return json;
}

function fillSelect(sel, options, labelFn = (x) => x, valueFn = (x) => x) {
  sel.innerHTML = "";
  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = valueFn(opt);
    o.textContent = labelFn(opt);
    sel.appendChild(o);
  });
}

async function init() {
  try {
    setStatus("loading sources…");
    sources = await fetchJSON("/api/sources");
  } catch (err) {
    setStatus("failed to load sources: " + err.message, "err");
    return;
  }

  fillSelect(els.roundSel, sources.rounds, (r) => "Round " + r.round, (r) => r.round);
  if (sources.rounds.length === 0) {
    setStatus("no round CSVs found", "err");
  } else {
    syncDays();
  }
  fillSelect(
    els.btSel,
    sources.backtests,
    (b) => `${b.folder}/${b.name}  (${b.size_kb} KB)`,
    (b) => b.path,
  );

  document.querySelectorAll("input[name=source]").forEach((r) =>
    r.addEventListener("change", onSourceChange),
  );
  els.roundSel.addEventListener("change", syncDays);
  els.loadBtn.addEventListener("click", load);
  els.productSel.addEventListener("change", renderAll);

  setStatus("ready — pick a source and hit Load.", "ok");
}

function onSourceChange() {
  const s = els.source();
  els.roundGrp.classList.toggle("hidden", s !== "round");
  els.btGrp.classList.toggle("hidden", s !== "backtest");
}

function syncDays() {
  const r = sources.rounds.find((x) => x.round === els.roundSel.value);
  if (!r) return;
  fillSelect(els.daySel, r.days, (d) => "Day " + d, (d) => d);
}

async function load() {
  try {
    setStatus("loading…");
    const s = els.source();
    const url =
      s === "round"
        ? `/api/round?round=${encodeURIComponent(els.roundSel.value)}&day=${encodeURIComponent(els.daySel.value)}`
        : `/api/backtest?file=${encodeURIComponent(els.btSel.value)}`;
    current = await fetchJSON(url);
  } catch (err) {
    setStatus("load failed: " + err.message, "err");
    return;
  }

  const products = current.prices.products;
  fillSelect(els.productSel, products);
  if (products.length === 0) {
    setStatus("no products in source", "err");
    return;
  }
  // Prefer the product with the largest PnL magnitude if backtest, else first.
  let best = products[0];
  if (current.has_submission) {
    let bestAbs = -1;
    for (const p of products) {
      const pnl = current.prices.data[p].pnl;
      const final = pnl[pnl.length - 1] ?? 0;
      if (Math.abs(final) > bestAbs) {
        bestAbs = Math.abs(final);
        best = p;
      }
    }
  }
  els.productSel.value = best;

  renderAll();
  setStatus(
    `${current.has_submission ? "Backtest" : "Round data"} — ${products.length} products, ` +
    `${current.trades.length} trades`,
    "ok",
  );
}

function ownTrades(product) {
  return current.trades.filter((t) => t.symbol === product && t.buyer === "SUBMISSION");
}
function selfSells(product) {
  return current.trades.filter((t) => t.symbol === product && t.seller === "SUBMISSION");
}
function marketTrades(product) {
  return current.trades.filter(
    (t) => t.symbol === product && t.buyer !== "SUBMISSION" && t.seller !== "SUBMISSION",
  );
}

function renderAll() {
  const product = els.productSel.value;
  if (!product || !current) return;
  renderPrice(product);
  renderDepth(product);
  renderSpread(product);
  renderPnL(product);
  renderTradeDistribution(product);
  renderSummary(product);
}

function renderPrice(product) {
  const d = current.prices.data[product];
  const traces = [
    { x: d.t, y: d.mid, name: "mid", mode: "lines", line: { color: "#58a6ff", width: 1.8 } },
    { x: d.t, y: d.bid1, name: "best bid", mode: "lines", line: { color: "#3fb950", width: 1 }, opacity: 0.75 },
    { x: d.t, y: d.ask1, name: "best ask", mode: "lines", line: { color: "#f85149", width: 1 }, opacity: 0.75 },
  ];

  if (current.has_submission) {
    const buys = ownTrades(product);
    const sells = selfSells(product);
    if (buys.length) {
      traces.push({
        x: buys.map((t) => t.timestamp),
        y: buys.map((t) => t.price),
        name: "own buys",
        mode: "markers",
        marker: { color: "#3fb950", symbol: "triangle-up", size: 9, line: { color: "#0e1117", width: 1 } },
        text: buys.map((t) => `qty ${t.quantity} @ ${t.price}`),
        hoverinfo: "text+x",
      });
    }
    if (sells.length) {
      traces.push({
        x: sells.map((t) => t.timestamp),
        y: sells.map((t) => t.price),
        name: "own sells",
        mode: "markers",
        marker: { color: "#f85149", symbol: "triangle-down", size: 9, line: { color: "#0e1117", width: 1 } },
        text: sells.map((t) => `qty ${t.quantity} @ ${t.price}`),
        hoverinfo: "text+x",
      });
    }
  }

  Plotly.react("chart-price", traces, {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "timestamp" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "price" },
  }, CONFIG);
}

function renderDepth(product) {
  const d = current.prices.data[product];
  Plotly.react("chart-depth", [
    { x: d.t, y: d.bid_depth, name: "bid volume", mode: "lines", stackgroup: "bids",
      line: { color: "#3fb950", width: 0 }, fillcolor: "rgba(63,185,80,0.45)" },
    { x: d.t, y: d.ask_depth.map((v) => -v), name: "ask volume", mode: "lines", stackgroup: "asks",
      line: { color: "#f85149", width: 0 }, fillcolor: "rgba(248,81,73,0.45)" },
  ], {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "timestamp" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "volume (asks as negative)" },
  }, CONFIG);
}

function renderSpread(product) {
  const d = current.prices.data[product];
  Plotly.react("chart-spread", [
    { x: d.t, y: d.spread, mode: "lines", line: { color: "#d29922", width: 1.3 }, name: "spread" },
  ], {
    ...DARK_LAYOUT,
    showlegend: false,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "timestamp" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "ask − bid" },
  }, CONFIG);
}

function renderPnL(product) {
  const d = current.prices.data[product];
  const traces = [
    { x: d.t, y: d.pnl, name: "PnL", mode: "lines", line: { color: "#58a6ff", width: 1.6 } },
  ];

  if (current.has_submission) {
    // Build running position from SUBMISSION trades.
    const buys = ownTrades(product);
    const sells = selfSells(product);
    const moves = [
      ...buys.map((t) => ({ t: t.timestamp, q: t.quantity })),
      ...sells.map((t) => ({ t: t.timestamp, q: -t.quantity })),
    ].sort((a, b) => a.t - b.t);
    let pos = 0;
    const xs = [];
    const ys = [];
    for (const m of moves) {
      pos += m.q;
      xs.push(m.t);
      ys.push(pos);
    }
    traces.push({
      x: xs,
      y: ys,
      name: "position",
      mode: "lines",
      line: { color: "#d29922", width: 1.2, shape: "hv" },
      yaxis: "y2",
    });
  }

  Plotly.react("chart-pnl", traces, {
    ...DARK_LAYOUT,
    xaxis: { ...DARK_LAYOUT.xaxis, title: "timestamp" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "PnL (seashells)" },
    yaxis2: {
      title: "position",
      overlaying: "y",
      side: "right",
      gridcolor: "transparent",
      zerolinecolor: "#444",
    },
  }, CONFIG);
}

function renderTradeDistribution(product) {
  const market = marketTrades(product);
  const own = current.has_submission
    ? current.trades.filter((t) => t.symbol === product && (t.buyer === "SUBMISSION" || t.seller === "SUBMISSION"))
    : [];
  const traces = [
    { x: market.map((t) => t.quantity), name: "market trades", type: "histogram",
      marker: { color: "#58a6ff" }, opacity: 0.75, nbinsx: 30 },
  ];
  if (own.length) {
    traces.push({
      x: own.map((t) => t.quantity),
      name: "own trades",
      type: "histogram",
      marker: { color: "#f0883e" },
      opacity: 0.8,
      nbinsx: 30,
    });
  }
  Plotly.react("chart-trades", traces, {
    ...DARK_LAYOUT,
    barmode: "overlay",
    xaxis: { ...DARK_LAYOUT.xaxis, title: "trade size" },
    yaxis: { ...DARK_LAYOUT.yaxis, title: "count" },
  }, CONFIG);
}

function renderSummary(product) {
  const d = current.prices.data[product];
  const mids = d.mid.filter((v) => Number.isFinite(v));
  const spreads = d.spread.filter((v) => Number.isFinite(v));
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const stats = [
    { label: "Ticks", value: d.t.length.toLocaleString() },
    { label: "Mid (last)", value: mids.length ? mids[mids.length - 1].toFixed(2) : "—" },
    { label: "Mid (mean)", value: mids.length ? avg(mids).toFixed(2) : "—" },
    { label: "Mid σ", value: mids.length ? std(mids).toFixed(2) : "—" },
    { label: "Avg spread", value: spreads.length ? avg(spreads).toFixed(2) : "—" },
    { label: "Avg bid depth", value: avg(d.bid_depth).toFixed(1) },
    { label: "Avg ask depth", value: avg(d.ask_depth).toFixed(1) },
  ];

  if (current.has_submission) {
    const finalPnl = d.pnl[d.pnl.length - 1] ?? 0;
    const own = current.trades.filter((t) => t.symbol === product && (t.buyer === "SUBMISSION" || t.seller === "SUBMISSION"));
    const buys = ownTrades(product);
    const sells = selfSells(product);
    const ownVol = own.reduce((s, t) => s + t.quantity, 0);
    const pos = buys.reduce((s, t) => s + t.quantity, 0) - sells.reduce((s, t) => s + t.quantity, 0);
    stats.push(
      { label: "Final PnL", value: finalPnl.toFixed(0) },
      { label: "Own trades", value: own.length.toLocaleString() },
      { label: "Own volume", value: ownVol.toLocaleString() },
      { label: "Final position", value: pos.toString() },
    );
  } else {
    const total = current.trades.filter((t) => t.symbol === product);
    const vol = total.reduce((s, t) => s + (t.quantity || 0), 0);
    stats.push(
      { label: "Market trades", value: total.length.toLocaleString() },
      { label: "Market volume", value: vol.toLocaleString() },
    );
  }

  els.summary.innerHTML = stats
    .map((s) => `<div class="summary-card"><div class="label">${s.label}</div><div class="value">${s.value}</div></div>`)
    .join("");
}

function std(a) {
  if (a.length < 2) return 0;
  const mean = a.reduce((x, y) => x + y, 0) / a.length;
  const v = a.reduce((s, x) => s + (x - mean) ** 2, 0) / (a.length - 1);
  return Math.sqrt(v);
}

init();
