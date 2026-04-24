import type { Data, Layout } from 'plotly.js'
import { DARK_LAYOUT } from './chartConfig'
import { average, std } from './math'
import type { LoadResponse, ProductSeries, TradeRow } from './types'
import { marketTrades, ownTrades, selfSells } from './trades'

/** plotly.js typings expect `title: { text }`, not a bare string. */
function t(text: string) {
  return { text }
}

export type PriceLayerToggles = {
  mid: boolean
  bid: boolean
  ask: boolean
  ownBuys: boolean
  ownSells: boolean
}

export const DEFAULT_PRICE_LAYERS: PriceLayerToggles = {
  mid: true,
  bid: true,
  ask: true,
  ownBuys: true,
  ownSells: true,
}

function price(
  d: ProductSeries,
  hasSubmission: boolean,
  trades: TradeRow[],
  product: string,
  layers: PriceLayerToggles,
): { data: Data[]; layout: Partial<Layout> } {
  const traces: Data[] = []

  if (layers.mid) {
    traces.push({
      x: d.t,
      y: d.mid,
      name: 'mid',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#4da3ff', width: 1.8 },
    })
  }
  if (layers.bid) {
    traces.push({
      x: d.t,
      y: d.bid1,
      name: 'best bid',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#3ecf8e', width: 1 },
      opacity: 0.8,
    })
  }
  if (layers.ask) {
    traces.push({
      x: d.t,
      y: d.ask1,
      name: 'best ask',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#f4727a', width: 1 },
      opacity: 0.8,
    })
  }

  if (hasSubmission) {
    const buys = ownTrades(trades, product)
    const sells = selfSells(trades, product)
    if (layers.ownBuys && buys.length) {
      traces.push({
        x: buys.map((tr) => tr.timestamp),
        y: buys.map((tr) => tr.price),
        name: 'own buys',
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#3ecf8e', symbol: 'triangle-up', size: 9, line: { color: '#0a0e14', width: 1 } },
        text: buys.map((tr) => `qty ${tr.quantity} @ ${tr.price}`),
        hoverinfo: 'x+text',
      })
    }
    if (layers.ownSells && sells.length) {
      traces.push({
        x: sells.map((tr) => tr.timestamp),
        y: sells.map((tr) => tr.price),
        name: 'own sells',
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#f4727a', symbol: 'triangle-down', size: 9, line: { color: '#0a0e14', width: 1 } },
        text: sells.map((tr) => `qty ${tr.quantity} @ ${tr.price}`),
        hoverinfo: 'x+text',
      })
    }
  }

  if (traces.length === 0) {
    traces.push({
      x: d.t,
      y: d.mid,
      name: 'mid',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#4da3ff', width: 1.2 },
    })
  }

  return {
    data: traces,
    layout: {
      ...DARK_LAYOUT,
      xaxis: { ...DARK_LAYOUT.xaxis, title: t('timestamp') },
      yaxis: { ...DARK_LAYOUT.yaxis, title: t('price') },
    },
  }
}

function depth(d: ProductSeries): { data: Data[]; layout: Partial<Layout> } {
  return {
    data: [
      {
        x: d.t,
        y: d.bid_depth,
        name: 'bid volume',
        type: 'scatter',
        mode: 'lines',
        stackgroup: 'bids',
        line: { color: '#3ecf8e', width: 0 },
        fillcolor: 'rgba(62, 207, 142, 0.4)',
      },
      {
        x: d.t,
        y: d.ask_depth.map((v) => -v),
        name: 'ask volume',
        type: 'scatter',
        mode: 'lines',
        stackgroup: 'asks',
        line: { color: '#f4727a', width: 0 },
        fillcolor: 'rgba(244, 114, 122, 0.4)',
      },
    ],
    layout: {
      ...DARK_LAYOUT,
      xaxis: { ...DARK_LAYOUT.xaxis, title: t('timestamp') },
      yaxis: { ...DARK_LAYOUT.yaxis, title: t('volume (asks as negative)') },
    },
  }
}

function spread(d: ProductSeries): { data: Data[]; layout: Partial<Layout> } {
  return {
    data: [
      {
        x: d.t,
        y: d.spread,
        name: 'spread',
        type: 'scatter',
        mode: 'lines',
        line: { color: '#e8b44c', width: 1.3 },
      },
    ],
    layout: {
      ...DARK_LAYOUT,
      showlegend: false,
      xaxis: { ...DARK_LAYOUT.xaxis, title: t('timestamp') },
      yaxis: { ...DARK_LAYOUT.yaxis, title: t('ask − bid') },
    },
  }
}

function pnl(
  d: ProductSeries,
  hasSubmission: boolean,
  trades: TradeRow[],
  product: string,
): { data: Data[]; layout: Partial<Layout> } {
  const traces: Data[] = [
    {
      x: d.t,
      y: d.pnl,
      name: 'PnL',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#4da3ff', width: 1.6 },
    },
  ]

  if (hasSubmission) {
    const buys = ownTrades(trades, product)
    const sells = selfSells(trades, product)
    const moves = [
      ...buys.map((t) => ({ t: t.timestamp, q: t.quantity })),
      ...sells.map((t) => ({ t: t.timestamp, q: -t.quantity })),
    ].sort((a, b) => a.t - b.t)
    let pos = 0
    const xs: number[] = []
    const ys: number[] = []
    for (const m of moves) {
      pos += m.q
      xs.push(m.t)
      ys.push(pos)
    }
    traces.push({
      x: xs,
      y: ys,
      name: 'position',
      type: 'scatter',
      mode: 'lines',
      line: { color: '#e8b44c', width: 1.2, shape: 'hv' },
      yaxis: 'y2',
    })
  }

  return {
    data: traces,
    layout: {
      ...DARK_LAYOUT,
      xaxis: { ...DARK_LAYOUT.xaxis, title: t('timestamp') },
      yaxis: { ...DARK_LAYOUT.yaxis, title: t('PnL (seashells)') },
      yaxis2: {
        title: t('position'),
        overlaying: 'y',
        side: 'right',
        gridcolor: 'transparent',
        zerolinecolor: '#444',
      },
    },
  }
}

function tradeDistribution(
  hasSubmission: boolean,
  trades: TradeRow[],
  product: string,
): { data: Data[]; layout: Partial<Layout> } {
  const market = marketTrades(trades, product)
  const own = hasSubmission
    ? trades.filter(
        (t) => t.symbol === product && (t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION'),
      )
    : []
  const traces: Data[] = [
    {
      x: market.map((tr) => tr.quantity),
      name: 'market trades',
      type: 'histogram',
      marker: { color: '#4da3ff' },
      opacity: 0.75,
      nbinsx: 30,
    } as Data,
  ]
  if (own.length) {
    traces.push({
      x: own.map((tr) => tr.quantity),
      name: 'own trades',
      type: 'histogram',
      marker: { color: '#f59e4a' },
      opacity: 0.8,
      nbinsx: 30,
    } as Data)
  }
  return {
    data: traces,
    layout: {
      ...DARK_LAYOUT,
      barmode: 'overlay',
      xaxis: { ...DARK_LAYOUT.xaxis, title: t('trade size') },
      yaxis: { ...DARK_LAYOUT.yaxis, title: t('count') },
    },
  }
}

export type SummaryItem = { label: string; value: string }

function buildSummary(
  current: LoadResponse,
  d: ProductSeries,
  product: string,
): SummaryItem[] {
  const mids = d.mid.filter((v): v is number => v !== null && Number.isFinite(v))
  const spreads = d.spread.filter((v): v is number => v !== null && Number.isFinite(v))
  const stats: SummaryItem[] = [
    { label: 'Ticks', value: d.t.length.toLocaleString() },
    { label: 'Mid (last)', value: mids.length ? mids[mids.length - 1]!.toFixed(2) : '—' },
    { label: 'Mid (mean)', value: mids.length ? average(mids).toFixed(2) : '—' },
    { label: 'Mid σ', value: mids.length ? std(mids).toFixed(2) : '—' },
    { label: 'Avg spread', value: spreads.length ? average(spreads).toFixed(2) : '—' },
    { label: 'Avg bid depth', value: average(d.bid_depth).toFixed(1) },
    { label: 'Avg ask depth', value: average(d.ask_depth).toFixed(1) },
  ]

  if (current.has_submission) {
    const pnlRow = d.pnl
    const finalPnl = pnlRow.length ? pnlRow[pnlRow.length - 1] ?? 0 : 0
    const own = current.trades.filter(
      (t) => t.symbol === product && (t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION'),
    )
    const buys = ownTrades(current.trades, product)
    const sells = selfSells(current.trades, product)
    const ownVol = own.reduce((s, t) => s + t.quantity, 0)
    const pos =
      buys.reduce((s, t) => s + t.quantity, 0) - sells.reduce((s, t) => s + t.quantity, 0)
    stats.push(
      { label: 'Final PnL', value: Number(finalPnl).toFixed(0) },
      { label: 'Own trades', value: own.length.toLocaleString() },
      { label: 'Own volume', value: ownVol.toLocaleString() },
      { label: 'Final position', value: String(pos) },
    )
    if (current.sandbox_count != null) {
      stats.push({ label: 'Sandbox log lines', value: String(current.sandbox_count) })
    }
  } else {
    const total = current.trades.filter((t) => t.symbol === product)
    const vol = total.reduce((s, t) => s + (t.quantity || 0), 0)
    stats.push(
      { label: 'Market trades', value: total.length.toLocaleString() },
      { label: 'Market volume', value: vol.toLocaleString() },
    )
  }

  return stats
}

export function buildAllCharts(
  current: LoadResponse,
  product: string,
  priceLayers: PriceLayerToggles = DEFAULT_PRICE_LAYERS,
): {
  price: { data: Data[]; layout: Partial<Layout> }
  depth: { data: Data[]; layout: Partial<Layout> }
  spread: { data: Data[]; layout: Partial<Layout> }
  pnl: { data: Data[]; layout: Partial<Layout> }
  trades: { data: Data[]; layout: Partial<Layout> }
  summary: SummaryItem[]
} {
  const d = current.prices.data[product]
  if (!d) {
    return {
      price: { data: [], layout: DARK_LAYOUT },
      depth: { data: [], layout: DARK_LAYOUT },
      spread: { data: [], layout: DARK_LAYOUT },
      pnl: { data: [], layout: DARK_LAYOUT },
      trades: { data: [], layout: DARK_LAYOUT },
      summary: [],
    }
  }
  return {
    price: price(d, current.has_submission, current.trades, product, priceLayers),
    depth: depth(d),
    spread: spread(d),
    pnl: pnl(d, current.has_submission, current.trades, product),
    trades: tradeDistribution(current.has_submission, current.trades, product),
    summary: buildSummary(current, d, product),
  }
}
