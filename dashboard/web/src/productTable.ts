import { average, std } from './math'
import type { LoadResponse, ProductSeries } from './types'
import { ownTrades, selfSells } from './trades'

export type ProductTableRow = {
  product: string
  ticks: number
  lastMid: string
  midSigma: string
  avgSpread: string
  tradeCount: number
  volume: number
  finalPnl: string | null
  position: string | null
}

function rowForProduct(current: LoadResponse, product: string, d: ProductSeries): ProductTableRow {
  const mids = d.mid.filter((v): v is number => v !== null && Number.isFinite(v))
  const spreads = d.spread.filter((v): v is number => v !== null && Number.isFinite(v))
  const lastMid = mids.length ? mids[mids.length - 1]!.toFixed(2) : '—'
  const midSig = mids.length ? std(mids).toFixed(2) : '—'
  const avgSpread = spreads.length ? average(spreads).toFixed(2) : '—'

  if (current.has_submission) {
    const pnlRow = d.pnl
    const finalPnl = pnlRow.length ? pnlRow[pnlRow.length - 1] ?? 0 : 0
    const own = current.trades.filter(
      (t) => t.symbol === product && (t.buyer === 'SUBMISSION' || t.seller === 'SUBMISSION'),
    )
    const buys = ownTrades(current.trades, product)
    const sells = selfSells(current.trades, product)
    const pos =
      buys.reduce((s, t) => s + t.quantity, 0) - sells.reduce((s, t) => s + t.quantity, 0)
    return {
      product,
      ticks: d.t.length,
      lastMid,
      midSigma: midSig,
      avgSpread,
      tradeCount: own.length,
      volume: own.reduce((s, t) => s + t.quantity, 0),
      finalPnl: Number(finalPnl).toFixed(0),
      position: String(pos),
    }
  }

  const total = current.trades.filter((t) => t.symbol === product)
  const vol = total.reduce((s, t) => s + (t.quantity || 0), 0)
  return {
    product,
    ticks: d.t.length,
    lastMid,
    midSigma: midSig,
    avgSpread,
    tradeCount: total.length,
    volume: vol,
    finalPnl: null,
    position: null,
  }
}

export function buildProductTable(current: LoadResponse | null): ProductTableRow[] {
  if (!current) return []
  return current.prices.products
    .map((p) => {
      const d = current.prices.data[p]
      if (!d) return null
      return rowForProduct(current, p, d)
    })
    .filter((r): r is ProductTableRow => r !== null)
}

export type TableSortKey = keyof ProductTableRow

export function sortTableRows(
  rows: ProductTableRow[],
  key: TableSortKey,
  dir: 'asc' | 'desc',
): ProductTableRow[] {
  const mult = dir === 'asc' ? 1 : -1
  const out = [...rows]
  out.sort((a, b) => {
    const va = a[key]
    const vb = b[key]
    if (key === 'finalPnl' || key === 'position') {
      if (va === null && vb === null) {
        return 0
      }
      if (va === null) {
        return 1
      }
      if (vb === null) {
        return -1
      }
      const na = Number.parseFloat(String(va))
      const nb = Number.parseFloat(String(vb))
      if (na === nb) {
        return 0
      }
      return na < nb ? -mult : mult
    }
    if (typeof va === 'number' && typeof vb === 'number') {
      if (va === vb) {
        return 0
      }
      return va < vb ? -mult : mult
    }
    if (key === 'lastMid' || key === 'midSigma' || key === 'avgSpread') {
      const na = parseFloat(String(va)) || 0
      const nb = parseFloat(String(vb)) || 0
      if (na === nb) {
        return String(va).localeCompare(String(vb)) * mult
      }
      return na < nb ? -mult : mult
    }
    return String(va).localeCompare(String(vb)) * mult
  })
  return out
}
