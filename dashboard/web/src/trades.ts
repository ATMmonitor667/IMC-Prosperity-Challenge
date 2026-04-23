import type { TradeRow } from './types'

export function ownTrades(trades: TradeRow[], product: string): TradeRow[] {
  return trades.filter((t) => t.symbol === product && t.buyer === 'SUBMISSION')
}

export function selfSells(trades: TradeRow[], product: string): TradeRow[] {
  return trades.filter((t) => t.symbol === product && t.seller === 'SUBMISSION')
}

export function marketTrades(trades: TradeRow[], product: string): TradeRow[] {
  return trades.filter(
    (t) => t.symbol === product && t.buyer !== 'SUBMISSION' && t.seller !== 'SUBMISSION',
  )
}
