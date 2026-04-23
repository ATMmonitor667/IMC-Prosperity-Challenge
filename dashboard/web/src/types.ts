export type ProductSeries = {
  t: number[]
  mid: (number | null)[]
  bid1: (number | null)[]
  ask1: (number | null)[]
  bid2: (number | null)[]
  ask2: (number | null)[]
  bid3: (number | null)[]
  ask3: (number | null)[]
  spread: (number | null)[]
  bid_depth: number[]
  ask_depth: number[]
  pnl: (number | null)[]
}

export type TradeRow = {
  symbol: string
  buyer: string
  seller: string
  timestamp: number
  price: number
  quantity: number
}

export type RoundInfo = { round: string; days: number[] }

export type BacktestFile = { path: string; name: string; folder: string; size_kb: number }

export type SourcesResponse = { rounds: RoundInfo[]; backtests: BacktestFile[] }

export type LoadResponse = {
  prices: { products: string[]; data: Record<string, ProductSeries> }
  trades: TradeRow[]
  has_submission: boolean
  sandbox_count?: number
}
