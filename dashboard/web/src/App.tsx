import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { buildAllCharts, DEFAULT_PRICE_LAYERS, type PriceLayerToggles } from './buildCharts'
import { PlotlyChart } from './PlotlyChart'
import { parseXRangeFromRelayout } from './plotlyRelayout'
import { buildProductTable, sortTableRows, type ProductTableRow, type TableSortKey } from './productTable'
import { withTimeXRange } from './timeAxisLayout'
import type { LoadResponse, RoundInfo, SourcesResponse } from './types'

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  const json: unknown = await res.json()
  if (!res.ok) {
    const err = json as { error?: string }
    throw new Error(err.error || res.statusText)
  }
  return json as T
}

function bestProductId(load: LoadResponse): string {
  const products = load.prices.products
  if (products.length === 0) return ''
  if (!load.has_submission) return products[0]!
  let best = products[0]!
  let bestAbs = -1
  for (const p of products) {
    const pnl = load.prices.data[p]?.pnl
    const final = pnl?.length ? pnl[pnl.length - 1] ?? 0 : 0
    if (Math.abs(final) > bestAbs) {
      bestAbs = Math.abs(final)
      best = p
    }
  }
  return best
}

type SourceMode = 'round' | 'backtest'

function SectionTitle({ children, subtitle }: { children: string; subtitle?: string }) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-2">
      <h2 className="m-0 text-xs font-semibold tracking-[0.12em] text-[var(--color-muted)] uppercase">
        {children}
      </h2>
      {subtitle ? <span className="text-[0.7rem] text-white/30">{subtitle}</span> : null}
    </div>
  )
}

function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-white/10 bg-[var(--color-elevated)]/70 shadow-[0_0_0_1px_rgba(255,255,255,0.03)_inset] backdrop-blur-md ${className}`}
    >
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-br from-white/5 to-transparent" />
      <div className="relative">{children}</div>
    </div>
  )
}

function StatusPill({ ok, children }: { ok: boolean | null; children: string }) {
  const cls =
    ok === null ? 'bg-white/10 text-white/50' : ok ? 'bg-emerald-500/15 text-emerald-200' : 'bg-rose-500/20 text-rose-200'
  return (
    <span
      className={`inline-flex max-w-full items-center gap-1.5 truncate rounded-full border border-white/10 px-3 py-1 text-xs ${cls}`}
    >
      <span
        className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
          ok === null ? 'bg-white/40' : ok ? 'bg-emerald-400' : 'bg-rose-400'
        }`}
        aria-hidden
      />
      {children}
    </span>
  )
}

const LAYER_OPTIONS: { key: keyof PriceLayerToggles; label: string; submissionOnly?: boolean }[] = [
  { key: 'mid', label: 'Mid' },
  { key: 'bid', label: 'Bid' },
  { key: 'ask', label: 'Ask' },
  { key: 'ownBuys', label: 'Own buys', submissionOnly: true },
  { key: 'ownSells', label: 'Own sells', submissionOnly: true },
]

const TABLE_COLUMNS: { key: TableSortKey; label: string; align: 'left' | 'right' }[] = [
  { key: 'product', label: 'Product', align: 'left' },
  { key: 'ticks', label: 'Ticks', align: 'right' },
  { key: 'lastMid', label: 'Last mid', align: 'right' },
  { key: 'midSigma', label: 'Mid σ', align: 'right' },
  { key: 'avgSpread', label: 'Avg spr.', align: 'right' },
  { key: 'tradeCount', label: 'Trades', align: 'right' },
  { key: 'volume', label: 'Volume', align: 'right' },
  { key: 'finalPnl', label: 'Final PnL', align: 'right' },
  { key: 'position', label: 'Position', align: 'right' },
]

function toggleLayer(prev: PriceLayerToggles, key: keyof PriceLayerToggles): PriceLayerToggles {
  const next = { ...prev, [key]: !prev[key] }
  if (!next.mid && !next.bid && !next.ask && !next.ownBuys && !next.ownSells) {
    return { ...next, mid: true }
  }
  return next
}

export default function App() {
  const [sources, setSources] = useState<SourcesResponse | null>(null)
  const [sourceMode, setSourceMode] = useState<SourceMode>('round')
  const [roundId, setRoundId] = useState('1')
  const [day, setDay] = useState(0)
  const [backtestPath, setBacktestPath] = useState('')
  const [product, setProduct] = useState('')
  const [current, setCurrent] = useState<LoadResponse | null>(null)
  const [status, setStatus] = useState({ text: 'Starting…', ok: null as boolean | null })
  const [xRange, setXRange] = useState<[number, number] | null>(null)
  const [priceLayers, setPriceLayers] = useState<PriceLayerToggles>(DEFAULT_PRICE_LAYERS)
  const [tableSort, setTableSort] = useState<{ key: TableSortKey; dir: 'asc' | 'desc' }>({
    key: 'product',
    dir: 'asc',
  })

  const rounds: RoundInfo[] = sources?.rounds ?? []
  const selectedRound = useMemo(
    () => rounds.find((r) => r.round === roundId) ?? rounds[0],
    [rounds, roundId],
  )
  const days = selectedRound?.days ?? []

  useEffect(() => {
    void (async () => {
      try {
        setStatus({ text: 'Loading sources…', ok: null })
        const s = await fetchJSON<SourcesResponse>('/api/sources')
        setSources(s)
        if (s.rounds.length) {
          const r0 = s.rounds[0]!
          setRoundId(r0.round)
          setDay(r0.days[0] ?? 0)
        }
        if (s.backtests.length) {
          setBacktestPath(s.backtests[0]!.path)
        }
        setStatus({ text: 'Ready — pick a source and load data.', ok: true })
      } catch (e) {
        setStatus({ text: `Failed: ${(e as Error).message}`, ok: false })
      }
    })()
  }, [])

  useEffect(() => {
    if (days.length && !days.includes(day)) {
      setDay(days[0]!)
    }
  }, [days, day])

  const load = useCallback(async () => {
    try {
      setStatus({ text: 'Loading…', ok: null })
      const url =
        sourceMode === 'round'
          ? `/api/round?round=${encodeURIComponent(roundId)}&day=${encodeURIComponent(String(day))}`
          : `/api/backtest?file=${encodeURIComponent(backtestPath)}`
      const data = await fetchJSON<LoadResponse>(url)
      setCurrent(data)
      setXRange(null)
      setPriceLayers(DEFAULT_PRICE_LAYERS)
      const prods = data.prices.products
      if (prods.length === 0) {
        setProduct('')
        setStatus({ text: 'No products in source.', ok: false })
        return
      }
      setProduct(bestProductId(data))
      const kind = data.has_submission ? 'Backtest' : 'Round data'
      setStatus({
        text: `${kind} — ${prods.length} products, ${data.trades.length} trades`,
        ok: true,
      })
    } catch (e) {
      setStatus({ text: `Load failed: ${(e as Error).message}`, ok: false })
    }
  }, [sourceMode, roundId, day, backtestPath])

  const tableRows = useMemo(() => {
    const raw = buildProductTable(current)
    return sortTableRows(raw, tableSort.key, tableSort.dir)
  }, [current, tableSort])

  const onHeaderClick = (key: TableSortKey) => {
    setTableSort((s) => {
      if (s.key === key) {
        return { key, dir: s.dir === 'asc' ? 'desc' : 'asc' }
      }
      return { key, dir: 'asc' }
    })
  }

  const handlePriceRelayout = useCallback((ev: Readonly<Record<string, unknown>>) => {
    const p = parseXRangeFromRelayout(ev)
    if (p === null) {
      return
    }
    if (p === 'autorange') {
      setXRange(null)
    } else {
      setXRange(p)
    }
  }, [])

  const charts = useMemo(() => {
    if (!current || !product) {
      return null
    }
    return buildAllCharts(current, product, priceLayers)
  }, [current, product, priceLayers])

  const priceLayout = useMemo(
    () => (charts ? withTimeXRange(charts.price.layout, xRange) : {}),
    [charts, xRange],
  )
  const depthLayout = useMemo(
    () => (charts ? withTimeXRange(charts.depth.layout, xRange) : {}),
    [charts, xRange],
  )
  const spreadLayout = useMemo(
    () => (charts ? withTimeXRange(charts.spread.layout, xRange) : {}),
    [charts, xRange],
  )
  const pnlLayout = useMemo(
    () => (charts ? withTimeXRange(charts.pnl.layout, xRange) : {}),
    [charts, xRange],
  )

  return (
    <div className="mx-auto min-h-screen max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-8 flex flex-col gap-3 border-b border-white/10 pb-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="mb-1 text-xs font-medium tracking-widest text-white/40 uppercase">IMC Prosperity</p>
          <h1 className="m-0 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
            Algorithmic trend dashboard
          </h1>
          <p className="mt-1 max-w-xl text-sm text-white/50">
            Inspect round CSVs and backtest logs: prices, book depth, spreads, PnL, and your own fills. Zoom the price
            chart to sync the time window on the panels below; double-click a chart to reset the X axis.
          </p>
        </div>
        <StatusPill ok={status.ok}>{status.text}</StatusPill>
      </header>

      <div className="mb-6 grid gap-4 lg:grid-cols-[1fr_auto]">
        <Panel>
          <div className="flex flex-wrap items-end gap-4 p-4 sm:p-5">
            <div className="flex min-w-[12rem] flex-col gap-2">
              <span className="text-[0.65rem] font-medium tracking-widest text-white/40 uppercase">Source</span>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setSourceMode('round')}
                  className={`rounded-xl border px-3 py-2 text-sm font-medium transition ${
                    sourceMode === 'round'
                      ? 'border-[var(--color-accent)]/40 bg-[var(--color-accent)]/20 text-white'
                      : 'border-white/10 bg-white/5 text-white/60 hover:border-white/20 hover:text-white/90'
                  }`}
                >
                  Round data
                </button>
                <button
                  type="button"
                  onClick={() => setSourceMode('backtest')}
                  className={`rounded-xl border px-3 py-2 text-sm font-medium transition ${
                    sourceMode === 'backtest'
                      ? 'border-[var(--color-accent)]/40 bg-[var(--color-accent)]/20 text-white'
                      : 'border-white/10 bg-white/5 text-white/60 hover:border-white/20 hover:text-white/90'
                  }`}
                >
                  Backtest log
                </button>
              </div>
            </div>

            {sourceMode === 'round' ? (
              <>
                <label className="flex min-w-[6rem] flex-col gap-1.5 text-xs text-white/50">
                  Round
                  <select
                    className="rounded-xl border border-white/10 bg-[#0a0e14] px-3 py-2 text-sm text-white outline-none focus:border-[var(--color-accent)]/50"
                    value={roundId}
                    onChange={(e) => setRoundId(e.target.value)}
                    disabled={!rounds.length}
                  >
                    {rounds.map((r) => (
                      <option key={r.round} value={r.round}>
                        Round {r.round}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex min-w-[6rem] flex-col gap-1.5 text-xs text-white/50">
                  Day
                  <select
                    className="rounded-xl border border-white/10 bg-[#0a0e14] px-3 py-2 text-sm text-white outline-none focus:border-[var(--color-accent)]/50"
                    value={day}
                    onChange={(e) => setDay(Number(e.target.value))}
                    disabled={!days.length}
                  >
                    {days.map((d) => (
                      <option key={d} value={d}>
                        Day {d}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            ) : (
              <label className="flex min-w-[16rem] max-w-md flex-1 flex-col gap-1.5 text-xs text-white/50">
                Backtest file
                <select
                  className="max-w-full rounded-xl border border-white/10 bg-[#0a0e14] px-3 py-2 text-sm text-white outline-none focus:border-[var(--color-accent)]/50"
                  value={backtestPath}
                  onChange={(e) => setBacktestPath(e.target.value)}
                  disabled={!sources?.backtests.length}
                >
                  {(sources?.backtests ?? []).map((b) => (
                    <option key={b.path} value={b.path}>
                      {b.folder}/{b.name} ({b.size_kb} KB)
                    </option>
                  ))}
                </select>
              </label>
            )}

            <label className="flex min-w-[9rem] flex-col gap-1.5 text-xs text-white/50">
              Product
              <select
                className="rounded-xl border border-white/10 bg-[#0a0e14] px-3 py-2 text-sm text-white outline-none focus:border-[var(--color-accent)]/50"
                value={product}
                onChange={(e) => setProduct(e.target.value)}
                disabled={!current}
              >
                {(current?.prices.products ?? []).map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>

            <div className="ms-auto">
              <button
                type="button"
                onClick={() => void load()}
                className="rounded-xl bg-gradient-to-b from-[#5eb0ff] to-[#2d8ae6] px-5 py-2.5 text-sm font-semibold text-[#0a1628] shadow-lg shadow-cyan-500/10 transition hover:brightness-110"
              >
                Load
              </button>
            </div>
          </div>
        </Panel>
      </div>

      {tableRows.length > 0 ? (
        <Panel className="mb-5 overflow-x-auto">
          <div className="p-3 sm:p-4">
            <SectionTitle subtitle="Click a row to select product · click header to sort">
              All products
            </SectionTitle>
            <table className="w-full min-w-[720px] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-white/10 text-white/50">
                  {TABLE_COLUMNS.map((c) => (
                    <th
                      key={c.key}
                      className={`whitespace-nowrap py-2 pr-3 text-[0.65rem] font-medium tracking-wider uppercase ${
                        c.align === 'right' ? 'text-right' : 'text-left'
                      } ${c.key === 'finalPnl' || c.key === 'position' ? 'max-sm:hidden' : ''}`}
                    >
                      <button
                        type="button"
                        className={`inline-flex w-full items-center gap-1 text-white/50 hover:text-white/90 ${
                          c.align === 'right' ? 'justify-end' : 'justify-start'
                        }`}
                        onClick={() => onHeaderClick(c.key)}
                      >
                        {c.label}
                        {tableSort.key === c.key ? (tableSort.dir === 'asc' ? ' \u00b7↑' : ' \u00b7↓') : ''}
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row) => (
                  <tr
                    key={row.product}
                    onClick={() => setProduct(row.product)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        setProduct(row.product)
                      }
                    }}
                    className={`cursor-pointer border-b border-white/5 font-mono text-[0.8rem] tabular-nums transition hover:bg-white/[0.06] ${
                      row.product === product ? 'bg-[var(--color-accent)]/10' : ''
                    }`}
                    role="button"
                    tabIndex={0}
                    aria-current={row.product === product}
                  >
                    {TABLE_COLUMNS.map((c) => (
                      <td
                        key={c.key}
                        className={`py-2 pr-3 ${c.align === 'right' ? 'text-right' : ''} ${
                          c.key === 'finalPnl' || c.key === 'position' ? 'max-sm:hidden' : ''
                        } text-white/90 ${c.key === 'product' ? 'font-sans' : ''}`}
                      >
                        {formatTableCell(row, c.key)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      ) : null}

      {charts && charts.summary.length > 0 ? (
        <div className="mb-6 grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8">
          {charts.summary.map((s) => (
            <div
              key={s.label}
              className="rounded-xl border border-white/8 bg-white/[0.04] px-3 py-2.5 backdrop-blur-sm"
            >
              <div className="text-[0.65rem] font-medium tracking-wider text-white/40 uppercase">{s.label}</div>
              <div className="mt-0.5 font-mono text-base font-semibold text-white tabular-nums">{s.value}</div>
            </div>
          ))}
        </div>
      ) : null}

      {charts && product ? (
        <div className="flex flex-col gap-5">
          <Panel>
            <div className="p-3 sm:p-4">
              <div className="mb-2 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <SectionTitle>Price and your trades</SectionTitle>
                  <p className="m-0 text-[0.7rem] text-white/40">Drag to zoom, box-select, or use mode bar · layers:</p>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  {LAYER_OPTIONS.map((opt) => {
                    const dis = opt.submissionOnly && !current?.has_submission
                    const on = priceLayers[opt.key]
                    return (
                      <label
                        key={opt.key}
                        className={`inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-[#0a0e14] px-2.5 py-1 text-xs ${
                          dis ? 'cursor-not-allowed opacity-40' : 'cursor-pointer hover:border-white/20'
                        }`}
                      >
                        <input
                          type="checkbox"
                          className="accent-sky-400"
                          checked={on}
                          disabled={!!dis}
                          onChange={() => setPriceLayers((p) => toggleLayer(p, opt.key))}
                        />
                        {opt.label}
                      </label>
                    )
                  })}
                  <button
                    type="button"
                    className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1 text-xs text-white/80 hover:border-white/30"
                    onClick={() => setXRange(null)}
                    title="Reset time zoom on all linked charts (same as double-click autoscale on X)"
                  >
                    Reset time window
                  </button>
                </div>
              </div>
              <PlotlyChart
                className="h-[min(40vh,420px)] w-full"
                data={charts.price.data}
                layout={priceLayout}
                onRelayout={handlePriceRelayout}
              />
            </div>
          </Panel>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel>
              <div className="p-3 sm:p-4">
                <SectionTitle subtitle="Time axis linked from price">Order-book depth</SectionTitle>
                <PlotlyChart
                  className="h-[min(32vh,320px)] w-full"
                  data={charts.depth.data}
                  layout={depthLayout}
                />
              </div>
            </Panel>
            <Panel>
              <div className="p-3 sm:p-4">
                <SectionTitle subtitle="Time axis linked from price">Bid–ask spread</SectionTitle>
                <PlotlyChart
                  className="h-[min(32vh,320px)] w-full"
                  data={charts.spread.data}
                  layout={spreadLayout}
                />
              </div>
            </Panel>
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <Panel>
              <div className="p-3 sm:p-4">
                <SectionTitle subtitle="Time axis linked from price">PnL and position</SectionTitle>
                <PlotlyChart className="h-[min(32vh,320px)] w-full" data={charts.pnl.data} layout={pnlLayout} />
              </div>
            </Panel>
            <Panel>
              <div className="p-3 sm:p-4">
                <SectionTitle>Trade size distribution (full sample)</SectionTitle>
                <PlotlyChart
                  className="h-[min(32vh,320px)] w-full"
                  data={charts.trades.data}
                  layout={charts.trades.layout}
                />
              </div>
            </Panel>
          </div>
        </div>
      ) : (
        <Panel className="p-8 text-center">
          <p className="m-0 text-sm text-white/50">
            Load a round or a backtest log to render charts. Run the API server:{' '}
            <code className="rounded bg-white/10 px-1.5 py-0.5 text-white/80">python dashboard/server.py</code>
          </p>
        </Panel>
      )}
    </div>
  )
}

function formatTableCell(row: ProductTableRow, key: TableSortKey): string {
  if (key === 'finalPnl' || key === 'position') {
    if (row[key] === null) {
      return '—'
    }
  }
  if (key === 'volume' || key === 'tradeCount' || key === 'ticks') {
    return (row[key] as number).toLocaleString()
  }
  if (key === 'product') {
    return row.product
  }
  return String(row[key])
}
