import { useEffect, useLayoutEffect, useRef } from 'react'
import type { Config, Data, Layout, PlotlyHTMLElement, PlotRelayoutEvent } from 'plotly.js'
import { PLOTLY_CONFIG } from './chartConfig'

type PlotlyChartProps = {
  data: Data[]
  layout: Partial<Layout>
  className?: string
  config?: Partial<Config>
  /** Subscribes to time-axis relayout (zoom/pan) — use on the main price chart to drive other panels. */
  onRelayout?: (ev: Readonly<Record<string, unknown>>) => void
}

export function PlotlyChart({ data, layout, className, config, onRelayout }: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const el = ref.current
    const Plotly = window.Plotly
    if (!el || !Plotly) {
      return
    }
    const pEl = el as unknown as PlotlyHTMLElement
    const merged: Partial<Config> = { ...PLOTLY_CONFIG, ...config }
    void Plotly.react(pEl, data, layout, merged)

    if (onRelayout) {
      const handler = (ev: PlotRelayoutEvent) => onRelayout(ev as unknown as Readonly<Record<string, unknown>>)
      pEl.on('plotly_relayout', handler)
    }

    return () => {
      pEl.removeAllListeners('plotly_relayout')
    }
  }, [data, layout, config, onRelayout])

  useEffect(
    () => () => {
      const el = ref.current
      if (el && window.Plotly) {
        void window.Plotly.purge(el)
      }
    },
    [],
  )

  return <div className={className} ref={ref} />
}
