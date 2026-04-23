import { useLayoutEffect, useRef } from 'react'
import type { Config, Data, Layout } from 'plotly.js'
import { PLOTLY_CONFIG } from './chartConfig'

type PlotlyChartProps = {
  data: Data[]
  layout: Partial<Layout>
  className?: string
  config?: Partial<Config>
}

export function PlotlyChart({ data, layout, className, config }: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement>(null)

  useLayoutEffect(() => {
    const el = ref.current
    const Plotly = window.Plotly
    if (!el || !Plotly) return
    const merged: Partial<Config> = { ...PLOTLY_CONFIG, ...config }
    void Plotly.react(el, data, layout, merged)
    return () => {
      void Plotly.purge(el)
    }
  }, [data, layout, config])

  return <div className={className} ref={ref} />
}
