import type { Layout } from 'plotly.js'

export const PLOTLY_CONFIG = { displaylogo: false, responsive: true }

export const DARK_LAYOUT: Partial<Layout> = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#b8c4d4', size: 11, family: 'Instrument Sans, system-ui, sans-serif' },
  margin: { l: 55, r: 20, t: 8, b: 40 },
  xaxis: { gridcolor: 'rgba(38, 48, 64, 0.7)', zerolinecolor: 'rgba(38, 48, 64, 0.7)' },
  yaxis: { gridcolor: 'rgba(38, 48, 64, 0.7)', zerolinecolor: 'rgba(38, 48, 64, 0.7)' },
  legend: { orientation: 'h', y: -0.22, font: { size: 11 } },
}
