import type { PlotlyStatic } from 'plotly.js'

declare global {
  interface Window {
    Plotly: PlotlyStatic
  }
}

export {}
