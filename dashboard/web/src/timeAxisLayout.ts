import type { Layout } from 'plotly.js'

const TIME_UIREVISION = 'time-sync-v1'

/**
 * Share one X window across time-based charts. `xRange` null = full (auto) range.
 * Histograms use a different x-axis and should not call this.
 */
export function withTimeXRange(
  layout: Partial<Layout>,
  xRange: [number, number] | null,
): Partial<Layout> {
  if (xRange === null) {
    return {
      ...layout,
      uirevision: TIME_UIREVISION,
      xaxis: {
        ...layout.xaxis,
        autorange: true,
        range: undefined,
      },
    }
  }
  return {
    ...layout,
    uirevision: TIME_UIREVISION,
    xaxis: {
      ...layout.xaxis,
      autorange: false,
      range: [xRange[0], xRange[1]],
    },
  }
}
