/** Parse X zoom from `plotly_relayout` on a cartesian time-series chart. */
export function parseXRangeFromRelayout(ev: Readonly<Record<string, unknown>>): 'autorange' | [number, number] | null {
  if (ev['xaxis.autorange'] === true) {
    return 'autorange'
  }
  const a = ev['xaxis.range[0]']
  const b = ev['xaxis.range[1]']
  if (a === undefined && b === undefined) {
    return null
  }
  const n0 = typeof a === 'number' ? a : a !== undefined ? Number.parseFloat(String(a)) : Number.NaN
  const n1 = typeof b === 'number' ? b : b !== undefined ? Number.parseFloat(String(b)) : Number.NaN
  if (Number.isFinite(n0) && Number.isFinite(n1)) {
    return [n0, n1]
  }
  return null
}
