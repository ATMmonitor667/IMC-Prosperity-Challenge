export function std(a: number[]): number {
  if (a.length < 2) return 0
  const mean = a.reduce((x, y) => x + y, 0) / a.length
  const v = a.reduce((s, x) => s + (x - mean) ** 2, 0) / (a.length - 1)
  return Math.sqrt(v)
}

export function average(a: number[]): number {
  if (a.length === 0) return Number.NaN
  return a.reduce((x, y) => x + y, 0) / a.length
}
