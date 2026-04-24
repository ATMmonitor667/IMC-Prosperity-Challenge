"""Quick Round 3 diagnostics: product list, return correlations, spreads, lead-lag.

  python analyze_r3_data.py [day]
  day defaults to 0
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    day = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    path = ROOT / f"prices_round_3_day_{day}.csv"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    df = pd.read_csv(path, sep=";")
    print(f"File: {path.name}  rows={len(df):,}")
    prods = sorted(df["product"].unique())
    print("Products:", prods)
    print("Timestamps:", int(df["timestamp"].min()), "…", int(df["timestamp"].max()))

    w = df.pivot_table(index="timestamp", columns="product", values="mid_price", aggfunc="first")
    r = w.pct_change().replace([np.inf, -np.inf], np.nan)

    print("\n--- Spread (ask1 - bid1) mean by product ---")
    df = df.copy()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    for pr in prods:
        s = df.loc[df["product"] == pr, "spread"]
        print(f"  {pr:24s}  mean {float(s.mean()):.2f}  median {float(s.median()):.0f}")

    print("\n--- Return vol (1e4 * std of pct_change), rough 'bp' scale ---")
    for c in w.columns:
        v = float(r[c].std())
        if np.isnan(v):
            v = 0.0
        print(f"  {c:24s}  {1e4 * v:.2f}")

    c = r.corr()
    pairs: list[tuple[float, str, str]] = []
    for a, b in itertools.combinations(c.columns, 2):
        v = c.loc[a, b]
        if np.isfinite(v):
            pairs.append((float(v), a, b))
    pairs.sort(reverse=True)

    print("\n--- Highest return correlations ---")
    for v, a, b in pairs[:10]:
        print(f"  {v:6.3f}  {a:24s}  {b}")
    print("\n--- Lowest return correlations ---")
    for v, a, b in pairs[-6:]:
        print(f"  {v:6.3f}  {a:24s}  {b}")

    if "VELVETFRUIT_EXTRACT" in w.columns and "VEV_5000" in w.columns:
        print("\n--- VELVET vs VEV_5000 return cross-corr (lags in rows = ticks) ---")
        a, b = r["VELVETFRUIT_EXTRACT"], r["VEV_5000"]
        for lag in range(-3, 4):
            if lag == 0:
                print(f"  lag {lag:2d}  (same) {a.corr(b):.4f}")
                continue
            if lag < 0:
                cc = a.corr(b.shift(-lag))
            else:
                cc = a.shift(lag).corr(b)
            print(f"  lag {lag:2d}  {cc if np.isfinite(cc) else float('nan'):.4f}")


if __name__ == "__main__":
    main()
