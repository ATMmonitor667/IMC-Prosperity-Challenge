"""Summarize IMC Prosperity 4 Round 2 historical price CSVs (prices_round_2_day_*.csv).

Run from repo root:  python ROUND2/analyze_round2_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    base = Path(__file__).resolve().parent
    files = sorted(base.glob("prices_round_2_day_*.csv"))
    if not files:
        print("No prices_round_2_day_*.csv in", base)
        return

    rows = []
    for f in files:
        df = pd.read_csv(f, sep=";")
        df["source"] = f.name
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)

    print("Files:", [f.name for f in files])
    print("Total rows:", len(all_df))
    print()

    for p in sorted(all_df["product"].unique()):
        sub = all_df[all_df["product"] == p]
        mid = sub["mid_price"].replace(0, np.nan).dropna()
        spr = (sub["ask_price_1"] - sub["bid_price_1"]).replace([np.inf, -np.inf], np.nan).dropna()
        print(f"=== {p} ===")
        print(f"  mid: mean={mid.mean():.2f}  std={mid.std():.2f}  min/max={mid.min():.0f}/{mid.max():.0f}")
        print(f"  spread (ask1-bid1): mean={spr.mean():.2f}  median={spr.median():.2f}")
        print()

    wide = all_df.pivot_table(
        index=["day", "timestamp"],
        columns="product",
        values="mid_price",
        aggfunc="first",
    )
    wide = wide.dropna(how="any")
    osm, pep = "ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"
    if osm in wide.columns and pep in wide.columns:
        mask = (wide[osm] > 0) & (wide[pep] > 0)
        wide = wide[mask]
        print("Paired observations (both mids > 0):", len(wide))
        print("Correlation (Osmium mid, Pepper mid):", wide[osm].corr(wide[pep]))
        print("By day:")
        for d in sorted(wide.index.get_level_values(0).unique()):
            w = wide.loc[d]
            print(f"  day {d}: n={len(w)}  corr={w[osm].corr(w[pep]):.4f}")

        x = wide[osm].values.astype(float)
        y = wide[pep].values.astype(float)
        b = np.cov(x, y, bias=True)[0, 1] / np.var(x)
        a = float(y.mean() - b * x.mean())
        resid = y - (a + b * x)
        print()
        print("=== OLS (pepper ~ a + b * osmium) ===")
        print(f"  b (hedge slope): {b:.6f}")
        print(f"  a (intercept):   {a:.2f}")
        print(f"  resid std:       {resid.std():.2f}")
        phi = np.dot(resid[:-1], resid[1:]) / np.dot(resid[:-1], resid[:-1])
        if 0 < phi < 1:
            hl = -np.log(2) / np.log(phi)
            print(f"  AR(1) on resid:  phi={phi:.4f}  half_life_ticks~={hl:.0f}")
        try:
            from statsmodels.tsa.stattools import adfuller

            adf_stat, adf_p, *_ = adfuller(resid, maxlag=20, regression="c", autolag="AIC")
            print(f"  ADF on resid:    stat={adf_stat:.4f}  p-value={adf_p:.4f}")
            print("  (Low p => stationary residuals / cointegration evidence.)")
        except ImportError:
            print("  (Install statsmodels for ADF on residuals.)")

        win = min(2000, len(x) // 3)
        if win >= 100:
            rolls = []
            for i in range(win, len(x), max(1, win // 4)):
                sl = slice(i - win, i)
                rolls.append(np.corrcoef(x[sl], y[sl])[0, 1])
            print()
            print(f"=== Rolling corr (window={win}) ===")
            print(f"  min={min(rolls):.4f}  max={max(rolls):.4f}  mean={float(np.mean(rolls)):.4f}")


if __name__ == "__main__":
    main()
