"""One-off: embed explanation markdown cells into round_4.ipynb (original 55 cells)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path


def md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text = text + "\n"
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "round_4.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    n0 = len(nb["cells"])
    if n0 != 55:
        print(f"Warning: expected 55 cells, found {n0} — indices may be off.")

    # insert[idx] = markdown inserted BEFORE original cell idx (0-based, original layout).
    inserts: dict[int, str] = {}

    inserts[1] = r"""**Notebook overview** — this file loads **Round 4 price** CSVs (semicolon-separated), builds a **stitched, full-product** time panel, runs **mean-reversion / VEV / beta** analysis, then loads **trades** and plots **who traded when** and **notional cashflows by trader ID**.

The cell below **imports** `pandas` / `numpy` / `matplotlib` for tables, numerics, and line plots.
"""

    inserts[2] = r"""**Load price files** — reads `prices_round_4_day_1/2/3.csv` from the current working directory (typically this folder). Each file is a long table: one row per `(timestamp, product)` with top-of-book columns and a `mid_price`.

'''
Data: adjust paths if the notebook is run from a different cwd. The separator is `;` as in many IMC-style exports. Large files — expect slow first read.
'''
"""

    inserts[3] = r"""**Preview (`head`)** — quick sanity check of column names: `day`, `timestamp`, `product`, `bid_price_*` / `ask_price_*` / volumes, `mid_price`, and `profit_and_loss` (often zero in raw snapshots used for EDA).
"""

    inserts[4] = r"""**Align timestamps across days** — each day uses `timestamp` 0, 100, …, 999900. Day 2 is shifted by `+1_000_000`, day 3 by `+2_000_000` so that after concatenation every row has a **unique** time key.

'''
Data analysis: the shift is a bookkeeping device so pivots and joins do not superimpose different sessions on the same index. It is not wall-clock time.
'''
"""

    inserts[5] = r"""**Stack and pivot** — `concat` all days, `pivot` to wide format (columns = product symbols, index = time). Then **filter** to rows where **all** of the named products have **strictly positive** mids.

'''
Data analysis: the “all products positive” filter defines a **common support** for cross-sectional stats. It drops ticks where any name is missing or at zero (e.g. illiquid or wing at floor), so correlations and betas are comparable — at the cost of a shorter time series.
'''
"""

    inserts[6] = r"""**Price plot: HYDROGEL_PACK** — `mid` over the full **offset** time index. Use it to see level, drifts, and whether day boundaries look like level shifts or continuous paths.
"""

    inserts[7] = r"""**Price plot: VELVETFRUIT_EXTRACT** — the “spot-like” name; compare volatility and any trend to HYDRO and to the **VEV_*** block later. This series drives synthetic **option intrinsic** in the VEV section.
"""

    inserts[9] = r"""**Code cell: single-name mean reversion (HYDRO + VELVET)** — runs *after* the **Single-Name Mean Reversion** section header. It builds `single_name_df` from the combined pivot, then for each name computes summary stats and a threshold study.

'''
Data analysis (concrete definitions):
- **ADF (`adfuller`) p-value** — from Augmented Dickey–Fuller on **levels**; very small p → reject unit root, consistent with **stationary** or at least *mean-reverting-like* data. On noisy mids, still interpret with care.
- **AR(1) / half-life** — OLS of `x_t` on `x_{t-1}`; if slope `β` is in (0,1), half-life in ticks = `-log(2)/log(β)` = typical time for an AR(1) shock to halve. If `β ≥ 1` or `β ≤ 0`, half-life is not finite in that toy model.
- **Rolling z-score** (window 1000): `z = (P - μ_roll) / σ_roll` for entry signals when |z| is “large.”
- **`entry_threshold_summary`** — for entry thresholds 1.5, 2, 2.5 σ: detects *crossings* of |z| above threshold; for each crossing looks **forward one half-life (rounded)** and records **signed** reversion in the direction a contrarian would want (`-sign(z) * (P_{t+h}-P_t)`), the fraction of times that move is **positive** (“hit rate”), and whether |z| returns inside a **0.5** band within that horizon. Also reports long-run **entry rate** of such crossings.
- **Figures** — z-score over time with ±2 and ±0.5 reference bands.

*Note: the prose in the section header mentions **Johansen**; this code cell does **not** run a Johansen test — it is **univariate** only.*
'''
"""

    inserts[11] = r"""**Code cell: VEV cross-section (after the VEV Strike Ladder section header)** — sorts `VEV_*` by strike, aligns with `VELVETFRUIT_EXTRACT`, and compares option mids to a **synthetic intrinsic** `max(velvet - K, 0)`.

'''
Data analysis:
- **intrinsic (per row)** — treats `VELVETFRUIT_EXTRACT` as the underlying **S**; each `VEV_*` column’s intrinsic is `max(S-K,0)` in **price** units, like a call payoff lower bound in the *same* numeraire as the option mid.
- **mean_time_value** — mean( mid − intrinsic ): average **excess** of the market mid over intrinsic — “time value / smile / microstructure” in one number per strike, under this crude decomposition.
- **corr_with_velvet / corr_with_intrinsic** — linear correlation of **option mid** with **spot** and with **synthetic intrinsic path**; shows whether the option tracks the spot or the *smoothed* intrinsic more closely.
- **diff_beta_to_velvet** — `cov(Δoption, Δvelvet) / var(Δvelvet)` = **slope of Δoption on Δvelvet** (empirical “delta in tick space” if you treat velvet diff as the factor).
- **pct_at_global_floor** — share of time the mid equals the **global minimum** over all VEV columns (e.g. wings parked at 0.5 in many samples).
- **monotonic_violations** — counts violations of **call monotonicity in strike** (higher strike should not have a **higher** price than a lower strike in a frictionless one-underlying world); positive counts point to microstructure, discreteness, or data quirks.
- **Plots** — (1) mean option vs mean intrinsic, (2) mean time value by strike, (3) diff beta by strike.
'''
"""

    inserts[13] = r"""**Code cell: VEV–VELVET betas (after the VEV beta section header)** — for each `VEV_*` column, aligns with velvet and reports:

- **price_change_beta** — `cov(ΔV, Δvelvet) / var(Δvelvet)` (same as before but in a single dedicated table).
- **return_beta** — the same with **percentage changes** (scale-free in price level).
- **change_correlation** — `corr(ΔV, Δvelvet)`.

'''
Data analysis: return-based beta is more comparable when tick sizes differ; price-change beta matches “dollar” intuition on this ladder. The plot overlays **price_change_beta** and **return_beta** by strike to see how the smile of sensitivity looks.
'''
"""

    inserts[14] = r"""**Trades section** — the next cells read **tape** files, preview columns, and define a **scatter timeline** of prints by **buyer** / **seller** id. This is for **flow** and **participation**, not the same as mark-to-market PnL of a strategy.
"""

    inserts[15] = r"""**Load `trades_round_4_day_*.csv`** — same `;` separator. Each row is a fill: who bought, who sold, `symbol` (product), `price`, `quantity`, and `timestamp`.
"""

    inserts[16] = r"""**Preview (day 3)** — `trades_three.head()` shows a sample of the schema; you can mirror with `trades_one` / `trades_two` to compare activity across days.
"""

    inserts[17] = r"""**Helper: `plot_product_activity(df, product_symbol)`** — long-form preparation for seaborn: duplicate each trade on the **buy** side and on the **sell** side, then `relplot` with time on x, price on y, **color = counterparty**, and **separate columns** for “Bought” vs “Sold.”

'''
Data analysis: the plot is **description**, not a structural model — it highlights **who** is on each side at which prices over time, whether prints cluster, and if certain IDs are persistently one-sided. It does *not* show inventory or PnL without a full position mark.
'''
"""

    inserts[18] = r"""**Day 1 trade timelines** — the following cells call `plot_product_activity(trades_one, ...)` for **HYDROGEL_PACK**, **VELVETFRUIT_EXTRACT**, and every **VEV_*** in order through **VEV_6500**. Use them to eyeball which names had the most prints and whether prices crossed wide spreads in the tape.
"""

    inserts[30] = r"""**Day 2 trade timelines** — same per-product `plot_product_activity` calls using **`trades_two`**. Pattern as day 1: HYDRO, VELVET, then the full VEV strip.
"""

    inserts[42] = r"""**Day 3 trade timelines (partial grid)** — the notebook only plots a **subset** of symbols on `trades_three` (e.g. **VEV_4500** through **VEV_6500** in the file as it stands). If you need **HYDROGEL / VELVET / VEV_4000** for day 3, add the corresponding `plot_product_activity(trades_three, '...')` cells here.
"""

    inserts[51] = r"""**Trader notional (day 1)** — for every fill, `total_value = price * quantity`, then **sum** buy-side cash into `Total Bought` by `buyer` and sell-side out of `Total Sold` by `seller`, `outer` merge, fill 0, and `Net = Sold − Bought` per `Trader` string.

'''
Data analysis: this is a **gross** cashflow summary in **notional** (labels say XIRECS) — *not* economic PnL unless you net positions, fees, and MTM. Use it to see **who** traded the most **value** and whether a trader is a net taker of liquidity in aggregate on day 1.
'''
"""

    inserts[52] = r"""**Trader notional (day 2)** — the **same** aggregation pattern as the previous cell applied to `trades_two` for a **second-session** view of the same `Trader` ids.
"""

    inserts[53] = r"""**Trader notional (day 3)** — same for `trades_three` (if some traders never appear, they may be missing or only all-zero after merge; `fillna(0)` keeps the table rectangular).
"""

    for idx in sorted(inserts.keys(), reverse=True):
        nb["cells"].insert(idx, md_cell(inserts[idx]))

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {path}; inserted {len(inserts)} markdown cells (was {n0}, now {len(nb['cells'])}).")


if __name__ == "__main__":
    main()
