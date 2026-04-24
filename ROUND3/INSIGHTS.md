# Round 3 — data structure, connections, and bot-relevant takeaways

This note ties together what the **prosperity price CSVs** show, how to **use the local dashboard** for intuition, and what to exploit in a **Round 3 trader**.

## What the files are

- **`prices_round_3_day_*.csv`**: 12 products × 10,000 ticks per day (`timestamp` 0…999900, step 100), `;`-separated. Columns match earlier rounds: book levels + `mid_price` + `profit_and_loss` (0 in historical data).
- **`trades_round_3_day_*.csv`**: Market prints (no buyer/seller in the sample; treat as *tape* for regime / liquidity).
- **`La_trahison_des_images.png`**: Magritte’s “this is not a pipe” (image ≠ object). In-game hint: **the `VEV_####` label is a strike *index* on a ladder, not a cash price in the same sense as the two “spot-like” products.** The ladder still co-moves as a **surface**.

## Product roles (abstraction for the bot)

| Group | Products | Role |
|--------|-----------|------|
| **Cash / “spot”** | `VELVETFRUIT_EXTRACT`, `HYDROGEL_PACK` | ~5k and ~10k level; **wide** top-of-book spreads (often ~5 and ~16 ticks in the data). |
| **VEV_4000 … VEV_6500** | 10 names | A **1-D strike/vol index**: numbers are **ladder keys** (4000→6500). |
| **Wings** | `VEV_6000`, `VEV_6500` | Mid ~0–1, **1-tick** spread, **~zero** percentage volatility (nearly static mids in the sample). |
| **Belly / wing** | `VEV_5200`–`VEV_5500` | Low dollar mids → **huge** % volatility (artefact of scale + noise), **tight** absolute spreads. |

Use this to **separate models**: do not one-shot “mean reversion on % returns” across all `VEV_*` with the same rule — scale and spread differ by an order of magnitude.

## Empirical relationships (day 0 sample; re-run `analyze_r3_data.py` on other days)

1. **VEV block moves together**  
   Neighboring and nearby strikes (e.g. 4000–5200) show **0.7–0.9** same-period **return** correlation. Treat as a **low-dimensional surface** (smooth across strikes) for **consistency / residual** signals: e.g. deviation of one strike from a spline or from neighbors = possible mean reversion *after* you normalize for level.

2. **VELVET links to mid-VEV, not to HYDROGEL**  
   `VELVETFRUIT_EXTRACT` returns correlate most with **`VEV_5000` / `VEV_5100`** (~0.75) in the same bar — consistent with a **“vol/vol index” tied to a spot-like name**. **HYDROGEL** returns are almost **uncorrelated** with the VEV block (~0.0) → **second, independent book**; price and risk **separate** from the vol ladder.

3. **No obvious lead–lag in one tick (100s)**  
   VELVET vs VEV_5000: cross-correlation at lags ±1…±3 is **~0** on day 0. So **no simple “look at VELVET, one tick later trade VEV”** at this resolution; any edge is more likely **co-movement, surface shape, or execution**, not 100-tick latency arbitrage in this dump.

4. **Spreads as a constraint**  
   `VEV_4000` and **HYDROGEL** have **~16–21** tick top-of-book spreads in the sample; `VEV_6000/6500` are **1 tick**. A bot must **size and quote** with **per-product edge ≥ half-spread+fees** to trade at all on wide books.

## How to use the dashboard

1. `python dashboard/server.py` (from repo root) after a `dashboard/web` build if the UI is stale.  
2. Select **Round 3**, day **0 / 1 / 2**, **Load**.  
3. **All-products table** → sort by **Avg spr.** to see which names are expensive to hit.  
4. **Product** = pick `VELVETFRUIT_EXTRACT` vs a `VEV_5000` and toggle **mid / bid / ask** to see co-movement; use **time zoom** on the price panel and link depth + spread.  
5. **Backtest logs**: put future `*.log` under `ROUND3/backtests/` (create folder) so the dashboard can list them — none shipped in this repo yet.

## Bot design hooks (concrete)

1. **Two-factor state**  
   - Factor A: `VELVET` (or log-mid) **level / return**.  
   - Factor B: **HYDROGEL** (orthogonal to VEV in corr).  
   - Factor C: **VEV surface** as vector along strikes; compress with **PCA / spline** or use **neighbors only** for residuals.

2. **Surface / calendar consistency**  
   If the game exposes multiple days, compare **day 0 vs 1 vs 2** shape and mean level — store running stats in the trader; use dashboard per day to eyeball **regime change**.

3. **Market making**  
   Prioritize **tight-spread** names for quotes (`VEV_6000/6500`, `VEV_5400/5500` in sample) unless limits force you to **HYDROGEL/VELVET**; then **widen** and reduce size to match the **observed spread** distribution.

4. **Risk**  
   **Do not** interpret huge % vol on VEV_5500 the same as VEV_5000 — **normalize** by mid level or work in **absolute** price and tick size.

5. **Execution**  
   The Magritte hint + structure suggest **don’t overfit the literal 4500/5000 numbers**; fit **proportions of the book and the neighbor structure** under whatever hidden fair-value process IMC actually uses.

## Reproduce numbers

```bash
python ROUND3/analyze_r3_data.py
```

## Trader

- `trader.py` — `Trader` class for the competition API; per-product edges tuned from the CSVs (see module docstring).  
- `datamodel.py` — same `TradingState` / `Order` / `OrderDepth` types as other rounds.  
- `run_bt.py` — injects all 12 `LIMITS` then runs the **Prosperity 4** backtester (`prosperity4bt`); IMC **game** round-3 data still live under `round3/`. **Verify position caps in the official brief** and update `_LIMITS_R3` / `trader._LIMITS` if they differ.

## Files in this folder

- `prices_round_3_day_*.csv`, `trades_round_3_day_*.csv` — historical data.  
- `trader.py`, `datamodel.py`, `run_bt.py` — submission-style bot and backtest wrapper.  
- `INSIGHTS.md` — this file.
