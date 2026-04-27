# Round 4 PnL Maximization Algorithm

## Summary
Build a hybrid market-making, statistical fair-value, and option-ladder strategy. Use HYDROGEL as an independent mean-reverting market-making product, treat VELVET as the underlying, and price all `VEV_*` products as call-like vouchers on VELVET with strict inventory and delta control.

Use the repo’s actual limits: `HYDROGEL_PACK=32`, `VELVETFRUIT_EXTRACT=32`, each `VEV_*=20`. The PDF’s conversion discussion is useful conceptually, but the local Round 4 CSV/backtester surface is mainly local order book plus VEV ladder behavior.

## Core Signals
- `HYDROGEL_PACK`: independent slow mean reversion. Mean price about `9994.65`, std `34.62`, half-life about `350` ticks, spread usually `16`.
- `VELVETFRUIT_EXTRACT`: underlying for vouchers. Mean about `5247.65`, std `18.08`, half-life about `352` ticks, spread usually `5`.
- `VEV_4000` and `VEV_4500`: near-pure intrinsic value. Use `VEV_4000 + 4000` and `VEV_4500 + 4500` as strong VELVET anchors.
- `VEV_5000` to `VEV_5500`: option time-value strip. Seed time values approximately: `5000:2.75`, `5100:11.5`, `5200:39`, `5300:34`, `5400:9`, `5500:2.7`.
- `VEV_6000` and `VEV_6500`: dead floor products. Fair value `0.5`; only quote `bid=0` to buy from sellers and `ask=1` to unwind longs.

## Implementation Changes
- Maintain per-product memory: `ema_mid`, `slow_mean`, `slow_var`, `vol`, `prev_mid`, `flow_signal`, `tv_ema`, `residual_vol`.
- Estimate underlying fair `S_hat` every tick from a clipped weighted basket: VELVET mid weight `4`, `VEV_4000+4000` weight `2`, `VEV_4500+4500` weight `2`, `VEV_5000+5000-tv_5000` weight `1`.
- For VEV fair values: `fair_K = max(S_hat - K, 0) + tv_ema[K]`, except `6000/6500 = 0.5`. Update `tv_ema` slowly from observed `mid - intrinsic`, clamp to sane bands, and enforce non-increasing voucher fair prices by strike.
- Add portfolio delta control using empirical deltas: `4000:.74`, `4500:.67`, `5000:.66`, `5100:.59`, `5200:.44`, `5300:.25`, `5400:.10`, `5500:.04`, wings `0`. Penalize new orders that increase excessive net delta.
- Add market-trade flow tilt from historical participant behavior: follow `Mark 14` in HYDRO/VEV4000, fade `Mark 38`; in VELVET follow `Mark 01`/`Mark 67` lightly and fade `Mark 55`/`Mark 49`. Decay flow by about `0.92` per tick and cap the price offset at `1.5 * vol`.
- Fair price formula per product: `expected = model_fair + micro_offset + mean_reversion_offset + flow_offset - inventory_skew - portfolio_delta_skew`.
- Use mean reversion only when strong: for HYDRO/VELVET, if `abs(z) > 1.5`, target inventory toward `-limit * clamp(z/3, -1, 1)` and add `-0.10 * (mid - slow_mean)` to fair, capped by volatility.
- Taker logic: buy asks when `ask <= expected - take_edge`; sell bids when `bid >= expected + take_edge`. Use larger edges for wide products: HYDRO `6`, VEV4000 `6`, VEV4500 `5`, VELVET `2`, VEV5000/5100 `2`, VEV5200+ `1`.
- Maker logic: quote only when edge exists after skew. Improve by one tick only if still at least `make_edge` from expected; otherwise join best bid/ask. Use small sizes: HYDRO/VELVET `3-5`, VEVs `1-3`.
- Cheap voucher rule: never open short positions in `VEV_5400+` unless bid is clearly above fair; prefer one-sided accumulation below fair and sell only to reduce inventory.
- Endgame rule: after timestamp `% 1_000_000 > 990_000`, stop opening marginal positions and skew all quotes toward reducing inventory and net delta.

## Test Plan
- Run current baseline first; existing latest log ends around `8925.5` total PnL, with gains in HYDRO/VELVET/VEV4000 and losses in VEV5100-VEV5400.
- Backtest day-by-day and all days together. Require no position-limit breaches, no crossed invalid quotes, stable JSON traderData size, and better per-product PnL on the near-ATM VEV strip.
- Ablate features in order: base fair/maker, VEV ladder fair, delta control, flow signal, cheap-voucher rules. Keep a feature only if it improves at least two of three days or materially reduces drawdown.
- Validate invariants with unit checks: VEV fair prices non-increasing by strike, wing fair fixed at `0.5`, no orders that increase inventory past limits, and portfolio delta skew changes sign correctly.

## Assumptions
- Target implementation is the local Round 4 backtester and repo data, not an unseen conversion-only spec.
- If live `state.observations.conversionObservations` exposes foreign bid/ask plus fees, add a secondary conversion fair interval: export fair `foreign_bid - transport - exportTariff`, import fair `foreign_ask + transport + importTariff`, then trade only when local prices clear that interval after rounding and edge.
- Avoid complex ML. The strongest robust edges are microstructure, participant flow, mean reversion, and VEV option parity.
