"""IMC Prosperity 4 — Round 2 trader: MM + statistical pairs overlay.

Offline analysis on prices_round_2_day_-1,0,1 (paired mids, same timestamp):

  • Pearson corr(osmium_mid, pepper_mid) ≈ -0.085 — misleadingly weak globally.
  • Rolling corr (2000-tick window) ranges ~[-0.83, +0.69]: relationship is
    strongly regime-dependent; a fixed global hedge is noisy.

  • OLS: pepper ≈ a + b * osmium with b ≈ -14.41, a ≈ 1.57e5.
    ADF on OLS residuals: p ≈ 0.77 — we do *not* rely on textbook cointegration
    for a static spread; instead we use an *online* EWMA mean/variance of the
    synthetic spread spread_t = pepper_mp - BETA * osmium_mp (BETA fixed to the
    pooled OLS slope as a scale anchor).

  • AR(1) on OLS residuals: phi ≈ 0.998 → any mean reversion is slow; overlay
    is gentle and gated by estimated spread vol and local correlation.

Live logic:
  • Base: mean-reverting MM on osmium + trend-aware MM on pepper (unchanged core).
  • Overlay: EWMA z-score of the synthetic spread adjusts pepper fair / target
    and nudges osmium fair. When |EWMA corr| of returns is tiny, the overlay
    is damped (no strong linear structure that tick).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# Pooled OLS hedge ratio (pepper on osmium mid) — scale anchor for spread
PAIR_BETA = -14.405893822008673
PAIR_ALPHA_MEAN = 0.004
PAIR_ALPHA_VAR = 0.018
PAIR_MIN_STD = 45.0
PAIR_WARMUP_TICKS = 80

# EWMA correlation of *returns* (microprice deltas); gate the pairs overlay
RET_ALPHA = 0.04
CORR_GATE = 0.12
PAIR_OVERLAY_SCALE = 1.0

# How strongly z-score steers pepper vs osmium (price units / inventory)
PEPPER_FAIR_Z = 10.0
PEPPER_TARGET_Z = 14
OSMIUM_FAIR_Z = 0.35

# Osmium base MM
OSMIUM_FAIR = 10_000.9
OSMIUM_EDGE = 7
OSMIUM_EDGE_FAR = 12
OSMIUM_SIZE_NEAR = 60
OSMIUM_SIZE_FAR = 20
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP = 0.5
OSMIUM_TAKE_SLOP = 1

# Pepper base MM
PEPPER_FAST_ALPHA = 0.45
PEPPER_SLOW_ALPHA = 0.08
PEPPER_SLOPE_ALPHA = 0.06
PEPPER_PRIOR_SLOPE = 0.0
PEPPER_FORECAST_TS_BASE = 800
PEPPER_FORECAST_TS_MAX = 1500
PEPPER_BID_EDGE_BASE = 3
PEPPER_ASK_EDGE_BASE = 4
PEPPER_SKEW_DENOM = 16
PEPPER_TARGET_MAX = 35
PEPPER_TREND_DEADZONE = 0.00030
PEPPER_VOL_ALPHA = 0.10
PEPPER_MIN_EDGE = 2
PEPPER_MAX_EDGE = 10
PEPPER_SIZE_BASE = 28


def _microprice(od: OrderDepth, best_bid: Optional[int], best_ask: Optional[int], mid: float) -> float:
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        if bv + av:
            return (best_bid * av + best_ask * bv) / (bv + av)
    return mid


def _update_pair_stats(memory: Dict[str, Any], osm_mp: float, pep_mp: float) -> float:
    """Update EWMA spread mean/var and return correlation-gated z-score."""
    spread = pep_mp - PAIR_BETA * osm_mp

    prev_osm = memory.get("prev_osm")
    prev_pep = memory.get("prev_pep")
    if prev_osm is not None and prev_pep is not None:
        dx = osm_mp - prev_osm
        dy = pep_mp - prev_pep
        mx = memory.get("rdx_ema", 0.0)
        my = memory.get("rdy_ema", 0.0)
        mx = (1.0 - RET_ALPHA) * mx + RET_ALPHA * dx
        my = (1.0 - RET_ALPHA) * my + RET_ALPHA * dy
        mxx = memory.get("rdx2_ema", 1e-9)
        myy = memory.get("rdy2_ema", 1e-9)
        mxy = memory.get("rdxy_ema", 0.0)
        mxx = (1.0 - RET_ALPHA) * mxx + RET_ALPHA * dx * dx
        myy = (1.0 - RET_ALPHA) * myy + RET_ALPHA * dy * dy
        mxy = (1.0 - RET_ALPHA) * mxy + RET_ALPHA * dx * dy
        memory["rdx_ema"], memory["rdy_ema"] = mx, my
        memory["rdx2_ema"], memory["rdy2_ema"], memory["rdxy_ema"] = mxx, myy, mxy
        vx = max(1e-9, mxx - mx * mx)
        vy = max(1e-9, myy - my * my)
        corr = (mxy - mx * my) / (math.sqrt(vx) * math.sqrt(vy))
        corr = max(-1.0, min(1.0, corr))
        memory["ret_corr"] = corr
    memory["prev_osm"], memory["prev_pep"] = osm_mp, pep_mp

    sm = memory.get("spr_mean")
    if sm is None:
        memory["spr_mean"] = spread
        memory["spr_var"] = PAIR_MIN_STD**2
        memory["pair_tick"] = 1
        return 0.0

    sm = (1.0 - PAIR_ALPHA_MEAN) * sm + PAIR_ALPHA_MEAN * spread
    memory["spr_mean"] = sm
    sv = memory.get("spr_var", PAIR_MIN_STD**2)
    sv = (1.0 - PAIR_ALPHA_VAR) * sv + PAIR_ALPHA_VAR * (spread - sm) ** 2
    memory["spr_var"] = max(sv, 1.0)
    memory["pair_tick"] = memory.get("pair_tick", 0) + 1

    std = math.sqrt(memory["spr_var"])
    std = max(std, PAIR_MIN_STD)
    z = (spread - sm) / std

    corr = memory.get("ret_corr", 0.0)
    gate = min(1.0, abs(corr) / CORR_GATE) if CORR_GATE > 0 else 0.0
    if memory.get("pair_tick", 0) < PAIR_WARMUP_TICKS:
        gate = 0.0

    z *= gate
    z = max(-3.0, min(3.0, z))
    return float(z * PAIR_OVERLAY_SCALE)


class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        micro: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            if not od.buy_orders and not od.sell_orders:
                continue
            bb = max(od.buy_orders) if od.buy_orders else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2.0
            elif bb is not None:
                mid = float(bb)
            else:
                mid = float(ba)
            micro[product] = _microprice(od, bb, ba, mid)

        pair_z = 0.0
        if OSMIUM in micro and PEPPER in micro:
            pair_z = _update_pair_stats(memory, micro[OSMIUM], micro[PEPPER])

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if not od.buy_orders and not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid = float(best_bid)
            else:
                mid = float(best_ask)

            if product == OSMIUM:
                result[product] = _trade_osmium(
                    od, position, best_bid, best_ask, mid, pair_z
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od,
                    position,
                    best_bid,
                    best_ask,
                    mid,
                    state.timestamp,
                    memory,
                    pair_z,
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


def _trade_osmium(
    od: OrderDepth,
    position: int,
    best_bid,
    best_ask,
    mid: float,
    pair_z: float,
) -> List[Order]:
    fair = OSMIUM_FAIR + OSMIUM_FAIR_Z * pair_z
    tot_b = sum(abs(v) for v in od.buy_orders.values())
    tot_a = sum(abs(v) for v in od.sell_orders.values())
    denom = tot_b + tot_a
    if denom > 0:
        fair += ((tot_b - tot_a) / denom) * OSMIUM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    for ask_price in sorted(od.sell_orders):
        if ask_price > fair + OSMIUM_TAKE_SLOP or buy_cap <= 0:
            break
        if ask_price >= fair and position >= 0:
            continue
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price < fair - OSMIUM_TAKE_SLOP or sell_cap <= 0:
            break
        if bid_price <= fair and position <= 0:
            continue
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    skew = position // OSMIUM_SKEW_DENOM
    fair_i = int(round(fair))
    post_bid_near = fair_i - OSMIUM_EDGE - skew
    post_ask_near = fair_i + OSMIUM_EDGE - skew
    post_bid_far = fair_i - OSMIUM_EDGE_FAR - skew
    post_ask_far = fair_i + OSMIUM_EDGE_FAR - skew

    if best_ask is not None and post_bid_near >= best_ask:
        post_bid_near = best_ask - 1
    if best_bid is not None and post_ask_near <= best_bid:
        post_ask_near = best_bid + 1
    if post_bid_near >= post_ask_near:
        post_bid_near = fair_i - 1
        post_ask_near = fair_i + 1

    near_buy = min(OSMIUM_SIZE_NEAR, buy_cap)
    near_sell = min(OSMIUM_SIZE_NEAR, sell_cap)
    if near_buy > 0:
        orders.append(Order(OSMIUM, post_bid_near, near_buy))
        buy_cap -= near_buy
    if near_sell > 0:
        orders.append(Order(OSMIUM, post_ask_near, -near_sell))
        sell_cap -= near_sell

    far_buy = min(OSMIUM_SIZE_FAR, buy_cap)
    far_sell = min(OSMIUM_SIZE_FAR, sell_cap)
    if far_buy > 0 and post_bid_far < post_bid_near:
        orders.append(Order(OSMIUM, post_bid_far, far_buy))
    if far_sell > 0 and post_ask_far > post_ask_near:
        orders.append(Order(OSMIUM, post_ask_far, -far_sell))

    return orders


def _trade_pepper(
    od: OrderDepth,
    position: int,
    best_bid,
    best_ask,
    mid: float,
    timestamp: int,
    memory: dict,
    pair_z: float,
) -> List[Order]:
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        price_in = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) else mid
    else:
        price_in = mid

    ema_prev = memory.get("pfast")
    slow_prev = memory.get("pslow")
    slope_prev = memory.get("pslope", PEPPER_PRIOR_SLOPE)
    vol_prev = memory.get("pvol", 1.0)
    ts_prev = memory.get("pts")

    if ema_prev is None:
        ema = price_in
        slow = price_in
        slope = PEPPER_PRIOR_SLOPE
        vol = 1.0
    else:
        ema = PEPPER_FAST_ALPHA * price_in + (1 - PEPPER_FAST_ALPHA) * ema_prev
        slow = PEPPER_SLOW_ALPHA * price_in + (1 - PEPPER_SLOW_ALPHA) * slow_prev
        dt = max(1, timestamp - ts_prev) if ts_prev is not None else 100
        inst_slope = (ema - ema_prev) / dt
        inst_slope = max(-0.01, min(0.01, inst_slope))
        slope = PEPPER_SLOPE_ALPHA * inst_slope + (1 - PEPPER_SLOPE_ALPHA) * slope_prev
        abs_err = abs(price_in - ema)
        vol = PEPPER_VOL_ALPHA * abs_err + (1 - PEPPER_VOL_ALPHA) * vol_prev

    memory["pfast"] = ema
    memory["pslow"] = slow
    memory["pslope"] = slope
    memory["pvol"] = vol
    memory["pts"] = timestamp

    trend_conf = abs(ema - slow)
    forecast_ts = PEPPER_FORECAST_TS_BASE + min(
        PEPPER_FORECAST_TS_MAX - PEPPER_FORECAST_TS_BASE,
        int(trend_conf * 180),
    )
    fair = ema + slope * forecast_ts
    fair -= PEPPER_FAIR_Z * pair_z

    if abs(slope) < PEPPER_TREND_DEADZONE:
        target = 0
    else:
        target = int(min(PEPPER_TARGET_MAX, abs(slope) * 22000))
        target = target if slope > 0 else -target
        if (target > 0 and ema < slow) or (target < 0 and ema > slow):
            target = 0

    target -= int(max(-PEPPER_TARGET_MAX, min(PEPPER_TARGET_MAX, PEPPER_TARGET_Z * pair_z)))

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position
    take_band = max(1.0, 0.80 * vol)

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair - take_band or buy_cap <= 0:
            break
        v = min(abs(od.sell_orders[ask_price]), buy_cap)
        if v > 0:
            orders.append(Order(PEPPER, ask_price, v))
            buy_cap -= v
            position += v

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair + take_band or sell_cap <= 0:
            break
        v = min(abs(od.buy_orders[bid_price]), sell_cap)
        if v > 0:
            orders.append(Order(PEPPER, bid_price, -v))
            sell_cap -= v
            position -= v

    skew = (position - target) // PEPPER_SKEW_DENOM
    edge_extra = int(min(3, max(0, round(vol / 2.0))))
    bid_edge = PEPPER_BID_EDGE_BASE + edge_extra
    ask_edge = PEPPER_ASK_EDGE_BASE + edge_extra
    if target > 0:
        bid_edge -= 1
        ask_edge += 1
    elif target < 0:
        bid_edge += 1
        ask_edge -= 1
    bid_edge = max(PEPPER_MIN_EDGE, min(PEPPER_MAX_EDGE, bid_edge))
    ask_edge = max(PEPPER_MIN_EDGE, min(PEPPER_MAX_EDGE, ask_edge))

    fair_i = int(round(fair))
    post_bid = fair_i - bid_edge - skew
    post_ask = fair_i + ask_edge - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = fair_i - 1
        post_ask = fair_i + 1

    inv_ratio = min(1.0, abs(position) / LIMIT)
    size_taper = 1.0 - 0.70 * inv_ratio
    base_size = max(6, int(round(PEPPER_SIZE_BASE * size_taper)))
    buy_bias = max(0, target - position) // 8
    sell_bias = max(0, position - target) // 8
    bid_size = min(buy_cap, base_size + buy_bias)
    ask_size = min(sell_cap, base_size + sell_bias)

    if bid_size > 0:
        orders.append(Order(PEPPER, post_bid, bid_size))
    if ask_size > 0:
        orders.append(Order(PEPPER, post_ask, -ask_size))

    return orders


def _load(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
