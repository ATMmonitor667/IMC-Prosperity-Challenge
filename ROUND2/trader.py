"""IMC Prosperity 4 - Round 2 trader.

Statistical analysis of prices_round_2_day_-1,0,1 (zero-mid rows filtered):

  ASH_COATED_OSMIUM:
      mid mean ~ 10,001, full-day std ~ 5, 200-MA detrended std ~ 3.5.
      median |tick return| = 1, lag-1 return autocorr = -0.50.
      -> pure tick-level mean reversion around a nearly constant anchor.

  INTARIAN_PEPPER_ROOT:
      mid drifts ~+1000 each day (day -1: 11k -> 12k; day 0: 12k -> 13k;
      day 1: 13k -> 14k). 200-MA detrended std ~ 2.2, tick MR is also strong
      (AC1 = -0.50), but the *day scale* drift dominates realized PnL: being
      structurally long 80 contracts for the full day captures ~+80k on
      price alone.

Strategy:

  Osmium:  mean-reverting market maker anchored to a slow EMA of microprice
           (fair is basically pinned at the empirical mean ~ 10,000.9). Two
           passive quote layers (near / far), inventory-skewed so we lean
           back toward zero.  Aggressive take when any book price crosses
           fair by at least 1 tick.

  Pepper:  fast/slow EMA + EWMA slope, with fair projected one forecast
           horizon ahead (fair = ema + slope * forecast_ts). Inventory
           target leans in the trend direction (up to +/- 50 contracts),
           passive quotes are asymmetric (tight on the trend-friendly
           side, wide on the trend-fighting side), and a far passive
           layer catches intermittent spread widenings.  Aggressive take
           when a book price is past fair by > 0.3 * local vol.

Two products stay decoupled: paired global corr is only ~-0.08 and rolling
corr swings across [-0.83, +0.69] (see analyze_round2_data.py), so no
static pairs trade is applied.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80


# ----------------------------- OSMIUM PARAMS ------------------------------ #
# Osmium mean mid ~ 10,000.9. Very slow EMA keeps fair pinned while tolerating
# any tiny secular drift; tick-level mean reversion is captured by a small
# take edge + two passive layers.
OSM_FAIR_PRIOR = 10_000.9
OSM_EMA_ALPHA = 0.02
OSM_VOL_ALPHA = 0.05
OSM_VOL_PRIOR = 3.5
OSM_MICRO_CAP = 0.5

OSM_TAKE_SLOP = 1
OSM_NEAR_EDGE = 7
OSM_FAR_EDGE = 12
OSM_NEAR_SIZE = 60
OSM_FAR_SIZE = 20
OSM_SKEW_DENOM = 40
OSM_INV_SOFT_CAP = 65

# ----------------------------- PEPPER PARAMS ------------------------------ #
PEP_FAST_ALPHA = 0.35
PEP_SLOW_ALPHA = 0.05
PEP_SLOPE_ALPHA = 0.04
PEP_PRIOR_SLOPE = 0.0
PEP_FORECAST_BASE = 1600
PEP_FORECAST_MAX = 1800
PEP_FORECAST_SCALE = 180
PEP_SLOPE_CLIP = 0.01

PEP_VOL_ALPHA = 0.10
PEP_VOL_PRIOR = 1.0

PEP_BID_EDGE_BASE = 2
PEP_ASK_EDGE_BASE = 5
PEP_MIN_EDGE = 2
PEP_MAX_EDGE = 10
PEP_SKEW_DENOM = 24
PEP_TARGET_MAX = 50
PEP_TREND_DEADZONE = 0.00015
PEP_SIZE_BASE = 34
PEP_SIZE_MAX_LEAN = 18
PEP_TAKE_VOL_K = 0.30
PEP_TARGET_GAIN = 26000          # target = TARGET_GAIN * slope (clipped)
PEP_FAR_EDGE_EXTRA = 4           # far layer sits this many ticks outside near
PEP_FAR_SIZE = 18
PEP_FAR_ENABLED = True


# =========================================================================== #
class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        mem = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            if not od.buy_orders and not od.sell_orders:
                result[product] = []
                continue

            bb = max(od.buy_orders) if od.buy_orders else None
            ba = min(od.sell_orders) if od.sell_orders else None
            if bb is not None and ba is not None:
                mid = (bb + ba) / 2.0
            elif bb is not None:
                mid = float(bb)
            else:
                mid = float(ba)

            micro = _microprice(od, bb, ba, mid)

            if product == OSMIUM:
                result[product] = _trade_osmium(od, pos, bb, ba, micro, mem)
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, pos, bb, ba, mid, micro, state.timestamp, mem
                )
            else:
                result[product] = []

        return result, 0, json.dumps(mem)


# =========================================================================== #
def _microprice(od: OrderDepth, bb: Optional[int], ba: Optional[int], mid: float) -> float:
    if bb is not None and ba is not None:
        bv = abs(od.buy_orders[bb])
        av = abs(od.sell_orders[ba])
        if bv + av:
            return (bb * av + ba * bv) / (bv + av)
    return mid


def _book_imbalance(od: OrderDepth) -> float:
    tb = sum(abs(v) for v in od.buy_orders.values())
    ta = sum(abs(v) for v in od.sell_orders.values())
    if tb + ta <= 0:
        return 0.0
    return (tb - ta) / (tb + ta)


# ---------------------------------- OSMIUM -------------------------------- #
def _trade_osmium(
    od: OrderDepth, pos: int, bb, ba, micro: float, mem: Dict[str, Any]
) -> List[Order]:
    ema = mem.get("o_ema", OSM_FAIR_PRIOR)
    vol = mem.get("o_vol", OSM_VOL_PRIOR)
    ema = (1 - OSM_EMA_ALPHA) * ema + OSM_EMA_ALPHA * micro
    dev = abs(micro - ema)
    vol = (1 - OSM_VOL_ALPHA) * vol + OSM_VOL_ALPHA * dev
    vol = max(1.0, vol)
    mem["o_ema"] = ema
    mem["o_vol"] = vol

    fair = ema + _book_imbalance(od) * OSM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - pos
    sell_cap = LIMIT + pos

    # --- aggressive take: eat prices past fair by >= OSM_TAKE_SLOP
    for ap in sorted(od.sell_orders):
        if ap > fair - OSM_TAKE_SLOP or buy_cap <= 0:
            break
        # avoid extending already-long position via crossing AT fair
        if ap >= fair and pos >= 0:
            continue
        v = min(abs(od.sell_orders[ap]), buy_cap)
        if v > 0:
            orders.append(Order(OSMIUM, ap, v))
            buy_cap -= v
            pos += v

    for bp in sorted(od.buy_orders, reverse=True):
        if bp < fair + OSM_TAKE_SLOP or sell_cap <= 0:
            break
        if bp <= fair and pos <= 0:
            continue
        v = min(abs(od.buy_orders[bp]), sell_cap)
        if v > 0:
            orders.append(Order(OSMIUM, bp, -v))
            sell_cap -= v
            pos -= v

    # --- passive quotes: two layers (near + far), inventory-skewed
    skew = pos // OSM_SKEW_DENOM
    fair_i = int(round(fair))
    post_bid_near = fair_i - OSM_NEAR_EDGE - skew
    post_ask_near = fair_i + OSM_NEAR_EDGE - skew
    post_bid_far = fair_i - OSM_FAR_EDGE - skew
    post_ask_far = fair_i + OSM_FAR_EDGE - skew

    if ba is not None and post_bid_near >= ba:
        post_bid_near = ba - 1
    if bb is not None and post_ask_near <= bb:
        post_ask_near = bb + 1
    if post_bid_near >= post_ask_near:
        post_bid_near = fair_i - 1
        post_ask_near = fair_i + 1

    inv_abs = abs(pos)
    if inv_abs > OSM_INV_SOFT_CAP:
        shrink = max(0.0, 1.0 - (inv_abs - OSM_INV_SOFT_CAP) / max(1, LIMIT - OSM_INV_SOFT_CAP))
    else:
        shrink = 1.0
    bid_shrink = shrink if pos > 0 else 1.0
    ask_shrink = shrink if pos < 0 else 1.0

    near_b = min(buy_cap, int(round(OSM_NEAR_SIZE * bid_shrink)))
    near_s = min(sell_cap, int(round(OSM_NEAR_SIZE * ask_shrink)))
    if near_b > 0:
        orders.append(Order(OSMIUM, post_bid_near, near_b))
        buy_cap -= near_b
    if near_s > 0:
        orders.append(Order(OSMIUM, post_ask_near, -near_s))
        sell_cap -= near_s

    far_b = min(buy_cap, int(round(OSM_FAR_SIZE * bid_shrink)))
    far_s = min(sell_cap, int(round(OSM_FAR_SIZE * ask_shrink)))
    if far_b > 0 and post_bid_far < post_bid_near:
        orders.append(Order(OSMIUM, post_bid_far, far_b))
    if far_s > 0 and post_ask_far > post_ask_near:
        orders.append(Order(OSMIUM, post_ask_far, -far_s))

    return orders


# ---------------------------------- PEPPER -------------------------------- #
def _trade_pepper(
    od: OrderDepth,
    pos: int,
    bb,
    ba,
    mid: float,
    micro: float,
    timestamp: int,
    mem: Dict[str, Any],
) -> List[Order]:
    price_in = micro

    ema = mem.get("p_fast")
    slow = mem.get("p_slow")
    slope = mem.get("p_slope", PEP_PRIOR_SLOPE)
    vol = mem.get("p_vol", PEP_VOL_PRIOR)
    ts_prev = mem.get("p_ts")

    if ema is None:
        ema_prev = price_in
        ema = price_in
        slow = price_in
        slope = PEP_PRIOR_SLOPE
        vol = PEP_VOL_PRIOR
    else:
        ema_prev = ema
        ema = PEP_FAST_ALPHA * price_in + (1 - PEP_FAST_ALPHA) * ema_prev
        slow = PEP_SLOW_ALPHA * price_in + (1 - PEP_SLOW_ALPHA) * slow
        dt = max(1, timestamp - ts_prev) if ts_prev is not None else 100
        inst_slope = (ema - ema_prev) / dt
        inst_slope = max(-PEP_SLOPE_CLIP, min(PEP_SLOPE_CLIP, inst_slope))
        slope = PEP_SLOPE_ALPHA * inst_slope + (1 - PEP_SLOPE_ALPHA) * slope
        abs_err = abs(price_in - ema_prev)
        vol = PEP_VOL_ALPHA * abs_err + (1 - PEP_VOL_ALPHA) * vol

    mem["p_fast"] = ema
    mem["p_slow"] = slow
    mem["p_slope"] = slope
    mem["p_vol"] = max(0.5, vol)
    mem["p_ts"] = timestamp

    trend_conf = abs(ema - slow)
    forecast_ts = PEP_FORECAST_BASE + min(
        PEP_FORECAST_MAX - PEP_FORECAST_BASE, int(trend_conf * PEP_FORECAST_SCALE)
    )
    fair = ema + slope * forecast_ts

    # --- inventory target: lean in trend direction (pepper drift is persistent)
    if abs(slope) < PEP_TREND_DEADZONE:
        target = 0
    else:
        target = int(min(PEP_TARGET_MAX, abs(slope) * PEP_TARGET_GAIN))
        target = target if slope > 0 else -target
        # only aggressively lean if fast & slow agree on direction
        if (target > 0 and ema < slow) or (target < 0 and ema > slow):
            target = target // 3  # weaker conviction

    orders: List[Order] = []
    buy_cap = LIMIT - pos
    sell_cap = LIMIT + pos
    take_band = max(1.0, PEP_TAKE_VOL_K * vol)

    # --- AGGRESSIVE TAKE against fair forecast
    for ap in sorted(od.sell_orders):
        if ap >= fair - take_band or buy_cap <= 0:
            break
        v = min(abs(od.sell_orders[ap]), buy_cap)
        if v > 0:
            orders.append(Order(PEPPER, ap, v))
            buy_cap -= v
            pos += v

    for bp in sorted(od.buy_orders, reverse=True):
        if bp <= fair + take_band or sell_cap <= 0:
            break
        v = min(abs(od.buy_orders[bp]), sell_cap)
        if v > 0:
            orders.append(Order(PEPPER, bp, -v))
            sell_cap -= v
            pos -= v

    # --- passive quotes, inv-skewed toward `target`
    skew = (pos - target) // PEP_SKEW_DENOM
    edge_extra = int(min(3, max(0, round(vol / 2.0))))
    bid_edge = PEP_BID_EDGE_BASE + edge_extra
    ask_edge = PEP_ASK_EDGE_BASE + edge_extra
    if target > 0:
        bid_edge -= 1
        ask_edge += 1
    elif target < 0:
        bid_edge += 1
        ask_edge -= 1
    bid_edge = max(PEP_MIN_EDGE, min(PEP_MAX_EDGE, bid_edge))
    ask_edge = max(PEP_MIN_EDGE, min(PEP_MAX_EDGE, ask_edge))

    fair_i = int(round(fair))
    post_bid = fair_i - bid_edge - skew
    post_ask = fair_i + ask_edge - skew
    if ba is not None and post_bid >= ba:
        post_bid = ba - 1
    if bb is not None and post_ask <= bb:
        post_ask = bb + 1
    if post_bid >= post_ask:
        post_bid = fair_i - 1
        post_ask = fair_i + 1

    inv_ratio = min(1.0, abs(pos) / LIMIT)
    size_taper = 1.0 - 0.65 * inv_ratio
    base_size = max(8, int(round(PEP_SIZE_BASE * size_taper)))
    gap = target - pos
    if gap > 0:
        buy_lean = min(PEP_SIZE_MAX_LEAN, gap // 3)
        sell_lean = 0
    elif gap < 0:
        buy_lean = 0
        sell_lean = min(PEP_SIZE_MAX_LEAN, (-gap) // 3)
    else:
        buy_lean = sell_lean = 0
    bid_size = min(buy_cap, base_size + buy_lean)
    ask_size = min(sell_cap, base_size + sell_lean)

    if bid_size > 0:
        orders.append(Order(PEPPER, post_bid, bid_size))
        buy_cap -= bid_size
    if ask_size > 0:
        orders.append(Order(PEPPER, post_ask, -ask_size))
        sell_cap -= ask_size

    # --- optional far layer: sits a few ticks outside the near quote and
    # catches momentary spread widenings without consuming near inventory.
    if PEP_FAR_ENABLED:
        far_bid = post_bid - PEP_FAR_EDGE_EXTRA
        far_ask = post_ask + PEP_FAR_EDGE_EXTRA
        far_bid_sz = min(buy_cap, PEP_FAR_SIZE)
        far_ask_sz = min(sell_cap, PEP_FAR_SIZE)
        if far_bid_sz > 0 and far_bid < post_bid:
            orders.append(Order(PEPPER, far_bid, far_bid_sz))
        if far_ask_sz > 0 and far_ask > post_ask:
            orders.append(Order(PEPPER, far_ask, -far_ask_sz))

    return orders


# =========================================================================== #
def _load(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
