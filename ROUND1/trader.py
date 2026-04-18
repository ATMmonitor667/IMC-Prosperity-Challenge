"""Round 1 trader with risk-aware market making.

This version keeps the OSMIUM fixed-fair edge capture, but makes PEPPER
materially safer by removing hardcoded long-only behavior.

PEPPER now uses:
- fast/slow EMA regime filter
- bounded slope forecast horizon
- adaptive (sign-aware) inventory target
- volatility-aware quote widening
- inventory tapering for passive quote size

These controls reduce directional tail risk and improve robustness against
trend reversals while preserving opportunity capture during stable drift.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


# ══════════════════════════════════════════════════════════════
#   PRODUCTS  /  GLOBALS
# ══════════════════════════════════════════════════════════════

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# ── Osmium parameters ─────────────────────────────────────────
OSMIUM_FAIR        = 10_000.2   # empirical long-run mean
OSMIUM_EDGE = 7
OSMIUM_EDGE_FAR = 12
OSMIUM_SIZE_NEAR = 60
OSMIUM_SIZE_FAR = 20
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP   = 0.5
OSMIUM_TAKE_SLOP = 1

# ── Pepper parameters ─────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════

class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

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
                    od, position, best_bid, best_ask, mid
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid, state.timestamp, memory
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


# ══════════════════════════════════════════════════════════════
#   OSMIUM — fixed-fair aggressive MM with two-layer ladder
# ══════════════════════════════════════════════════════════════

def _trade_osmium(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
) -> List[Order]:
    fair = OSMIUM_FAIR

    # Microprice tilt (very mild)
    tot_b = sum(abs(v) for v in od.buy_orders.values())
    tot_a = sum(abs(v) for v in od.sell_orders.values())
    denom = tot_b + tot_a
    if denom > 0:
        imbalance = (tot_b - tot_a) / denom
        fair += imbalance * OSMIUM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    # Phase 1: AGGRESSIVE sweep. Take asks up to fair + OSMIUM_TAKE_SLOP
    # and bids down to fair - OSMIUM_TAKE_SLOP. Lag-1 autocorr = -0.5,
    # so any print away from 10,000 has strong reversion expectancy.
    for ask_price in sorted(od.sell_orders):
        if ask_price > fair + OSMIUM_TAKE_SLOP or buy_cap <= 0:
            break
        # Discount: only take price-levels that are cheaper than fair
        # or where we need inventory (position still short of target).
        if ask_price >= fair and position >= 0:
            # neutral or long already; paying premium only makes sense
            # if price is clearly below 10000 — skip otherwise.
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

    # Phase 2: passive ladder with inventory skew
    skew = position // OSMIUM_SKEW_DENOM
    fair_i = int(round(fair))

    post_bid_near = fair_i - OSMIUM_EDGE - skew
    post_ask_near = fair_i + OSMIUM_EDGE - skew
    post_bid_far  = fair_i - OSMIUM_EDGE_FAR - skew
    post_ask_far  = fair_i + OSMIUM_EDGE_FAR - skew

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


# ══════════════════════════════════════════════════════════════
#   PEPPER — slope-projected fair + long-biased asymmetric MM
# ══════════════════════════════════════════════════════════════

def _trade_pepper(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
    timestamp: int, memory: dict,
) -> List[Order]:
    # ── Microprice input (robust to bid/ask bounce) ──────────
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        price_in = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) else mid
    else:
        price_in = mid

    # ── Fast/slow trend model and volatility estimate ─────────
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

    # ── Slope-projected fair (horizon adapts to trend strength) ─
    trend_conf = abs(ema - slow)
    forecast_ts = PEPPER_FORECAST_TS_BASE + min(
        PEPPER_FORECAST_TS_MAX - PEPPER_FORECAST_TS_BASE,
        int(trend_conf * 180),
    )
    fair = ema + slope * forecast_ts

    # Directional target is adaptive, not hardcoded long-only.
    if abs(slope) < PEPPER_TREND_DEADZONE:
        target = 0
    else:
        target = int(min(PEPPER_TARGET_MAX, abs(slope) * 22000))
        target = target if slope > 0 else -target
        # Regime check: if fast/slow disagree with slope, de-risk.
        if (target > 0 and ema < slow) or (target < 0 and ema > slow):
            target = 0

    # ── Sweep only when meaningfully mispriced vs fair ───────
    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position
    take_band = max(1.0, 0.80 * vol)

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair - take_band or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(PEPPER, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair + take_band or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(PEPPER, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # ── Inventory-aware quote placement ───────────────────────
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


# ══════════════════════════════════════════════════════════════
#   Utility
# ══════════════════════════════════════════════════════════════

def _load(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}