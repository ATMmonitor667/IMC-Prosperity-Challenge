"""Round 1 final trader — per-product strategies.

Products
--------
ASH_COATED_OSMIUM (OSMIUM):
    Stationary mean-reverting product centered very tightly on 10,000
    (std ~5, range ±25, ~16-tick quoted spread, lag-1 return autocorr
    ~-0.5 → textbook bid/ask bounce).  Strategy: fixed-fair aggressive
    market maker.  Anchor fair at 10,000 plus a tiny microprice tilt
    (±0.5) for queue priority, sweep any ask < fair / bid > fair, then
    post passive quotes at fair ± edge with inventory skew.

INTARIAN_PEPPER_ROOT (PEPPER):
    Strongly trending product (training data shows ≈ +1000 price units
    per day, i.e. slope ≈ 0.01 per timestamp unit) with residual
    micro-structure that looks like osmium (bid/ask bounce around the
    trend line).  Strategy: slope-adjusted-EMA fair value (learned
    online — no hard-coded drift), asymmetric edges (tight bid / wide
    ask in an uptrend and vice-versa), and a long-biased inventory
    target while slope is positive.  The slope is estimated from an
    exponentially-smoothed per-timestamp mid-price change, so it adapts
    if the live drift differs from the training sample.

No look-ahead
-------------
Every decision at tick t uses only:
  * the current order book at t,
  * our current position at t,
  * `state.timestamp` at t,
  * state persisted from ticks strictly before t via `traderData`.
No future prices, trades, or observations are accessed.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80


OSMIUM_FAIR = 10_000.0
OSMIUM_EDGE = 1
OSMIUM_SKEW_DIV = 20
OSMIUM_MICROPRICE_CAP = 0.5


PEPPER_FAST_ALPHA = 0.35
PEPPER_SLOPE_ALPHA = 0.01
PEPPER_FORECAST_HORIZON = 50
PEPPER_SLOPE_CAP = 0.05
PEPPER_SLOPE_DEADZONE = 0.0005

PEPPER_BASE_EDGE = 2
PEPPER_TREND_EDGE_WIDE = 5
PEPPER_TREND_EDGE_TIGHT = 2

PEPPER_INVENTORY_TARGET = 40
PEPPER_SKEW_DIV = 15


class Trader:
    """Entry point called by the exchange engine once per tick."""

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)
            else:
                result[product] = []
                continue

            if product == OSMIUM:
                result[product] = _trade_osmium(
                    od, position, best_bid, best_ask
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid,
                    state.timestamp, memory,
                )
            else:
                result[product] = _trade_generic(
                    product, od, position, best_bid, best_ask, mid
                )

        return result, 0, json.dumps(memory)


# ─────────────────────────────────────────────────────────────
#  ASH_COATED_OSMIUM — fixed-fair aggressive market maker
# ─────────────────────────────────────────────────────────────

def _trade_osmium(
    od: OrderDepth, position: int, best_bid, best_ask,
) -> List[Order]:
    fair = OSMIUM_FAIR

    total_bid_vol = sum(abs(v) for v in od.buy_orders.values())
    total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
    denom = total_bid_vol + total_ask_vol
    if denom > 0:
        imbalance = (total_bid_vol - total_ask_vol) / denom
        fair += imbalance * OSMIUM_MICROPRICE_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    skew = position // OSMIUM_SKEW_DIV

    post_bid = int(round(fair)) - OSMIUM_EDGE - skew
    post_ask = int(round(fair)) + OSMIUM_EDGE - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = int(round(fair)) - 1
        post_ask = int(round(fair)) + 1

    if buy_cap > 0:
        orders.append(Order(OSMIUM, post_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(OSMIUM, post_ask, -sell_cap))

    return orders


# ─────────────────────────────────────────────────────────────
#  INTARIAN_PEPPER_ROOT — slope-aware trend market maker
# ─────────────────────────────────────────────────────────────

def _trade_pepper(
    od: OrderDepth, position: int, best_bid, best_ask, mid: float,
    timestamp: int, memory: dict,
) -> List[Order]:

    # Microprice as the EMA input is less susceptible to bid/ask bounce.
    if best_bid is not None and best_ask is not None:
        bid_vol = abs(od.buy_orders[best_bid])
        ask_vol = abs(od.sell_orders[best_ask])
        denom = bid_vol + ask_vol
        if denom > 0:
            price_input = (best_bid * ask_vol + best_ask * bid_vol) / denom
        else:
            price_input = mid
    else:
        price_input = mid

    fast_prev = memory.get("pf")
    slope_prev = memory.get("psl")
    ts_prev = memory.get("pts")
    mid_prev = memory.get("pmid")

    if fast_prev is None:
        fast_ema = price_input
    else:
        fast_ema = (
            PEPPER_FAST_ALPHA * price_input + (1 - PEPPER_FAST_ALPHA) * fast_prev
        )

    # Online slope estimate in "price units per 1 timestamp unit".
    # Uses prior-tick mid, so still causal.
    if mid_prev is not None and ts_prev is not None and timestamp > ts_prev:
        dt = timestamp - ts_prev
        instant_slope = (price_input - mid_prev) / dt
        if slope_prev is None:
            slope = instant_slope
        else:
            slope = (
                PEPPER_SLOPE_ALPHA * instant_slope
                + (1 - PEPPER_SLOPE_ALPHA) * slope_prev
            )
    else:
        slope = slope_prev if slope_prev is not None else 0.0

    # Cap the slope so a spurious spike can't blow out the fair.
    if slope > PEPPER_SLOPE_CAP:
        slope = PEPPER_SLOPE_CAP
    elif slope < -PEPPER_SLOPE_CAP:
        slope = -PEPPER_SLOPE_CAP

    memory["pf"] = fast_ema
    memory["psl"] = slope
    memory["pts"] = timestamp
    memory["pmid"] = price_input

    # Forecast a small number of ticks ahead so we stop lagging the trend.
    fair = fast_ema + slope * PEPPER_FORECAST_HORIZON

    if slope > PEPPER_SLOPE_DEADZONE:
        bid_edge = PEPPER_TREND_EDGE_TIGHT
        ask_edge = PEPPER_TREND_EDGE_WIDE
        inventory_target = PEPPER_INVENTORY_TARGET
    elif slope < -PEPPER_SLOPE_DEADZONE:
        bid_edge = PEPPER_TREND_EDGE_WIDE
        ask_edge = PEPPER_TREND_EDGE_TIGHT
        inventory_target = -PEPPER_INVENTORY_TARGET
    else:
        bid_edge = PEPPER_BASE_EDGE
        ask_edge = PEPPER_BASE_EDGE
        inventory_target = 0

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(PEPPER, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(PEPPER, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # Skew is driven by *distance from inventory target*, not raw position,
    # so the trend-long bias is structural rather than accidental.
    skew = (position - inventory_target) // PEPPER_SKEW_DIV

    post_bid = int(round(fair)) - bid_edge - skew
    post_ask = int(round(fair)) + ask_edge - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = int(round(fair)) - 1
        post_ask = int(round(fair)) + 1

    if buy_cap > 0:
        orders.append(Order(PEPPER, post_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(PEPPER, post_ask, -sell_cap))

    return orders


# ─────────────────────────────────────────────────────────────
#  Fallback for any unexpected product
# ─────────────────────────────────────────────────────────────

def _trade_generic(
    product: str, od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
) -> List[Order]:
    fair = mid
    edge = 2

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    skew = position // 40

    post_bid = int(round(fair)) - edge - skew
    post_ask = int(round(fair)) + edge - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = int(round(fair)) - 1
        post_ask = int(round(fair)) + 1

    if buy_cap > 0:
        orders.append(Order(product, post_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(product, post_ask, -sell_cap))

    return orders


def _load(raw: str) -> dict:
    """Safely deserialise traderData JSON; return empty dict on failure."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
