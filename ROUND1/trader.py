"""Round 1 optimized trader — per-product strategies for maximum PnL.

ASH_COATED_OSMIUM (stable, mean ≈ 10,000, std ≈ 5, spread ≈ 16):
  Aggressive fixed-fair market maker with microprice adjustment.
  Sweeps all mispriced resting orders, then posts tight passive quotes
  with continuous inventory skew.  Edge = 1 captures the 16-tick spread
  many times per day.

INTARIAN_PEPPER_ROOT (trending, range 9,998 → 13,007, 17 mean crossings):
  Dual-EMA momentum trend follower.  Fast EMA (α = 0.35) tracks the
  current regime; slow EMA (α = 0.06) provides a trend baseline.
  Momentum = fast − slow shifts the fair value in the trend direction,
  making the bot aggressively take positions toward the trend.  Edges
  are asymmetric: tight toward the trend, wide against it.

State persisted as compact JSON in traderData (pepper EMAs only).
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

# ── Product names ──────────────────────────────────────────────
OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

# ── Hard position limit (per product, set by the exchange) ─────
LIMIT = 80

# ── Osmium parameters ─────────────────────────────────────────
OSMIUM_FAIR = 10_000.0     # long-run anchor (data mean = 10,000.20)
OSMIUM_EDGE = 1            # passive quote distance from fair
OSMIUM_SKEW_DIV = 20       # skew = position // div  → max ±4 at limit

# ── Pepper parameters ─────────────────────────────────────────
PEPPER_FAST_ALPHA = 0.35   # responsive to current price
PEPPER_SLOW_ALPHA = 0.06   # slow trend baseline
PEPPER_EDGE = 2            # base passive edge
PEPPER_SKEW_DIV = 15       # tighter skew for trending product  → max ±5
PEPPER_MOM_FAIR_SCALE = 0.02   # momentum → fair shift multiplier
PEPPER_MOM_FAIR_CAP = 5        # max absolute fair shift
PEPPER_MOM_EDGE_SCALE = 0.01   # momentum → edge asymmetry multiplier
PEPPER_MOM_EDGE_CAP = 3        # max edge asymmetry


class Trader:
    """Entry point called by the exchange engine once per tick."""

    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)

            # Need at least one side of the book
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
                    od, position, best_bid, best_ask, mid, memory
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid, memory
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


# ──────────────────────────────────────────────────────────────
#  ASH_COATED_OSMIUM — fixed-fair aggressive market maker
# ──────────────────────────────────────────────────────────────

def _trade_osmium(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float, memory: dict,
) -> List[Order]:
    fair = OSMIUM_FAIR

    # Microprice adjustment: nudge fair toward the heavier side of the book
    total_bid_vol = sum(abs(v) for v in od.buy_orders.values())
    total_ask_vol = sum(abs(v) for v in od.sell_orders.values())
    denom = total_bid_vol + total_ask_vol
    if denom > 0:
        imbalance = (total_bid_vol - total_ask_vol) / denom
        fair += imbalance * 0.5          # ±0.5 max — very subtle

    orders: List[Order] = []
    buy_cap = LIMIT - position           # room to buy before hitting +LIMIT
    sell_cap = LIMIT + position          # room to sell before hitting -LIMIT

    # ── Phase 1: sweep all mispriced resting orders ──
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

    # ── Phase 2: passive quoting with inventory skew ──
    skew = position // OSMIUM_SKEW_DIV   # positive when long → shift quotes down

    post_bid = int(round(fair)) - OSMIUM_EDGE - skew
    post_ask = int(round(fair)) + OSMIUM_EDGE - skew

    # Sanitize: never cross existing book or self-cross
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


# ──────────────────────────────────────────────────────────────
#  INTARIAN_PEPPER_ROOT — dual-EMA momentum trend follower
# ──────────────────────────────────────────────────────────────

def _trade_pepper(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float, memory: dict,
) -> List[Order]:

    # ── Compute microprice for a better EMA input ──
    if best_bid is not None and best_ask is not None:
        bid_vol = abs(od.buy_orders[best_bid])
        ask_vol = abs(od.sell_orders[best_ask])
        price_input = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
    else:
        price_input = mid

    # ── Update dual EMAs ──
    fast_prev = memory.get("pf")
    slow_prev = memory.get("ps")

    if fast_prev is None:
        fast_ema = price_input
        slow_ema = price_input
    else:
        fast_ema = PEPPER_FAST_ALPHA * price_input + (1 - PEPPER_FAST_ALPHA) * fast_prev
        slow_ema = PEPPER_SLOW_ALPHA * price_input + (1 - PEPPER_SLOW_ALPHA) * slow_prev

    memory["pf"] = fast_ema
    memory["ps"] = slow_ema

    # ── Momentum signal ──
    momentum = fast_ema - slow_ema

    # ── Fair value = fast EMA shifted by momentum ──
    #    Up-trend → fair pushed above fast EMA → aggressive buying
    #    Down-trend → fair pushed below fast EMA → aggressive selling
    trend_shift = min(abs(momentum) * PEPPER_MOM_FAIR_SCALE, PEPPER_MOM_FAIR_CAP)
    if momentum > 0:
        fair = fast_ema + trend_shift
    else:
        fair = fast_ema - trend_shift

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    # ── Phase 1: sweep mispriced orders vs momentum-adjusted fair ──
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

    # ── Phase 2: momentum-adjusted passive quoting ──
    skew = position // PEPPER_SKEW_DIV

    # Asymmetric edges: tight toward trend, wide against trend
    mom_edge_raw = min(abs(momentum) * PEPPER_MOM_EDGE_SCALE, PEPPER_MOM_EDGE_CAP)
    mom_edge = int(round(mom_edge_raw))

    if momentum > 0:                        # UP-TREND
        bid_edge = max(1, PEPPER_EDGE - mom_edge)   # tighter bid  → eager to buy
        ask_edge = max(1, PEPPER_EDGE + mom_edge)   # wider ask    → hold longs
    else:                                    # DOWN-TREND
        bid_edge = max(1, PEPPER_EDGE + mom_edge)   # wider bid    → avoid buying
        ask_edge = max(1, PEPPER_EDGE - mom_edge)   # tighter ask  → eager to sell

    post_bid = int(round(fair)) - bid_edge - skew
    post_ask = int(round(fair)) + ask_edge - skew

    # Sanitize
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


# ── Utility ────────────────────────────────────────────────────

def _load(raw: str) -> dict:
    """Safely deserialise traderData JSON; return empty dict on failure."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
