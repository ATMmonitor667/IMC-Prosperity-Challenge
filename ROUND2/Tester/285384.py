"""IMC Prosperity 4 — Round 2 simple market-making bot.

Data (see analyze_round2_data.py on prices_round_2_day_-1,0,1.csv):
  ASH_COATED_OSMIUM: mean mid ~10,001, std ~5, typical top-of-book spread ~16.
  INTARIAN_PEPPER_ROOT: mean mid ~12,500, std ~800+, spread ~14 — trending.

Paired mid correlation across days is weak overall (~-0.08), so we run
separate strategies per symbol (not a static pairs trade).

Osmium: fixed-fair mean reversion + two-layer passive quotes (same structure
as Round 1, fair anchored to Round 2 empirical mean).

Pepper: microprice-driven fast/slow EMA, slope-projected fair, adaptive
inventory target, vol-aware edges (same family as Round 1 trader.py).
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# Osmium — Round 2 empirical mean ~10000.88 (see analyze_round2_data.py)
OSMIUM_FAIR = 10_000.9
OSMIUM_EDGE = 7
OSMIUM_EDGE_FAR = 12
OSMIUM_SIZE_NEAR = 60
OSMIUM_SIZE_FAR = 20
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP = 0.5
OSMIUM_TAKE_SLOP = 1

# Pepper
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


def _trade_osmium(
    od: OrderDepth,
    position: int,
    best_bid,
    best_ask,
    mid: float,
) -> List[Order]:
    fair = OSMIUM_FAIR
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

    if abs(slope) < PEPPER_TREND_DEADZONE:
        target = 0
    else:
        target = int(min(PEPPER_TARGET_MAX, abs(slope) * 22000))
        target = target if slope > 0 else -target
        if (target > 0 and ema < slow) or (target < 0 and ema > slow):
            target = 0

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