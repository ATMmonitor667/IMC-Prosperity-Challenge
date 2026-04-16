
import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

OSMIUM_FAIR = 10_000.0
OSMIUM_EDGE = 7

PEPPER_ALPHA = 0.7
PEPPER_EDGE = 5

OSMIUM_SKEW_DENOM = 40
PEPPER_SKEW_DENOM = 60


class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid = (best_bid + best_ask) / 2.0

            if product == OSMIUM:
                fair = OSMIUM_FAIR
                edge = OSMIUM_EDGE
                skew_denom = OSMIUM_SKEW_DENOM
            elif product == PEPPER:
                prev = memory.get("pepper_ema")
                ema = mid if prev is None else PEPPER_ALPHA * mid + (1 - PEPPER_ALPHA) * prev
                memory["pepper_ema"] = ema
                fair = ema
                edge = PEPPER_EDGE
                skew_denom = PEPPER_SKEW_DENOM
            else:
                fair = mid
                edge = 2
                skew_denom = 40

            result[product] = _quote(
                product, od, position, fair, edge, best_bid, best_ask, skew_denom
            )

        return result, 0, json.dumps(memory)


def _quote(
    product: str,
    od: OrderDepth,
    position: int,
    fair: float,
    edge: int,
    best_bid: int,
    best_ask: int,
    skew_denom: int,
) -> List[Order]:
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position
    orders: List[Order] = []

    for ask_price in sorted(od.sell_orders.keys()):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(product, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders.keys(), reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(product, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    skew = position // skew_denom

    post_bid = int(round(fair - edge)) - skew
    post_ask = int(round(fair + edge)) - skew

    if post_bid >= best_ask:
        post_bid = best_ask - 1
    if post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = int(round(fair)) - edge
        post_ask = int(round(fair)) + edge

    if buy_cap > 0:
        orders.append(Order(product, post_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(product, post_ask, -sell_cap))

    return orders


def _load(raw: str) -> Dict[str, float]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
