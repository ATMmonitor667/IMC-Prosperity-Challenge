"""Test: just buy pepper aggressively to +80 and hold. No MM, no asks.
This establishes the theoretical trend-only PnL ceiling."""

import json
from typing import Dict, List, Tuple
from datamodel import Order, OrderDepth, TradingState


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        for product, od in state.order_depths.items():
            orders: List[Order] = []
            if product == "INTARIAN_PEPPER_ROOT":
                pos = state.position.get(product, 0)
                buy_cap = 80 - pos
                # Take every ask we can, regardless of price
                for ask, vol in sorted(od.sell_orders.items()):
                    if buy_cap <= 0:
                        break
                    take = min(abs(vol), buy_cap)
                    orders.append(Order(product, ask, take))
                    buy_cap -= take
                # Also post a passive bid at best_bid+1 with remaining cap
                if buy_cap > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders)
                    orders.append(Order(product, best_bid + 1, buy_cap))
            elif product == "ASH_COATED_OSMIUM":
                # do nothing on osmium
                pass
            result[product] = orders
        return result, 0, ""
