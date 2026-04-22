import jsonpickle
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple


'''
   I will be implementing common trading strategies here on my own, With the help
   of cursor AI's autocomplete, the strategies are named here on the side and I will
   be implementing them here on my own one by one.add()
   
   There are two different products for which i will be implementing the strategies for,
   OSMIUM and PEPPER.
   OSMIUM is a product which is a metal and is used in the production of other metals.
   PEPPER is a product which is a spice and is used in the production of other spices.
   The strategies are named here on the side and I will be implementing them here on my own one by one.
   The strategies are named here on the side and I will be implementing them here on my own one by one.
'''

class Trader:
    # Change these to choose which generic algorithm each product uses.
    # Valid choices: 1, 2, 3, or 4.
    # Example: OSMIUM_ALGORITHM = 1 and PEPPER_ALGORITHM = 4.
    OSMIUM_ALGORITHM = 4
    PEPPER_ALGORITHM = 4

    def __init__(self, state=None):
        self.PRODUCTS = {
            "OSMIUM": "ASH_COATED_OSMIUM",
            "PEPPER": "INTARIAN_PEPPER_ROOT",
        }
        self.POSITION_LIMITS = {
            self.PRODUCTS["OSMIUM"]: 80,
            self.PRODUCTS["PEPPER"]: 80,
        }
        self.orders = []
        self.EMA = {}
        self.n_seen = {}
        self.sigma = {}
        self.mu = {}
        self.slow_ema = {}
        self.algo4_seen = {}
        self.algo4_target = {}
        if state is not None:
            for product in state.order_depths:
                self._ensure_product_state(product)

    def _normalize_product(self, product: str):
        if product in self.POSITION_LIMITS:
            return product
        return self.PRODUCTS.get(product)

    def _ensure_product_state(self, product: str) -> None:
        self.EMA.setdefault(product, None)
        self.n_seen.setdefault(product, 0)
        self.mu.setdefault(product, None)
        self.sigma.setdefault(product, None)
        self.slow_ema.setdefault(product, None)
        self.algo4_seen.setdefault(product, 0)
        self.algo4_target.setdefault(product, 0)

    def _load_trader_data(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            decoded = jsonpickle.decode(trader_data)
        except Exception:
            return
        if not isinstance(decoded, dict):
            return
        ema = decoded.get("EMA")
        n_seen = decoded.get("n_seen")
        mu = decoded.get("mu")
        sigma = decoded.get("sigma")
        slow_ema = decoded.get("slow_ema")
        algo4_seen = decoded.get("algo4_seen")
        algo4_target = decoded.get("algo4_target")
        if isinstance(ema, dict):
            self.EMA.update(ema)
        if isinstance(n_seen, dict):
            self.n_seen.update(n_seen)
        if isinstance(mu, dict):
            self.mu.update(mu)
        if isinstance(sigma, dict):
            self.sigma.update(sigma)
        if isinstance(slow_ema, dict):
            self.slow_ema.update(slow_ema)
        if isinstance(algo4_seen, dict):
            self.algo4_seen.update(algo4_seen)
        if isinstance(algo4_target, dict):
            self.algo4_target.update(algo4_target)

    def _dump_trader_data(self) -> str:
        return jsonpickle.encode(
            {
                "EMA": self.EMA,
                "n_seen": self.n_seen,
                "mu": self.mu,
                "sigma": self.sigma,
                "slow_ema": self.slow_ema,
                "algo4_seen": self.algo4_seen,
                "algo4_target": self.algo4_target,
            }
        )

    def update_EMA(self, product: str, mid: float, alpha: float) -> float:
        self._ensure_product_state(product)
        if self.EMA[product] is None:
            self.EMA[product] = float(mid)
        else:
            self.EMA[product] = alpha * mid + (1.0 - alpha) * self.EMA[product]
        self.n_seen[product] += 1
        return self.EMA[product]

    def compute_mid(self, state: TradingState, product: str):
        product = self._normalize_product(product)
        if product is None or product not in state.order_depths:
            return None
        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    def _clamp_quote_prices(
        self, bid_price: int, ask_price: int, order_depth: OrderDepth
    ) -> Tuple[int, int]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_ask is not None and bid_price >= best_ask:
            bid_price = best_ask - 1
        if best_bid is not None and ask_price <= best_bid:
            ask_price = best_bid + 1
        if bid_price >= ask_price:
            bid_price = ask_price - 1
        return bid_price, ask_price

    def get_position(self, symbol: str, state: TradingState) -> int:
        symbol = self._normalize_product(symbol)
        if symbol is None:
            return 0
        return state.position.get(symbol, 0)

    def _algorithm_1(self, state: TradingState, product: str) -> List[Order]:
        product = self._normalize_product(product)
        if product is None or product not in state.order_depths:
            return []

        order_depth = state.order_depths[product]
        mid = self.compute_mid(state, product)
        if mid is None:
            return []

        position = self.get_position(product, state)
        limit = self.POSITION_LIMITS[product]
        capacity_buy = limit - position
        capacity_sell = limit + position

        #==============================================================
        # Algorithm 1: Static Quote Market Making (Baseline)
        # Product-neutral: same spread and size for any product passed in.
        #==============================================================
        spread = 3
        size = 5
        orders: List[Order] = []

        bid_price = int(mid - spread)
        ask_price = int(mid + spread)
        bid_price, ask_price = self._clamp_quote_prices(
            bid_price, ask_price, order_depth
        )
        bid_size = min(size, capacity_buy)
        ask_size = min(size, capacity_sell)

        if bid_size > 0:
            orders.append(Order(product, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask_price, -ask_size))
        return orders

    def algorithm_2(self, state: TradingState, product: str) -> List[Order]:
        product = self._normalize_product(product)
        if product is None or product not in state.order_depths:
            return []

        order_depth = state.order_depths[product]
        mid = self.compute_mid(state, product)
        if mid is None:
            return []

        #==============================================================
        # Algorithm 2: EMA Fair + Inventory-Skewed Quoting
        # Product-neutral: same EMA length, spread, skew, and size.
        #==============================================================
        n = 20
        spread = 1
        size = 5
        alpha = 2 / (n + 1)
        fair_price = self.update_EMA(product, mid, alpha)
        if self.n_seen[product] < n:
            fair_price = mid

        position = self.get_position(product, state)
        limit = self.POSITION_LIMITS[product]
        capacity_buy = limit - position
        capacity_sell = limit + position

        skew = -1 * (position / limit) * spread
        skew = max(-spread, min(spread, skew))

        bid_price = int(fair_price + skew - spread)
        ask_price = int(fair_price + skew + spread)
        bid_price, ask_price = self._clamp_quote_prices(
            bid_price, ask_price, order_depth
        )
        bid_size = min(size, capacity_buy)
        ask_size = min(size, capacity_sell)

        orders: List[Order] = []
        if bid_size > 0:
            orders.append(Order(product, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask_price, -ask_size))
        return orders
    
    def algorithm_3(self, state: TradingState, product: str) -> List[Order]:
        product = self._normalize_product(product)
        if product is None or product not in state.order_depths:
            return []

        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None and best_ask is None:
            return []

        mid = self.compute_mid(state, product)
        if mid is None:
            return []

        if best_bid is None:
            fair_price = float(best_ask)
        elif best_ask is None:
            fair_price = float(best_bid)
        else:
            size_at_bid = abs(order_depth.buy_orders[best_bid])
            size_at_ask = abs(order_depth.sell_orders[best_ask])
            total_size = size_at_bid + size_at_ask
            if total_size > 0:
                fair_price = (
                    size_at_ask * best_bid + size_at_bid * best_ask
                ) / total_size
            else:
                fair_price = mid

        position = self.get_position(product, state)
        limit = self.POSITION_LIMITS[product]
        capacity_buy = limit - position
        capacity_sell = limit + position

        #==============================================================
        # Algorithm 3: Microprice + Aggressive Take-Then-Make
        # Product-neutral: same edges and sizes for any product passed in.
        #==============================================================
        take_edge = 1
        make_spread = 1
        take_size = 10
        make_size = 5
        orders: List[Order] = []

        remaining_take = take_size
        for ask_price in sorted(order_depth.sell_orders.keys()):
            if capacity_buy <= 0 or remaining_take <= 0:
                break
            if ask_price > fair_price - take_edge:
                break
            available_size = abs(order_depth.sell_orders[ask_price])
            quantity = min(available_size, capacity_buy, remaining_take)
            if quantity > 0:
                orders.append(Order(product, ask_price, quantity))
                capacity_buy -= quantity
                remaining_take -= quantity

        remaining_take = take_size
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if capacity_sell <= 0 or remaining_take <= 0:
                break
            if bid_price < fair_price + take_edge:
                break
            available_size = abs(order_depth.buy_orders[bid_price])
            quantity = min(available_size, capacity_sell, remaining_take)
            if quantity > 0:
                orders.append(Order(product, bid_price, -quantity))
                capacity_sell -= quantity
                remaining_take -= quantity

        if best_bid is None or best_ask is None or best_bid >= best_ask:
            return orders

        bid_price = int(fair_price - make_spread)
        ask_price = int(fair_price + make_spread)
        bid_price, ask_price = self._clamp_quote_prices(
            bid_price, ask_price, order_depth
        )

        bid_size = min(make_size, capacity_buy)
        ask_size = min(make_size, capacity_sell)
        if bid_size > 0:
            orders.append(Order(product, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask_price, -ask_size))

        return orders
    def algorithm_4(self, state: TradingState, product: str) -> List[Order]:
        product = self._normalize_product(product)
        if product is None or product not in state.order_depths:
            return []

        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return []

        mid = self.compute_mid(state, product)
        if mid is None:
            return []

        self._ensure_product_state(product)

        #==============================================================
        # Algorithm 4: Mean-Reversion Z-Score with Drift Handling
        # Osmium uses raw mid. Pepper uses mid minus a slow EMA to remove
        # its long drift before computing the z-score.
        #==============================================================
        W_fast = 200
        L_slow = 500
        z_in = 1.5
        z_out = 0.3
        z_max = 3.0
        q_max = 20
        sigma_min = 0.5

        if product == self.PRODUCTS["PEPPER"]:
            slow_alpha = 2 / (L_slow + 1)
            if self.slow_ema[product] is None:
                self.slow_ema[product] = mid
            else:
                self.slow_ema[product] = (
                    slow_alpha * mid + (1 - slow_alpha) * self.slow_ema[product]
                )
            series = mid - self.slow_ema[product]
            warmup = L_slow
        else:
            series = mid
            warmup = W_fast

        alpha = 2 / (W_fast + 1)
        old_mu = self.mu[product]
        old_var = self.sigma[product]
        self.algo4_seen[product] += 1

        if old_mu is None:
            self.mu[product] = series
            self.sigma[product] = 0.0
            return []

        if old_var is None:
            old_var = 0.0

        new_mu = alpha * series + (1 - alpha) * old_mu
        new_var = (1 - alpha) * (old_var + alpha * (series - old_mu) ** 2)
        self.mu[product] = new_mu
        self.sigma[product] = new_var

        if self.algo4_seen[product] < warmup:
            return []

        sigma = max(new_var, 0.0) ** 0.5
        if sigma < sigma_min:
            return []

        z_score = (series - new_mu) / sigma
        previous_target = self.algo4_target.get(product, 0)

        if z_score > z_in:
            target = -int(round(q_max * min(abs(z_score), z_max) / z_max))
        elif z_score < -z_in:
            target = int(round(q_max * min(abs(z_score), z_max) / z_max))
        elif abs(z_score) < z_out:
            target = 0
        else:
            target = previous_target

        self.algo4_target[product] = target

        position = self.get_position(product, state)
        trade_qty = target - position
        limit = self.POSITION_LIMITS[product]
        capacity_buy = limit - position
        capacity_sell = limit + position

        orders: List[Order] = []
        if trade_qty > 0:
            quantity = min(trade_qty, capacity_buy)
            if quantity > 0:
                orders.append(Order(product, best_bid, quantity))
        elif trade_qty < 0:
            quantity = min(-trade_qty, capacity_sell)
            if quantity > 0:
                orders.append(Order(product, best_ask, -quantity))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        self._load_trader_data(state.traderData)
        result: Dict[str, List[Order]] = {}
        self.orders = []

        for product in state.order_depths:
            if product == self.PRODUCTS["OSMIUM"]:
                orders = self._trade_osmium(state)
            elif product == self.PRODUCTS["PEPPER"]:
                orders = self._trade_pepper(state)
            else:
                orders = []

            result[product] = orders
            self.orders.extend(orders)

        return result, 0, self._dump_trader_data()

    def _trade_osmium(self, state: TradingState) -> List[Order]:
        product = self.PRODUCTS["OSMIUM"]
        if self.OSMIUM_ALGORITHM == 1:
            return self._algorithm_1(state, product)
        if self.OSMIUM_ALGORITHM == 2:
            return self.algorithm_2(state, product)
        if self.OSMIUM_ALGORITHM == 3:
            return self.algorithm_3(state, product)
        if self.OSMIUM_ALGORITHM == 4:
            return self.algorithm_4(state, product)
        return []

    def _trade_pepper(self, state: TradingState) -> List[Order]:
        product = self.PRODUCTS["PEPPER"]
        if self.PEPPER_ALGORITHM == 1:
            return self._algorithm_1(state, product)
        if self.PEPPER_ALGORITHM == 2:
            return self.algorithm_2(state, product)
        if self.PEPPER_ALGORITHM == 3:
            return self.algorithm_3(state, product)
        if self.PEPPER_ALGORITHM == 4:
            return self.algorithm_4(state, product)
        return []

    def _trade_adaptive(self, od: OrderDepth, position: int, best_bid: float, best_ask: float, mid: float) -> List[Order]:
        pass

    def get_orders(self, symbol: str) -> List[Order]:
        symbol = self._normalize_product(symbol)
        if symbol is None:
            return []
        return [order for order in self.orders if order.symbol == symbol]
