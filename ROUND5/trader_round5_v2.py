from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import jsonpickle
import math


class Trader:
    """
    IMC Prosperity 4 Round 5 improved adaptive trader.

    Improvements over v1:
    - Uses the actual Round 5 hard limit discovered from the uploaded run: 10 units/product.
    - Fixes the biggest v1 issue: no more 30+ unit desired positions that get rejected.
    - Learns product-specific residual baselines before trading relative value, so it does not
      confuse a permanently cheaper variant with a temporarily mispriced variant.
    - Adds stronger trend/leader-lagger logic because the uploaded run shows many products trend
      for long parts of the day.
    - Adds online model-quality scoring: if our alpha for a product has bad forward markouts,
      the bot automatically scales that product down.
    - Keeps counterparty-ID learning, but treats it as optional because many Round 5 logs expose
      blank buyer/seller IDs except SUBMISSION.
    """

    DEFAULT_LIMIT = 10
    ACTIVE_LIMIT_FRACTION = 1.00

    # Histories are intentionally compact to avoid traderData bloat.
    PRICE_HISTORY = 72
    RESID_HISTORY = 72
    RETURN_HISTORY = 72

    # Online markout windows.
    SIGNAL_LOOKAHEAD = 500
    TRADE_LOOKAHEAD = 500

    MAX_PENDING_SIGNALS = 900
    MAX_PENDING_TRADES = 900
    MAX_SEEN_TRADES = 2000

    # Warm-up prevents raw family level bias at the start.
    MIN_HISTORY_FOR_RELVAL = 22
    MIN_HISTORY_FOR_TREND = 14

    # Signal weights. Trend is stronger in v2 because the uploaded data showed meaningful
    # persistent moves; relative value is still useful but gated harder.
    W_TREND = 1.35
    W_RELVAL = 0.85
    W_FAMILY_TREND = 0.45
    W_FLOW = 1.20
    W_IMBALANCE = 0.22
    W_INVENTORY = 0.42

    TAKE_THRESHOLD = 1.15
    PASSIVE_THRESHOLD = 0.35
    RELVAL_Z_ENTER = 1.25

    # Exact product list from the uploaded Round 5 run. All had exchange limit 10.
    POSITION_LIMITS: Dict[str, int] = {
        "GALAXY_SOUNDS_BLACK_HOLES": 10,
        "GALAXY_SOUNDS_DARK_MATTER": 10,
        "GALAXY_SOUNDS_PLANETARY_RINGS": 10,
        "GALAXY_SOUNDS_SOLAR_FLAMES": 10,
        "GALAXY_SOUNDS_SOLAR_WINDS": 10,
        "MICROCHIP_CIRCLE": 10,
        "MICROCHIP_OVAL": 10,
        "MICROCHIP_RECTANGLE": 10,
        "MICROCHIP_SQUARE": 10,
        "MICROCHIP_TRIANGLE": 10,
        "OXYGEN_SHAKE_CHOCOLATE": 10,
        "OXYGEN_SHAKE_EVENING_BREATH": 10,
        "OXYGEN_SHAKE_GARLIC": 10,
        "OXYGEN_SHAKE_MINT": 10,
        "OXYGEN_SHAKE_MORNING_BREATH": 10,
        "PANEL_1X2": 10,
        "PANEL_1X4": 10,
        "PANEL_2X2": 10,
        "PANEL_2X4": 10,
        "PANEL_4X4": 10,
        "PEBBLES_L": 10,
        "PEBBLES_M": 10,
        "PEBBLES_S": 10,
        "PEBBLES_XL": 10,
        "PEBBLES_XS": 10,
        "ROBOT_DISHES": 10,
        "ROBOT_IRONING": 10,
        "ROBOT_LAUNDRY": 10,
        "ROBOT_MOPPING": 10,
        "ROBOT_VACUUMING": 10,
        "SLEEP_POD_COTTON": 10,
        "SLEEP_POD_LAMB_WOOL": 10,
        "SLEEP_POD_NYLON": 10,
        "SLEEP_POD_POLYESTER": 10,
        "SLEEP_POD_SUEDE": 10,
        "SNACKPACK_CHOCOLATE": 10,
        "SNACKPACK_PISTACHIO": 10,
        "SNACKPACK_RASPBERRY": 10,
        "SNACKPACK_STRAWBERRY": 10,
        "SNACKPACK_VANILLA": 10,
        "TRANSLATOR_ASTRO_BLACK": 10,
        "TRANSLATOR_ECLIPSE_CHARCOAL": 10,
        "TRANSLATOR_GRAPHITE_MIST": 10,
        "TRANSLATOR_SPACE_GRAY": 10,
        "TRANSLATOR_VOID_BLUE": 10,
        "UV_VISOR_AMBER": 10,
        "UV_VISOR_MAGENTA": 10,
        "UV_VISOR_ORANGE": 10,
        "UV_VISOR_RED": 10,
        "UV_VISOR_YELLOW": 10,
    }

    # ----------------------------- memory -----------------------------

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "price_hist": {},       # product -> recent fair values
            "resid_hist": {},       # product -> recent log-family residuals
            "ret_hist": {},         # product -> recent fair diffs
            "model_perf": {},       # product -> {"score": float, "n": int}
            "pending_signals": [],  # delayed alpha markout events
            "counterparty": {},     # key -> {"score": float, "n": int}
            "pending_trades": [],   # delayed trader-ID markout events
            "seen_trades": [],
        }

    def _load_memory(self, trader_data: str) -> Dict[str, Any]:
        if not trader_data:
            return self._default_memory()
        try:
            mem = jsonpickle.decode(trader_data)
            if not isinstance(mem, dict):
                return self._default_memory()
            base = self._default_memory()
            base.update(mem)
            return base
        except Exception:
            return self._default_memory()

    def _save_memory(self, mem: Dict[str, Any]) -> str:
        for bucket, max_len in (
            ("price_hist", self.PRICE_HISTORY),
            ("resid_hist", self.RESID_HISTORY),
            ("ret_hist", self.RETURN_HISTORY),
        ):
            for product in list(mem.get(bucket, {}).keys()):
                mem[bucket][product] = mem[bucket][product][-max_len:]

        mem["pending_signals"] = mem.get("pending_signals", [])[-self.MAX_PENDING_SIGNALS:]
        mem["pending_trades"] = mem.get("pending_trades", [])[-self.MAX_PENDING_TRADES:]
        mem["seen_trades"] = mem.get("seen_trades", [])[-self.MAX_SEEN_TRADES:]

        # Drop very stale/weak counterparty entries if the object grows.
        if len(mem.get("counterparty", {})) > 1200:
            items = list(mem["counterparty"].items())
            items.sort(key=lambda kv: abs(float(kv[1].get("score", 0.0))) * max(1, int(kv[1].get("n", 0))), reverse=True)
            mem["counterparty"] = dict(items[:800])

        return jsonpickle.encode(mem)

    # ----------------------------- utilities -----------------------------

    def _limit(self, product: str) -> int:
        return max(1, int(self.POSITION_LIMITS.get(product, self.DEFAULT_LIMIT) * self.ACTIVE_LIMIT_FRACTION))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 1.0
        m = self._mean(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        return max(1.0, math.sqrt(var))

    def _ema(self, xs: List[float], span: int) -> float:
        if not xs:
            return 0.0
        alpha = 2.0 / (span + 1.0)
        out = xs[0]
        for x in xs[1:]:
            out = alpha * x + (1.0 - alpha) * out
        return out

    def _append(self, mem: Dict[str, Any], bucket: str, product: str, value: float, max_len: int) -> None:
        if product not in mem[bucket]:
            mem[bucket][product] = []
        mem[bucket][product].append(float(value))
        if len(mem[bucket][product]) > max_len:
            mem[bucket][product] = mem[bucket][product][-max_len:]

    # ----------------------------- book features -----------------------------

    def _best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def _fair_value(self, depth: OrderDepth) -> Optional[float]:
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)
        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)

        mid = 0.5 * (best_bid + best_ask)
        if bid_vol + ask_vol <= 0:
            return mid

        micro = (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)

        # Add a light level-2/3 weighted book fair if available.
        bid_notional = 0.0
        bid_qty = 0.0
        for px, qty in depth.buy_orders.items():
            q = max(0, qty)
            bid_notional += px * q
            bid_qty += q

        ask_notional = 0.0
        ask_qty = 0.0
        for px, qty in depth.sell_orders.items():
            q = max(0, -qty)
            ask_notional += px * q
            ask_qty += q

        if bid_qty > 0 and ask_qty > 0:
            book_mid = 0.5 * (bid_notional / bid_qty + ask_notional / ask_qty)
            return 0.55 * micro + 0.30 * mid + 0.15 * book_mid
        return 0.70 * micro + 0.30 * mid

    def _spread(self, depth: OrderDepth) -> float:
        best_bid, _, best_ask, _ = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return 2.0
        return max(1.0, float(best_ask - best_bid))

    def _imbalance(self, depth: OrderDepth) -> float:
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return 0.0
        den = bid_vol + ask_vol
        if den <= 0:
            return 0.0
        return (bid_vol - ask_vol) / den

    # ----------------------------- families -----------------------------

    def _family_key(self, product: str) -> str:
        multi = ("GALAXY_SOUNDS", "OXYGEN_SHAKE", "SLEEP_POD")
        for pref in multi:
            if product.startswith(pref + "_"):
                return pref
        return product.split("_")[0]

    def _family_log_means(self, fairs: Dict[str, float]) -> Dict[str, float]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for product, fair in fairs.items():
            if fair > 0:
                groups[self._family_key(product)].append(math.log(fair))
        return {fam: self._mean(vals) for fam, vals in groups.items() if vals}

    def _family_trend(self, mem: Dict[str, Any], product: str) -> float:
        fam = self._family_key(product)
        vals = []
        for other, hist in mem.get("price_hist", {}).items():
            if other == product or self._family_key(other) != fam or len(hist) < self.MIN_HISTORY_FOR_TREND:
                continue
            vol = self._std([hist[i] - hist[i - 1] for i in range(1, len(hist))][-24:])
            vals.append((self._ema(hist[-6:], 4) - self._ema(hist[-24:], 14)) / max(1.0, vol))
        if not vals:
            return 0.0
        return self._clip(self._mean(vals), -3.0, 3.0)

    # ----------------------------- online model performance -----------------------------

    def _resolve_signal_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        still = []
        for ev in mem.get("pending_signals", []):
            product = ev.get("product")
            if product not in fairs:
                still.append(ev)
                continue
            if timestamp - int(ev.get("timestamp", timestamp)) < self.SIGNAL_LOOKAHEAD:
                still.append(ev)
                continue

            old_fair = float(ev.get("fair", fairs[product]))
            old_alpha = float(ev.get("alpha", 0.0))
            old_vol = max(1.0, float(ev.get("vol", 1.0)))
            markout = self._clip((fairs[product] - old_fair) * self._clip(old_alpha, -3.0, 3.0) / old_vol, -4.0, 4.0)

            rec = mem["model_perf"].get(product, {"score": 0.0, "n": 0})
            n = int(rec.get("n", 0))
            ew = 0.10 if n < 8 else 0.045
            rec["score"] = (1.0 - ew) * float(rec.get("score", 0.0)) + ew * markout
            rec["n"] = min(9999, n + 1)
            mem["model_perf"][product] = rec

        mem["pending_signals"] = still[-self.MAX_PENDING_SIGNALS:]

    def _product_scale(self, mem: Dict[str, Any], product: str) -> float:
        rec = mem.get("model_perf", {}).get(product)
        if not rec:
            return 1.0
        n = int(rec.get("n", 0))
        if n < 6:
            return 1.0
        score = float(rec.get("score", 0.0))
        # Bad markouts quickly reduce product risk; good markouts allow modestly larger confidence.
        if score < -0.70:
            return 0.20
        if score < -0.35:
            return 0.40
        if score < -0.10:
            return 0.70
        if score > 0.90:
            return 1.25
        if score > 0.35:
            return 1.12
        return 1.0

    # ----------------------------- counterparty model -----------------------------

    def _trade_key(self, product: str, tr: Any) -> str:
        return f"{product}|{getattr(tr, 'timestamp', 0)}|{getattr(tr, 'buyer', '')}|{getattr(tr, 'seller', '')}|{getattr(tr, 'price', 0)}|{getattr(tr, 'quantity', 0)}"

    def _add_counterparty_score(self, mem: Dict[str, Any], product: str, trader_id: str, impact: float) -> None:
        if not trader_id or trader_id == "SUBMISSION":
            return
        fam = self._family_key(product)
        for key, weight in ((f"P:{product}:{trader_id}", 1.0), (f"F:{fam}:{trader_id}", 0.65), (f"G:{trader_id}", 0.35)):
            rec = mem["counterparty"].get(key, {"score": 0.0, "n": 0})
            n = int(rec.get("n", 0))
            ew = 0.12 if n < 8 else 0.05
            rec["score"] = (1.0 - ew) * float(rec.get("score", 0.0)) + ew * self._clip(weight * impact, -5.0, 5.0)
            rec["n"] = min(9999, n + 1)
            mem["counterparty"][key] = rec

    def _counterparty_score(self, mem: Dict[str, Any], product: str, trader_id: str) -> float:
        if not trader_id or trader_id == "SUBMISSION":
            return 0.0
        fam = self._family_key(product)
        keys = (f"P:{product}:{trader_id}", f"F:{fam}:{trader_id}", f"G:{trader_id}")
        weights = (0.58, 0.30, 0.12)
        out = 0.0
        for key, w in zip(keys, weights):
            rec = mem.get("counterparty", {}).get(key)
            if not rec:
                continue
            n = int(rec.get("n", 0))
            trust = min(1.0, n / 8.0)
            out += w * trust * float(rec.get("score", 0.0))
        return self._clip(out, -4.0, 4.0)

    def _resolve_trade_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        still = []
        for ev in mem.get("pending_trades", []):
            product = ev.get("product")
            if product not in fairs:
                still.append(ev)
                continue
            if timestamp - int(ev.get("timestamp", timestamp)) < self.TRADE_LOOKAHEAD:
                still.append(ev)
                continue
            old_fair = float(ev.get("fair", fairs[product]))
            side = float(ev.get("side", 0.0))
            vol = max(1.0, float(ev.get("vol", 1.0)))
            impact = side * (fairs[product] - old_fair) / vol
            self._add_counterparty_score(mem, product, ev.get("trader", ""), impact)
        mem["pending_trades"] = still[-self.MAX_PENDING_TRADES:]

    def _record_market_trades(self, mem: Dict[str, Any], state: TradingState, fairs: Dict[str, float], vols: Dict[str, float]) -> None:
        seen = set(mem.get("seen_trades", []))
        seen_list = list(mem.get("seen_trades", []))

        for product, trades in state.market_trades.items():
            if product not in fairs:
                continue
            fair = fairs[product]
            vol = vols.get(product, 1.0)
            for tr in trades:
                key = self._trade_key(product, tr)
                if key in seen:
                    continue
                seen.add(key)
                seen_list.append(key)

                buyer = getattr(tr, "buyer", "")
                seller = getattr(tr, "seller", "")
                if (not buyer or buyer == "SUBMISSION") and (not seller or seller == "SUBMISSION"):
                    continue

                price = float(getattr(tr, "price", fair))
                qty = max(1, abs(int(getattr(tr, "quantity", 1))))
                scale = min(2.0, math.sqrt(qty))

                # If buyer paid above fair, treat as buyer-initiated. If seller hit below fair,
                # treat as seller-initiated.
                if buyer and buyer != "SUBMISSION" and price >= fair:
                    mem["pending_trades"].append({
                        "product": product,
                        "timestamp": state.timestamp,
                        "trader": buyer,
                        "side": 1.0,
                        "fair": fair,
                        "vol": vol / scale,
                    })
                if seller and seller != "SUBMISSION" and price <= fair:
                    mem["pending_trades"].append({
                        "product": product,
                        "timestamp": state.timestamp,
                        "trader": seller,
                        "side": -1.0,
                        "fair": fair,
                        "vol": vol / scale,
                    })

        mem["seen_trades"] = seen_list[-self.MAX_SEEN_TRADES:]
        mem["pending_trades"] = mem.get("pending_trades", [])[-self.MAX_PENDING_TRADES:]

    def _flow_alpha(self, mem: Dict[str, Any], product: str, trades: List[Any], fair: float) -> float:
        out = 0.0
        for tr in trades:
            price = float(getattr(tr, "price", fair))
            qty = max(1, abs(int(getattr(tr, "quantity", 1))))
            scale = min(1.7, math.sqrt(qty))
            buyer = getattr(tr, "buyer", "")
            seller = getattr(tr, "seller", "")
            if price >= fair:
                out += scale * self._counterparty_score(mem, product, buyer)
            if price <= fair:
                out -= scale * self._counterparty_score(mem, product, seller)
        return self._clip(out, -5.0, 5.0)

    # ----------------------------- alpha -----------------------------

    def _trend_alpha(self, hist: List[float], vol: float) -> float:
        if len(hist) < self.MIN_HISTORY_FOR_TREND:
            return 0.0
        fast = self._ema(hist[-7:], 5)
        slow = self._ema(hist[-28:], 18) if len(hist) >= 28 else self._ema(hist, 14)
        slope = (fast - slow) / max(1.0, vol)

        # Confirm with last few returns so we do not chase one bad spike.
        recent = (hist[-1] - hist[-5]) / max(1.0, vol) if len(hist) >= 5 else 0.0
        if slope * recent < 0:
            slope *= 0.45
        return self._clip(0.75 * slope + 0.25 * recent, -3.5, 3.5)

    def _relval_alpha(self, mem: Dict[str, Any], product: str, current_resid: float, trend_alpha: float) -> float:
        hist = mem.get("resid_hist", {}).get(product, [])
        if len(hist) < self.MIN_HISTORY_FOR_RELVAL:
            return 0.0

        # Use previous observations only; current residual is not in hist yet during alpha calculation.
        mu = self._mean(hist[-60:])
        sd = self._std(hist[-60:])
        z = (current_resid - mu) / max(0.00015, sd)

        if abs(z) < self.RELVAL_Z_ENTER:
            return 0.0

        # If a strong trend is pushing the same way as the residual, mean-reversion is dangerous.
        alpha = -z
        if alpha * trend_alpha < 0 and abs(trend_alpha) > 1.0:
            alpha *= 0.35
        elif alpha * trend_alpha > 0:
            alpha *= 1.15

        return self._clip(alpha, -3.5, 3.5)

    def _target_position(self, product: str, position: int, alpha: float, vol: float, spread: float, scale: float) -> int:
        limit = self._limit(product)
        if abs(alpha) < self.PASSIVE_THRESHOLD:
            # Mean-revert inventory toward zero when edge is weak.
            return int(round(position * 0.55))

        vol_penalty = 1.0 + 0.08 * max(0.0, vol - spread)
        raw = alpha / vol_penalty

        # Use only part of max limit unless signal is very strong.
        desired = raw * limit * 0.58 * scale
        if abs(alpha) > 2.2:
            desired = raw * limit * 0.78 * scale

        return int(self._clip(round(desired), -limit, limit))

    # ----------------------------- execution -----------------------------

    def _add_order(
        self,
        orders: List[Order],
        product: str,
        price: int,
        qty: int,
        start_pos: int,
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        if qty == 0:
            return buy_used, sell_used
        limit = self._limit(product)
        if qty > 0:
            room = limit - (start_pos + buy_used)
            q = min(qty, room)
            if q > 0:
                orders.append(Order(product, int(price), int(q)))
                buy_used += q
        else:
            room = limit + (start_pos - sell_used)
            q = min(-qty, room)
            if q > 0:
                orders.append(Order(product, int(price), int(-q)))
                sell_used += q
        return buy_used, sell_used

    def _execute(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        alpha: float,
        target: int,
        position: int,
        vol: float,
        spread: float,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_used = 0
        sell_used = 0

        # Forecasted fair move in ticks. This converts dimensionless alpha into an execution edge.
        move = self._clip(alpha, -3.0, 3.0) * max(1.0, min(4.0, 0.45 * vol + 0.55 * spread))
        predicted = fair + move

        # Aggressive taker leg only when the predicted edge beats spread/slippage.
        edge_cost = max(1.0, 0.38 * spread + 0.18 * vol)

        if target > position and alpha > self.TAKE_THRESHOLD:
            need = target - position
            for ask in sorted(depth.sell_orders.keys()):
                if need <= 0:
                    break
                ask_qty = -depth.sell_orders[ask]
                if ask <= predicted - edge_cost:
                    q = min(need, ask_qty)
                    before = buy_used
                    buy_used, sell_used = self._add_order(orders, product, ask, q, position, buy_used, sell_used)
                    filled = buy_used - before
                    need -= filled

        if target < position and alpha < -self.TAKE_THRESHOLD:
            need = position - target
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if need <= 0:
                    break
                bid_qty = depth.buy_orders[bid]
                if bid >= predicted + edge_cost:
                    q = min(need, bid_qty)
                    before = sell_used
                    buy_used, sell_used = self._add_order(orders, product, bid, -q, position, buy_used, sell_used)
                    filled = sell_used - before
                    need -= filled

        # Passive leg. Quote with inventory skew; make on both sides only in low-to-normal alpha.
        inv = position / max(1, self._limit(product))
        reservation = fair + 0.35 * move - inv * max(1.0, 0.50 * spread + 0.20 * vol)

        quote_half = max(1.0, 0.42 * spread, 0.22 * vol)
        bid_px = int(math.floor(reservation - quote_half))
        ask_px = int(math.ceil(reservation + quote_half))

        # Do not cross with passive orders.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)

        # Improve if spread is wide enough; otherwise join.
        if best_ask - best_bid >= 3:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)
        else:
            bid_px = min(bid_px, best_bid)
            ask_px = max(ask_px, best_ask)

        # Passive size prefers moving toward target, but still quotes an inventory-repair side.
        limit = self._limit(product)
        base = 1 if spread <= 2 else 2

        pos_after_worst_buy = position + buy_used
        pos_after_worst_sell = position - sell_used

        buy_room = limit - pos_after_worst_buy
        sell_room = limit + pos_after_worst_sell

        if target > position:
            buy_size = min(buy_room, max(base, target - position - buy_used))
            sell_size = min(sell_room, 1 if alpha > 0.75 else base)
        elif target < position:
            buy_size = min(buy_room, 1 if alpha < -0.75 else base)
            sell_size = min(sell_room, max(base, position - target - sell_used))
        else:
            buy_size = min(buy_room, base)
            sell_size = min(sell_room, base)

        # Avoid making the wrong side during strong one-sided flow.
        if buy_size > 0 and alpha > -1.35:
            buy_used, sell_used = self._add_order(orders, product, bid_px, int(buy_size), position, buy_used, sell_used)
        if sell_size > 0 and alpha < 1.35:
            buy_used, sell_used = self._add_order(orders, product, ask_px, -int(sell_size), position, buy_used, sell_used)

        return orders

    # ----------------------------- main -----------------------------

    def run(self, state: TradingState):
        mem = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        fairs: Dict[str, float] = {}
        spreads: Dict[str, float] = {}
        imbs: Dict[str, float] = {}
        vols: Dict[str, float] = {}

        # Current book features.
        for product, depth in state.order_depths.items():
            fair = self._fair_value(depth)
            if fair is None:
                continue
            fairs[product] = fair
            spreads[product] = self._spread(depth)
            imbs[product] = self._imbalance(depth)

            hist = mem.get("price_hist", {}).get(product, [])
            if len(hist) >= 3:
                rets = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
                vols[product] = self._std(rets[-self.RETURN_HISTORY:])
            else:
                vols[product] = max(1.0, spreads[product])

        # Resolve delayed online learning before computing the next signal.
        self._resolve_signal_markouts(mem, state.timestamp, fairs)
        self._resolve_trade_markouts(mem, state.timestamp, fairs)
        self._record_market_trades(mem, state, fairs, vols)

        family_logs = self._family_log_means(fairs)

        # Generate orders.
        for product, depth in state.order_depths.items():
            if product not in fairs:
                result[product] = []
                continue

            fair = fairs[product]
            spread = spreads.get(product, 1.0)
            vol = vols.get(product, spread)
            position = int(state.position.get(product, 0))

            hist = mem.get("price_hist", {}).get(product, [])

            if fair > 0:
                fam_resid = math.log(fair) - family_logs.get(self._family_key(product), math.log(fair))
            else:
                fam_resid = 0.0

            trend = self._trend_alpha(hist + [fair], vol)
            relval = self._relval_alpha(mem, product, fam_resid, trend)
            fam_trend = self._family_trend(mem, product)
            flow = self._flow_alpha(mem, product, state.market_trades.get(product, []), fair)
            imb = imbs.get(product, 0.0)
            inventory = -position / max(1, self._limit(product))

            alpha = (
                self.W_TREND * trend
                + self.W_RELVAL * relval
                + self.W_FAMILY_TREND * fam_trend
                + self.W_FLOW * flow
                + self.W_IMBALANCE * imb
                + self.W_INVENTORY * inventory
            )
            alpha = self._clip(alpha, -6.0, 6.0)

            scale = self._product_scale(mem, product)
            target = self._target_position(product, position, alpha, vol, spread, scale)

            result[product] = self._execute(product, depth, fair, alpha, target, position, vol, spread)

            # Store alpha markout event only when signal is meaningful.
            if abs(alpha) > 0.30:
                mem["pending_signals"].append({
                    "product": product,
                    "timestamp": state.timestamp,
                    "fair": fair,
                    "alpha": alpha,
                    "vol": max(1.0, vol),
                })

        # Update histories at the end of the tick, after alpha used previous state.
        for product, fair in fairs.items():
            old_hist = mem.get("price_hist", {}).get(product, [])
            if old_hist:
                self._append(mem, "ret_hist", product, fair - old_hist[-1], self.RETURN_HISTORY)
            self._append(mem, "price_hist", product, fair, self.PRICE_HISTORY)

            if fair > 0:
                resid = math.log(fair) - family_logs.get(self._family_key(product), math.log(fair))
                self._append(mem, "resid_hist", product, resid, self.RESID_HISTORY)

        trader_data = self._save_memory(mem)
        conversions = 0
        return result, conversions, trader_data
