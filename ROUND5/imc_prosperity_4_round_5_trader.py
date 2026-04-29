from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import jsonpickle
import math


class Trader:
    """
    IMC Prosperity 4 Round 5 adaptive trader.

    Core idea:
    1. Estimate a microstructure fair value from the current order book.
    2. Learn within-family relative value relationships online.
    3. Score buyer/seller IDs by whether their trades predict future price movement.
    4. Combine relative-value, counterparty-flow, imbalance, and inventory signals.
    5. Use aggressive orders only when edge is strong; otherwise make safely around fair value.

    This file is intentionally product-name dynamic. It does not require hardcoded Round 5
    product names, so it can trade PEBBLES_*, SNACKPACK_*, UV_VISOR_*, GALAXY_SOUNDS_*,
    MICROCHIP_*, OXYGEN_SHAKE_*, PANEL_*, ROBOT_*, SLEEP_POD_*, TRANSLATOR_*, or any
    other products that appear in state.order_depths.
    """

    # Use conservative soft limits when exact round limits are unknown.
    # If you know the official per-product limits, put them here.
    POSITION_LIMITS: Dict[str, int] = {
        # Example overrides if needed:
        # "PEBBLES_XS": 50,
        # "PEBBLES_S": 50,
        # "PEBBLES_M": 50,
        # "PEBBLES_L": 50,
        # "PEBBLES_XL": 50,
    }

    DEFAULT_LIMIT = 50
    ACTIVE_LIMIT_FRACTION = 0.65

    # Online-learning / signal parameters
    FAIR_HISTORY = 80
    RESID_HISTORY = 80
    VOL_HISTORY = 80
    LOOKAHEAD_TICKS = 300
    MAX_PENDING_EVENTS = 5000
    MAX_SEEN_TRADES = 12000

    # Signal weights. These are deliberately stable, not overfit.
    W_RELVAL = 1.15
    W_FLOW = 1.55
    W_IMBALANCE = 0.35
    W_MOMENTUM = 0.18
    W_INVENTORY = 0.22

    # Entry thresholds
    TAKE_THRESHOLD = 0.75
    MAKE_THRESHOLD = 0.30
    MIN_EDGE_TO_CROSS = 1.0

    def __init__(self):
        pass

    # ----------------------------- persistence -----------------------------

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "fair_hist": {},          # product -> list[float]
            "resid_hist": {},         # product -> list[float]
            "counterparty": {},       # key -> {score, n}
            "pending_events": [],     # unresolved trade impact events
            "seen_trades": [],        # recent trade keys to prevent double count
            "last_timestamp": -1,
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
        # Trim histories so traderData stays small.
        for k in list(mem.get("fair_hist", {}).keys()):
            mem["fair_hist"][k] = mem["fair_hist"][k][-self.FAIR_HISTORY:]
        for k in list(mem.get("resid_hist", {}).keys()):
            mem["resid_hist"][k] = mem["resid_hist"][k][-self.RESID_HISTORY:]
        mem["pending_events"] = mem.get("pending_events", [])[-self.MAX_PENDING_EVENTS:]
        mem["seen_trades"] = mem.get("seen_trades", [])[-self.MAX_SEEN_TRADES:]
        return jsonpickle.encode(mem)

    # ----------------------------- book features -----------------------------

    def _best_bid_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int, Optional[int], int]:
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        bid_vol = depth.buy_orders.get(best_bid, 0) if best_bid is not None else 0
        ask_vol = -depth.sell_orders.get(best_ask, 0) if best_ask is not None else 0
        return best_bid, bid_vol, best_ask, ask_vol

    def _book_fair_value(self, depth: OrderDepth) -> Optional[float]:
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)

        if best_bid is None and best_ask is None:
            return None
        if best_bid is None:
            return float(best_ask)
        if best_ask is None:
            return float(best_bid)

        mid = (best_bid + best_ask) / 2.0
        if bid_vol + ask_vol <= 0:
            return mid

        # Microprice leans toward the side with less visible liquidity because
        # heavy bid volume means upward pressure, heavy ask volume means downward pressure.
        micro = (best_ask * bid_vol + best_bid * ask_vol) / (bid_vol + ask_vol)
        return 0.65 * micro + 0.35 * mid

    def _spread(self, depth: OrderDepth) -> float:
        best_bid, _, best_ask, _ = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return 2.0
        return max(1.0, float(best_ask - best_bid))

    def _imbalance(self, depth: OrderDepth) -> float:
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)
        den = bid_vol + ask_vol
        if best_bid is None or best_ask is None or den <= 0:
            return 0.0
        return (bid_vol - ask_vol) / den

    # ----------------------------- statistics -----------------------------

    def _append_hist(self, mem: Dict[str, Any], bucket: str, product: str, value: float, max_len: int):
        if product not in mem[bucket]:
            mem[bucket][product] = []
        mem[bucket][product].append(float(value))
        if len(mem[bucket][product]) > max_len:
            mem[bucket][product] = mem[bucket][product][-max_len:]

    def _mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 1.0
        m = self._mean(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        return max(1.0, math.sqrt(var))

    def _momentum(self, xs: List[float]) -> float:
        if len(xs) < 5:
            return 0.0
        fast = self._mean(xs[-3:])
        slow = self._mean(xs[-12:]) if len(xs) >= 12 else self._mean(xs)
        return fast - slow

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ----------------------------- product families -----------------------------

    def _family_key(self, product: str) -> str:
        parts = product.split("_")
        if len(parts) <= 1:
            return product

        # Common size/design suffixes should not define the family.
        known_suffixes = {
            "XS", "S", "M", "L", "XL", "XXL",
            "1", "2", "3", "4", "5",
            "A", "B", "C", "D",
            "RED", "BLUE", "GREEN", "YELLOW",
        }
        if parts[-1] in known_suffixes:
            return "_".join(parts[:-1])
        return parts[0]

    def _family_baselines(self, fairs: Dict[str, float]) -> Dict[str, float]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for product, fv in fairs.items():
            groups[self._family_key(product)].append(fv)
        return {fam: self._mean(vals) for fam, vals in groups.items() if vals}

    # ----------------------------- counterparty model -----------------------------

    def _trade_key(self, product: str, tr: Any) -> str:
        return f"{product}|{getattr(tr, 'timestamp', 0)}|{getattr(tr, 'buyer', '')}|{getattr(tr, 'seller', '')}|{getattr(tr, 'price', 0)}|{getattr(tr, 'quantity', 0)}"

    def _add_counterparty_score(self, mem: Dict[str, Any], product: str, trader_id: str, impact: float):
        if trader_id is None or trader_id == "":
            return

        fam = self._family_key(product)
        for key in (f"P:{product}:{trader_id}", f"F:{fam}:{trader_id}", f"G:{trader_id}"):
            rec = mem["counterparty"].get(key, {"score": 0.0, "n": 0})
            n = int(rec.get("n", 0))
            old = float(rec.get("score", 0.0))
            # Conservative EWMA. More observations increase trust, but slowly.
            alpha = 0.10 if n < 8 else 0.04
            rec["score"] = (1.0 - alpha) * old + alpha * self._clip(impact, -6.0, 6.0)
            rec["n"] = min(9999, n + 1)
            mem["counterparty"][key] = rec

    def _counterparty_predictiveness(self, mem: Dict[str, Any], product: str, trader_id: str) -> float:
        if trader_id is None or trader_id == "":
            return 0.0
        fam = self._family_key(product)
        keys = [f"P:{product}:{trader_id}", f"F:{fam}:{trader_id}", f"G:{trader_id}"]
        weights = [0.55, 0.30, 0.15]
        out = 0.0
        for key, w in zip(keys, weights):
            rec = mem["counterparty"].get(key)
            if not rec:
                continue
            n = int(rec.get("n", 0))
            trust = min(1.0, n / 10.0)
            out += w * trust * float(rec.get("score", 0.0))
        return out

    def _resolve_pending_trade_impacts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]):
        still_pending = []
        for ev in mem.get("pending_events", []):
            product = ev.get("product")
            if product not in fairs:
                still_pending.append(ev)
                continue
            age = timestamp - int(ev.get("timestamp", timestamp))
            if age < self.LOOKAHEAD_TICKS:
                still_pending.append(ev)
                continue

            current_fv = fairs[product]
            old_fv = float(ev.get("fair", current_fv))
            signed_future_move = float(ev.get("side", 0.0)) * (current_fv - old_fv)
            qty_scale = math.sqrt(max(1.0, abs(float(ev.get("quantity", 1.0)))))
            impact = signed_future_move * min(3.0, qty_scale)
            self._add_counterparty_score(mem, product, ev.get("trader", ""), impact)

        mem["pending_events"] = still_pending[-self.MAX_PENDING_EVENTS:]

    def _record_new_market_trades(self, mem: Dict[str, Any], state: TradingState, fairs: Dict[str, float]):
        seen = set(mem.get("seen_trades", []))
        new_seen = list(mem.get("seen_trades", []))

        for product, trades in state.market_trades.items():
            if product not in fairs:
                continue
            fair = fairs[product]
            for tr in trades:
                key = self._trade_key(product, tr)
                if key in seen:
                    continue
                seen.add(key)
                new_seen.append(key)

                price = float(getattr(tr, "price", fair))
                qty = int(getattr(tr, "quantity", 0))
                buyer = getattr(tr, "buyer", "")
                seller = getattr(tr, "seller", "")
                ts = int(getattr(tr, "timestamp", state.timestamp))

                # Infer aggressor direction using trade price versus current fair.
                # If trade is above fair, buyer likely paid up: bullish informed-flow candidate.
                # If below fair, seller likely hit bid: bearish informed-flow candidate.
                if price >= fair:
                    mem["pending_events"].append({
                        "product": product,
                        "timestamp": ts,
                        "trader": buyer,
                        "side": 1.0,
                        "fair": fair,
                        "quantity": qty,
                    })
                if price <= fair:
                    mem["pending_events"].append({
                        "product": product,
                        "timestamp": ts,
                        "trader": seller,
                        "side": -1.0,
                        "fair": fair,
                        "quantity": qty,
                    })

        mem["seen_trades"] = new_seen[-self.MAX_SEEN_TRADES:]
        mem["pending_events"] = mem.get("pending_events", [])[-self.MAX_PENDING_EVENTS:]

    def _current_flow_alpha(self, mem: Dict[str, Any], product: str, trades: List[Any], fair: float) -> float:
        alpha = 0.0
        for tr in trades:
            price = float(getattr(tr, "price", fair))
            qty = abs(int(getattr(tr, "quantity", 1)))
            scale = min(2.5, math.sqrt(max(1, qty)))
            buyer = getattr(tr, "buyer", "")
            seller = getattr(tr, "seller", "")

            if price >= fair:
                alpha += scale * self._counterparty_predictiveness(mem, product, buyer)
            if price <= fair:
                # A predictive seller means future downward movement, so subtract.
                alpha -= scale * self._counterparty_predictiveness(mem, product, seller)
        return self._clip(alpha, -8.0, 8.0)

    # ----------------------------- risk and execution -----------------------------

    def _limit(self, product: str) -> int:
        hard = self.POSITION_LIMITS.get(product, self.DEFAULT_LIMIT)
        return max(1, int(hard * self.ACTIVE_LIMIT_FRACTION))

    def _target_position(self, product: str, position: int, alpha: float, vol: float, spread: float) -> int:
        limit = self._limit(product)
        regime_penalty = 1.0
        if vol > 2.5 * spread:
            regime_penalty = 1.8
        elif vol > 1.5 * spread:
            regime_penalty = 1.35

        raw = alpha / max(1.0, vol * regime_penalty)
        target = int(round(self._clip(raw * limit * 0.42, -limit, limit)))

        # Do not flip violently on tiny signals.
        if abs(alpha) < self.MAKE_THRESHOLD:
            target = int(round(position * 0.70))
        return max(-limit, min(limit, target))

    def _take_liquidity(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        target: int,
        position: int,
        alpha: float,
        vol: float,
    ) -> Tuple[List[Order], int]:
        orders: List[Order] = []
        limit = self._limit(product)
        pos = position

        edge_needed = self.MIN_EDGE_TO_CROSS + 0.25 * vol

        # Buy underpriced asks or when alpha is strongly positive.
        if target > pos and depth.sell_orders:
            remaining = min(target - pos, limit - pos)
            for ask in sorted(depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                ask_vol = -depth.sell_orders[ask]
                edge = fair - ask
                if edge >= edge_needed or alpha >= self.TAKE_THRESHOLD + self._spread(depth):
                    qty = min(remaining, ask_vol)
                    if qty > 0:
                        orders.append(Order(product, ask, qty))
                        pos += qty
                        remaining -= qty

        # Sell overpriced bids or when alpha is strongly negative.
        if target < pos and depth.buy_orders:
            remaining = min(pos - target, limit + pos)
            for bid in sorted(depth.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break
                bid_vol = depth.buy_orders[bid]
                edge = bid - fair
                if edge >= edge_needed or alpha <= -self.TAKE_THRESHOLD - self._spread(depth):
                    qty = min(remaining, bid_vol)
                    if qty > 0:
                        orders.append(Order(product, bid, -qty))
                        pos -= qty
                        remaining -= qty

        return orders, pos

    def _make_market(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        target: int,
        position_after_takes: int,
        alpha: float,
        vol: float,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol, best_ask, ask_vol = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        limit = self._limit(product)
        pos = position_after_takes
        spread = max(1, best_ask - best_bid)

        # Inventory skew: if long, quote lower; if short, quote higher.
        inv_frac = pos / max(1, limit)
        skew = inv_frac * max(1.0, 0.35 * spread + 0.25 * vol)
        reservation = fair - skew

        # Alpha shifts reservation in the signal direction.
        reservation += self._clip(alpha, -2.0, 2.0) * 0.25

        # Wider in volatile/toxic regimes, tighter in calm wide-spread regimes.
        half_spread = max(1.0, 0.45 * spread, 0.45 * vol)
        bid_px = int(math.floor(reservation - half_spread))
        ask_px = int(math.ceil(reservation + half_spread))

        # Avoid crossing with passive quotes.
        bid_px = min(bid_px, best_ask - 1)
        ask_px = max(ask_px, best_bid + 1)

        # Join/improve if spread is wide enough.
        if spread >= 3:
            bid_px = max(bid_px, best_bid + 1)
            ask_px = min(ask_px, best_ask - 1)
        else:
            bid_px = min(bid_px, best_bid)
            ask_px = max(ask_px, best_ask)

        # Size quotes based on target and remaining inventory room.
        base_size = max(1, int(limit * 0.12))
        buy_room = limit - pos
        sell_room = limit + pos

        if target > pos:
            buy_size = min(buy_room, base_size + abs(target - pos))
            sell_size = min(sell_room, max(1, base_size // 2))
        elif target < pos:
            buy_size = min(buy_room, max(1, base_size // 2))
            sell_size = min(sell_room, base_size + abs(target - pos))
        else:
            buy_size = min(buy_room, base_size)
            sell_size = min(sell_room, base_size)

        # Do not make aggressively when signal is strongly one-sided against that side.
        if buy_size > 0 and alpha > -self.TAKE_THRESHOLD:
            orders.append(Order(product, bid_px, buy_size))
        if sell_size > 0 and alpha < self.TAKE_THRESHOLD:
            orders.append(Order(product, ask_px, -sell_size))

        return orders

    # ----------------------------- main loop -----------------------------

    def run(self, state: TradingState):
        mem = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        # 1. Compute current fair values.
        fairs: Dict[str, float] = {}
        spreads: Dict[str, float] = {}
        imbalances: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            fv = self._book_fair_value(depth)
            if fv is None:
                continue
            fairs[product] = fv
            spreads[product] = self._spread(depth)
            imbalances[product] = self._imbalance(depth)

        # 2. Update counterparty model using matured previous trades, then record new trades.
        self._resolve_pending_trade_impacts(mem, state.timestamp, fairs)
        self._record_new_market_trades(mem, state, fairs)

        # 3. Build family baselines and historical features.
        family_mu = self._family_baselines(fairs)

        for product, fair in fairs.items():
            self._append_hist(mem, "fair_hist", product, fair, self.FAIR_HISTORY)
            fam = self._family_key(product)
            residual = fair - family_mu.get(fam, fair)
            self._append_hist(mem, "resid_hist", product, residual, self.RESID_HISTORY)

        # 4. Generate orders product by product.
        for product, depth in state.order_depths.items():
            if product not in fairs:
                result[product] = []
                continue

            fair = fairs[product]
            spread = spreads.get(product, 1.0)
            position = int(state.position.get(product, 0))
            fair_hist = mem["fair_hist"].get(product, [])
            resid_hist = mem["resid_hist"].get(product, [])

            fam = self._family_key(product)
            residual = fair - family_mu.get(fam, fair)
            resid_mu = self._mean(resid_hist[:-1]) if len(resid_hist) > 3 else 0.0
            resid_std = self._std(resid_hist[:-1]) if len(resid_hist) > 5 else max(1.0, spread)
            z_resid = (residual - resid_mu) / max(1.0, resid_std)

            # Relative value: if product is high versus family, sell; low versus family, buy.
            relval_alpha = -z_resid

            # Counterparty flow: follow IDs whose trades have predicted future movement.
            trades = state.market_trades.get(product, [])
            flow_alpha = self._current_flow_alpha(mem, product, trades, fair)

            # Microstructure and momentum.
            imb_alpha = imbalances.get(product, 0.0)
            mom = self._momentum(fair_hist)
            vol = self._std([fair_hist[i] - fair_hist[i - 1] for i in range(1, len(fair_hist))][-self.VOL_HISTORY:]) if len(fair_hist) > 3 else max(1.0, spread)
            mom_alpha = mom / max(1.0, vol)

            # Inventory penalty: reduce longs, increase shorts.
            limit = self._limit(product)
            inv_alpha = -position / max(1.0, limit)

            alpha = (
                self.W_RELVAL * relval_alpha
                + self.W_FLOW * flow_alpha
                + self.W_IMBALANCE * imb_alpha
                + self.W_MOMENTUM * mom_alpha
                + self.W_INVENTORY * inv_alpha
            )
            alpha = self._clip(alpha, -12.0, 12.0)

            target = self._target_position(product, position, alpha, vol, spread)

            orders: List[Order] = []
            take_orders, pos_after = self._take_liquidity(product, depth, fair, target, position, alpha, vol)
            orders.extend(take_orders)

            # If the book is not extremely toxic, add passive quotes.
            # Avoid passive quoting in very high vol unless inventory needs repair.
            if vol <= 3.5 * spread or abs(position) > 0.45 * limit:
                orders.extend(self._make_market(product, depth, fair, target, pos_after, alpha, vol))

            result[product] = orders

        mem["last_timestamp"] = state.timestamp
        trader_data = self._save_memory(mem)
        conversions = 0
        return result, conversions, trader_data
