from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import jsonpickle
import math


class Trader:
    """
    IMC Prosperity 4 Round 5 v31 isolated microchip-square lift.

    This version is built from the only architecture that actually worked well in your
    uploaded runs: adaptive counterparty-aware relative value plus safe market making.

    Changes versus v10:
    - Keeps the original family-relative value and microprice engine.
    - Fixes the true Round 5 hard limit: 10 per product.
    - Clips every order before it is sent, so the exchange should not reject whole product legs.
    - Keeps v10 product priors, leak cuts, and execution shape.
    - Adds non-overfit anonymous tape-flow following and quote-toxicity guards based on live market structure.
    - Does not use exact timestamp exits, product-level curve-fitted peak locks, or one-run PnL stops.
    - Uses warm-up before trusting relative value so permanent level differences are not
      mistaken for mispricing.
    - Adds online alpha markout scoring. If a product starts losing live, its size decays.
    - Keeps trader-ID learning as an overlay, not the main signal, because the uploaded logs
      mostly had blank IDs.
    """

    HARD_LIMIT = 10
    FAIR_HISTORY = 90
    RESID_HISTORY = 90
    RET_HISTORY = 60
    MARKOUT_DELAY = 500
    CP_DELAY = 500

    MIN_REL_HISTORY = 18
    MIN_PRICE_HISTORY = 16

    MAX_PENDING_ALPHA = 1200
    MAX_PENDING_CP = 1600
    MAX_SEEN_TRADES = 5000

    FAMILY_PREFIXES = (
        "GALAXY_SOUNDS",
        "MICROCHIP",
        "OXYGEN_SHAKE",
        "PANEL",
        "PEBBLES",
        "ROBOT",
        "SLEEP_POD",
        "SNACKPACK",
        "TRANSLATOR",
        "UV_VISOR",
    )

    # These were persistent large losers under the profitable v1-style engine.
    # They are flattened if accidentally held and otherwise not traded.
    AVOID = {
        # Catastrophic losers from v1/v2 that v4 correctly removed.
        "TRANSLATOR_SPACE_GRAY",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "ROBOT_DISHES",
        "PANEL_2X2",
        "PANEL_1X2",
        "OXYGEN_SHAKE_MINT",
        "SLEEP_POD_LAMB_WOOL",

        # Surgical v6 pruning: these were the remaining negative contributors in
        # the 28.86k v4 run. Do not change the rest of v4's execution profile.
        "MICROCHIP_RECTANGLE",
        "PANEL_1X4",
        "UV_VISOR_YELLOW",
        "UV_VISOR_ORANGE",
        "MICROCHIP_TRIANGLE",
        "ROBOT_LAUNDRY",
        "PEBBLES_M",
        "PEBBLES_S",
        "OXYGEN_SHAKE_EVENING_BREATH",

        # v10 leak cut: v9 stayed under 30k mainly because these three
        # remaining products had negative realized expectancy in the latest run.
        "SLEEP_POD_COTTON",
        "SNACKPACK_RASPBERRY",
        "ROBOT_IRONING",
    }

    # Strong products from the 18k-PnL run and/or the collapsed run when the product itself
    # still showed strong edge. These receive higher caps but still obey HARD_LIMIT=10.
    STRONG = {
        "MICROCHIP_SQUARE",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "SLEEP_POD_COTTON",
        "UV_VISOR_RED",
        "SLEEP_POD_NYLON",
        "PEBBLES_XS",
        "MICROCHIP_RECTANGLE",
        "UV_VISOR_ORANGE",
        "TRANSLATOR_GRAPHITE_MIST",
        "OXYGEN_SHAKE_CHOCOLATE",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "ROBOT_IRONING",
        "TRANSLATOR_ASTRO_BLACK",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "MICROCHIP_TRIANGLE",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "PANEL_4X4",
        "ROBOT_MOPPING",
        "OXYGEN_SHAKE_GARLIC",
    }

    # v31 isolated, stateless, additive-only microstructure lift.
    # Previous basket-wide microstructure versions lost because they changed too many
    # products and increased bad participation. This keeps v17 exact everywhere except
    # MICROCHIP_SQUARE, the only product where the last run showed a large positive
    # response to microstructure lift.
    MICRO_LIFT_ONLY = {"MICROCHIP_SQUARE"}

    # Names that were unstable: trade them, but only with smaller caps unless live markouts improve.
    CAUTIOUS = {
        "PANEL_2X4",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_VANILLA",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "SLEEP_POD_POLYESTER",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "PEBBLES_L",
        "PEBBLES_M",
        "PEBBLES_S",
        "PEBBLES_XL",
        "ROBOT_LAUNDRY",
        "ROBOT_VACUUMING",
        "SLEEP_POD_SUEDE",
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_YELLOW",
        "PANEL_1X4",
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "TRANSLATOR_VOID_BLUE",
    }

    # Some products in the second run responded better to trend than to mean reversion.
    TREND_NAMES = {
        "MICROCHIP_RECTANGLE",
        "OXYGEN_SHAKE_GARLIC",
        "PANEL_4X4",
        "TRANSLATOR_VOID_BLUE",
        "UV_VISOR_AMBER",
        "SLEEP_POD_SUEDE",
        "PANEL_1X4",
        "ROBOT_VACUUMING",
    }

    FAMILY_NAMES = {
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_GRAPHITE_MIST",
        "UV_VISOR_RED",
        "UV_VISOR_ORANGE",
        "ROBOT_IRONING",
        "PANEL_4X4",
    }

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "fair_hist": {},
            "resid_hist": {},
            "ret_hist": {},
            "perf": {},
            "pending_alpha": [],
            "cp_score": {},
            "pending_cp": [],
            "seen_trades": [],
            "tape_flow": {},
        }

    def _load_memory(self, data: str) -> Dict[str, Any]:
        if not data:
            return self._default_memory()
        try:
            mem = jsonpickle.decode(data)
            if not isinstance(mem, dict):
                return self._default_memory()
            base = self._default_memory()
            base.update(mem)
            return base
        except Exception:
            return self._default_memory()

    def _save_memory(self, mem: Dict[str, Any]) -> str:
        for bucket, keep in (
            ("fair_hist", self.FAIR_HISTORY),
            ("resid_hist", self.RESID_HISTORY),
            ("ret_hist", self.RET_HISTORY),
        ):
            for product in list(mem.get(bucket, {}).keys()):
                mem[bucket][product] = mem[bucket][product][-keep:]

        mem["pending_alpha"] = mem.get("pending_alpha", [])[-self.MAX_PENDING_ALPHA:]
        mem["pending_cp"] = mem.get("pending_cp", [])[-self.MAX_PENDING_CP:]
        mem["seen_trades"] = mem.get("seen_trades", [])[-self.MAX_SEEN_TRADES:]

        if len(mem.get("cp_score", {})) > 1200:
            items = sorted(
                mem["cp_score"].items(),
                key=lambda kv: abs(float(kv[1].get("s", 0.0))) * max(1, int(kv[1].get("n", 0))),
                reverse=True,
            )
            mem["cp_score"] = dict(items[:800])

        return jsonpickle.encode(mem)

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 1.0
        m = self._mean(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        return max(1e-9, math.sqrt(var))

    def _ema(self, xs: List[float], span: int) -> float:
        if not xs:
            return 0.0
        a = 2.0 / (span + 1.0)
        out = xs[0]
        for x in xs[1:]:
            out = a * x + (1.0 - a) * out
        return out

    def _append(self, mem: Dict[str, Any], bucket: str, product: str, value: float, keep: int) -> None:
        if product not in mem[bucket]:
            mem[bucket][product] = []
        mem[bucket][product].append(float(value))
        if len(mem[bucket][product]) > keep:
            mem[bucket][product] = mem[bucket][product][-keep:]

    def _family(self, product: str) -> str:
        for pref in self.FAMILY_PREFIXES:
            if product.startswith(pref + "_"):
                return pref
        return product.split("_")[0]

    def _best(self, depth: OrderDepth) -> Tuple[Optional[int], int, Optional[int], int]:
        bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        bid_v = depth.buy_orders.get(bid, 0) if bid is not None else 0
        ask_v = -depth.sell_orders.get(ask, 0) if ask is not None else 0
        return bid, bid_v, ask, ask_v

    def _spread(self, depth: OrderDepth) -> float:
        bid, _, ask, _ = self._best(depth)
        if bid is None or ask is None:
            return 2.0
        return max(1.0, float(ask - bid))

    def _fair_value(self, depth: OrderDepth) -> Optional[float]:
        bid, bid_v, ask, ask_v = self._best(depth)
        if bid is None and ask is None:
            return None
        if bid is None:
            return float(ask)
        if ask is None:
            return float(bid)

        mid = 0.5 * (bid + ask)
        if bid_v + ask_v <= 0:
            return mid

        micro = (ask * bid_v + bid * ask_v) / (bid_v + ask_v)

        # Light level-2/3 fair value dampens one-level spoofiness without overfitting.
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
            depth_mid = 0.5 * (bid_notional / bid_qty + ask_notional / ask_qty)
            return 0.60 * micro + 0.30 * mid + 0.10 * depth_mid
        return 0.65 * micro + 0.35 * mid

    def _imbalance(self, depth: OrderDepth) -> float:
        bid, bid_v, ask, ask_v = self._best(depth)
        if bid is None or ask is None or bid_v + ask_v <= 0:
            return 0.0
        return (bid_v - ask_v) / (bid_v + ask_v)

    def _multi_level_imbalance(self, depth: OrderDepth) -> float:
        # Stateless depth-1/2/3 order-book imbalance. Positive means buy pressure.
        # This is intentionally used only for MICROCHIP_SQUARE in v31.
        if not depth.buy_orders or not depth.sell_orders:
            return 0.0
        bids = sorted(depth.buy_orders.items(), key=lambda kv: kv[0], reverse=True)[:3]
        asks = sorted(depth.sell_orders.items(), key=lambda kv: kv[0])[:3]
        weights = (0.62, 0.26, 0.12)
        score = 0.0
        wsum = 0.0
        for i in range(min(len(bids), len(asks), 3)):
            bv = max(0, int(bids[i][1]))
            av = max(0, -int(asks[i][1]))
            den = bv + av
            if den <= 0:
                continue
            w = weights[i]
            score += w * ((bv - av) / den)
            wsum += w
        if wsum <= 0:
            return 0.0
        return self._clip(score / wsum, -1.0, 1.0)

    def _family_log_means(self, fairs: Dict[str, float]) -> Dict[str, float]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for product, fair in fairs.items():
            if fair > 0:
                groups[self._family(product)].append(math.log(fair))
        return {fam: self._mean(vals) for fam, vals in groups.items() if vals}

    def _resolve_alpha_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        pending = []
        for ev in mem.get("pending_alpha", []):
            product = ev.get("p")
            if product not in fairs:
                pending.append(ev)
                continue
            if timestamp - int(ev.get("t", timestamp)) < self.MARKOUT_DELAY:
                pending.append(ev)
                continue

            move = fairs[product] - float(ev.get("fair", fairs[product]))
            sign = float(ev.get("sign", 0.0))
            vol = max(1.0, float(ev.get("vol", 1.0)))
            markout = self._clip(sign * move / vol, -4.0, 4.0)

            rec = mem["perf"].get(product, {"s": 0.0, "n": 0})
            n = int(rec.get("n", 0))
            lr = 0.10 if n < 8 else 0.045
            rec["s"] = (1.0 - lr) * float(rec.get("s", 0.0)) + lr * markout
            rec["n"] = min(9999, n + 1)
            mem["perf"][product] = rec

        mem["pending_alpha"] = pending[-self.MAX_PENDING_ALPHA:]

    def _product_prior_scale(self, product: str) -> float:
        if product in self.AVOID:
            return 0.0
        if product in self.STRONG:
            return 1.18
        if product in self.CAUTIOUS:
            return 0.72
        return 0.55

    def _live_perf_scale(self, mem: Dict[str, Any], product: str) -> float:
        rec = mem.get("perf", {}).get(product)
        if not rec or int(rec.get("n", 0)) < 7:
            return 1.0
        s = float(rec.get("s", 0.0))
        if s < -0.75:
            return 0.25
        if s < -0.35:
            return 0.50
        if s < -0.12:
            return 0.75
        if s > 0.75:
            return 1.18
        if s > 0.35:
            return 1.08
        return 1.0

    def _product_scale(self, mem: Dict[str, Any], product: str) -> float:
        return self._product_prior_scale(product) * self._live_perf_scale(mem, product)

    def _product_cap(self, product: str, scale: float) -> int:
        if scale <= 0:
            return 0
        if product in self.STRONG:
            return 8
        if product in self.CAUTIOUS:
            return 5
        return 4

    def _trade_key(self, product: str, tr: Any) -> str:
        return (
            f"{product}|{getattr(tr, 'timestamp', '')}|{getattr(tr, 'buyer', '')}|"
            f"{getattr(tr, 'seller', '')}|{getattr(tr, 'price', '')}|{getattr(tr, 'quantity', '')}"
        )

    def _update_cp_score(self, mem: Dict[str, Any], key: str, score: float) -> None:
        if not key:
            return
        rec = mem["cp_score"].get(key, {"s": 0.0, "n": 0})
        n = int(rec.get("n", 0))
        lr = 0.08 if n < 8 else 0.035
        rec["s"] = (1.0 - lr) * float(rec.get("s", 0.0)) + lr * self._clip(score, -4.0, 4.0)
        rec["n"] = min(9999, n + 1)
        mem["cp_score"][key] = rec

    def _resolve_cp_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        pending = []
        for ev in mem.get("pending_cp", []):
            product = ev.get("p")
            if product not in fairs:
                pending.append(ev)
                continue
            if timestamp - int(ev.get("t", timestamp)) < self.CP_DELAY:
                pending.append(ev)
                continue

            move = fairs[product] - float(ev.get("fair", fairs[product]))
            sign = float(ev.get("sign", 0.0))
            vol = max(1.0, float(ev.get("vol", 1.0)))
            markout = sign * move / vol
            self._update_cp_score(mem, ev.get("k", ""), markout)

        mem["pending_cp"] = pending[-self.MAX_PENDING_CP:]

    def _cp_lookup(self, mem: Dict[str, Any], product: str, trader_id: str) -> float:
        if not trader_id or trader_id == "SUBMISSION":
            return 0.0
        fam = self._family(product)
        keys = (
            (f"P|{product}|{trader_id}", 0.65),
            (f"F|{fam}|{trader_id}", 0.25),
            (f"G|{trader_id}", 0.10),
        )
        out = 0.0
        for key, w in keys:
            rec = mem.get("cp_score", {}).get(key)
            if not rec:
                continue
            n = int(rec.get("n", 0))
            trust = min(1.0, n / 10.0)
            out += w * trust * float(rec.get("s", 0.0))
        return self._clip(out, -3.0, 3.0)

    def _counterparty_flow(self, mem: Dict[str, Any], state: TradingState, product: str, fair: float, vol: float) -> float:
        trades = state.market_trades.get(product, [])

        tape = mem.setdefault("tape_flow", {})
        # Decay stale tape-flow every tick this product is processed, even if there are no prints.
        tape[product] = 0.88 * float(tape.get(product, 0.0))

        if not trades:
            return 0.0

        seen = set(mem.get("seen_trades", []))
        seen_list = list(mem.get("seen_trades", []))
        fam = self._family(product)
        flow = 0.0

        for tr in trades:
            key = self._trade_key(product, tr)
            is_new = key not in seen
            if is_new:
                seen.add(key)
                seen_list.append(key)

            buyer = getattr(tr, "buyer", "") or ""
            seller = getattr(tr, "seller", "") or ""

            price = float(getattr(tr, "price", fair))
            qty = max(1, abs(int(getattr(tr, "quantity", 1))))
            qscale = min(2.0, math.sqrt(qty))

            # Anonymous tape-flow fallback: in many submission logs non-SUBMISSION trader IDs
            # are blank. If a fresh external print happens above our fair, treat it as
            # short-horizon bullish pressure; below fair, bearish pressure.
            if is_new and buyer != "SUBMISSION" and seller != "SUBMISSION":
                edge = price - fair
                if abs(edge) >= max(0.25, 0.10 * vol):
                    signed = 1.0 if edge > 0 else -1.0
                    tape[product] = self._clip(float(tape.get(product, 0.0)) + signed * qscale, -6.0, 6.0)

            # If the IDs are fully anonymous, there is no named counterparty update to make,
            # but the tape-flow signal above still remains useful.
            if not buyer and not seller:
                continue
            if (buyer == "SUBMISSION") or (seller == "SUBMISSION"):
                continue

            # A trade through/above fair tags the buyer as potentially informed upward.
            if buyer and buyer != "SUBMISSION" and price >= fair:
                flow += qscale * self._cp_lookup(mem, product, buyer)
                if is_new:
                    for k in (f"P|{product}|{buyer}", f"F|{fam}|{buyer}", f"G|{buyer}"):
                        mem["pending_cp"].append({"k": k, "p": product, "t": state.timestamp, "fair": fair, "sign": 1.0, "vol": vol})

            # A trade through/below fair tags the seller as potentially informed downward.
            if seller and seller != "SUBMISSION" and price <= fair:
                flow -= qscale * self._cp_lookup(mem, product, seller)
                if is_new:
                    for k in (f"P|{product}|{seller}", f"F|{fam}|{seller}", f"G|{seller}"):
                        mem["pending_cp"].append({"k": k, "p": product, "t": state.timestamp, "fair": fair, "sign": -1.0, "vol": vol})

        mem["seen_trades"] = seen_list[-self.MAX_SEEN_TRADES:]
        return self._clip(flow, -5.0, 5.0)


    def _anonymous_tape_alpha(self, mem: Dict[str, Any], product: str) -> float:
        """
        Anonymous tape-flow signal.

        Public Round 5 discussion emphasizes that revealed trader IDs are the big new feature.
        In the logs we often only see blank IDs, so this is the robust fallback: follow
        aggressive anonymous prints for a short horizon without depending on exact names.
        """
        tape = mem.get("tape_flow", {})
        own = float(tape.get(product, 0.0))
        fam = self._family(product)

        fam_vals = []
        for other, val in tape.items():
            if other != product and self._family(other) == fam:
                fam_vals.append(float(val))
        fam_avg = self._mean(fam_vals) if fam_vals else 0.0

        return self._clip(0.28 * own + 0.12 * fam_avg, -1.35, 1.35)

    def _signals(
        self,
        mem: Dict[str, Any],
        product: str,
        fair: float,
        resid: float,
        depth: OrderDepth,
    ) -> Tuple[float, float, float, float, float]:
        hist = mem.get("fair_hist", {}).get(product, [])
        rhist = mem.get("resid_hist", {}).get(product, [])
        spread = self._spread(depth)

        if len(hist) >= 4:
            rets = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
            vol = max(1.0, self._std(rets[-self.RET_HISTORY:]))
        else:
            vol = max(1.0, spread)

        rel = 0.0
        if len(rhist) >= self.MIN_REL_HISTORY:
            mu = self._mean(rhist[-60:])
            sd = max(0.00015, self._std(rhist[-60:]))
            z = (resid - mu) / sd
            if abs(z) > 0.90:
                rel = self._clip(-z, -3.2, 3.2)

        own_mr = 0.0
        if len(hist) >= self.MIN_PRICE_HISTORY:
            mu = self._mean(hist[-36:])
            sd = max(1.0, self._std(hist[-36:]))
            z = (fair - mu) / sd
            if abs(z) > 0.75:
                own_mr = self._clip(-z, -2.6, 2.6)

        trend = 0.0
        if len(hist) >= self.MIN_PRICE_HISTORY:
            h = hist + [fair]
            fast = self._ema(h[-7:], 4)
            slow = self._ema(h[-28:], 14) if len(h) >= 28 else self._ema(h, 12)
            trend = self._clip((fast - slow) / vol, -2.8, 2.8)

        if product in self.MICRO_LIFT_ONLY:
            top_micro = self._imbalance(depth)
            depth_micro = self._multi_level_imbalance(depth)
            micro = self._clip(0.58 * top_micro + 0.42 * depth_micro, -1.0, 1.0)
        else:
            micro = self._clip(self._imbalance(depth), -1.0, 1.0)
        return rel, own_mr, trend, micro, vol

    def _combine_alpha(
        self,
        product: str,
        rel: float,
        own_mr: float,
        trend: float,
        micro: float,
        flow: float,
        tape: float,
        position: int,
        scale: float,
    ) -> float:
        cap = max(1.0, float(self._product_cap(product, scale)))
        inv = -position / cap

        # Original v1 bias: family relative value dominates. Mean reversion helps inside
        # each product. Trend is allowed only for names where the uploaded data showed it helps.
        if product in self.TREND_NAMES:
            alpha = 0.55 * rel + 0.25 * own_mr + 0.75 * trend + 0.25 * micro + 0.45 * flow + 0.32 * tape + 0.25 * inv
        elif product in self.FAMILY_NAMES:
            alpha = 1.20 * rel + 0.20 * own_mr + 0.18 * trend + 0.30 * micro + 0.45 * flow + 0.22 * tape + 0.25 * inv
        else:
            alpha = 1.10 * rel + 0.35 * own_mr + 0.10 * trend + 0.32 * micro + 0.45 * flow + 0.25 * tape + 0.28 * inv

        # If trend strongly disagrees with mean-reversion/relative-value, reduce conviction.
        rv = rel + 0.45 * own_mr
        if product not in self.TREND_NAMES and rv * trend < 0 and abs(trend) > 1.1:
            alpha *= 0.60

        # Do not fight strong short-horizon tape unless the structural RV signal is strong.
        if abs(tape) > 0.90 and (rv * tape) < 0 and abs(rv) < 1.35:
            alpha *= 0.72

        return self._clip(alpha * scale, -5.0, 5.0)

    def _target_position(self, product: str, position: int, alpha: float, scale: float, vol: float, spread: float) -> int:
        cap = self._product_cap(product, scale)
        if cap <= 0:
            return 0

        if abs(alpha) < 0.35:
            return int(round(0.55 * position))

        # Dimensionless alpha to bounded inventory. Keep enough room to quote both sides.
        vol_penalty = 1.0 + 0.04 * max(0.0, vol - spread)
        desired = (alpha / 2.25) * cap / vol_penalty

        if abs(alpha) > 2.2:
            desired *= 1.20

        return int(self._clip(round(desired), -cap, cap))

    def _add_order(
        self,
        orders: List[Order],
        product: str,
        price: int,
        qty: int,
        position: int,
        buy_used: int,
        sell_used: int,
    ) -> Tuple[int, int]:
        if qty > 0:
            room = self.HARD_LIMIT - (position + buy_used)
            q = min(int(qty), room)
            if q > 0:
                orders.append(Order(product, int(price), int(q)))
                buy_used += q
        elif qty < 0:
            room = self.HARD_LIMIT + (position - sell_used)
            q = min(int(-qty), room)
            if q > 0:
                orders.append(Order(product, int(price), int(-q)))
                sell_used += q
        return buy_used, sell_used

    def _flatten_orders(self, product: str, depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        bid, _, ask, _ = self._best(depth)
        if position == 0 or bid is None or ask is None:
            return orders
        buy_used = 0
        sell_used = 0
        if position > 0:
            self._add_order(orders, product, bid, -min(position, 3), position, buy_used, sell_used)
        else:
            self._add_order(orders, product, ask, min(-position, 3), position, buy_used, sell_used)
        return orders

    def _make_orders(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        alpha: float,
        target: int,
        position: int,
        vol: float,
        spread: float,
        scale: float,
        tape: float = 0.0,
    ) -> List[Order]:
        bid, bid_v, ask, ask_v = self._best(depth)
        orders: List[Order] = []
        if bid is None or ask is None:
            return orders

        buy_used = 0
        sell_used = 0

        # Selective taker leg. This is intentionally small; most edge in the good run
        # came from passive fills, not paying spread repeatedly.
        pred_move = self._clip(alpha, -3.0, 3.0) * max(1.0, 0.35 * spread + 0.20 * vol)
        pred_fair = fair + pred_move
        take_cost = max(1.0, 0.50 * spread + 0.08 * vol)

        if product not in self.CAUTIOUS and abs(alpha) > 1.55 and (abs(tape) < 0.55 or tape * alpha > -0.10):
            if target > position:
                need = min(2, target - position)
                for px in sorted(depth.sell_orders.keys()):
                    if need <= 0:
                        break
                    available = -depth.sell_orders[px]
                    if px <= pred_fair - take_cost:
                        before = buy_used
                        buy_used, sell_used = self._add_order(
                            orders, product, px, min(need, available), position, buy_used, sell_used
                        )
                        need -= buy_used - before

            elif target < position:
                need = min(2, position - target)
                for px in sorted(depth.buy_orders.keys(), reverse=True):
                    if need <= 0:
                        break
                    available = depth.buy_orders[px]
                    if px >= pred_fair + take_cost:
                        before = sell_used
                        buy_used, sell_used = self._add_order(
                            orders, product, px, -min(need, available), position, buy_used, sell_used
                        )
                        need -= sell_used - before

        # Passive market making with alpha and inventory skew.
        inv_frac = position / float(self.HARD_LIMIT)
        reservation = fair + 0.30 * alpha * max(1.0, spread) - inv_frac * max(1.0, 0.75 * spread + 0.15 * vol)

        if spread >= 3:
            bid_px = min(ask - 1, max(bid + 1, int(math.floor(reservation - 0.45 * spread))))
            ask_px = max(bid + 1, min(ask - 1, int(math.ceil(reservation + 0.45 * spread))))
        else:
            bid_px = min(bid, int(math.floor(reservation - 1)))
            ask_px = max(ask, int(math.ceil(reservation + 1)))

        if bid_px >= ask_px:
            bid_px = bid
            ask_px = ask

        # v31 isolated microchip-square quote placement lift.
        # It never reduces a quote side and never affects any other product.
        # The condition requires alpha and depth pressure to agree, so this is not a
        # blind size increase.
        if product in self.MICRO_LIFT_ONLY and spread >= 4 and abs(alpha) > 0.70:
            depth_micro = self._multi_level_imbalance(depth)
            if alpha * depth_micro > 0.55:
                if alpha > 0:
                    bid_px = min(ask - 1, bid_px + 1)
                elif alpha < 0:
                    ask_px = max(bid + 1, ask_px - 1)

        cap = self._product_cap(product, scale)
        base = 2 if product in self.STRONG and abs(alpha) > 0.75 else 1

        if target > position:
            buy_sz = min(base, max(1, target - position - buy_used))
            sell_sz = 0 if alpha > 0.95 else 1
        elif target < position:
            buy_sz = 0 if alpha < -0.95 else 1
            sell_sz = min(base, max(1, position - target - sell_used))
        else:
            buy_sz = 1 if alpha > -1.25 else 0
            sell_sz = 1 if alpha < 1.25 else 0

        # v31 isolated size lift for MICROCHIP_SQUARE only.
        # Add size only when current depth pressure agrees with alpha and there is room.
        if product in self.MICRO_LIFT_ONLY and abs(alpha) > 0.80:
            depth_micro = self._multi_level_imbalance(depth)
            if alpha * depth_micro > 0.65:
                if alpha > 0 and position < cap - 1:
                    buy_sz = max(buy_sz, min(2, cap - position - buy_used))
                elif alpha < 0 and position > -cap + 1:
                    sell_sz = max(sell_sz, min(2, cap + position - sell_used))

        # If near cap, prioritize inventory repair side.
        if position >= cap - 1:
            buy_sz = 0
            sell_sz = max(1, sell_sz)
        if position <= -cap + 1:
            sell_sz = 0
            buy_sz = max(1, buy_sz)

        # Non-overfit quote-toxicity guard.
        # A lot of the losses came from passively quoting both sides while the visible book
        # was leaning hard against one side. This rule does not use timestamps or fitted
        # product exits; it only reacts to current top-of-book pressure.
        top_den = max(1, bid_v + ask_v)
        top_imb = (bid_v - ask_v) / top_den

        # If the bid side is thin and ask liquidity dominates, avoid adding passive bids
        # unless alpha strongly says we should buy or we are repairing a short.
        if top_imb < -0.58 and alpha < 0.90 and position >= 0:
            buy_sz = 0

        # If the ask side is thin and bid liquidity dominates, avoid adding passive asks
        # unless alpha strongly says we should sell or we are repairing a long.
        if top_imb > 0.58 and alpha > -0.90 and position <= 0:
            sell_sz = 0

        # Tape-flow toxicity guard. Do not passively join the side that fresh prints
        # are moving against, unless we are repairing inventory or alpha is clearly stronger.
        if tape < -0.95 and alpha < 1.15 and position >= 0:
            buy_sz = 0
        if tape > 0.95 and alpha > -1.15 and position <= 0:
            sell_sz = 0

        # On tight-spread cautious products, tiny two-sided quotes often churn without edge.
        # Keep inventory-repair quotes, but do not open new two-sided risk on weak signals.
        if product in self.CAUTIOUS and spread <= 2 and abs(alpha) < 0.65:
            if position >= 0:
                buy_sz = 0
            if position <= 0:
                sell_sz = 0

        if buy_sz > 0 and alpha > -1.55:
            buy_used, sell_used = self._add_order(orders, product, bid_px, buy_sz, position, buy_used, sell_used)
        if sell_sz > 0 and alpha < 1.55:
            buy_used, sell_used = self._add_order(orders, product, ask_px, -sell_sz, position, buy_used, sell_used)

        return orders

    def run(self, state: TradingState):
        mem = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        fairs: Dict[str, float] = {}
        spreads: Dict[str, float] = {}

        for product, depth in state.order_depths.items():
            fair = self._fair_value(depth)
            if fair is not None and fair > 0:
                fairs[product] = fair
                spreads[product] = self._spread(depth)

        family_mu = self._family_log_means(fairs)

        self._resolve_alpha_markouts(mem, state.timestamp, fairs)
        self._resolve_cp_markouts(mem, state.timestamp, fairs)

        for product, depth in state.order_depths.items():
            if product not in fairs:
                result[product] = []
                continue

            position = int(state.position.get(product, 0))
            fair = fairs[product]
            spread = spreads.get(product, 1.0)

            if product in self.AVOID:
                result[product] = self._flatten_orders(product, depth, position)
                continue

            resid = math.log(fair) - family_mu.get(self._family(product), math.log(fair))
            rel, own_mr, trend, micro, vol = self._signals(mem, product, fair, resid, depth)
            flow = self._counterparty_flow(mem, state, product, fair, vol)
            tape = self._anonymous_tape_alpha(mem, product)
            scale = self._product_scale(mem, product)
            alpha = self._combine_alpha(product, rel, own_mr, trend, micro, flow, tape, position, scale)

            target = self._target_position(product, position, alpha, scale, vol, spread)
            result[product] = self._make_orders(product, depth, fair, alpha, target, position, vol, spread, scale, tape)

            if scale > 0.0 and abs(alpha) > 0.45:
                mem["pending_alpha"].append({
                    "p": product,
                    "t": state.timestamp,
                    "fair": fair,
                    "sign": 1.0 if alpha > 0 else -1.0,
                    "vol": max(1.0, vol),
                })

        for product, fair in fairs.items():
            old = mem.get("fair_hist", {}).get(product, [])
            if old:
                self._append(mem, "ret_hist", product, fair - old[-1], self.RET_HISTORY)
            self._append(mem, "fair_hist", product, fair, self.FAIR_HISTORY)

            resid = math.log(fair) - family_mu.get(self._family(product), math.log(fair))
            self._append(mem, "resid_hist", product, resid, self.RESID_HISTORY)

        return result, 0, self._save_memory(mem)