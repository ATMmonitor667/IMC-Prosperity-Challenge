from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import jsonpickle
import math


class Trader:
    """
    IMC Prosperity 4 Round 5 Sigma Wolf.

    Built on the actual 570532/v18 tape-core because that was the proven profitable
    architecture in the uploaded official run logs.

    Research-backed additions are deliberately narrow:
    - compact JSON traderData with bounded caches,
    - severe-only side adverse-fill control,
    - inventory-age target decay without exact timestamp exits,
    - a branch-only PANEL_2X2 geometry fair model,
    - no branch takeover of normal core products.
    """

    HARD_LIMIT = 10
    FAIR_HISTORY = 55
    RESID_HISTORY = 45
    RET_HISTORY = 60
    MARKOUT_DELAY = 500
    CP_DELAY = 500

    MIN_REL_HISTORY = 18
    MIN_PRICE_HISTORY = 16

    FILL_DELAY = 500

    MAX_PENDING_ALPHA = 220
    MAX_PENDING_CP = 220
    MAX_PENDING_FILL = 500
    MAX_SEEN_TRADES = 120
    MAX_SEEN_OWN = 300

    INVENTORY_AGE_SOFT = 4000
    INVENTORY_AGE_HARD = 8000
    SIDE_TOXICITY_BLOCK = -1.10

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

    PANEL_FEATURES = {
        "PANEL_1X2": (1.0, 3.0, 2.0),
        "PANEL_2X2": (1.0, 4.0, 4.0),
        "PANEL_1X4": (1.0, 5.0, 4.0),
        "PANEL_2X4": (1.0, 6.0, 8.0),
        "PANEL_4X4": (1.0, 8.0, 16.0),
    }

    BRANCH_CAP = {
        "PANEL_2X2": 6,
    }

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "fair_hist": {},
            "resid_hist": {},
            "perf": {},
            "pending_alpha": [],
            "cp_score": {},
            "pending_cp": [],
            "pending_fill": [],
            "seen_trades": [],
            "seen_own": [],
            "tape_flow": {},
            "fill_quality": {},
            "position_state": {},
            "last_quotes": {},
        }

    def _load_memory(self, data: str) -> Dict[str, Any]:
        if not data:
            return self._default_memory()
        try:
            mem = json.loads(data)
        except Exception:
            try:
                mem = jsonpickle.decode(data)
            except Exception:
                return self._default_memory()
        try:
            if not isinstance(mem, dict):
                return self._default_memory()
            base = self._default_memory()
            base.update(mem)
            return base
        except Exception:
            return self._default_memory()

    def _save_memory(self, mem: Dict[str, Any]) -> str:
        mem.pop("ret_hist", None)
        for bucket, keep in (
            ("fair_hist", self.FAIR_HISTORY),
            ("resid_hist", self.RESID_HISTORY),
        ):
            for product in list(mem.get(bucket, {}).keys()):
                mem[bucket][product] = mem[bucket][product][-keep:]

        mem["pending_alpha"] = mem.get("pending_alpha", [])[-self.MAX_PENDING_ALPHA:]
        mem["pending_cp"] = mem.get("pending_cp", [])[-self.MAX_PENDING_CP:]
        mem["pending_fill"] = mem.get("pending_fill", [])[-self.MAX_PENDING_FILL:]
        mem["seen_trades"] = mem.get("seen_trades", [])[-self.MAX_SEEN_TRADES:]
        mem["seen_own"] = mem.get("seen_own", [])[-self.MAX_SEEN_OWN:]
        if len(mem.get("tape_flow", {})) > 120:
            items = sorted(mem["tape_flow"].items(), key=lambda kv: abs(float(kv[1])), reverse=True)
            mem["tape_flow"] = dict(items[:80])

        if len(mem.get("cp_score", {})) > 100:
            items = sorted(
                mem["cp_score"].items(),
                key=lambda kv: abs(float(kv[1].get("s", 0.0))) * max(1, int(kv[1].get("n", 0))),
                reverse=True,
            )
            mem["cp_score"] = dict(items[:80])

        if len(mem.get("last_quotes", {})) > 80:
            mem["last_quotes"] = dict(list(mem["last_quotes"].items())[-80:])

        try:
            return json.dumps(mem, separators=(",", ":"))
        except Exception:
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
        if bucket == "resid_hist":
            stored = round(float(value), 8)
        else:
            stored = round(float(value), 3)
        mem[bucket][product].append(stored)
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

    def _short_key(self, *parts: Any) -> str:
        h = 2166136261
        for part in parts:
            for ch in str(part):
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
            h ^= 255
            h = (h * 16777619) & 0xFFFFFFFF
        return format(h, "08x")

    def _trade_key(self, product: str, tr: Any) -> str:
        return self._short_key(
            product,
            getattr(tr, "timestamp", ""),
            getattr(tr, "buyer", ""),
            getattr(tr, "seller", ""),
            getattr(tr, "price", ""),
            getattr(tr, "quantity", ""),
        )

    def _own_trade_key(self, product: str, tr: Any) -> str:
        return self._short_key(
            product,
            getattr(tr, "timestamp", ""),
            getattr(tr, "buyer", ""),
            getattr(tr, "seller", ""),
            getattr(tr, "price", ""),
            getattr(tr, "quantity", ""),
        )

    def _cp_key(self, scope: str, name: str, trader_id: str) -> str:
        return self._short_key(scope, name, trader_id)

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
            (self._cp_key("P", product, trader_id), 0.65),
            (self._cp_key("F", fam, trader_id), 0.25),
            (self._cp_key("G", "", trader_id), 0.10),
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
        """
        Robust Round-5 flow signal.

        The failed tape version ignored blank-ID trades before extracting anonymous
        tape-flow. In Round 5 many trades have empty buyer/seller fields, so this
        function first updates anonymous tape from price-vs-fair for every new trade,
        then adds named trader-ID alpha only when IDs are actually available.
        """
        tape = mem.setdefault("tape_flow", {})
        tape[product] = 0.90 * float(tape.get(product, 0.0))

        trades = state.market_trades.get(product, [])
        if not trades:
            fam_vals = [
                float(v)
                for p, v in tape.items()
                if p != product and self._family(p) == self._family(product)
            ]
            fam_tape = self._mean(fam_vals) if fam_vals else 0.0
            return self._clip(0.35 * float(tape.get(product, 0.0)) + 0.10 * fam_tape, -2.0, 2.0)

        seen = set(mem.get("seen_trades", []))
        seen_list = list(mem.get("seen_trades", []))
        fam = self._family(product)
        id_flow = 0.0

        for tr in trades:
            key = self._trade_key(product, tr)
            is_new = key not in seen
            if is_new:
                seen.add(key)
                seen_list.append(key)

            price = float(getattr(tr, "price", fair))
            qty = max(1, abs(int(getattr(tr, "quantity", 1))))
            qscale = min(2.0, math.sqrt(qty))

            # Anonymous tape-flow works even when buyer/seller IDs are blank.
            # Only count meaningful prints away from fair so mid/noisy trades do not dominate.
            if is_new:
                edge = (price - fair) / max(1.0, vol)
                signed = 0.0
                if edge > 0.15:
                    signed = 1.0
                elif edge < -0.15:
                    signed = -1.0
                if signed != 0.0:
                    tape[product] = self._clip(float(tape.get(product, 0.0)) + signed * qscale, -6.0, 6.0)

            buyer = getattr(tr, "buyer", "") or ""
            seller = getattr(tr, "seller", "") or ""

            if buyer and buyer != "SUBMISSION" and price >= fair:
                id_flow += qscale * self._cp_lookup(mem, product, buyer)
                if is_new:
                    for k in (
                        self._cp_key("P", product, buyer),
                        self._cp_key("F", fam, buyer),
                        self._cp_key("G", "", buyer),
                    ):
                        mem["pending_cp"].append({"k": k, "p": product, "t": state.timestamp, "fair": fair, "sign": 1.0, "vol": vol})

            if seller and seller != "SUBMISSION" and price <= fair:
                id_flow -= qscale * self._cp_lookup(mem, product, seller)
                if is_new:
                    for k in (
                        self._cp_key("P", product, seller),
                        self._cp_key("F", fam, seller),
                        self._cp_key("G", "", seller),
                    ):
                        mem["pending_cp"].append({"k": k, "p": product, "t": state.timestamp, "fair": fair, "sign": -1.0, "vol": vol})

        mem["seen_trades"] = seen_list[-self.MAX_SEEN_TRADES:]

        fam_vals = [
            float(v)
            for p, v in tape.items()
            if p != product and self._family(p) == self._family(product)
        ]
        fam_tape = self._mean(fam_vals) if fam_vals else 0.0
        tape_alpha = 0.35 * float(tape.get(product, 0.0)) + 0.10 * fam_tape
        return self._clip(id_flow + tape_alpha, -5.0, 5.0)

    def _product_vol(self, mem: Dict[str, Any], product: str, fallback: float = 1.0) -> float:
        hist = mem.get("fair_hist", {}).get(product, [])
        if len(hist) >= 4:
            rets = [float(hist[i]) - float(hist[i - 1]) for i in range(1, len(hist))]
            return max(1.0, self._std(rets[-self.RET_HISTORY:]))
        return max(1.0, fallback)

    def _register_own_fills(self, mem: Dict[str, Any], state: TradingState, fairs: Dict[str, float]) -> None:
        seen = set(mem.get("seen_own", []))
        seen_list = list(mem.get("seen_own", []))
        pending = mem.setdefault("pending_fill", [])
        last_quotes = mem.get("last_quotes", {})

        for product, trades in getattr(state, "own_trades", {}).items():
            if product not in fairs:
                continue
            for tr in trades:
                key = self._own_trade_key(product, tr)
                if key in seen:
                    continue
                seen.add(key)
                seen_list.append(key)

                buyer = getattr(tr, "buyer", "") or ""
                seller = getattr(tr, "seller", "") or ""
                qty = abs(int(getattr(tr, "quantity", 0)))
                if qty <= 0:
                    continue
                if buyer == "SUBMISSION":
                    side = 1.0
                    side_name = "bid"
                elif seller == "SUBMISSION":
                    side = -1.0
                    side_name = "ask"
                else:
                    continue

                quote_px = last_quotes.get(product, {}).get(side_name)
                if quote_px is not None and int(quote_px) != int(getattr(tr, "price", 0)):
                    continue

                pending.append({
                    "p": product,
                    "side": side,
                    "side_name": side_name,
                    "t": state.timestamp,
                    "fair": float(fairs[product]),
                    "vol": self._product_vol(mem, product, 1.0),
                })

        mem["seen_own"] = seen_list[-self.MAX_SEEN_OWN:]
        mem["pending_fill"] = pending[-self.MAX_PENDING_FILL:]

    def _resolve_fill_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        pending = []
        fill_quality = mem.setdefault("fill_quality", {})
        for ev in mem.get("pending_fill", []):
            product = ev.get("p")
            if product not in fairs:
                pending.append(ev)
                continue
            if timestamp - int(ev.get("t", timestamp)) < self.FILL_DELAY:
                pending.append(ev)
                continue

            side = float(ev.get("side", 0.0))
            side_name = str(ev.get("side_name", "bid"))
            move = float(fairs[product]) - float(ev.get("fair", fairs[product]))
            vol = max(1.0, float(ev.get("vol", 1.0)))
            markout = self._clip(side * move / vol, -4.0, 4.0)
            product_quality = fill_quality.setdefault(product, {})
            rec = product_quality.get(side_name, {"s": 0.0, "n": 0})
            n = int(rec.get("n", 0))
            lr = 0.12 if n < 8 else 0.05
            rec["s"] = (1.0 - lr) * float(rec.get("s", 0.0)) + lr * markout
            rec["n"] = min(9999, n + 1)
            product_quality[side_name] = rec

        mem["pending_fill"] = pending[-self.MAX_PENDING_FILL:]

    def _fill_quality(self, mem: Dict[str, Any], product: str, side_name: str) -> float:
        rec = mem.get("fill_quality", {}).get(product, {}).get(side_name)
        if not rec or int(rec.get("n", 0)) < 8:
            return 0.0
        return self._clip(float(rec.get("s", 0.0)), -3.0, 3.0)

    def _record_passive_quotes(
        self,
        mem: Dict[str, Any],
        product: str,
        depth: OrderDepth,
        orders: List[Order],
    ) -> None:
        bid, _, ask, _ = self._best(depth)
        last_quotes = mem.setdefault("last_quotes", {})
        if bid is None or ask is None:
            last_quotes.pop(product, None)
            return

        quote: Dict[str, int] = {}
        for order in orders:
            qty = int(order.quantity)
            px = int(order.price)
            if qty > 0 and px < ask:
                quote["bid"] = max(px, int(quote.get("bid", px)))
            elif qty < 0 and px > bid:
                quote["ask"] = min(px, int(quote.get("ask", px)))

        if quote:
            last_quotes[product] = quote
        else:
            last_quotes.pop(product, None)

    def _family_stress(self, mem: Dict[str, Any], product: str, fairs: Dict[str, float]) -> float:
        """
        Mild non-overfit family-volatility control.
        If many names in the same family are moving sharply at the same time,
        raw cross-sectional relative value becomes less reliable. We scale it down,
        but never turn it off completely.
        """
        fam = self._family(product)
        moves = []
        for p, fv in fairs.items():
            if self._family(p) != fam:
                continue
            hist = mem.get("fair_hist", {}).get(p, [])
            if not hist:
                continue
            moves.append(abs(fv - hist[-1]))
        if len(moves) < 3:
            return 0.0
        return self._mean(moves)

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
        position: int,
        scale: float,
    ) -> float:
        cap = max(1.0, float(self._product_cap(product, scale)))
        inv = -position / cap

        # Original v1 bias: family relative value dominates. Mean reversion helps inside
        # each product. Trend is allowed only for names where the uploaded data showed it helps.
        if product in self.TREND_NAMES:
            alpha = 0.55 * rel + 0.25 * own_mr + 0.75 * trend + 0.25 * micro + 0.45 * flow + 0.25 * inv
        elif product in self.FAMILY_NAMES:
            alpha = 1.20 * rel + 0.20 * own_mr + 0.18 * trend + 0.30 * micro + 0.45 * flow + 0.25 * inv
        else:
            alpha = 1.10 * rel + 0.35 * own_mr + 0.10 * trend + 0.32 * micro + 0.45 * flow + 0.28 * inv

        # If trend strongly disagrees with mean-reversion/relative-value, reduce conviction.
        rv = rel + 0.45 * own_mr
        if product not in self.TREND_NAMES and rv * trend < 0 and abs(trend) > 1.1:
            alpha *= 0.60

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

    def _target_with_age(
        self,
        mem: Dict[str, Any],
        product: str,
        timestamp: int,
        position: int,
        alpha: float,
        target: int,
    ) -> int:
        state = mem.setdefault("position_state", {})
        if position == 0:
            state.pop(product, None)
            return target

        sign = 1 if position > 0 else -1
        rec = state.get(product)
        if not isinstance(rec, dict) or int(rec.get("sign", 0)) != sign:
            rec = {"sign": sign, "t": int(timestamp), "last": int(position)}
        elif abs(position) > abs(int(rec.get("last", 0))):
            rec["t"] = int(timestamp)
            rec["last"] = int(position)
        else:
            rec["last"] = int(position)
        state[product] = rec

        age = int(timestamp) - int(rec.get("t", timestamp))
        unsupported_inventory = sign * alpha < -0.15
        if age >= self.INVENTORY_AGE_HARD and unsupported_inventory:
            return int(round(0.25 * target))
        if age >= self.INVENTORY_AGE_SOFT and unsupported_inventory:
            return int(round(0.50 * target))
        return target

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

    def _solve_linear_system(self, a: List[List[float]], b: List[float]) -> Optional[List[float]]:
        n = len(b)
        matrix = [row[:] + [b[i]] for i, row in enumerate(a)]
        for col in range(n):
            pivot = col
            best = abs(matrix[col][col])
            for row in range(col + 1, n):
                value = abs(matrix[row][col])
                if value > best:
                    pivot = row
                    best = value
            if best < 1e-9:
                return None
            if pivot != col:
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            div = matrix[col][col]
            for c in range(col, n + 1):
                matrix[col][c] /= div
            for row in range(n):
                if row == col:
                    continue
                factor = matrix[row][col]
                if abs(factor) < 1e-12:
                    continue
                for c in range(col, n + 1):
                    matrix[row][c] -= factor * matrix[col][c]
        return [matrix[i][n] for i in range(n)]

    def _panel_geometry_fair(self, fairs: Dict[str, float], target: str = "PANEL_2X2") -> Optional[float]:
        if target not in self.PANEL_FEATURES:
            return None
        rows = []
        ys = []
        for product, features in self.PANEL_FEATURES.items():
            if product == target:
                continue
            if product not in fairs:
                return None
            rows.append([float(x) for x in features])
            ys.append(float(fairs[product]))

        k = 3
        xtx = [[0.0 for _ in range(k)] for _ in range(k)]
        xty = [0.0 for _ in range(k)]
        for row, y in zip(rows, ys):
            for i in range(k):
                xty[i] += row[i] * y
                for j in range(k):
                    xtx[i][j] += row[i] * row[j]
        for i in range(k):
            xtx[i][i] += 1e-4
        beta = self._solve_linear_system(xtx, xty)
        if beta is None:
            return None
        features = [float(x) for x in self.PANEL_FEATURES[target]]
        return sum(beta[i] * features[i] for i in range(k))

    def _panel_branch_orders(
        self,
        product: str,
        depth: OrderDepth,
        fairs: Dict[str, float],
        position: int,
    ) -> List[Order]:
        orders: List[Order] = []
        bid, _, ask, _ = self._best(depth)
        if bid is None or ask is None:
            return orders
        model_fair = self._panel_geometry_fair(fairs, product)
        if model_fair is None or not math.isfinite(model_fair):
            return self._flatten_orders(product, depth, position)

        spread = max(1.0, float(ask - bid))
        mid = 0.5 * (bid + ask)
        cap = int(self.BRANCH_CAP.get(product, 4))
        reservation = float(model_fair) - 1.0 * spread * position
        buy_signal = reservation - mid >= 1.0 * spread
        sell_signal = mid - reservation >= 1.0 * spread

        buy_used = 0
        sell_used = 0
        if buy_signal and position < cap:
            buy_used, sell_used = self._add_order(orders, product, bid, 1, position, buy_used, sell_used)
        if sell_signal and position > -cap:
            buy_used, sell_used = self._add_order(orders, product, ask, -1, position, buy_used, sell_used)

        if not orders:
            if position > cap:
                self._add_order(orders, product, bid, -min(2, position - cap), position, buy_used, sell_used)
            elif position < -cap:
                self._add_order(orders, product, ask, min(2, -cap - position), position, buy_used, sell_used)
        return orders

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
        mem: Optional[Dict[str, Any]] = None,
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

        if product not in self.CAUTIOUS and abs(alpha) > 1.55:
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

        if mem is not None:
            bid_quality = self._fill_quality(mem, product, "bid")
            ask_quality = self._fill_quality(mem, product, "ask")
            if bid_quality < self.SIDE_TOXICITY_BLOCK and position >= 0:
                buy_sz = 0
            if ask_quality < self.SIDE_TOXICITY_BLOCK and position <= 0:
                sell_sz = 0

        # If near cap, prioritize inventory repair side.
        if position >= cap - 1:
            buy_sz = 0
            sell_sz = max(1, sell_sz)
        if position <= -cap + 1:
            sell_sz = 0
            buy_sz = max(1, buy_sz)

        if buy_sz > 0 and alpha > -1.55:
            buy_used, sell_used = self._add_order(orders, product, bid_px, buy_sz, position, buy_used, sell_used)
        if sell_sz > 0 and alpha < 1.55:
            buy_used, sell_used = self._add_order(orders, product, ask_px, -sell_sz, position, buy_used, sell_used)

        return orders

    def _snapshot_family_positions(
        self, positions: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        family_net: Dict[str, int] = defaultdict(int)
        family_gross: Dict[str, int] = defaultdict(int)

        for product, qty in positions.items():
            quantity = int(qty)
            if quantity == 0:
                continue
            family = self._family(product)
            family_net[family] += quantity
            family_gross[family] += abs(quantity)

        ordered = sorted(
            family_gross,
            key=lambda family: (family_gross[family], abs(family_net[family]), family),
            reverse=True,
        )
        return (
            {family: family_net[family] for family in ordered},
            {family: family_gross[family] for family in ordered},
        )

    def _snapshot_family_orders(self, orders_by_product: Dict[str, List[Order]]) -> Dict[str, int]:
        family_orders: Dict[str, int] = defaultdict(int)

        for product, orders in orders_by_product.items():
            if not orders:
                continue
            net_qty = sum(int(order.quantity) for order in orders)
            if net_qty == 0:
                continue
            family_orders[self._family(product)] += net_qty

        ordered = sorted(
            family_orders,
            key=lambda family: (abs(family_orders[family]), family),
            reverse=True,
        )
        return {family: family_orders[family] for family in ordered}

    def _emit_run_log(
        self,
        state: TradingState,
        orders_by_product: Dict[str, List[Order]],
        diagnostics: Dict[str, Dict[str, float]],
    ) -> None:
        family_pos, family_gross = self._snapshot_family_positions(state.position)
        family_orders = self._snapshot_family_orders(orders_by_product)

        top_signals = []
        for product, diag in sorted(
            diagnostics.items(), key=lambda item: abs(item[1]["alpha"]), reverse=True
        )[:6]:
            top_signals.append(
                [
                    product,
                    round(diag["alpha"], 3),
                    int(diag["target"]),
                    int(diag["position"]),
                    round(diag["scale"], 3),
                ]
            )

        payload = {
            "kind": "family_snapshot",
            "ts": state.timestamp,
            "family_pos": family_pos,
            "family_gross": family_gross,
            "family_orders": family_orders,
            "top_signals": top_signals,
        }
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))

    def run(self, state: TradingState):
        mem = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        diagnostics: Dict[str, Dict[str, float]] = {}

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
        self._resolve_fill_markouts(mem, state.timestamp, fairs)
        self._register_own_fills(mem, state, fairs)

        for product, depth in state.order_depths.items():
            if product not in fairs:
                result[product] = []
                continue

            position = int(state.position.get(product, 0))
            fair = fairs[product]
            spread = spreads.get(product, 1.0)

            if product in self.BRANCH_CAP:
                orders = self._panel_branch_orders(product, depth, fairs, position)
                result[product] = orders
                self._record_passive_quotes(mem, product, depth, orders)
                continue

            if product in self.AVOID:
                orders = self._flatten_orders(product, depth, position)
                result[product] = orders
                self._record_passive_quotes(mem, product, depth, orders)
                continue

            resid = math.log(fair) - family_mu.get(self._family(product), math.log(fair))
            rel, own_mr, trend, micro, vol = self._signals(mem, product, fair, resid, depth)

            # Mild family-stress scaling. This avoids the previous failed versions'
            # mistake of shutting down quotes; it only reduces relative-value weight
            # during broad family moves where cross-sectional mean reversion is less reliable.
            fam_stress = self._family_stress(mem, product, fairs)
            if fam_stress > 3.0:
                rel *= 0.70
            elif fam_stress > 2.0:
                rel *= 0.85

            flow = self._counterparty_flow(mem, state, product, fair, vol)
            scale = self._product_scale(mem, product)
            alpha = self._combine_alpha(product, rel, own_mr, trend, micro, flow, position, scale)

            target = self._target_position(product, position, alpha, scale, vol, spread)
            orders = self._make_orders(product, depth, fair, alpha, target, position, vol, spread, scale)
            result[product] = orders
            self._record_passive_quotes(mem, product, depth, orders)
            diagnostics[product] = {
                "alpha": alpha,
                "target": float(target),
                "position": float(position),
                "scale": scale,
            }

            if scale > 0.0 and abs(alpha) > 0.45:
                mem["pending_alpha"].append({
                    "p": product,
                    "t": state.timestamp,
                    "fair": fair,
                    "sign": 1.0 if alpha > 0 else -1.0,
                    "vol": max(1.0, vol),
                })

        for product, fair in fairs.items():
            self._append(mem, "fair_hist", product, fair, self.FAIR_HISTORY)

            if product in self.AVOID or product in self.BRANCH_CAP:
                continue
            resid = math.log(fair) - family_mu.get(self._family(product), math.log(fair))
            self._append(mem, "resid_hist", product, resid, self.RESID_HISTORY)

        # Keep submission output quiet; diagnostics are intentionally not printed.

        return result, 0, self._save_memory(mem)
