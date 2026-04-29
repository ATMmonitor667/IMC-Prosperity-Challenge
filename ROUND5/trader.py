from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import jsonpickle
import math


class Trader:
    LIMIT = 10
    FLATTEN_START = 98500
    HIST = 96
    RESID_HIST = 96
    MARKOUT_DELAY = 500

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

    # Evidence-backed hard avoids from repeated losses in both uploaded runs
    AVOID = {
        "TRANSLATOR_SPACE_GRAY",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_DARK_MATTER",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "ROBOT_DISHES",
        "PANEL_1X2",
        "PANEL_2X2",
        "MICROCHIP_OVAL",
        "MICROCHIP_CIRCLE",
    }

    # Passive-only names: positive in uploads, but directional edge usually too small to justify crossing
    PASSIVE_ONLY = {
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_ASTRO_BLACK",
        "UV_VISOR_RED",
        "UV_VISOR_ORANGE",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    }

    # Product-specific alpha modes from uploaded data
    MODE = {
        "MICROCHIP_SQUARE": "mr",
        "MICROCHIP_RECTANGLE": "trend",
        "PEBBLES_XS": "mr",
        "OXYGEN_SHAKE_CHOCOLATE": "mr",
        "OXYGEN_SHAKE_GARLIC": "trend",
        "TRANSLATOR_VOID_BLUE": "mr",
        "ROBOT_MOPPING": "mr",
        "ROBOT_IRONING": "family",
        "PANEL_4X4": "family",
        "GALAXY_SOUNDS_SOLAR_WINDS": "mr",
        "TRANSLATOR_ECLIPSE_CHARCOAL": "mr",
        "TRANSLATOR_ASTRO_BLACK": "family",
        "UV_VISOR_RED": "family",
        "UV_VISOR_ORANGE": "family",
        "GALAXY_SOUNDS_BLACK_HOLES": "mr",
        "GALAXY_SOUNDS_SOLAR_FLAMES": "mr",
    }

    def _default_mem(self) -> Dict[str, Any]:
        return {
            "fair": {},          # product -> list[float]
            "resid": {},         # product -> list[float]
            "perf": {},          # product -> {"s": float, "n": int}
            "pending_alpha": [], # delayed alpha markouts
            "cp_score": {},      # key -> score
            "cp_pending": [],    # delayed counterparty markouts
        }

    def _load(self, s: str) -> Dict[str, Any]:
        if not s:
            return self._default_mem()
        try:
            obj = jsonpickle.decode(s)
            if not isinstance(obj, dict):
                return self._default_mem()
            base = self._default_mem()
            base.update(obj)
            return base
        except Exception:
            return self._default_mem()

    def _save(self, mem: Dict[str, Any]) -> str:
        for bucket, keep in (("fair", self.HIST), ("resid", self.RESID_HIST)):
            for p in list(mem[bucket].keys()):
                mem[bucket][p] = mem[bucket][p][-keep:]
        mem["pending_alpha"] = mem["pending_alpha"][-1200:]
        mem["cp_pending"] = mem["cp_pending"][-2000:]
        return jsonpickle.encode(mem)

    def _family(self, product: str) -> str:
        for pref in self.FAMILY_PREFIXES:
            if product.startswith(pref + "_"):
                return pref
        return product.split("_")[0]

    def _mean(self, xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _std(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 1.0
        m = self._mean(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        return max(1e-9, math.sqrt(v))

    def _ema(self, xs: List[float], span: int) -> float:
        if not xs:
            return 0.0
        a = 2.0 / (span + 1.0)
        out = xs[0]
        for x in xs[1:]:
            out = a * x + (1.0 - a) * out
        return out

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _append(self, mem: Dict[str, Any], bucket: str, product: str, value: float, keep: int) -> None:
        if product not in mem[bucket]:
            mem[bucket][product] = []
        mem[bucket][product].append(float(value))
        mem[bucket][product] = mem[bucket][product][-keep:]

    def _best(self, depth: OrderDepth) -> Tuple[int, int, int, int]:
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

    def _fair(self, depth: OrderDepth) -> float:
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
        return 0.65 * micro + 0.35 * mid

    def _imbalance(self, depth: OrderDepth) -> float:
        bid, bid_v, ask, ask_v = self._best(depth)
        if bid is None or ask is None or bid_v + ask_v <= 0:
            return 0.0
        return (bid_v - ask_v) / (bid_v + ask_v)

    def _resolve_alpha_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        keep = []
        for ev in mem["pending_alpha"]:
            p = ev["p"]
            if p not in fairs or timestamp - ev["t"] < self.MARKOUT_DELAY:
                keep.append(ev)
                continue
            move = fairs[p] - ev["fair"]
            score = self._clip((move * ev["sign"]) / max(1.0, ev["vol"]), -4.0, 4.0)
            rec = mem["perf"].get(p, {"s": 0.0, "n": 0})
            lr = 0.08 if rec["n"] < 10 else 0.04
            rec["s"] = (1.0 - lr) * rec["s"] + lr * score
            rec["n"] = min(9999, rec["n"] + 1)
            mem["perf"][p] = rec
        mem["pending_alpha"] = keep

    def _resolve_counterparty_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        keep = []
        for ev in mem["cp_pending"]:
            p = ev["p"]
            if p not in fairs or timestamp - ev["t"] < self.MARKOUT_DELAY:
                keep.append(ev)
                continue
            move = fairs[p] - ev["fair"]
            score = self._clip((move * ev["sign"]) / max(1.0, ev["vol"]), -4.0, 4.0)
            old = float(mem["cp_score"].get(ev["k"], 0.0))
            mem["cp_score"][ev["k"]] = 0.96 * old + 0.04 * score
        mem["cp_pending"] = keep

    def _product_scale(self, mem: Dict[str, Any], product: str) -> float:
        if product in self.AVOID:
            return 0.0
        rec = mem["perf"].get(product, {"s": 0.0, "n": 0})
        if rec["n"] < 8:
            return 1.0
        s = rec["s"]
        if s < -0.6:
            return 0.35
        if s < -0.2:
            return 0.65
        if s > 0.5:
            return 1.15
        return 1.0

    def _counterparty_flow(self, state: TradingState, mem: Dict[str, Any], product: str, fair: float, vol: float) -> float:
        flow = 0.0
        fam = self._family(product)
        for tr in state.market_trades.get(product, []):
            qty = max(1, int(getattr(tr, "quantity", 0)))
            w = min(1.5, 0.25 + qty / 20.0)
            buyer = getattr(tr, "buyer", "") or ""
            seller = getattr(tr, "seller", "") or ""

            if buyer and buyer != "SUBMISSION":
                kp = f"P|{product}|{buyer}"
                kf = f"F|{fam}|{buyer}"
                sc = 0.7 * float(mem["cp_score"].get(kp, 0.0)) + 0.3 * float(mem["cp_score"].get(kf, 0.0))
                flow += w * sc
                mem["cp_pending"].append({"k": kp, "p": product, "t": state.timestamp, "fair": fair, "sign": +1.0, "vol": vol})
                mem["cp_pending"].append({"k": kf, "p": product, "t": state.timestamp, "fair": fair, "sign": +1.0, "vol": vol})

            if seller and seller != "SUBMISSION":
                kp = f"P|{product}|{seller}"
                kf = f"F|{fam}|{seller}"
                sc = 0.7 * float(mem["cp_score"].get(kp, 0.0)) + 0.3 * float(mem["cp_score"].get(kf, 0.0))
                flow -= w * sc
                mem["cp_pending"].append({"k": kp, "p": product, "t": state.timestamp, "fair": fair, "sign": -1.0, "vol": vol})
                mem["cp_pending"].append({"k": kf, "p": product, "t": state.timestamp, "fair": fair, "sign": -1.0, "vol": vol})

        return self._clip(flow, -3.0, 3.0)

    def _signals(self, mem: Dict[str, Any], product: str, fair: float, resid: float, depth: OrderDepth) -> Tuple[float, float, float, float, float]:
        hist = mem["fair"].get(product, [])
        rh = mem["resid"].get(product, [])
        spread = self._spread(depth)
        imb = self._imbalance(depth)

        if len(hist) >= 6:
            rets = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
            vol = max(1.0, self._std(rets[-40:]))
        else:
            vol = max(1.0, spread)

        mr = 0.0
        if len(hist) >= 20:
            mu = self._mean(hist[-36:])
            sd = max(1.0, self._std(hist[-36:]))
            mr = self._clip(-(fair - mu) / sd, -3.0, 3.0)

        trend = 0.0
        if len(hist) >= 20:
            fast = self._ema((hist + [fair])[-8:], 4)
            slow = self._ema((hist + [fair])[-28:], 12)
            trend = self._clip((fast - slow) / vol, -3.0, 3.0)

        family = 0.0
        if len(rh) >= 20:
            mu = self._mean(rh[-48:])
            sd = max(0.00015, self._std(rh[-48:]))
            family = self._clip(-(resid - mu) / sd, -3.0, 3.0)

        micro = self._clip(0.9 * imb, -1.5, 1.5)
        return mr, trend, family, micro, vol

    def _alpha(self, product: str, mode: str, mr: float, trend: float, family: float, micro: float, cp: float) -> float:
        if mode == "trend":
            return self._clip(1.00 * trend + 0.20 * micro + 0.30 * cp - 0.10 * mr, -4.0, 4.0)
        if mode == "family":
            return self._clip(1.00 * family + 0.15 * trend + 0.15 * micro + 0.25 * cp, -4.0, 4.0)
        # default: mean reversion
        return self._clip(1.00 * mr + 0.30 * family + 0.15 * micro + 0.25 * cp - 0.05 * trend, -4.0, 4.0)

    def _cap(self, product: str) -> int:
        return 4 if product in self.PASSIVE_ONLY else 6

    def _target(self, product: str, position: int, alpha: float) -> int:
        cap = self._cap(product)
        if abs(alpha) < 0.45:
            return int(round(0.5 * position))
        raw = alpha / 2.0
        return int(self._clip(round(raw * cap), -cap, cap))

    def _add(self, orders: List[Order], product: str, price: int, qty: int, pos: int, buy_used: int, sell_used: int):
        if qty > 0:
            room = self.LIMIT - (pos + buy_used)
            q = min(qty, room)
            if q > 0:
                orders.append(Order(product, int(price), int(q)))
                buy_used += q
        elif qty < 0:
            room = self.LIMIT + (pos - sell_used)
            q = min(-qty, room)
            if q > 0:
                orders.append(Order(product, int(price), int(-q)))
                sell_used += q
        return buy_used, sell_used

    def _make_orders(self, product: str, depth: OrderDepth, fair: float, alpha: float, target: int, pos: int, vol: float, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        bid, bid_v, ask, ask_v = self._best(depth)
        if bid is None or ask is None:
            return orders

        spread = max(1.0, ask - bid)
        buy_used = 0
        sell_used = 0

        if timestamp >= self.FLATTEN_START:
            if pos > 0:
                self._add(orders, product, bid, -min(pos, 3), pos, buy_used, sell_used)
            elif pos < 0:
                self._add(orders, product, ask, min(-pos, 3), pos, buy_used, sell_used)
            return orders

        # Small selective taker leg for directional names only
        if product not in self.PASSIVE_ONLY and abs(alpha) > 1.6:
            pred = fair + alpha * max(1.0, 0.35 * spread + 0.20 * vol)
            take_cost = 0.55 * spread

            if target > pos:
                need = min(2, target - pos)
                for px in sorted(depth.sell_orders.keys()):
                    if need <= 0:
                        break
                    avail = -depth.sell_orders[px]
                    if px <= pred - take_cost:
                        before = buy_used
                        buy_used, sell_used = self._add(orders, product, px, min(need, avail), pos, buy_used, sell_used)
                        need -= buy_used - before

            if target < pos:
                need = min(2, pos - target)
                for px in sorted(depth.buy_orders.keys(), reverse=True):
                    if need <= 0:
                        break
                    avail = depth.buy_orders[px]
                    if px >= pred + take_cost:
                        before = sell_used
                        buy_used, sell_used = self._add(orders, product, px, -min(need, avail), pos, buy_used, sell_used)
                        need -= sell_used - before

        # Passive skewed reservation pricing
        inv = pos / float(self.LIMIT)
        reservation = fair + 0.28 * alpha * spread - inv * max(1.0, 0.75 * spread)

        if spread >= 3:
            bid_px = min(ask - 1, max(bid + 1, int(math.floor(reservation - 0.45 * spread))))
            ask_px = max(bid + 1, min(ask - 1, int(math.ceil(reservation + 0.45 * spread))))
        else:
            bid_px = min(bid, int(math.floor(reservation - 1)))
            ask_px = max(ask, int(math.ceil(reservation + 1)))

        base = 1 if product in self.PASSIVE_ONLY else 2

        if target > pos:
            buy_sz = min(base, max(1, target - pos - buy_used))
            sell_sz = 0 if alpha > 0.8 else 1
        elif target < pos:
            buy_sz = 0 if alpha < -0.8 else 1
            sell_sz = min(base, max(1, pos - target - sell_used))
        else:
            buy_sz = 1 if alpha >= -1.2 else 0
            sell_sz = 1 if alpha <= 1.2 else 0

        if buy_sz > 0 and alpha > -1.5:
            buy_used, sell_used = self._add(orders, product, bid_px, buy_sz, pos, buy_used, sell_used)
        if sell_sz > 0 and alpha < 1.5:
            buy_used, sell_used = self._add(orders, product, ask_px, -sell_sz, pos, buy_used, sell_used)

        return orders

    def run(self, state: TradingState):
        mem = self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        fairs: Dict[str, float] = {}
        fam_logs: Dict[str, List[float]] = defaultdict(list)

        for product, depth in state.order_depths.items():
            fair = self._fair(depth)
            if fair is not None and fair > 0:
                fairs[product] = fair
                fam_logs[self._family(product)].append(math.log(fair))

        fam_mu = {f: self._mean(v) for f, v in fam_logs.items() if v}

        self._resolve_alpha_markouts(mem, state.timestamp, fairs)
        self._resolve_counterparty_markouts(mem, state.timestamp, fairs)

        for product, depth in state.order_depths.items():
            if product not in fairs or product in self.AVOID:
                result[product] = []
                continue

            fair = fairs[product]
            pos = int(state.position.get(product, 0))
            resid = math.log(fair) - fam_mu.get(self._family(product), math.log(fair))

            mr, trend, family, micro, vol = self._signals(mem, product, fair, resid, depth)
            cp = self._counterparty_flow(state, mem, product, fair, vol)
            mode = self.MODE.get(product, "mr")
            alpha = self._alpha(product, mode, mr, trend, family, micro, cp)
            alpha *= self._product_scale(mem, product)

            target = self._target(product, pos, alpha)
            result[product] = self._make_orders(product, depth, fair, alpha, target, pos, vol, state.timestamp)

            if abs(alpha) > 0.5:
                mem["pending_alpha"].append({
                    "p": product,
                    "t": state.timestamp,
                    "fair": fair,
                    "sign": 1.0 if alpha > 0 else -1.0,
                    "vol": vol,
                })

        for product, fair in fairs.items():
            self._append(mem, "fair", product, fair, self.HIST)
            resid = math.log(fair) - fam_mu.get(self._family(product), math.log(fair))
            self._append(mem, "resid", product, resid, self.RESID_HIST)

        return result, 0, self._save(mem)