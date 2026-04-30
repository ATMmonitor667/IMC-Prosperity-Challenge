from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import jsonpickle
import math


class Trader:
    """
    IMC Prosperity 4 Round 5 final trader.

    This version is built from the only architecture that actually worked well in your
    uploaded runs: adaptive counterparty-aware relative value plus safe market making.

    Final synthesis:
    - Keeps the original family-relative value and microprice engine.
    - Fixes the true Round 5 hard limit: 10 per product.
    - Clips every order before it is sent, so the exchange should not reject whole product legs.
    - Starts from the best uploaded run architecture, but removes exact timestamp exits.
    - Uses state-based risk reduction from signal conflict, live markouts, inventory age, and volatility.
    - Adds side-specific adverse-selection tracking for passive bid/ask fills.
    - Adds family-regime scaling and capped anonymous tape-flow.
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
    MAX_PENDING_SIDE = 1600
    MAX_SEEN_TRADES = 5000
    MAX_SEEN_OWN_TRADES = 2500
    INVENTORY_AGE_LIMIT = 5000

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

    # These names start with a low prior, but can earn some size back via live markouts.
    AVOID = {
        "TRANSLATOR_SPACE_GRAY",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "ROBOT_DISHES",
        "PANEL_2X2",
        "PANEL_1X2",
        "OXYGEN_SHAKE_MINT",
        "SLEEP_POD_LAMB_WOOL",
        "MICROCHIP_RECTANGLE",
        "PANEL_1X4",
        "UV_VISOR_YELLOW",
        "UV_VISOR_ORANGE",
        "MICROCHIP_TRIANGLE",
        "ROBOT_LAUNDRY",
        "PEBBLES_M",
        "PEBBLES_S",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "SLEEP_POD_COTTON",
        "SNACKPACK_RASPBERRY",
        "ROBOT_IRONING",
    }

    # Keep hard pruning only for repeated severe losers.
    HARD_AVOID = {
        "TRANSLATOR_SPACE_GRAY",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "ROBOT_DISHES",
        "PANEL_2X2",
        "PANEL_1X2",
        "OXYGEN_SHAKE_MINT",
        "SLEEP_POD_LAMB_WOOL",
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

    RESEARCH_FAMILIES = {
        "SNACKPACKS": [
            "SNACKPACK_CHOCOLATE",
            "SNACKPACK_VANILLA",
            "SNACKPACK_PISTACHIO",
            "SNACKPACK_STRAWBERRY",
            "SNACKPACK_RASPBERRY",
        ],
        "PANELS": [
            "PANEL_1X2",
            "PANEL_2X2",
            "PANEL_1X4",
            "PANEL_2X4",
            "PANEL_4X4",
        ],
        "UV_VISORS": [
            "UV_VISOR_YELLOW",
            "UV_VISOR_AMBER",
            "UV_VISOR_ORANGE",
            "UV_VISOR_RED",
            "UV_VISOR_MAGENTA",
        ],
        "TRANSLATORS": [
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_VOID_BLUE",
        ],
        "GALAXY_SOUNDS": [
            "GALAXY_SOUNDS_DARK_MATTER",
            "GALAXY_SOUNDS_BLACK_HOLES",
            "GALAXY_SOUNDS_PLANETARY_RINGS",
            "GALAXY_SOUNDS_SOLAR_WINDS",
            "GALAXY_SOUNDS_SOLAR_FLAMES",
        ],
        "SLEEP_PODS": [
            "SLEEP_POD_SUEDE",
            "SLEEP_POD_LAMB_WOOL",
            "SLEEP_POD_POLYESTER",
            "SLEEP_POD_NYLON",
            "SLEEP_POD_COTTON",
        ],
        "MICROCHIPS": [
            "MICROCHIP_CIRCLE",
            "MICROCHIP_OVAL",
            "MICROCHIP_SQUARE",
            "MICROCHIP_RECTANGLE",
            "MICROCHIP_TRIANGLE",
        ],
    }

    PANEL_FEATURES = {
        "PANEL_1X2": (1.0, 3.0, 2.0),
        "PANEL_2X2": (1.0, 4.0, 4.0),
        "PANEL_1X4": (1.0, 5.0, 4.0),
        "PANEL_2X4": (1.0, 6.0, 8.0),
        "PANEL_4X4": (1.0, 8.0, 16.0),
    }

    PASSIVE_FV_CONFIG = {
        "PANEL_2X2": {
            "family": "PANELS",
            "edge": 0.25,
            "skew": 0.75,
            "cap": 6,
            "mode": "panel_geometry",
            "warmup": 0,
        },
        "UV_VISOR_MAGENTA": {
            "family": "UV_VISORS",
            "edge": 0.25,
            "skew": 0.75,
            "cap": 5,
            "mode": "rolling_regression",
            "warmup": 45,
        },
        "GALAXY_SOUNDS_SOLAR_FLAMES": {
            "family": "GALAXY_SOUNDS",
            "edge": 0.25,
            "skew": 0.75,
            "cap": 5,
            "mode": "rolling_regression",
            "warmup": 45,
        },
        "SNACKPACK_PISTACHIO": {
            "family": "SNACKPACKS",
            "edge": 0.50,
            "skew": 0.25,
            "cap": 4,
            "mode": "rolling_regression",
            "warmup": 45,
        },
    }

    PAIR_BRANCHES = (
        {
            "left": "SLEEP_POD_POLYESTER",
            "right": "SLEEP_POD_NYLON",
            "lookback": 90,
            "entry": 2.5,
            "exit": 1.0,
            "strength": 1.20,
        },
        {
            "left": "MICROCHIP_CIRCLE",
            "right": "MICROCHIP_RECTANGLE",
            "lookback": 90,
            "entry": 2.5,
            "exit": 0.5,
            "strength": 1.10,
        },
        {
            "left": "TRANSLATOR_ECLIPSE_CHARCOAL",
            "right": "TRANSLATOR_VOID_BLUE",
            "lookback": 90,
            "entry": 2.5,
            "exit": 1.0,
            "strength": 1.00,
        },
    )

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
            "side_perf": {},
            "pending_side": [],
            "seen_own_trades": [],
            "position_state": {},
            "last_quotes": {},
            "tape_flow": {},
            "pair_state": {},
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
        mem["pending_side"] = mem.get("pending_side", [])[-self.MAX_PENDING_SIDE:]
        mem["seen_trades"] = mem.get("seen_trades", [])[-self.MAX_SEEN_TRADES:]
        mem["seen_own_trades"] = mem.get("seen_own_trades", [])[-self.MAX_SEEN_OWN_TRADES:]

        if len(mem.get("cp_score", {})) > 1200:
            items = sorted(
                mem["cp_score"].items(),
                key=lambda kv: abs(float(kv[1].get("s", 0.0))) * max(1, int(kv[1].get("n", 0))),
                reverse=True,
            )
            mem["cp_score"] = dict(items[:800])

        if len(mem.get("side_perf", {})) > 200:
            items = sorted(
                mem["side_perf"].items(),
                key=lambda kv: max(
                    abs(float(kv[1].get("bid", {}).get("s", 0.0))),
                    abs(float(kv[1].get("ask", {}).get("s", 0.0))),
                ),
                reverse=True,
            )
            mem["side_perf"] = dict(items[:150])

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

    def _family_log_means(self, fairs: Dict[str, float]) -> Dict[str, float]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for product, fair in fairs.items():
            if fair > 0:
                groups[self._family(product)].append(math.log(fair))
        return {fam: self._mean(vals) for fam, vals in groups.items() if vals}

    def _solve_linear_system(self, matrix: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
        n = len(rhs)
        if n == 0:
            return []
        aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]

        for col in range(n):
            pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
            if abs(aug[pivot][col]) < 1e-9:
                return None
            if pivot != col:
                aug[col], aug[pivot] = aug[pivot], aug[col]

            div = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= div

            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                if factor == 0:
                    continue
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        return [aug[i][n] for i in range(n)]

    def _least_squares_coeffs(
        self,
        rows: List[List[float]],
        y_values: List[float],
        ridge: float = 1e-5,
    ) -> Optional[List[float]]:
        if not rows or len(rows) != len(y_values):
            return None
        width = len(rows[0])
        if width == 0 or len(rows) < width:
            return None

        xtx = [[0.0 for _ in range(width)] for _ in range(width)]
        xty = [0.0 for _ in range(width)]
        for row, y in zip(rows, y_values):
            if len(row) != width:
                return None
            for i in range(width):
                xty[i] += row[i] * y
                for j in range(width):
                    xtx[i][j] += row[i] * row[j]

        for i in range(width):
            xtx[i][i] += ridge

        return self._solve_linear_system(xtx, xty)

    def _dot(self, left: List[float], right: List[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _rolling_regression_fair(
        self,
        mem: Dict[str, Any],
        product: str,
        config: Dict[str, Any],
        fairs: Dict[str, float],
    ) -> Optional[float]:
        family = str(config["family"])
        products = [p for p in self.RESEARCH_FAMILIES[family] if p != product]
        if product not in fairs or any(p not in fairs for p in products):
            return None

        histories = [mem.get("fair_hist", {}).get(product, [])]
        histories.extend(mem.get("fair_hist", {}).get(p, []) for p in products)
        min_len = min(len(hist) for hist in histories)
        warmup = int(config.get("warmup", 45))
        if min_len < warmup:
            return None

        lookback = min(90, min_len)
        y_hist = histories[0][-lookback:]
        hedge_histories = [hist[-lookback:] for hist in histories[1:]]
        rows: List[List[float]] = []
        for i in range(lookback):
            rows.append([1.0] + [float(hist[i]) for hist in hedge_histories])

        coeffs = self._least_squares_coeffs(rows, [float(v) for v in y_hist], ridge=0.05)
        if coeffs is None:
            return None

        estimate = self._dot(coeffs, [1.0] + [float(fairs[p]) for p in products])
        family_fairs = [float(fairs[p]) for p in self.RESEARCH_FAMILIES[family] if p in fairs]
        if not family_fairs:
            return None

        lo = min(family_fairs) - 2.0 * max(1.0, self._std(family_fairs))
        hi = max(family_fairs) + 2.0 * max(1.0, self._std(family_fairs))
        return self._clip(estimate, lo, hi)

    def _panel_geometry_fair(self, product: str, fairs: Dict[str, float]) -> Optional[float]:
        if product not in self.PANEL_FEATURES:
            return None
        rows = []
        y_values = []
        for panel, features in self.PANEL_FEATURES.items():
            if panel == product:
                continue
            if panel not in fairs:
                return None
            rows.append([float(x) for x in features])
            y_values.append(float(fairs[panel]))

        coeffs = self._least_squares_coeffs(rows, y_values, ridge=1e-4)
        if coeffs is None:
            return None
        return self._dot(coeffs, [float(x) for x in self.PANEL_FEATURES[product]])

    def _passive_branch_fair(
        self,
        mem: Dict[str, Any],
        product: str,
        fairs: Dict[str, float],
    ) -> Optional[float]:
        config = self.PASSIVE_FV_CONFIG.get(product)
        if not config:
            return None
        mode = str(config.get("mode", "rolling_regression"))
        if mode == "panel_geometry":
            return self._panel_geometry_fair(product, fairs)
        return self._rolling_regression_fair(mem, product, config, fairs)

    def _make_passive_fair_value_orders(
        self,
        product: str,
        depth: OrderDepth,
        model_fair: float,
        position: int,
        config: Dict[str, Any],
    ) -> List[Order]:
        bid, _, ask, _ = self._best(depth)
        orders: List[Order] = []
        if bid is None or ask is None:
            return orders

        spread = max(1.0, float(ask - bid))
        mid = 0.5 * (bid + ask)
        cap = int(config.get("cap", 4))
        edge = float(config.get("edge", 0.5))
        skew = float(config.get("skew", 0.5))
        reservation = float(model_fair) - skew * spread * position

        buy_signal = reservation - mid >= edge * spread
        sell_signal = mid - reservation >= edge * spread

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

    def _pair_alpha_overlays(self, mem: Dict[str, Any], fairs: Dict[str, float]) -> Dict[str, float]:
        overlays: Dict[str, float] = defaultdict(float)
        pair_state = mem.setdefault("pair_state", {})

        for config in self.PAIR_BRANCHES:
            left = str(config["left"])
            right = str(config["right"])
            if left not in fairs or right not in fairs:
                continue
            left_hist = mem.get("fair_hist", {}).get(left, [])
            right_hist = mem.get("fair_hist", {}).get(right, [])
            n = min(len(left_hist), len(right_hist), int(config.get("lookback", 90)))
            if n < 45:
                continue

            left_vals = [float(v) for v in left_hist[-n:]]
            right_vals = [float(v) for v in right_hist[-n:]]
            right_mu = self._mean(right_vals)
            left_mu = self._mean(left_vals)
            right_var = sum((x - right_mu) ** 2 for x in right_vals)
            if right_var <= 1e-9:
                beta = 1.0
            else:
                beta = sum((l - left_mu) * (r - right_mu) for l, r in zip(left_vals, right_vals)) / right_var
            if not math.isfinite(beta):
                beta = 1.0

            residuals = [l - beta * r for l, r in zip(left_vals, right_vals)]
            resid_mu = self._mean(residuals)
            resid_sd = max(1.0, self._std(residuals))
            zscore = (float(fairs[left]) - beta * float(fairs[right]) - resid_mu) / resid_sd

            key = f"{left}|{right}"
            side = int(pair_state.get(key, 0))
            entry = float(config.get("entry", 2.5))
            exit_z = float(config.get("exit", 1.0))
            if zscore >= entry:
                side = -1
            elif zscore <= -entry:
                side = 1
            elif abs(zscore) <= exit_z:
                side = 0
            pair_state[key] = side

            if side == 0:
                continue

            hedge_sign = -1.0 if beta >= 0 else 1.0
            strength = float(config.get("strength", 1.0))
            if abs(zscore) > entry + 0.75:
                strength *= 1.25
            overlays[left] += side * strength
            overlays[right] += side * hedge_sign * strength * self._clip(abs(beta), 0.55, 1.35)

        return {product: self._clip(value, -2.2, 2.2) for product, value in overlays.items()}

    def _family_regimes(self, mem: Dict[str, Any], fairs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for product, fair in fairs.items():
            groups[self._family(product)].append((product, fair))

        regimes: Dict[str, Dict[str, float]] = {}
        for fam, items in groups.items():
            pressures = []
            directions = []
            for product, fair in items:
                hist = mem.get("fair_hist", {}).get(product, [])
                if not hist:
                    continue
                ret = fair - float(hist[-1])
                rh = mem.get("ret_hist", {}).get(product, [])
                vol = max(1.0, self._std(rh[-self.RET_HISTORY:]) if len(rh) >= 3 else abs(ret))
                pressures.append(abs(ret) / vol)
                if abs(ret) > 0.20 * vol:
                    directions.append(1.0 if ret > 0 else -1.0)

            if not pressures:
                regimes[fam] = {"rel": 1.0, "trend": 1.0, "micro": 1.0}
                continue

            pressure = self._mean(pressures)
            coherence = abs(sum(directions)) / len(directions) if directions else 0.0
            stress = self._clip(pressure * coherence, 0.0, 3.0)
            regimes[fam] = {
                "rel": self._clip(1.0 / (1.0 + 0.42 * stress), 0.45, 1.0),
                "trend": self._clip(1.0 + 0.22 * stress, 1.0, 1.40),
                "micro": self._clip(1.0 + 0.12 * stress, 1.0, 1.25),
            }
        return regimes

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
            rec["peak"] = max(float(rec.get("peak", rec["s"])), float(rec["s"]))
            rec["n"] = min(9999, n + 1)
            mem["perf"][product] = rec

        mem["pending_alpha"] = pending[-self.MAX_PENDING_ALPHA:]

    def _product_prior_scale(self, product: str) -> float:
        if product in self.HARD_AVOID:
            return 0.0
        if product in self.AVOID:
            return 0.42
        if product in self.STRONG:
            return 1.12
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
        prior = self._product_prior_scale(product)
        if prior <= 0.0:
            return 0.0
        live = self._live_perf_scale(mem, product)
        return self._clip(prior * (0.55 + 0.45 * live), 0.12, 1.24)

    def _product_cap(self, product: str, scale: float) -> int:
        if scale <= 0:
            return 0
        return int(self._clip(round(2 + 5 * scale), 2, 8))

    def _trade_key(self, product: str, tr: Any) -> str:
        return (
            f"{product}|{getattr(tr, 'timestamp', '')}|{getattr(tr, 'buyer', '')}|"
            f"{getattr(tr, 'seller', '')}|{getattr(tr, 'price', '')}|{getattr(tr, 'quantity', '')}"
        )

    def _own_trade_key(self, product: str, tr: Any) -> str:
        return self._trade_key(product, tr)

    def _update_side_score(self, mem: Dict[str, Any], product: str, side: str, score: float) -> None:
        bucket = mem.setdefault("side_perf", {}).setdefault(product, {})
        rec = bucket.get(side, {"s": 0.0, "n": 0})
        n = int(rec.get("n", 0))
        lr = 0.12 if n < 8 else 0.045
        rec["s"] = (1.0 - lr) * float(rec.get("s", 0.0)) + lr * self._clip(score, -4.0, 4.0)
        rec["n"] = min(9999, n + 1)
        bucket[side] = rec

    def _side_quality(self, mem: Dict[str, Any], product: str, side: str) -> float:
        rec = mem.get("side_perf", {}).get(product, {}).get(side)
        if not rec:
            return 0.0
        n = int(rec.get("n", 0))
        trust = min(1.0, n / 8.0)
        return self._clip(trust * float(rec.get("s", 0.0)), -2.0, 2.0)

    def _resolve_side_markouts(self, mem: Dict[str, Any], timestamp: int, fairs: Dict[str, float]) -> None:
        pending = []
        for ev in mem.get("pending_side", []):
            product = ev.get("p")
            if product not in fairs:
                pending.append(ev)
                continue
            if timestamp - int(ev.get("t", timestamp)) < self.MARKOUT_DELAY:
                pending.append(ev)
                continue

            side = ev.get("side", "")
            if side not in ("bid", "ask"):
                continue
            sign = 1.0 if side == "bid" else -1.0
            move = fairs[product] - float(ev.get("fair", fairs[product]))
            vol = max(1.0, float(ev.get("vol", 1.0)))
            self._update_side_score(mem, product, side, sign * move / vol)

        mem["pending_side"] = pending[-self.MAX_PENDING_SIDE:]

    def _record_own_passive_fills(
        self,
        mem: Dict[str, Any],
        state: TradingState,
        fairs: Dict[str, float],
        spreads: Dict[str, float],
    ) -> None:
        seen = set(mem.get("seen_own_trades", []))
        seen_list = list(mem.get("seen_own_trades", []))
        last_quotes = mem.get("last_quotes", {})

        for product, trades in (state.own_trades or {}).items():
            if product not in fairs:
                continue
            quote = last_quotes.get(product, {})
            bid_quote = quote.get("bid")
            ask_quote = quote.get("ask")
            fair = fairs[product]
            vol = max(1.0, spreads.get(product, 1.0))

            for tr in trades:
                key = self._own_trade_key(product, tr)
                if key in seen:
                    continue
                seen.add(key)
                seen_list.append(key)

                buyer = getattr(tr, "buyer", "") or ""
                seller = getattr(tr, "seller", "") or ""
                price = int(getattr(tr, "price", 0))
                side = ""

                if buyer == "SUBMISSION" and bid_quote is not None and price <= int(bid_quote) + 1:
                    side = "bid"
                elif seller == "SUBMISSION" and ask_quote is not None and price >= int(ask_quote) - 1:
                    side = "ask"
                elif not buyer and not seller:
                    if bid_quote is not None and price <= int(bid_quote) + 1:
                        side = "bid"
                    elif ask_quote is not None and price >= int(ask_quote) - 1:
                        side = "ask"

                if side:
                    mem["pending_side"].append({
                        "p": product,
                        "side": side,
                        "t": state.timestamp,
                        "fair": fair,
                        "vol": vol,
                    })

        mem["seen_own_trades"] = seen_list[-self.MAX_SEEN_OWN_TRADES:]

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
        trades = (state.market_trades or {}).get(product, [])
        seen = set(mem.get("seen_trades", []))
        seen_list = list(mem.get("seen_trades", []))
        fam = self._family(product)
        flow = 0.0
        tape = mem.setdefault("tape_flow", {})
        tape[product] = 0.88 * float(tape.get(product, 0.0))

        for tr in trades:
            key = self._trade_key(product, tr)
            is_new = key not in seen
            if is_new:
                seen.add(key)
                seen_list.append(key)

            price = float(getattr(tr, "price", fair))
            qty = max(1, abs(int(getattr(tr, "quantity", 1))))
            qscale = min(2.0, math.sqrt(qty))
            if is_new:
                signed = 0.0
                edge = (price - fair) / max(1.0, vol)
                if edge > 0.15:
                    signed = 1.0
                elif edge < -0.15:
                    signed = -1.0
                if signed:
                    tape[product] = self._clip(float(tape.get(product, 0.0)) + signed * qscale, -6.0, 6.0)

            buyer = getattr(tr, "buyer", "") or ""
            seller = getattr(tr, "seller", "") or ""
            if (not buyer or buyer == "SUBMISSION") and (not seller or seller == "SUBMISSION"):
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

        fam_vals = [
            float(v)
            for p, v in tape.items()
            if p != product and self._family(p) == fam
        ]
        fam_tape = self._mean(fam_vals) if fam_vals else 0.0
        tape_alpha = self._clip(float(tape.get(product, 0.0)) / 3.0 + 0.25 * fam_tape / 3.0, -2.0, 2.0)
        return self._clip(flow + 0.55 * tape_alpha, -5.0, 5.0)

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
        regime: Dict[str, float],
    ) -> float:
        cap = max(1.0, float(self._product_cap(product, scale)))
        inv = -position / cap
        rel_scale = float(regime.get("rel", 1.0))
        trend_scale = float(regime.get("trend", 1.0))
        micro_scale = float(regime.get("micro", 1.0))
        rel *= rel_scale

        # Original v1 bias: family relative value dominates. Mean reversion helps inside
        # each product. Trend is allowed only for names where the uploaded data showed it helps.
        if product in self.TREND_NAMES:
            alpha = 0.55 * rel + 0.25 * own_mr + 0.75 * trend_scale * trend + 0.25 * micro_scale * micro + 0.45 * flow + 0.25 * inv
        elif product in self.FAMILY_NAMES:
            alpha = 1.20 * rel + 0.20 * own_mr + 0.18 * trend_scale * trend + 0.30 * micro_scale * micro + 0.45 * flow + 0.25 * inv
        else:
            alpha = 1.10 * rel + 0.35 * own_mr + 0.10 * trend_scale * trend + 0.32 * micro_scale * micro + 0.45 * flow + 0.28 * inv

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

    def _update_inventory_state(self, mem: Dict[str, Any], timestamp: int, positions: Dict[str, int]) -> None:
        state = mem.setdefault("position_state", {})
        products = set(state.keys()) | set(positions.keys())
        for product in products:
            pos = int(positions.get(product, 0))
            rec = state.get(product, {"q": 0, "since": timestamp})
            old = int(rec.get("q", 0))
            if pos == 0:
                state[product] = {"q": 0, "since": timestamp}
            elif old == 0 or old * pos < 0:
                state[product] = {"q": pos, "since": timestamp}
            else:
                rec["q"] = pos
                state[product] = rec

    def _inventory_age(self, mem: Dict[str, Any], product: str, timestamp: int) -> int:
        rec = mem.get("position_state", {}).get(product)
        if not rec:
            return 0
        return max(0, timestamp - int(rec.get("since", timestamp)))

    def _state_adjust_target(
        self,
        mem: Dict[str, Any],
        product: str,
        position: int,
        target: int,
        alpha: float,
        rel: float,
        own_mr: float,
        trend: float,
        vol: float,
        spread: float,
        timestamp: int,
    ) -> Tuple[int, bool]:
        if position == 0 and abs(alpha) >= 0.55:
            return target, False

        rv = rel + 0.45 * own_mr
        conflict = rv * trend < 0.0 and abs(trend) > 0.85 and abs(rv) > 0.35
        unsupported_inventory = position != 0 and position * alpha <= 0.15 * abs(position)
        age = self._inventory_age(mem, product, timestamp)
        rec = mem.get("perf", {}).get(product, {"s": 0.0, "n": 0})
        live_s = float(rec.get("s", 0.0))
        peak_s = float(rec.get("peak", live_s))
        live_bad = int(rec.get("n", 0)) >= 8 and live_s < -0.10
        live_giveback = int(rec.get("n", 0)) >= 10 and peak_s > 0.35 and live_s < peak_s - 0.35
        noisy = vol > max(2.0, 1.35 * spread)

        repair_only = False
        adjusted = target

        if abs(alpha) < 0.45:
            adjusted = int(round(0.45 * position))
            repair_only = True

        if conflict and (live_bad or live_giveback or noisy or abs(alpha) < 0.90):
            adjusted = int(round(0.50 * adjusted))
            repair_only = True

        if age >= self.INVENTORY_AGE_LIMIT and unsupported_inventory:
            adjusted = int(round(0.35 * position))
            repair_only = True
        elif age >= 2 * self.INVENTORY_AGE_LIMIT and position != 0 and abs(alpha) < 0.85:
            adjusted = int(round(0.50 * position))
            repair_only = True

        return adjusted, repair_only

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
        mem: Dict[str, Any],
        repair_only: bool,
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

        # If near cap, prioritize inventory repair side.
        if position >= cap - 1:
            buy_sz = 0
            sell_sz = max(1, sell_sz)
        if position <= -cap + 1:
            sell_sz = 0
            buy_sz = max(1, buy_sz)

        top_den = max(1, bid_v + ask_v)
        top_imb = (bid_v - ask_v) / top_den

        if top_imb < -0.58 and alpha < 0.90 and position >= 0:
            buy_sz = 0
        if top_imb > 0.58 and alpha > -0.90 and position <= 0:
            sell_sz = 0

        if product in self.CAUTIOUS and spread <= 2 and abs(alpha) < 0.65:
            if position >= 0:
                buy_sz = 0
            if position <= 0:
                sell_sz = 0

        bid_quality = self._side_quality(mem, product, "bid")
        ask_quality = self._side_quality(mem, product, "ask")
        bid_conf = alpha + 0.30 * top_imb + bid_quality
        ask_conf = -alpha - 0.30 * top_imb + ask_quality

        if repair_only or abs(alpha) < 0.50:
            if position >= 0:
                buy_sz = 0
            if position <= 0:
                sell_sz = 0

        if bid_conf < -0.25 and position >= 0:
            buy_sz = 0
        if ask_conf < -0.25 and position <= 0:
            sell_sz = 0

        if bid_quality < -0.55 and position >= 0:
            buy_sz = 0
        if ask_quality < -0.55 and position <= 0:
            sell_sz = 0

        if buy_sz > 0 and alpha > -1.55:
            buy_used, sell_used = self._add_order(orders, product, bid_px, buy_sz, position, buy_used, sell_used)
        if sell_sz > 0 and alpha < 1.55:
            buy_used, sell_used = self._add_order(orders, product, ask_px, -sell_sz, position, buy_used, sell_used)

        return orders

    def _passive_quote_record(self, orders: List[Order], bid: int, ask: int) -> Dict[str, int]:
        record: Dict[str, int] = {}
        bids = [int(o.price) for o in orders if int(o.quantity) > 0 and int(o.price) < ask]
        asks = [int(o.price) for o in orders if int(o.quantity) < 0 and int(o.price) > bid]
        if bids:
            record["bid"] = max(bids)
        if asks:
            record["ask"] = min(asks)
        return record

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
        family_regimes = self._family_regimes(mem, fairs)

        self._update_inventory_state(mem, state.timestamp, state.position)
        self._resolve_alpha_markouts(mem, state.timestamp, fairs)
        self._resolve_cp_markouts(mem, state.timestamp, fairs)
        self._resolve_side_markouts(mem, state.timestamp, fairs)
        self._record_own_passive_fills(mem, state, fairs, spreads)
        pair_overlays = self._pair_alpha_overlays(mem, fairs)

        next_quotes: Dict[str, Dict[str, int]] = {}

        for product, depth in state.order_depths.items():
            if product not in fairs:
                result[product] = []
                continue

            position = int(state.position.get(product, 0))
            fair = fairs[product]
            spread = spreads.get(product, 1.0)

            passive_config = self.PASSIVE_FV_CONFIG.get(product)
            if passive_config:
                branch_fair = self._passive_branch_fair(mem, product, fairs)
                if branch_fair is not None:
                    branch_orders = self._make_passive_fair_value_orders(
                        product,
                        depth,
                        branch_fair,
                        position,
                        passive_config,
                    )
                    if branch_orders or product in self.HARD_AVOID:
                        result[product] = branch_orders
                        bid, _, ask, _ = self._best(depth)
                        if bid is not None and ask is not None:
                            quote_record = self._passive_quote_record(branch_orders, bid, ask)
                            if quote_record:
                                next_quotes[product] = quote_record
                        continue

            if product in self.HARD_AVOID:
                result[product] = self._flatten_orders(product, depth, position)
                continue

            resid = math.log(fair) - family_mu.get(self._family(product), math.log(fair))
            rel, own_mr, trend, micro, vol = self._signals(mem, product, fair, resid, depth)
            flow = self._counterparty_flow(mem, state, product, fair, vol)
            scale = self._product_scale(mem, product)
            alpha = self._combine_alpha(
                product,
                rel,
                own_mr,
                trend,
                micro,
                flow,
                position,
                scale,
                family_regimes.get(self._family(product), {}),
            )
            alpha = self._clip(alpha + pair_overlays.get(product, 0.0), -5.0, 5.0)

            target = self._target_position(product, position, alpha, scale, vol, spread)
            target, repair_only = self._state_adjust_target(
                mem, product, position, target, alpha, rel, own_mr, trend, vol, spread, state.timestamp
            )
            orders = self._make_orders(product, depth, fair, alpha, target, position, vol, spread, scale, mem, repair_only)
            result[product] = orders

            bid, _, ask, _ = self._best(depth)
            if bid is not None and ask is not None:
                quote_record = self._passive_quote_record(orders, bid, ask)
                if quote_record:
                    next_quotes[product] = quote_record

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

        mem["last_quotes"] = next_quotes

        return result, 0, self._save_memory(mem)
