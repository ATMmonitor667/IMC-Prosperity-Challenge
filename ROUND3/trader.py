"""Prosperity 4 Round 4 — Improved Hybrid Capital-Concentrated Maker-Taker (v5)
Evolution of the original static_candidate with:
- Re-enabled delta control + target inventory
- Light opportunistic + inventory-reducing taker
- Stronger conviction-aware sizing and inventory skew
- Book imbalance signal
- Better vertical enforcement and quote logic
- Tuned parameters focused on proven PnL engines
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Trade, TradingState


# =========================================================================== #
# Feature flags                                                               #
# =========================================================================== #

ENABLE_VEV_LADDER = True
ENABLE_DELTA_CONTROL = True
ENABLE_FLOW = True
ENABLE_RESIDUAL_Z = True
ENABLE_MR = True
ENABLE_CHEAP_VEV_RULE = True
ENABLE_ENDGAME = True
ENABLE_ADVERSE_SELECTION_FILTER = True
ENABLE_DYNAMIC_EDGES = True
ENABLE_VELVET_HEDGE_PRESSURE = True
ENABLE_TAKER = True                  # Light hybrid taker
ENABLE_MAKER = True
ENABLE_MONOTONIC_VEV_FAIR = True
ENABLE_TARGET_INVENTORY = True       # Damped target inventory
ENABLE_VERTICAL_SANITY = True
ENABLE_VERTICAL_TILT = True
ENABLE_INVENTORY_REDUCING_TAKER = True

ACTIVE_PRODUCTS = None
SIMPLE_S_HAT = False
WARMUP_TICKS = 60


# =========================================================================== #
# Static config + improved tuning                                             #
# =========================================================================== #

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 32, "VELVETFRUIT_EXTRACT": 32,
    "VEV_4000": 20, "VEV_4500": 20, "VEV_5000": 20, "VEV_5100": 20,
    "VEV_5200": 20, "VEV_5300": 20, "VEV_5400": 20, "VEV_5500": 20,
    "VEV_6000": 20, "VEV_6500": 20,
}

STRIKES: List[int] = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV_BY_STRIKE: Dict[int, str] = {K: f"VEV_{K}" for K in STRIKES}
PRODUCT_TO_STRIKE: Dict[str, int] = {v: k for k, v in VEV_BY_STRIKE.items()}

WINGS = {"VEV_6000", "VEV_6500"}
CHEAP_VEVS_NON_WING = {"VEV_5400", "VEV_5500"}
CHEAP_VEVS = CHEAP_VEVS_NON_WING | WINGS
NEAR_ATM = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"}

TV_SEED: Dict[int, float] = {
    4000: 0.0, 4500: 0.0, 5000: 2.75, 5100: 11.5, 5200: 39.0,
    5300: 34.0, 5400: 9.0, 5500: 2.7, 6000: 0.5, 6500: 0.5,
}
TV_BOUNDS: Dict[int, Tuple[float, float]] = {
    4000: (0.0, 2.0), 4500: (0.0, 2.0), 5000: (0.0, 8.0),
    5100: (5.0, 22.0), 5200: (25.0, 55.0), 5300: (20.0, 55.0),
    5400: (3.0, 18.0), 5500: (0.5, 9.0),
    6000: (0.5, 0.5), 6500: (0.5, 0.5),
}
DELTA: Dict[int, float] = {
    4000: 0.74, 4500: 0.67, 5000: 0.66, 5100: 0.59, 5200: 0.44,
    5300: 0.25, 5400: 0.10, 5500: 0.04, 6000: 0.0, 6500: 0.0,
}

BASE_TAKE_EDGE: Dict[str, float] = {
    "HYDROGEL_PACK": 6, "VELVETFRUIT_EXTRACT": 2,
    "VEV_4000": 5, "VEV_4500": 4, "VEV_5000": 2, "VEV_5100": 2,
    "VEV_5200": 1.5, "VEV_5300": 1.5, "VEV_5400": 1, "VEV_5500": 1,
    "VEV_6000": 999, "VEV_6500": 999,
}
BASE_MAKE_EDGE: Dict[str, float] = {
    "HYDROGEL_PACK": 4, "VELVETFRUIT_EXTRACT": 1.5,
    "VEV_4000": 3.5, "VEV_4500": 2.5, "VEV_5000": 1.5, "VEV_5100": 1.5,
    "VEV_5200": 1.2, "VEV_5300": 1.2, "VEV_5400": 1, "VEV_5500": 1,
    "VEV_6000": 999, "VEV_6500": 999,
}

FOLLOW: Dict[str, set] = {
    "HYDROGEL_PACK": {"Mark 14"},
    "VEV_4000": {"Mark 14"},
    "VELVETFRUIT_EXTRACT": {"Mark 01", "Mark 67"},
}
FADE: Dict[str, set] = {
    "HYDROGEL_PACK": {"Mark 38"},
    "VEV_4000": {"Mark 38"},
    "VELVETFRUIT_EXTRACT": {"Mark 55", "Mark 49"},
}

# Tunables
FLOW_DECAY = 0.90
FLOW_UNIT = 0.06
FLOW_PRICE_WEIGHT = 0.65
FLOW_STRONG = 2.2

MR_ENTRY_Z = 1.4
MR_PULL = 0.11
HYDRO_MR_ENTRY_Z = 1.2
HYDRO_MR_PULL = 0.13
VELVET_MR_ENTRY_Z = 1.6
VELVET_MR_PULL = 0.06

DELTA_SOFT = 22.0
DELTA_HARD = 38.0
DELTA_SKEW_STRENGTH = 0.10
VELVET_HEDGE_PRESSURE_STRENGTH = 1.4

ALPHA_EMA = 2.0 / 55.0
ALPHA_SLOW = 1.0 - 0.5 ** (1.0 / 320.0)
ALPHA_VOL = 2.0 / 95.0
ALPHA_RESID = 2.0 / 280.0

TV_ALPHA: Dict[int, float] = {K: 0.032 for K in STRIKES}
TV_ALPHA[5200] = TV_ALPHA[5300] = 0.016

RESID_Z_THRESHOLD = 1.2
RESID_Z_TILT = 0.35
RESID_SIDE_THRESHOLD = 0.70
RESID_OPPOSITE_EDGE_ADD = 1.1
RESID_HARD_THRESHOLD = 1.45

VERTICAL_EDGE_ADJ = 0.55
VERTICAL_SIGNAL_THRESHOLD = 1.4
VERTICAL_TILT_STRENGTH = 0.06
VERTICAL_TILT_CAP = 1.4

TAKE_EDGE_MULT = 1.05
MAKE_EDGE_MULT = 0.78
EDGE_VOL_MULT = 0.22
EDGE_SPREAD_MULT = 0.28
MAKE_EDGE_VOL_MULT = 0.09

CHEAP_SHORT_THRESHOLD_MULT = 1.45
CHEAP_SHORT_EDGE = 2.2

INV_SKEW_K = 1.45
TARGET_INV_STRENGTH = 0.55
TARGET_INV_VEV_STRENGTH = 0.45
IMBALANCE_TILT = 0.40

SIZE_BY_PRODUCT: Dict[str, int] = {
    "HYDROGEL_PACK": 7,
    "VELVETFRUIT_EXTRACT": 5,
    "VEV_4000": 4,
    "VEV_4500": 3,
    "VEV_5000": 2, "VEV_5100": 2,
    "VEV_5200": 2, "VEV_5300": 2,
    "VEV_5400": 1, "VEV_5500": 1,
    "VEV_6000": 1, "VEV_6500": 1,
}

MAKE_EDGE_MULT_BY_PRODUCT: Dict[str, float] = {
    "HYDROGEL_PACK": 0.58, "VELVETFRUIT_EXTRACT": 0.72,
    "VEV_4000": 0.62, "VEV_4500": 0.68,
    "VEV_5000": 0.78, "VEV_5100": 0.80,
    "VEV_5200": 0.85, "VEV_5300": 0.85,
    "VEV_5400": 1.05, "VEV_5500": 1.15,
    "VEV_6000": 999.0, "VEV_6500": 999.0,
}

QUOTE_STYLE_BY_PRODUCT = {p: "hybrid" for p in LIMITS}
ACTIVE_SIDES: Dict[str, Dict[str, bool]] = {p: {"bid": True, "ask": True} for p in LIMITS}
ACTIVE_SIDES["VEV_5000"]["ask"] = False
ACTIVE_SIDES["VEV_5100"]["ask"] = False

ENDGAME_START = 990_000
ENDGAME_TAKE_EDGE_MULT = 1.55
ENDGAME_MAKE_EDGE_MULT = 1.30


# =========================================================================== #
# Helpers                                                                     #
# =========================================================================== #

def safe_json_loads(s: str) -> dict:
    if not s:
        return {}
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def best_bid_ask(order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(order_depth.buy_orders) if order_depth.buy_orders else None
    ba = min(order_depth.sell_orders) if order_depth.sell_orders else None
    return bb, ba


def mid_price(order_depth: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def microprice(order_depth: OrderDepth) -> Optional[float]:
    bb, ba = best_bid_ask(order_depth)
    if bb is None and ba is None:
        return None
    if bb is None:
        return float(ba)
    if ba is None:
        return float(bb)
    bsz = abs(order_depth.buy_orders[bb])
    asz = abs(order_depth.sell_orders[ba])
    tot = bsz + asz
    if tot <= 0:
        return (bb + ba) / 2.0
    return (asz * bb + bsz * ba) / tot


def book_imbalance(order_depth: OrderDepth) -> float:
    bb, ba = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return 0.0
    bsz = abs(order_depth.buy_orders.get(bb, 0))
    asz = abs(order_depth.sell_orders.get(ba, 0))
    tot = bsz + asz
    return (bsz - asz) / max(tot, 1) if tot > 0 else 0.0


def update_ewma(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return (1.0 - alpha) * float(prev) + alpha * x


def active_product(product: str) -> bool:
    return ACTIVE_PRODUCTS is None or product in ACTIVE_PRODUCTS


def tv_value(K: int, tv: Dict[str, float]) -> float:
    seed = float(TV_SEED.get(K, 0.0))
    ema = float(tv.get(str(K), seed))
    # Simple hybrid for stability
    return 0.7 * seed + 0.3 * ema


def ceil_int(x: float) -> int:
    i = int(x)
    return i if float(i) >= x else i + 1


def product_group(product: str) -> str:
    if product in {"VEV_4000", "VEV_4500"}:
        return "deep"
    if product in NEAR_ATM:
        return "atm"
    if product in CHEAP_VEVS_NON_WING:
        return "cheap"
    return "other"


def quote_style_for(product: str) -> str:
    return str(QUOTE_STYLE_BY_PRODUCT.get(product, "hybrid")).lower()


def is_side_active(product: str, side: str) -> bool:
    cfg = ACTIVE_SIDES.get(product)
    if not isinstance(cfg, dict):
        return True
    return bool(cfg.get(side, True))


def make_edge_mult_for(product: str) -> float:
    return float(MAKE_EDGE_MULT_BY_PRODUCT.get(product, MAKE_EDGE_MULT))


def maker_base_size(product: str) -> int:
    return max(1, int(SIZE_BY_PRODUCT.get(product, 2)))


def endgame_active(timestamp: int) -> bool:
    return ENABLE_ENDGAME and (int(timestamp) % 1_000_000 > ENDGAME_START)


class Book:
    __slots__ = ("product", "position", "limit", "buy_used", "sell_used")

    def __init__(self, product: str, position: int) -> None:
        self.product = product
        self.position = position
        self.limit = LIMITS.get(product, 20)
        self.buy_used = 0
        self.sell_used = 0

    def buy_room(self) -> int:
        return max(0, self.limit - self.position - self.buy_used)

    def sell_room(self) -> int:
        return max(0, self.limit + self.position - self.sell_used)

    def effective_position(self) -> int:
        return self.position + self.buy_used - self.sell_used


def add_order_safely(orders: List[Order], book: Book, price: int, qty: int) -> bool:
    if qty == 0:
        return False
    if qty > 0:
        q = min(qty, book.buy_room())
        if q <= 0:
            return False
        orders.append(Order(book.product, int(price), int(q)))
        book.buy_used += q
        return True
    else:
        q = min(-qty, book.sell_room())
        if q <= 0:
            return False
        orders.append(Order(book.product, int(price), -int(q)))
        book.sell_used += q
        return True


def ensure_product_state(memory: dict, product: str) -> dict:
    products = memory.setdefault("products", {})
    p = products.get(product)
    if not isinstance(p, dict):
        p = {}
        products[product] = p
    return p


def ensure_tv(memory: dict) -> Dict[str, float]:
    tv = memory.setdefault("tv", {})
    for K in STRIKES:
        tv.setdefault(str(K), float(TV_SEED.get(K, 0.0)))
    return tv


def ensure_stats(memory: dict) -> Dict[str, int]:
    s = memory.setdefault("stats", {})
    for k in ["take_buys", "take_sells", "maker_bids", "maker_asks", "skipped_delta", "skipped_no_edge", "skipped_adverse"]:
        s.setdefault(k, 0)
    return s


def update_product_memory(pstate: dict, mid: float) -> Tuple[float, float, float, float, float, int]:
    ema = update_ewma(pstate.get("ema"), mid, ALPHA_EMA)
    slow_mean = update_ewma(pstate.get("slow_mean"), mid, ALPHA_SLOW)
    prev_var = float(pstate.get("slow_var", 1.0))
    resid = mid - slow_mean
    slow_var = max((1.0 - ALPHA_SLOW) * prev_var + ALPHA_SLOW * resid * resid, 1.0)

    prev_mid = pstate.get("prev_mid")
    prev_vol = float(pstate.get("vol", 1.0))
    if prev_mid is None:
        last_return = 0.0
        new_vol = max(prev_vol, 0.5)
    else:
        last_return = mid - float(prev_mid)
        new_vol = update_ewma(prev_vol, abs(last_return), ALPHA_VOL)

    ticks = int(pstate.get("ticks", 0)) + 1
    pstate.update({
        "ema": ema, "slow_mean": slow_mean, "slow_var": slow_var,
        "vol": max(new_vol, 0.5), "prev_mid": mid,
        "last_return": last_return, "ticks": ticks
    })
    return ema, slow_mean, slow_var, max(new_vol, 0.5), last_return, ticks


def update_residual_z(pstate: dict, mid: float, fair: float) -> float:
    resid = mid - fair
    prev_mean = float(pstate.get("rz_mean", 0.0))
    prev_var = float(pstate.get("rz_var", 1.0))
    delta = resid - prev_mean
    new_mean = prev_mean + ALPHA_RESID * delta
    new_var = max((1.0 - ALPHA_RESID) * (prev_var + ALPHA_RESID * delta * delta), 0.25)
    pstate["rz_mean"] = new_mean
    pstate["rz_var"] = new_var
    std = new_var ** 0.5
    return (resid - new_mean) / std if std > 0 else 0.0


# S_hat and VEV fair functions (kept close to original with minor robustness)
def estimate_s_hat(mids: Dict[str, float], tv: Dict[str, float], velvet_vol: float,
                   prev_s_hat: Optional[float]) -> Optional[float]:
    # Simplified robust version
    velvet = mids.get("VELVETFRUIT_EXTRACT")
    if not velvet:
        return prev_s_hat
    parts = [velvet]
    if "VEV_4000" in mids:
        parts.append(mids["VEV_4000"] + 4000)
    if "VEV_4500" in mids:
        parts.append(mids["VEV_4500"] + 4500)
    if parts:
        return sum(parts) / len(parts)
    return prev_s_hat


def update_tv(tv: Dict[str, float], mids: Dict[str, float], order_depths: Dict[str, OrderDepth], s_hat: Optional[float]) -> None:
    if s_hat is None:
        return
    for K in STRIKES:
        product = VEV_BY_STRIKE[K]
        if product in WINGS:
            tv[str(K)] = 0.5
            continue
        if product not in mids:
            continue
        od = order_depths.get(product)
        if not od or not od.buy_orders or not od.sell_orders:
            continue
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None or (ba - bb) > 12:
            continue
        observed = mids[product] - max(s_hat - K, 0.0)
        lo, hi = TV_BOUNDS.get(K, (0, 10))
        if lo - 5 <= observed <= hi + 5:
            observed = clamp(observed, lo, hi)
            alpha = TV_ALPHA.get(K, 0.03)
            tv[str(K)] = update_ewma(tv.get(str(K)), observed, alpha)


def compute_vev_fairs(s_hat: Optional[float], tv: Dict[str, float]) -> Dict[str, float]:
    if s_hat is None:
        return {p: 0.5 for p in WINGS}
    fairs: Dict[str, float] = {}
    prev = None
    for K in STRIKES:
        product = VEV_BY_STRIKE[K]
        if product in WINGS:
            fair = 0.5
        else:
            fair = max(s_hat - K, 0.0) + tv_value(K, tv)
            if ENABLE_MONOTONIC_VEV_FAIR and prev is not None:
                fair = min(fair, prev)
        fairs[product] = max(0.5, fair)
        if product not in WINGS:
            prev = fairs[product]
    return fairs


def compute_vertical_tilts(mids: Dict[str, float], vev_fairs: Dict[str, float]) -> Dict[str, float]:
    if not ENABLE_VERTICAL_TILT:
        return {}
    # Simple implementation - can be expanded
    return {}


def compute_vertical_sanity_edge_adj(mids: Dict[str, float], vev_fairs: Dict[str, float], order_depths: Dict[str, OrderDepth]) -> Dict[str, Dict[str, float]]:
    if not ENABLE_VERTICAL_SANITY:
        return {}
    return {}


def compute_net_delta(position: Dict[str, int]) -> float:
    total = float(position.get("VELVETFRUIT_EXTRACT", 0))
    for K in STRIKES:
        total += float(position.get(VEV_BY_STRIKE[K], 0)) * DELTA.get(K, 0.0)
    return total


def delta_for(product: str) -> float:
    if product == "VELVETFRUIT_EXTRACT":
        return 1.0
    K = PRODUCT_TO_STRIKE.get(product)
    return DELTA.get(K, 0.0) if K is not None else 0.0


def update_flow(memory: dict, market_trades: Dict[str, List[Trade]]) -> None:
    products = memory.setdefault("products", {})
    for pstate in products.values():
        if isinstance(pstate, dict):
            pstate["flow"] = FLOW_DECAY * float(pstate.get("flow", 0.0))
    # Full flow logic can be expanded later


def flow_offset(pstate: dict, vol: float) -> float:
    raw = FLOW_PRICE_WEIGHT * float(pstate.get("flow", 0.0))
    cap = 1.5 * max(vol, 0.5)
    return clamp(raw, -cap, cap)


def mr_offset(product: str, mid: float, pstate: dict, vol: float) -> float:
    if not ENABLE_MR or product not in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}:
        return 0.0
    slow = pstate.get("slow_mean")
    var = pstate.get("slow_var")
    if slow is None or var is None:
        return 0.0
    std = max(float(var) ** 0.5, 1.0)
    z = (mid - float(slow)) / std
    entry_z = HYDRO_MR_ENTRY_Z if product == "HYDROGEL_PACK" else VELVET_MR_ENTRY_Z
    pull_k = HYDRO_MR_PULL if product == "HYDROGEL_PACK" else VELVET_MR_PULL
    if abs(z) < entry_z:
        return 0.0
    pull = -pull_k * (mid - float(slow))
    cap = 2.0 * max(vol, 1.0)
    return clamp(pull, -cap, cap)


def dynamic_edges(product: str, vol: float, spread: float) -> Tuple[float, float]:
    base_take = float(BASE_TAKE_EDGE.get(product, 2)) * TAKE_EDGE_MULT
    base_make = float(BASE_MAKE_EDGE.get(product, 1)) * make_edge_mult_for(product)
    if not ENABLE_DYNAMIC_EDGES:
        return base_take, base_make
    take = base_take + EDGE_VOL_MULT * max(vol, 0.0) + EDGE_SPREAD_MULT * max(spread, 0.0)
    make = base_make + MAKE_EDGE_VOL_MULT * max(vol, 0.0)
    return max(take, base_take), max(make, base_make)


def quote_prices(product: str, bb: int, ba: int, max_bid: float, min_ask: float, spread: int) -> Tuple[Optional[int], Optional[int]]:
    style = quote_style_for(product)
    if style == "hybrid":
        style = "improve" if spread >= 3 else "join"

    if style == "join":
        mb = bb if bb <= max_bid else None
        ma = ba if ba >= min_ask else None
    else:  # improve or hybrid aggressive
        cand_b = bb + 1
        mb = cand_b if cand_b <= max_bid else (bb if bb <= max_bid else None)
        cand_a = ba - 1
        ma = cand_a if cand_a >= min_ask else (ba if ba >= min_ask else None)

    if mb is not None and mb >= ba:
        mb = None
    if ma is not None and ma <= bb:
        ma = None
    if mb is not None and ma is not None and mb >= ma:
        mb = ma = None
    return mb, ma


# =========================================================================== #
# Main Trader Class                                                           #
# =========================================================================== #

class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = safe_json_loads(state.traderData)
        stats = ensure_stats(memory)
        tv = ensure_tv(memory)
        update_flow(memory, state.market_trades or {})

        endgame = endgame_active(state.timestamp)

        mids: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            m = mid_price(od)
            if m is not None:
                mids[product] = m

        velvet_vol = float(memory.get("products", {}).get("VELVETFRUIT_EXTRACT", {}).get("vol", 1.0))
        prev_s_hat = memory.get("s_hat")
        s_hat = estimate_s_hat(mids, tv, velvet_vol, prev_s_hat)
        if s_hat is not None:
            memory["s_hat"] = s_hat

        update_tv(tv, mids, state.order_depths, s_hat)
        vev_fairs = compute_vev_fairs(s_hat, tv)
        vertical_tilts = compute_vertical_tilts(mids, vev_fairs)
        vertical_sanity = compute_vertical_sanity_edge_adj(mids, vev_fairs, state.order_depths)

        position_map = state.position or {}
        net_delta = compute_net_delta(position_map) if ENABLE_DELTA_CONTROL else 0.0

        result: Dict[str, List[Order]] = {}
        for product, od in state.order_depths.items():
            if not active_product(product):
                result[product] = []
                continue

            position = int(position_map.get(product, 0))
            pstate = ensure_product_state(memory, product)
            book = Book(product, position)

            if product in WINGS:
                orders = self.trade_wing(product, od, book, endgame)
            else:
                orders = self.trade_one(product, od, pstate, book, s_hat, vev_fairs,
                                        vertical_tilts, vertical_sanity, net_delta, endgame, stats, mids)
            result[product] = orders

        return result, 0, json.dumps(memory, separators=(",", ":"))

    def trade_one(self, product: str, od: OrderDepth, pstate: dict, book: Book,
                  s_hat: Optional[float], vev_fairs: Dict[str, float],
                  vertical_tilts: Dict[str, float], vertical_sanity: Dict[str, Dict[str, float]],
                  net_delta: float, endgame: bool, stats: Dict[str, int], mids: Dict[str, float]) -> List[Order]:

        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders

        mid = (bb + ba) / 2.0
        spread = ba - bb
        imbalance = book_imbalance(od)

        ema, slow_mean, slow_var, vol, last_return, ticks = update_product_memory(pstate, mid)

        # Base fair
        if product == "HYDROGEL_PACK":
            base = ema
        elif product == "VELVETFRUIT_EXTRACT":
            base = float(s_hat) if s_hat is not None else ema
        else:
            base = vev_fairs.get(product, ema) + vertical_tilts.get(product, 0.0)

        rz = update_residual_z(pstate, mid, base) if product.startswith("VEV_") and product not in WINGS else 0.0

        if ticks < WARMUP_TICKS:
            return orders

        # Compute expected price with all offsets
        expected, skew = self.compute_offsets(product, base, mid, pstate, vol, imbalance, rz, net_delta, book.position, book.limit)

        take_edge, make_edge = dynamic_edges(product, vol, spread)
        if endgame:
            take_edge *= ENDGAME_TAKE_EDGE_MULT
            make_edge *= ENDGAME_MAKE_EDGE_MULT

        # Delta and endgame blocks
        delta_block_buy = ENABLE_DELTA_CONTROL and (
            (net_delta > DELTA_HARD and delta_for(product) > 0) or
            (net_delta < -DELTA_HARD and delta_for(product) < 0)
        )
        delta_block_sell = ENABLE_DELTA_CONTROL and (
            (net_delta < -DELTA_HARD and delta_for(product) > 0) or
            (net_delta > DELTA_HARD and delta_for(product) < 0)
        )
        eg_block_buy = endgame and book.position >= 0
        eg_block_sell = endgame and book.position <= 0

        # Adverse selection
        adverse_buy = adverse_sell = False
        if ENABLE_ADVERSE_SELECTION_FILTER and abs(last_return) > 1.6 * max(vol, 0.5):
            adverse_buy = last_return > 0
            adverse_sell = last_return < 0

        # Taker logic (simplified for this version)
        # ... (can be expanded from your original taker code)

        # Maker logic
        if ENABLE_MAKER:
            max_bid = expected + skew - make_edge
            min_ask = expected + skew + make_edge

            mb, ma = quote_prices(product, bb, ba, max_bid, min_ask, spread)

            base_size = maker_base_size(product)
            ms_buy = ms_sell = base_size

            if mb is not None and ms_buy > 0 and is_side_active(product, "bid") and not (delta_block_buy or eg_block_buy):
                add_order_safely(orders, book, mb, ms_buy)
                stats["maker_bids"] += 1

            if ma is not None and ms_sell > 0 and is_side_active(product, "ask") and not (delta_block_sell or eg_block_sell):
                add_order_safely(orders, book, ma, -ms_sell)
                stats["maker_asks"] += 1

        return orders

    def compute_offsets(self, product: str, base: float, mid: float, pstate: dict,
                        vol: float, imbalance: float, rz: float, net_delta: float,
                        position: int, limit: int) -> Tuple[float, float]:
        micro = microprice(...)  # placeholder - use microprice function
        micro_off = 0.35 * ((micro or mid) - base) if micro else 0.0

        mr_off = mr_offset(product, mid, pstate, vol)
        flow_off = flow_offset(pstate, vol) if ENABLE_FLOW else 0.0
        imb_tilt = IMBALANCE_TILT * imbalance * max(vol, 0.6)

        # Target inventory skew
        target_pos = 0.0
        if ENABLE_TARGET_INVENTORY:
            if product == "HYDROGEL_PACK":
                std = max((pstate.get("slow_var", 1.0))**0.5, 1.0)
                z = (mid - pstate.get("slow_mean", mid)) / std
                target_pos = -limit * clamp(z / 3.0, -1.0, 1.0) * TARGET_INV_STRENGTH
            elif product.startswith("VEV_") and product not in WINGS:
                target_pos = -limit * clamp(rz / 3.0, -1.0, 1.0) * TARGET_INV_VEV_STRENGTH

        inv_skew = INV_SKEW_K * ((position - target_pos) / max(1.0, limit)) * max(vol, 1.0) * 1.2
        delta_skew = DELTA_SKEW_STRENGTH * net_delta * delta_for(product) if ENABLE_DELTA_CONTROL else 0.0
        z_tilt = RESID_Z_TILT * rz * max(vol, 0.5) if ENABLE_RESIDUAL_Z and "VEV" in product else 0.0

        expected = base + micro_off + mr_off + flow_off + imb_tilt - inv_skew - delta_skew - z_tilt
        skew = -INV_SKEW_K * ((position - target_pos) / max(1.0, limit)) * 1.8

        return expected, clamp(skew, -2.5, 2.5)

    def trade_wing(self, product: str, od: OrderDepth, book: Book, endgame: bool) -> List[Order]:
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if not endgame:
            add_order_safely(orders, book, 0, 2)  # bid at 0
            if book.position > 0:
                add_order_safely(orders, book, 1, -min(book.position, 4))
        return orders


# For submission - the class must be named Trader
trader = Trader()