"""Prosperity 4 Round 4 trader (v3 - maker-first hybrid).

Measured direction:
- Per-tick reservation accounting (Book) is the single source of capacity.
- HYDROGEL remains the main independent market-making book.
- VELVET plus deep ITM VEV anchors feed the VEV fair-value ladder.
- Maker quotes drive PnL; aggressive taker logic is disabled by default.
- Delta hard gates, VELVET hedge pressure, and adverse-selection skips are disabled.
- Residual VEV tilt, MR, dynamic maker edges, TV updates, and inventory skew stay active.
- Lightweight diagnostics counters in traderData.
- Feature flags for clean ablation.

Runtime depends only on the prosperity datamodel.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, Trade, TradingState


# =========================================================================== #
# Feature flags                                                                #
# =========================================================================== #

ENABLE_VEV_LADDER = True
ENABLE_DELTA_CONTROL = False
ENABLE_FLOW = False
ENABLE_RESIDUAL_Z = True
ENABLE_MR = True
ENABLE_CHEAP_VEV_RULE = True
ENABLE_ENDGAME = True
ENABLE_ADVERSE_SELECTION_FILTER = False
ENABLE_DYNAMIC_EDGES = True
ENABLE_VELVET_HEDGE_PRESSURE = False
ENABLE_TAKER = False
ENABLE_MAKER = True
ENABLE_MONOTONIC_VEV_FAIR = True

# Test switches; None means trade every listed product.
ACTIVE_PRODUCTS = None
SIMPLE_S_HAT = False


# =========================================================================== #
# Static config                                                                #
# =========================================================================== #

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 32,
    "VELVETFRUIT_EXTRACT": 32,
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
    "VEV_4000": 6, "VEV_4500": 5, "VEV_5000": 2, "VEV_5100": 2,
    "VEV_5200": 1, "VEV_5300": 1, "VEV_5400": 1, "VEV_5500": 1,
    "VEV_6000": 999, "VEV_6500": 999,
}
BASE_MAKE_EDGE: Dict[str, float] = {
    "HYDROGEL_PACK": 4, "VELVETFRUIT_EXTRACT": 1.5,
    "VEV_4000": 4, "VEV_4500": 3, "VEV_5000": 1.5, "VEV_5100": 1.5,
    "VEV_5200": 1, "VEV_5300": 1, "VEV_5400": 1, "VEV_5500": 1,
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

# Tunables.
FLOW_DECAY = 0.92
FLOW_UNIT = 0.05
FLOW_PRICE_WEIGHT = 0.6
FLOW_STRONG = 2.0  # |flow| above which adverse-selection kicks in

MR_ENTRY_Z = 1.5
MR_PULL = 0.10
HYDRO_MR_ENTRY_Z = 1.25
HYDRO_MR_PULL = 0.12
VELVET_MR_ENTRY_Z = 1.75
VELVET_MR_PULL = 0.05
MR_CAP_VOL_MULT = 1.0

DELTA_SOFT = 25.0
DELTA_HARD = 40.0
DELTA_SKEW_STRENGTH = 0.08
VELVET_HEDGE_PRESSURE_STRENGTH = 1.25

ALPHA_EMA = 2.0 / (60.0 + 1.0)
ALPHA_SLOW = 1.0 - 0.5 ** (1.0 / 350.0)   # half-life ~350
ALPHA_VOL = 2.0 / (100.0 + 1.0)
ALPHA_RESID = 2.0 / (300.0 + 1.0)         # residual z EWMA span ~300

# Per-strike TV alpha (slower at high-gamma 5200/5300).
TV_ALPHA: Dict[int, float] = {K: 0.03 for K in STRIKES}
TV_ALPHA[5200] = 0.015
TV_ALPHA[5300] = 0.015

# Residual z thresholds and tilt strength.
RESID_Z_THRESHOLD = 1.25
RESID_Z_TILT = 0.30  # how strongly z shifts expected (× vol)

# Dynamic edge multipliers.
TAKE_EDGE_MULT = 1.0
MAKE_EDGE_MULT = 0.75
EDGE_VOL_MULT = 0.20
EDGE_SPREAD_MULT = 0.25
MAKE_EDGE_VOL_MULT = 0.08
CHEAP_SHORT_THRESHOLD_MULT = 1.5
CHEAP_SHORT_EDGE = 2.0

# Maker quote placement: improve, join, center, hybrid.
QUOTE_STYLE = "hybrid"
# Step 4F: revert VEV_5400/5500 to hybrid (was "join"; unproven).
QUOTE_STYLE_BY_PRODUCT = {
    "HYDROGEL_PACK": "hybrid",
    "VELVETFRUIT_EXTRACT": "hybrid",
    "VEV_4000": "hybrid",
    "VEV_4500": "hybrid",
    "VEV_5000": "hybrid",
    "VEV_5100": "hybrid",
    "VEV_5200": "hybrid",
    "VEV_5300": "hybrid",
    "VEV_5400": "hybrid",
    "VEV_5500": "hybrid",
}
SIDE_EDGE_ADJ = {
    "HYDROGEL_PACK": {"bid": 0.0, "ask": 0.0},
    "VELVETFRUIT_EXTRACT": {"bid": 0.0, "ask": 0.0},
    "VEV_4000": {"bid": 0.0, "ask": 0.0},
    "VEV_4500": {"bid": 0.0, "ask": 0.0},
    "VEV_5000": {"bid": 0.0, "ask": 0.0},
    "VEV_5100": {"bid": 0.0, "ask": 0.0},
    "VEV_5200": {"bid": 0.0, "ask": 0.0},
    "VEV_5300": {"bid": 0.0, "ask": 0.0},
    "VEV_5400": {"bid": 0.0, "ask": 0.0},
    "VEV_5500": {"bid": 0.0, "ask": 0.0},
}

# Step 5: per-product, per-side activation. Default all True so disabling a side
# is opt-in. Disable a side only after fill diagnostics show it loses money on
# at least two days.
ACTIVE_SIDES: Dict[str, Dict[str, bool]] = {
    "HYDROGEL_PACK":      {"bid": True, "ask": True},
    "VELVETFRUIT_EXTRACT":{"bid": True, "ask": True},
    "VEV_4000":           {"bid": True, "ask": True},
    "VEV_4500":           {"bid": True, "ask": True},
    "VEV_5000":           {"bid": True, "ask": True},
    "VEV_5100":           {"bid": True, "ask": True},
    "VEV_5200":           {"bid": True, "ask": True},
    "VEV_5300":           {"bid": True, "ask": True},
    "VEV_5400":           {"bid": True, "ask": True},
    "VEV_5500":           {"bid": True, "ask": True},
}

# Step 6: residual side gate. Replaces vertical tilt's coarse fair shift with a
# per-side maker permission on VEV non-wing strikes:
#   rz > +threshold  (voucher rich)  -> require extra edge for bid
#   rz < -threshold  (voucher cheap) -> require extra edge for ask
#   |rz| > hard_threshold            -> disable opposite side outright
RESID_SIDE_GATE = True
RESID_SIDE_THRESHOLD = 0.75
RESID_OPPOSITE_EDGE_ADD = 1.0
RESID_HARD_THRESHOLD = 1.5

# Step 7: vertical no-arbitrage sanity. Lightweight per-side edge nudge for
# adjacent strike pairs based on (mid_K1 - mid_K2) - (fair_K1 - fair_K2).
ENABLE_VERTICAL_SANITY = True
VERTICAL_EDGE_ADJ = 0.5
VERTICAL_SIGNAL_THRESHOLD = 1.5
VERTICAL_SANITY_SPREAD_CAP = 6  # skip pair if either spread > this

# Steps 8-9: per-product maker edge multipliers. Default flag off so the global
# MAKE_EDGE_MULT / MAKE_EDGE_VOL_MULT continue to govern edges (preserves the
# 1600 baseline). Flip ENABLE_PER_PRODUCT_MAKER_EDGE to switch to the dicts.
ENABLE_PER_PRODUCT_MAKER_EDGE = False
MAKE_EDGE_MULT_BY_PRODUCT: Dict[str, float] = {
    "HYDROGEL_PACK": 0.65,
    "VELVETFRUIT_EXTRACT": 0.75,
    "VEV_4000": 0.65,
    "VEV_4500": 0.65,
    "VEV_5000": 0.75,
    "VEV_5100": 0.75,
    "VEV_5200": 0.85,
    "VEV_5300": 0.85,
    "VEV_5400": 1.00,
    "VEV_5500": 1.00,
}
MAKE_EDGE_VOL_MULT_BY_PRODUCT: Dict[str, float] = {
    "HYDROGEL_PACK": 0.05,
    "VELVETFRUIT_EXTRACT": 0.05,
    "VEV_4000": 0.05,
    "VEV_4500": 0.05,
    "VEV_5000": 0.08,
    "VEV_5100": 0.08,
    "VEV_5200": 0.10,
    "VEV_5300": 0.10,
    "VEV_5400": 0.12,
    "VEV_5500": 0.12,
}

# Step 13: target-inventory skew. When on, the maker skew center moves from 0
# toward a product-specific target derived from mean-reversion z (HYDRO) or
# residual z (VEVs). Default off until backtested.
ENABLE_TARGET_INVENTORY = True
TARGET_INV_STRENGTH = 0.5
TARGET_INV_VEV_STRENGTH = 0.5  # extra dampener for VEVs (small target on ATM/cheap)
TARGET_INV_EDGE_ADJ = 0.45

# Adaptive maker engine. It learns compact product/side markout quality from
# own fills and only changes maker edge/size; it never opens through takers.
ENABLE_ADAPTIVE_MARKOUT = True
ENABLE_ADAPTIVE_CONFIDENCE = True
ENABLE_ADAPTIVE_STATE_MACHINE = True
ENABLE_QUOTE_SCORING = True
ENABLE_STRIP_BALANCE = True
ENABLE_ADAPTIVE_SECOND_LEVEL = False
MARKOUT_HORIZON = 50
MARKOUT_PENDING_MAX = 100
MARKOUT_SEEN_MAX = 240
MARKOUT_ALPHA = 0.05
ADAPT_MIN_FILLS = 5
ADAPT_STRONG_FILLS = 10
ADAPT_GOOD_MARKOUT = 0.5
ADAPT_BAD_MARKOUT = -0.5
ADAPT_TOXIC_MARKOUT = -1.5
ADAPT_GOOD_WIN = 0.55
ADAPT_BAD_WIN = 0.45
ADAPT_EDGE_MIN = -0.75
ADAPT_EDGE_MAX = 2.0
ADAPT_SIZE_MIN = 0.25
ADAPT_SIZE_MAX = 1.5
ADAPT_COOLDOWN_TICKS = 500
SCORE_THRESHOLD = 0.0
CONF_LOW = 0.45
CONF_SKIP = 0.25
CONF_EDGE_PENALTY = 0.75
CONF_SIZE_MIN = 0.50
STRIP_BALANCE_THRESHOLD = 0.55
STRIP_BALANCE_EDGE = 0.50

# Maker sizes by product group.
HYDRO_SIZE = 5
VELVET_SIZE = 4
DEEP_VEV_SIZE = 2
ATM_VEV_SIZE = 2
CHEAP_VEV_SIZE = 1
WING_BID_SIZE = 2
WING_UNWIND_SIZE = 4
ENABLE_INVENTORY_BUCKET_SIZING = False

# VEV fair-value controls.
TV_MODE = "ema"  # fixed, ema, hybrid
TV_HYBRID_SEED_WEIGHT = 0.70
S_HAT_MODE = "global"  # global, strike_local
S_WEIGHT_VELVET = 4.0
S_WEIGHT_VEV4000 = 2.0
S_WEIGHT_VEV4500 = 2.0
S_WEIGHT_VEV5000 = 1.0
S_HAT_WEIGHTS_BY_GROUP = {
    "deep": {"velvet": 5.0, "vev4000": 2.0, "vev4500": 2.0, "vev5000": 0.0},
    "atm": {"velvet": 4.0, "vev4000": 2.0, "vev4500": 2.0, "vev5000": 1.0},
    "cheap": {"velvet": 6.0, "vev4000": 1.0, "vev4500": 1.0, "vev5000": 0.0},
}

# Step 4F: vertical tilt off by default. Replaced by ENABLE_VERTICAL_SANITY.
ENABLE_VERTICAL_TILT = False
VERTICAL_TILT_STRENGTH = 0.05
VERTICAL_TILT_CAP = 1.5

# Step 4F + Step 14: second-level quotes off by default. Re-enable per-product
# only after fill diagnostics show high PnL, low inventory risk, and high fill
# rate on that product.
ENABLE_SECOND_LEVEL_QUOTES = False
SECOND_LEVEL_PRODUCTS: set = set()
SECOND_LEVEL_SIZE_MULT = 0.5
SECOND_LEVEL_MAX_ABS_INV_RATIO = 0.70

# Step 15: conservative inventory-reducing taker. Off by default.
# When on, only fires to reduce inventory toward 0, requires edge >= 2 * make
# edge, and never sells a long cheap VEV (cheap-VEV unwinds use the maker).
ENABLE_INVENTORY_REDUCING_TAKER = False
INV_TAKER_EDGE_MULT = 2.0
ENABLE_EXTREME_EDGE_TAKER = False
RESCUE_TAKER_EDGE_MULT = 3.0
EXTREME_TAKER_EDGE_MULT = 3.0

# Endgame controls.
ENDGAME_MODE = "risk_reduce_only"  # risk_reduce_only, normal_until_last_5k, off
ENDGAME_TAKE_EDGE_MULT = 1.5
ENDGAME_MAKE_EDGE_MULT = 1.25

# Adverse-selection threshold.
ADV_RETURN_VOL_MULT = 1.5

# Microprice tilt and inventory skew strength.
MICRO_TILT = 0.30
INV_SKEW_K = 1.2

# Warmup.
WARMUP_TICKS = 60

# Endgame.
ENDGAME_START = 990_000

# TV-quality spread cap multiplier (× base_make_edge, with a floor of 6).
TV_SPREAD_FLOOR = 6.0
TV_SPREAD_MULT = 4.0


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

def safe_json_loads(s: str) -> dict:
    if not s:
        return {}
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except (ValueError, TypeError):
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


def update_ewma(prev: Optional[float], x: float, alpha: float) -> float:
    if prev is None:
        return x
    return (1.0 - alpha) * float(prev) + alpha * x


def active_product(product: str) -> bool:
    return ACTIVE_PRODUCTS is None or product in ACTIVE_PRODUCTS


def submission_side(t: Trade) -> Optional[str]:
    if str(t.buyer) == "SUBMISSION":
        return "bid"
    if str(t.seller) == "SUBMISSION":
        return "ask"
    return None


def trade_key(product: str, t: Trade, side: str) -> str:
    return f"{product}:{side}:{int(t.timestamp)}:{int(t.price)}:{int(t.quantity)}"


def tv_value(K: int, tv: Dict[str, float]) -> float:
    seed = float(TV_SEED[K])
    ema = float(tv.get(str(K), seed))
    mode = str(TV_MODE).lower()
    if mode == "fixed":
        return seed
    if mode == "hybrid":
        seed_w = clamp(float(TV_HYBRID_SEED_WEIGHT), 0.0, 1.0)
        return seed_w * seed + (1.0 - seed_w) * ema
    return ema


def ceil_int(x: float) -> int:
    i = int(x)
    return i if float(i) >= x else i + 1


def product_group(product: str) -> str:
    if product in {"VEV_4000", "VEV_4500"}:
        return "deep"
    if product in {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"}:
        return "atm"
    if product in CHEAP_VEVS_NON_WING:
        return "cheap"
    return "other"


def quote_style_for(product: str) -> str:
    return str(QUOTE_STYLE_BY_PRODUCT.get(product, QUOTE_STYLE)).lower()


def side_edge_adj(product: str, side: str) -> float:
    return float(SIDE_EDGE_ADJ.get(product, {}).get(side, 0.0))


def is_side_active(product: str, side: str) -> bool:
    """Step 5 — per-product, per-side maker activation."""
    cfg = ACTIVE_SIDES.get(product)
    if not isinstance(cfg, dict):
        return True
    return bool(cfg.get(side, True))


def make_edge_mult_for(product: str) -> float:
    if ENABLE_PER_PRODUCT_MAKER_EDGE:
        return float(MAKE_EDGE_MULT_BY_PRODUCT.get(product, MAKE_EDGE_MULT))
    return float(MAKE_EDGE_MULT)


def make_edge_vol_mult_for(product: str) -> float:
    if ENABLE_PER_PRODUCT_MAKER_EDGE:
        return float(MAKE_EDGE_VOL_MULT_BY_PRODUCT.get(product, MAKE_EDGE_VOL_MULT))
    return float(MAKE_EDGE_VOL_MULT)


def maker_base_size(product: str) -> int:
    if product == "HYDROGEL_PACK":
        return HYDRO_SIZE
    if product == "VELVETFRUIT_EXTRACT":
        return VELVET_SIZE
    if product in CHEAP_VEVS_NON_WING:
        return CHEAP_VEV_SIZE
    if product in NEAR_ATM:
        return ATM_VEV_SIZE
    if product.startswith("VEV_"):
        return DEEP_VEV_SIZE
    return 1


def inventory_bucket_size(base_size: int, side: str, position: int, limit: int) -> int:
    if not ENABLE_INVENTORY_BUCKET_SIZING:
        return base_size
    if limit <= 0:
        return base_size
    reduces = (side == "bid" and position < 0) or (side == "ask" and position > 0)
    abs_ratio = abs(position) / float(limit)
    if reduces:
        mult = 1.25
    elif abs_ratio > 0.75:
        mult = 0.25
    elif abs_ratio > 0.50:
        mult = 0.50
    else:
        mult = 1.0
    return max(1, int(round(base_size * mult)))


def endgame_active(timestamp: int) -> bool:
    mode = str(ENDGAME_MODE).lower()
    if mode == "off":
        return False
    if mode == "normal_until_last_5k":
        return (int(timestamp) % 1_000_000) > 995_000
    return (int(timestamp) % 1_000_000) > ENDGAME_START


def ensure_maker_stats(memory: dict) -> dict:
    stats = memory.get("maker_stats")
    if not isinstance(stats, dict):
        stats = {}
        memory["maker_stats"] = stats
    for product in LIMITS:
        p = stats.get(product)
        if not isinstance(p, dict):
            p = {}
            stats[product] = p
        for side in ("bid", "ask"):
            s = p.get(side)
            if not isinstance(s, dict):
                s = {
                    "fills": 0,
                    "qty": 0,
                    "markout_ema": 0.0,
                    "markout_var": 1.0,
                    "win_ema": 0.5,
                    "last_fill_tick": 0,
                    "cool_until": 0,
                }
                p[side] = s
    return stats


def update_markout_stat(side_state: dict, markout: float, qty: int, tick: int) -> None:
    fills = int(side_state.get("fills", 0)) + 1
    old = float(side_state.get("markout_ema", 0.0))
    delta = markout - old
    new = old + MARKOUT_ALPHA * delta
    old_var = float(side_state.get("markout_var", 1.0))
    new_var = max((1.0 - MARKOUT_ALPHA) * (old_var + MARKOUT_ALPHA * delta * delta), 0.05)
    win = 1.0 if markout > 0.0 else 0.0
    old_win = float(side_state.get("win_ema", 0.5))

    side_state["fills"] = fills
    side_state["qty"] = int(side_state.get("qty", 0)) + int(qty)
    side_state["markout_ema"] = new
    side_state["markout_var"] = new_var
    side_state["win_ema"] = (1.0 - MARKOUT_ALPHA) * old_win + MARKOUT_ALPHA * win
    side_state["last_fill_tick"] = tick
    if fills >= ADAPT_STRONG_FILLS and new < ADAPT_TOXIC_MARKOUT:
        side_state["cool_until"] = tick + ADAPT_COOLDOWN_TICKS


def process_fill_markouts(memory: dict, own_trades: Dict[str, List[Trade]], mids: Dict[str, float], tick: int) -> None:
    if not ENABLE_ADAPTIVE_MARKOUT:
        return
    maker_stats = ensure_maker_stats(memory)

    pending = memory.get("pending_fills")
    if not isinstance(pending, list):
        pending = []
    remaining = []
    for ev in pending:
        if not isinstance(ev, dict):
            continue
        product = str(ev.get("p", ""))
        side = str(ev.get("s", ""))
        fill_tick = int(ev.get("t", tick))
        if tick - fill_tick < MARKOUT_HORIZON or product not in mids:
            remaining.append(ev)
            continue
        price = float(ev.get("px", 0.0))
        qty = int(ev.get("q", 1))
        future_mid = float(mids[product])
        markout = future_mid - price if side == "bid" else price - future_mid
        side_state = maker_stats.get(product, {}).get(side)
        if isinstance(side_state, dict):
            update_markout_stat(side_state, markout, qty, tick)

    seen = memory.get("seen_fills")
    if not isinstance(seen, list):
        seen = []
    seen_set = set(str(x) for x in seen)

    for product, trades in (own_trades or {}).items():
        for t in trades:
            if not isinstance(t, Trade):
                continue
            side = submission_side(t)
            if side is None:
                continue
            key = trade_key(product, t, side)
            if key in seen_set:
                continue
            seen.append(key)
            seen_set.add(key)
            remaining.append({
                "p": product,
                "s": side,
                "px": int(t.price),
                "q": max(1, abs(int(t.quantity))),
                "t": tick,
                "m": float(mids.get(product, t.price)),
            })

    memory["pending_fills"] = remaining[-MARKOUT_PENDING_MAX:]
    memory["seen_fills"] = seen[-MARKOUT_SEEN_MAX:]


def side_state_name(side_state: dict, tick: int) -> str:
    if not ENABLE_ADAPTIVE_STATE_MACHINE:
        return "NORMAL"
    if int(side_state.get("cool_until", 0)) > tick:
        return "COOLDOWN"
    fills = int(side_state.get("fills", 0))
    if fills < ADAPT_MIN_FILLS:
        return "NORMAL"
    markout = float(side_state.get("markout_ema", 0.0))
    win = float(side_state.get("win_ema", 0.5))
    if markout > ADAPT_GOOD_MARKOUT and win > ADAPT_GOOD_WIN:
        return "AGGRESSIVE"
    if markout < ADAPT_BAD_MARKOUT or win < ADAPT_BAD_WIN:
        return "DEFENSIVE"
    return "NORMAL"


def adaptive_side_controls(maker_stats: dict, product: str, side: str, position: int, limit: int, tick: int) -> dict:
    out = {"edge_adj": 0.0, "size_mult": 1.0, "state": "NORMAL", "active": True}
    if not ENABLE_ADAPTIVE_MARKOUT:
        return out
    side_state = maker_stats.get(product, {}).get(side, {})
    if not isinstance(side_state, dict):
        return out
    state = side_state_name(side_state, tick)
    out["state"] = state
    fills = int(side_state.get("fills", 0))
    markout = float(side_state.get("markout_ema", 0.0))
    win = float(side_state.get("win_ema", 0.5))

    if fills >= ADAPT_MIN_FILLS:
        if markout > ADAPT_GOOD_MARKOUT and win > ADAPT_GOOD_WIN:
            out["edge_adj"] -= 0.50
            out["size_mult"] *= 1.20
        elif markout < ADAPT_BAD_MARKOUT or win < ADAPT_BAD_WIN:
            out["edge_adj"] += 0.75
            out["size_mult"] *= 0.50
    if state == "COOLDOWN":
        reduces = (side == "bid" and position < 0) or (side == "ask" and position > 0)
        out["active"] = reduces
        out["edge_adj"] += 1.25
        out["size_mult"] *= 0.25
    elif state == "AGGRESSIVE":
        out["edge_adj"] -= 0.25
        out["size_mult"] *= 1.10
    elif state == "DEFENSIVE":
        out["edge_adj"] += 0.50
        out["size_mult"] *= 0.70

    out["edge_adj"] = clamp(float(out["edge_adj"]), ADAPT_EDGE_MIN, ADAPT_EDGE_MAX)
    out["size_mult"] = clamp(float(out["size_mult"]), ADAPT_SIZE_MIN, ADAPT_SIZE_MAX)
    return out


def product_confidence(product: str, spread: int, vol: float, last_return: float, rz: float, s_hat: Optional[float]) -> float:
    if not ENABLE_ADAPTIVE_CONFIDENCE:
        return 1.0
    conf = 1.0
    spread_ref = max(2.0, 3.0 * float(BASE_MAKE_EDGE.get(product, 1.0)))
    if spread > spread_ref:
        conf -= min(0.30, 0.04 * (spread - spread_ref))
    if abs(last_return) > 2.5 * max(vol, 0.5):
        conf -= 0.25
    if product.startswith("VEV_"):
        if s_hat is None:
            conf -= 0.30
        if abs(rz) > 2.5:
            conf -= 0.15
    return clamp(conf, 0.0, 1.0)


def strip_positions(position: Dict[str, int]) -> Dict[str, int]:
    return {
        "deep": int(position.get("VEV_4000", 0)) + int(position.get("VEV_4500", 0)),
        "atm": sum(int(position.get(VEV_BY_STRIKE[K], 0)) for K in [5000, 5100, 5200, 5300]),
        "cheap": int(position.get("VEV_5400", 0)) + int(position.get("VEV_5500", 0)),
    }


def strip_balance_edge(product: str, side: str, strips: Dict[str, int]) -> float:
    if not ENABLE_STRIP_BALANCE or not product.startswith("VEV_") or product in WINGS:
        return 0.0
    group = product_group(product)
    group_limit = {"deep": 40.0, "atm": 80.0, "cheap": 40.0}.get(group)
    if not group_limit:
        return 0.0
    ratio = float(strips.get(group, 0)) / group_limit
    if ratio > STRIP_BALANCE_THRESHOLD and side == "bid":
        return STRIP_BALANCE_EDGE
    if ratio < -STRIP_BALANCE_THRESHOLD and side == "ask":
        return STRIP_BALANCE_EDGE
    return 0.0


# --- Reservation tracker --------------------------------------------------- #

class Book:
    """Per-product remaining capacity, accounting for orders queued this tick."""
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
        """Net position if all queued orders fill."""
        return self.position + self.buy_used - self.sell_used


def add_order_safely(
    orders: List[Order],
    book: Book,
    price: int,
    qty: int,
) -> bool:
    """Append an order if size > 0 and within remaining capacity. Updates book."""
    if qty == 0:
        return False
    if qty > 0:
        q = min(qty, book.buy_room())
        if q <= 0:
            return False
        orders.append(Order(book.product, int(price), int(q)))
        book.buy_used += q
        return True
    q = min(-qty, book.sell_room())
    if q <= 0:
        return False
    orders.append(Order(book.product, int(price), -int(q)))
    book.sell_used += q
    return True


# --- Memory ---------------------------------------------------------------- #

def ensure_product_state(memory: dict, product: str) -> dict:
    products = memory.setdefault("products", {})
    p = products.get(product)
    if not isinstance(p, dict):
        p = {}
        products[product] = p
    return p


def ensure_tv(memory: dict) -> Dict[str, float]:
    tv = memory.get("tv")
    if not isinstance(tv, dict):
        tv = {}
        memory["tv"] = tv
    for K, seed in TV_SEED.items():
        tv.setdefault(str(K), seed)
    return tv


def ensure_stats(memory: dict) -> Dict[str, int]:
    s = memory.get("stats")
    if not isinstance(s, dict):
        s = {
            "take_buys": 0, "take_sells": 0,
            "maker_bids": 0, "maker_asks": 0,
            "skipped_delta": 0, "skipped_no_edge": 0,
            "skipped_adverse": 0,
        }
        memory["stats"] = s
    return s


def update_product_memory(pstate: dict, mid: float) -> Tuple[float, float, float, float, float, int]:
    """Update EMA, slow_mean/var, vol, last_return, ticks. Returns key metrics."""
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
    pstate["ema"] = ema
    pstate["slow_mean"] = slow_mean
    pstate["slow_var"] = slow_var
    pstate["vol"] = max(new_vol, 0.5)
    pstate["prev_mid"] = mid
    pstate["last_return"] = last_return
    pstate["ticks"] = ticks
    return ema, slow_mean, slow_var, max(new_vol, 0.5), last_return, ticks


def update_residual_z(pstate: dict, mid: float, fair: float) -> float:
    """EWMA residual mean and variance; returns standardized z."""
    resid = mid - fair
    prev_mean = float(pstate.get("rz_mean", 0.0))
    prev_var = float(pstate.get("rz_var", 1.0))
    delta = resid - prev_mean
    new_mean = prev_mean + ALPHA_RESID * delta
    new_var = max(
        (1.0 - ALPHA_RESID) * (prev_var + ALPHA_RESID * delta * delta),
        0.25,
    )
    pstate["rz_mean"] = new_mean
    pstate["rz_var"] = new_var
    std = new_var ** 0.5
    if std <= 0:
        return 0.0
    return (resid - new_mean) / std


# =========================================================================== #
# S_hat / VEV fairs                                                            #
# =========================================================================== #

def estimate_s_hat(
    mids: Dict[str, float],
    tv: Dict[str, float],
    velvet_vol: float,
    prev_s_hat: Optional[float],
    weights: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """Two-pass weighted mean with outlier rejection; falls back to VELVET / prev."""
    w = weights or {
        "velvet": S_WEIGHT_VELVET,
        "vev4000": S_WEIGHT_VEV4000,
        "vev4500": S_WEIGHT_VEV4500,
        "vev5000": S_WEIGHT_VEV5000,
    }
    parts: List[Tuple[float, float]] = []
    velvet = mids.get("VELVETFRUIT_EXTRACT")
    if velvet is not None and w.get("velvet", 0.0) > 0:
        parts.append((velvet, float(w["velvet"])))
    if "VEV_4000" in mids and w.get("vev4000", 0.0) > 0:
        parts.append((mids["VEV_4000"] + 4000.0, float(w["vev4000"])))
    if "VEV_4500" in mids and w.get("vev4500", 0.0) > 0:
        parts.append((mids["VEV_4500"] + 4500.0, float(w["vev4500"])))
    if "VEV_5000" in mids and w.get("vev5000", 0.0) > 0:
        parts.append((mids["VEV_5000"] + 5000.0 - tv_value(5000, tv), float(w["vev5000"])))

    if not parts:
        return velvet if velvet is not None else prev_s_hat

    if SIMPLE_S_HAT:
        return velvet if velvet is not None else prev_s_hat

    # Pass 1: weighted mean.
    s = sum(w * v for v, w in parts)
    w_tot = sum(w for _, w in parts)
    prelim = s / w_tot if w_tot > 0 else (velvet if velvet is not None else prev_s_hat)
    if prelim is None:
        return prev_s_hat

    # Pass 2: drop outliers > max(8, 2.5 * velvet_vol) from the preliminary mean.
    cutoff = max(8.0, 2.5 * max(float(velvet_vol), 0.5))
    kept = [(v, w) for v, w in parts if abs(v - prelim) <= cutoff]
    if len(kept) < 2:
        return velvet if velvet is not None else prev_s_hat

    s = sum(w * v for v, w in kept)
    w_tot = sum(w for _, w in kept)
    return s / w_tot if w_tot > 0 else prelim


def estimate_local_s_hats(
    mids: Dict[str, float],
    tv: Dict[str, float],
    velvet_vol: float,
    prev_s_hat: Optional[float],
    global_s_hat: Optional[float],
) -> Optional[Dict[str, float]]:
    if str(S_HAT_MODE).lower() != "strike_local":
        return None
    group_s: Dict[str, Optional[float]] = {}
    for group, weights in S_HAT_WEIGHTS_BY_GROUP.items():
        group_s[group] = estimate_s_hat(mids, tv, velvet_vol, prev_s_hat, weights)

    out: Dict[str, float] = {}
    for K in STRIKES:
        product = VEV_BY_STRIKE[K]
        g = product_group(product)
        s = group_s.get(g) or global_s_hat
        if s is not None:
            out[product] = float(s)
    return out


def update_tv(
    tv: Dict[str, float],
    mids: Dict[str, float],
    order_depths: Dict[str, OrderDepth],
    s_hat: Optional[float],
) -> None:
    """Quality-filtered TV update; pin wings; respect TV_BOUNDS."""
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
        if od is None or not od.buy_orders or not od.sell_orders:
            continue
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            continue
        spread = ba - bb
        spread_cap = max(TV_SPREAD_FLOOR, TV_SPREAD_MULT * BASE_MAKE_EDGE.get(product, 1))
        if spread > spread_cap:
            continue

        observed = mids[product] - max(s_hat - K, 0.0)
        lo, hi = TV_BOUNDS[K]
        # Reject ticks far outside generous bounds (likely bad data).
        if not (lo - 5.0 <= observed <= hi + 5.0):
            continue
        observed = clamp(observed, lo, hi)
        alpha = TV_ALPHA.get(K, 0.03)
        tv[str(K)] = update_ewma(tv.get(str(K), TV_SEED[K]), observed, alpha)


def compute_vev_fairs(
    s_hat: Optional[float],
    tv: Dict[str, float],
    local_s_hats: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Voucher fair = max(S_hat - K, 0) + tv_K, monotone non-increasing by K. Wings = 0.5."""
    if s_hat is None:
        return {p: 0.5 for p in WINGS}
    fairs: Dict[str, float] = {}
    prev: Optional[float] = None
    for K in STRIKES:
        product = VEV_BY_STRIKE[K]
        if product in WINGS:
            fair = 0.5
        else:
            s_for_product = local_s_hats.get(product, float(s_hat)) if local_s_hats else float(s_hat)
            fair = max(s_for_product - K, 0.0) + tv_value(K, tv)
            if ENABLE_MONOTONIC_VEV_FAIR and prev is not None:
                fair = min(fair, prev)
        fairs[product] = max(0.5, fair)
        if product not in WINGS:
            prev = fairs[product]
    # Wings are pinned regardless of monotone pass.
    for w in WINGS:
        fairs[w] = 0.5
    return fairs


def compute_vertical_tilts(
    mids: Dict[str, float],
    vev_fairs: Dict[str, float],
) -> Dict[str, float]:
    if not ENABLE_VERTICAL_TILT or VERTICAL_TILT_STRENGTH <= 0:
        return {}
    tilts: Dict[str, float] = {}
    local = [5000, 5100, 5200, 5300, 5400, 5500]
    for k1, k2 in zip(local, local[1:]):
        p1 = VEV_BY_STRIKE[k1]
        p2 = VEV_BY_STRIKE[k2]
        if p1 not in mids or p2 not in mids or p1 not in vev_fairs or p2 not in vev_fairs:
            continue
        vertical_market = mids[p1] - mids[p2]
        vertical_fair = vev_fairs[p1] - vev_fairs[p2]
        residual = vertical_market - vertical_fair
        tilt = clamp(VERTICAL_TILT_STRENGTH * residual, -VERTICAL_TILT_CAP, VERTICAL_TILT_CAP)
        tilts[p1] = tilts.get(p1, 0.0) - tilt
        tilts[p2] = tilts.get(p2, 0.0) + tilt
    return tilts


def compute_vertical_sanity_edge_adj(
    mids: Dict[str, float],
    vev_fairs: Dict[str, float],
    order_depths: Dict[str, OrderDepth],
) -> Dict[str, Dict[str, float]]:
    """Step 7 — per-side maker edge adjustment from cross-strike richness.

    For adjacent strikes K1 < K2 we compare the market vertical
    (mid_K1 - mid_K2) against the fair vertical (fair_K1 - fair_K2). When
    the residual exceeds VERTICAL_SIGNAL_THRESHOLD the rich leg is allowed
    to shave some ask edge and the cheap leg is allowed to shave bid edge.
    Negative values relax the side (more aggressive maker quote); we never
    add edge here. Skip the pair when either book is missing or the spread
    on either side is wider than VERTICAL_SANITY_SPREAD_CAP — both signals
    that the fair comparison is unreliable.
    """
    if not ENABLE_VERTICAL_SANITY:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    local = [5000, 5100, 5200, 5300, 5400, 5500]
    for k1, k2 in zip(local, local[1:]):
        p1 = VEV_BY_STRIKE[k1]
        p2 = VEV_BY_STRIKE[k2]
        if p1 not in mids or p2 not in mids:
            continue
        if p1 not in vev_fairs or p2 not in vev_fairs:
            continue
        od1 = order_depths.get(p1)
        od2 = order_depths.get(p2)
        if od1 is None or od2 is None:
            continue
        bb1, ba1 = best_bid_ask(od1)
        bb2, ba2 = best_bid_ask(od2)
        if bb1 is None or ba1 is None or bb2 is None or ba2 is None:
            continue
        if (ba1 - bb1) > VERTICAL_SANITY_SPREAD_CAP:
            continue
        if (ba2 - bb2) > VERTICAL_SANITY_SPREAD_CAP:
            continue

        residual = (mids[p1] - mids[p2]) - (vev_fairs[p1] - vev_fairs[p2])
        if residual >= VERTICAL_SIGNAL_THRESHOLD:
            # K1 rich, K2 cheap.
            out.setdefault(p1, {"bid": 0.0, "ask": 0.0})["ask"] -= VERTICAL_EDGE_ADJ
            out.setdefault(p2, {"bid": 0.0, "ask": 0.0})["bid"] -= VERTICAL_EDGE_ADJ
        elif residual <= -VERTICAL_SIGNAL_THRESHOLD:
            # K1 cheap, K2 rich.
            out.setdefault(p1, {"bid": 0.0, "ask": 0.0})["bid"] -= VERTICAL_EDGE_ADJ
            out.setdefault(p2, {"bid": 0.0, "ask": 0.0})["ask"] -= VERTICAL_EDGE_ADJ
    return out


def residual_side_adjustments(rz: float) -> Tuple[float, float, bool, bool]:
    """Step 6 — return (bid_edge_add, ask_edge_add, disable_bid, disable_ask)."""
    if not RESID_SIDE_GATE:
        return 0.0, 0.0, False, False
    bid_add = 0.0
    ask_add = 0.0
    disable_bid = False
    disable_ask = False
    if rz > RESID_SIDE_THRESHOLD:
        # Voucher rich: prefer asks, require extra edge to bid.
        bid_add = RESID_OPPOSITE_EDGE_ADD
        if rz > RESID_HARD_THRESHOLD:
            disable_bid = True
    elif rz < -RESID_SIDE_THRESHOLD:
        # Voucher cheap: prefer bids, require extra edge to ask.
        ask_add = RESID_OPPOSITE_EDGE_ADD
        if rz < -RESID_HARD_THRESHOLD:
            disable_ask = True
    return bid_add, ask_add, disable_bid, disable_ask


# =========================================================================== #
# Delta / Flow / MR                                                            #
# =========================================================================== #

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


def flow_weight(product: str, trader: str) -> float:
    if trader in FOLLOW.get(product, set()):
        return 1.0
    if trader in FADE.get(product, set()):
        return -1.0
    return 0.0


def update_flow(memory: dict, market_trades: Dict[str, List[Trade]]) -> None:
    products = memory.setdefault("products", {})
    for raw in products.values():
        if isinstance(raw, dict):
            raw["flow"] = FLOW_DECAY * float(raw.get("flow", 0.0))
    if not ENABLE_FLOW:
        return
    for product, trades in market_trades.items():
        if product not in FOLLOW and product not in FADE:
            continue
        pstate = ensure_product_state(memory, product)
        signal = float(pstate.get("flow", 0.0))
        for t in trades:
            if not isinstance(t, Trade):
                continue
            qty = max(1, abs(int(t.quantity)))
            buyer_w = flow_weight(product, str(t.buyer))
            seller_w = flow_weight(product, str(t.seller))
            signal += FLOW_UNIT * qty * (buyer_w - seller_w)
        pstate["flow"] = clamp(signal, -8.0, 8.0)


def flow_offset(pstate: dict, vol: float) -> float:
    raw = FLOW_PRICE_WEIGHT * float(pstate.get("flow", 0.0))
    cap = 1.5 * max(vol, 0.5)
    return clamp(raw, -cap, cap)


def mr_offset(product: str, mid: float, pstate: dict, vol: float) -> float:
    if not ENABLE_MR:
        return 0.0
    if product not in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}:
        return 0.0
    slow = pstate.get("slow_mean")
    var = pstate.get("slow_var")
    if slow is None or var is None:
        return 0.0
    std = max(float(var) ** 0.5, 1.0)
    z = (mid - float(slow)) / std
    if product == "HYDROGEL_PACK":
        entry_z = HYDRO_MR_ENTRY_Z
        pull_k = HYDRO_MR_PULL
    elif product == "VELVETFRUIT_EXTRACT":
        entry_z = VELVET_MR_ENTRY_Z
        pull_k = VELVET_MR_PULL
    else:
        entry_z = MR_ENTRY_Z
        pull_k = MR_PULL
    if abs(z) < entry_z:
        return 0.0
    pull = -pull_k * (mid - float(slow))
    cap = MR_CAP_VOL_MULT * 2.0 * max(vol, 1.0)
    return clamp(pull, -cap, cap)


# =========================================================================== #
# Dynamic edges                                                                #
# =========================================================================== #

def dynamic_edges(product: str, vol: float, spread: float) -> Tuple[float, float]:
    base_take = float(BASE_TAKE_EDGE.get(product, 2)) * TAKE_EDGE_MULT
    base_make = float(BASE_MAKE_EDGE.get(product, 1)) * make_edge_mult_for(product)
    if not ENABLE_DYNAMIC_EDGES:
        return base_take, base_make
    take = base_take + EDGE_VOL_MULT * max(vol, 0.0) + EDGE_SPREAD_MULT * max(spread, 0.0)
    make = base_make + make_edge_vol_mult_for(product) * max(vol, 0.0)
    return max(take, base_take), max(make, base_make)


def quote_prices(
    product: str,
    bb: int,
    ba: int,
    max_bid: float,
    min_ask: float,
    spread: int,
) -> Tuple[Optional[int], Optional[int]]:
    style = quote_style_for(product)
    if style == "hybrid":
        style = "improve" if spread >= 3 else "join"

    if style == "join":
        mb = bb if bb <= max_bid else None
        ma = ba if ba >= min_ask else None
    elif style in {"center", "centered"}:
        mb = int(max_bid)
        ma = ceil_int(min_ask)
    else:
        cand_b = bb + 1
        if cand_b <= max_bid:
            mb = cand_b
        elif bb <= max_bid:
            mb = bb
        else:
            mb = None

        cand_a = ba - 1
        if cand_a >= min_ask:
            ma = cand_a
        elif ba >= min_ask:
            ma = ba
        else:
            ma = None

    if mb is not None and mb >= ba:
        mb = None
    if ma is not None and ma <= bb:
        ma = None
    if mb is not None and ma is not None and mb >= ma:
        if spread <= 2 and bb <= max_bid and ba >= min_ask:
            mb, ma = bb, ba
        else:
            mb, ma = None, None
    return mb, ma


def best_scored_quote(
    product: str,
    side: str,
    bb: int,
    ba: int,
    expected: float,
    skew: float,
    edge_required: float,
    score_bonus: float,
) -> Optional[int]:
    if side == "bid":
        candidates = {bb, bb + 1, int(expected + skew - edge_required)}
        valid: List[Tuple[int, float]] = []
        for price in candidates:
            if price <= 0 or price >= ba:
                continue
            raw_edge = expected + skew - float(price)
            score = raw_edge - edge_required + score_bonus
            if score >= SCORE_THRESHOLD:
                valid.append((int(price), score))
        if not valid:
            return None
        return max(p for p, _ in valid)

    candidates = {ba, ba - 1, ceil_int(expected + skew + edge_required)}
    valid = []
    for price in candidates:
        if price <= bb:
            continue
        raw_edge = float(price) - (expected + skew)
        score = raw_edge - edge_required + score_bonus
        if score >= SCORE_THRESHOLD:
            valid.append((int(price), score))
    if not valid:
        return None
    return min(p for p, _ in valid)


def quote_prices_scored(
    product: str,
    bb: int,
    ba: int,
    expected: float,
    skew: float,
    bid_edge: float,
    ask_edge: float,
    bid_score_bonus: float,
    ask_score_bonus: float,
) -> Tuple[Optional[int], Optional[int]]:
    mb = best_scored_quote(product, "bid", bb, ba, expected, skew, bid_edge, bid_score_bonus)
    ma = best_scored_quote(product, "ask", bb, ba, expected, skew, ask_edge, ask_score_bonus)
    if mb is not None and ma is not None and mb >= ma:
        mb, ma = None, None
    return mb, ma


# =========================================================================== #
# Trader                                                                       #
# =========================================================================== #

class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = safe_json_loads(state.traderData)
        memory.setdefault("products", {})
        stats = ensure_stats(memory)
        tick = int(memory.get("tick", 0)) + 1
        memory["tick"] = tick

        update_flow(memory, state.market_trades or {})
        tv = ensure_tv(memory)

        endgame = ENABLE_ENDGAME and endgame_active(int(state.timestamp))

        mids: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            m = mid_price(od)
            if m is not None:
                mids[product] = m
        process_fill_markouts(memory, state.own_trades or {}, mids, tick)
        maker_stats = ensure_maker_stats(memory)

        velvet_pstate = memory.get("products", {}).get("VELVETFRUIT_EXTRACT") or {}
        velvet_vol = float(velvet_pstate.get("vol", 1.0))
        prev_s_hat = memory.get("s_hat")

        s_hat = estimate_s_hat(mids, tv, velvet_vol, prev_s_hat)
        if s_hat is not None:
            memory["s_hat"] = s_hat
        update_tv(tv, mids, state.order_depths, s_hat)

        local_s_hats = estimate_local_s_hats(mids, tv, velvet_vol, prev_s_hat, s_hat)
        vev_fairs = compute_vev_fairs(s_hat, tv, local_s_hats) if ENABLE_VEV_LADDER else {p: 0.5 for p in WINGS}
        vertical_tilts = compute_vertical_tilts(mids, vev_fairs)
        vertical_sanity = compute_vertical_sanity_edge_adj(mids, vev_fairs, state.order_depths)

        position_map: Dict[str, int] = state.position or {}
        net_delta = compute_net_delta(position_map) if ENABLE_DELTA_CONTROL else 0.0
        strip_pos = strip_positions(position_map)

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
                orders = self.trade_one(
                    product=product,
                    od=od,
                    pstate=pstate,
                    book=book,
                    s_hat=s_hat,
                    vev_fairs=vev_fairs,
                    vertical_tilts=vertical_tilts,
                    vertical_sanity=vertical_sanity,
                    net_delta=net_delta,
                    maker_stats=maker_stats,
                    strip_pos=strip_pos,
                    tick=tick,
                    endgame=endgame,
                    stats=stats,
                )
            result[product] = orders

        # Cap diagnostic counters so traderData never balloons.
        for k, v in list(stats.items()):
            if isinstance(v, int) and v > 1_000_000_000:
                stats[k] = 1_000_000_000

        return result, 0, json.dumps(memory, separators=(",", ":"))

    # ---- per-product main path ------------------------------------------- #

    def trade_one(
        self,
        product: str,
        od: OrderDepth,
        pstate: dict,
        book: Book,
        s_hat: Optional[float],
        vev_fairs: Dict[str, float],
        vertical_tilts: Dict[str, float],
        vertical_sanity: Dict[str, Dict[str, float]],
        net_delta: float,
        maker_stats: dict,
        strip_pos: Dict[str, int],
        tick: int,
        endgame: bool,
        stats: Dict[str, int],
    ) -> List[Order]:
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders
        mid = (bb + ba) / 2.0
        spread = ba - bb

        ema, slow_mean, slow_var, vol, last_return, ticks = update_product_memory(pstate, mid)

        # Base fair per family.
        if product == "HYDROGEL_PACK":
            base = ema
        elif product == "VELVETFRUIT_EXTRACT":
            base = float(s_hat) if s_hat is not None else ema
        else:
            base = vev_fairs.get(product, ema) + vertical_tilts.get(product, 0.0)

        # Residual z update — kept warm even during warmup.
        rz = 0.0
        is_vev_non_wing = product.startswith("VEV_") and product not in WINGS
        if is_vev_non_wing:
            rz = update_residual_z(pstate, mid, base)

        if ticks < WARMUP_TICKS:
            return orders

        position = book.position
        limit = book.limit

        # ---- Offsets ----------------------------------------------------- #
        micro = microprice(od) or mid
        micro_offset = MICRO_TILT * (micro - base)

        flow_off = flow_offset(pstate, vol) if ENABLE_FLOW else 0.0
        mr_off = mr_offset(product, mid, pstate, vol)

        inv_skew = INV_SKEW_K * (position / max(1.0, float(limit))) * max(vol, 1.0)

        my_delta = delta_for(product)
        delta_skew = 0.0
        if ENABLE_DELTA_CONTROL:
            delta_skew = DELTA_SKEW_STRENGTH * net_delta * my_delta

        velvet_hedge = 0.0
        if ENABLE_VELVET_HEDGE_PRESSURE and product == "VELVETFRUIT_EXTRACT":
            hp = clamp(net_delta / DELTA_HARD, -1.5, 1.5)
            raw = VELVET_HEDGE_PRESSURE_STRENGTH * hp * max(vol, 1.0)
            velvet_hedge = clamp(raw, -2.0 * max(vol, 1.0), 2.0 * max(vol, 1.0))

        z_tilt = 0.0
        if ENABLE_RESIDUAL_Z and is_vev_non_wing:
            z_tilt = RESID_Z_TILT * rz * max(vol, 0.5)

        expected = (
            base + micro_offset + mr_off + flow_off
            - inv_skew - delta_skew - velvet_hedge - z_tilt
        )

        # ---- Edges ------------------------------------------------------- #
        take_edge, make_edge = dynamic_edges(product, vol, spread)
        if endgame:
            take_edge *= ENDGAME_TAKE_EDGE_MULT
            make_edge *= ENDGAME_MAKE_EDGE_MULT

        # ---- Gates ------------------------------------------------------- #
        delta_block_buy = ENABLE_DELTA_CONTROL and (
            (net_delta > DELTA_HARD and my_delta > 0)
            or (net_delta < -DELTA_HARD and my_delta < 0)
        )
        delta_block_sell = ENABLE_DELTA_CONTROL and (
            (net_delta < -DELTA_HARD and my_delta > 0)
            or (net_delta > DELTA_HARD and my_delta < 0)
        )
        eg_block_buy = endgame and position >= 0
        eg_block_sell = endgame and position <= 0

        # Adverse-selection signals.
        adverse_buy = False
        adverse_sell = False
        if ENABLE_ADVERSE_SELECTION_FILTER:
            if abs(last_return) > ADV_RETURN_VOL_MULT * max(vol, 0.5):
                if last_return > 0:
                    adverse_buy = True
                elif last_return < 0:
                    adverse_sell = True
            flow_v = float(pstate.get("flow", 0.0))
            if flow_v > FLOW_STRONG:
                adverse_sell = True
            elif flow_v < -FLOW_STRONG:
                adverse_buy = True

        cheap_protect = ENABLE_CHEAP_VEV_RULE and product in CHEAP_VEVS_NON_WING
        cheap_fair = vev_fairs.get(product, base)

        # ---- Takers ------------------------------------------------------ #
        # Step 15: when only the inventory-reducing taker is enabled, the buy
        # rescue requires excess >= INV_TAKER_EDGE_MULT * make_edge, never
        # opens new positions (capped at |position|), and is allowed on cheap
        # VEVs because it covers shorts.
        _, base_make_edge = dynamic_edges(product, vol, spread)
        if endgame:
            base_make_edge *= ENDGAME_MAKE_EDGE_MULT
        rescue_buy = ENABLE_INVENTORY_REDUCING_TAKER and position < 0
        extreme_buy = ENABLE_EXTREME_EDGE_TAKER
        if (ENABLE_TAKER or rescue_buy or extreme_buy) and not (delta_block_buy or eg_block_buy):
            buy_take_edge = take_edge
            if not ENABLE_TAKER:
                if rescue_buy:
                    buy_take_edge = INV_TAKER_EDGE_MULT * base_make_edge
                else:
                    buy_take_edge *= EXTREME_TAKER_EDGE_MULT
            for ap in sorted(od.sell_orders):
                if book.buy_room() <= 0:
                    break
                eff_pos = book.effective_position()
                # Aggressive cheap-VEV buyback when short.
                allow_aggressive_buy = (
                    cheap_protect and eff_pos < 0 and ap <= cheap_fair + 1
                )
                if ap > expected - buy_take_edge and not allow_aggressive_buy:
                    break

                # Edge strength → size cap (tighter for near-ATM).
                excess = (expected - buy_take_edge) - ap
                if product in NEAR_ATM:
                    if excess >= buy_take_edge:
                        max_size = 3
                    elif excess >= 0.5 * buy_take_edge:
                        max_size = 2
                    else:
                        max_size = 1
                else:
                    max_size = limit
                if allow_aggressive_buy:
                    max_size = limit
                if not ENABLE_TAKER:
                    max_size = min(max_size, max(1, abs(position)))

                avail = abs(od.sell_orders[ap])
                q = min(avail, book.buy_room(), max_size)
                if q > 0 and add_order_safely(orders, book, ap, q):
                    stats["take_buys"] = stats.get("take_buys", 0) + 1

        # Step 15: rescue-sell forbidden on cheap VEVs (never sell-rescue a long
        # cheap VEV via taker; let the maker handle that side).
        rescue_sell = (
            ENABLE_INVENTORY_REDUCING_TAKER
            and position > 0
            and product not in CHEAP_VEVS_NON_WING
        )
        extreme_sell = ENABLE_EXTREME_EDGE_TAKER
        if (ENABLE_TAKER or rescue_sell or extreme_sell) and not (delta_block_sell or eg_block_sell):
            sell_take_edge = take_edge
            if not ENABLE_TAKER:
                if rescue_sell:
                    sell_take_edge = INV_TAKER_EDGE_MULT * base_make_edge
                else:
                    sell_take_edge *= EXTREME_TAKER_EDGE_MULT
            for bp in sorted(od.buy_orders, reverse=True):
                if book.sell_room() <= 0:
                    break
                if bp < expected + sell_take_edge:
                    break

                eff_pos = book.effective_position()
                if cheap_protect and eff_pos <= 0:
                    # Hard cheap-VEV short rule: bid >= fair + max(2, 1.5*take_edge)
                    # AND voucher must look genuinely overpriced (rz > threshold).
                    short_threshold = cheap_fair + max(2.0, CHEAP_SHORT_THRESHOLD_MULT * sell_take_edge)
                    if bp < short_threshold:
                        break
                    if ENABLE_RESIDUAL_Z and rz < RESID_Z_THRESHOLD:
                        break

                # Allow aggressive cheap-VEV unwind when long.
                allow_aggressive_sell = (
                    cheap_protect and eff_pos > 0 and bp >= cheap_fair + 1
                )

                excess = bp - (expected + sell_take_edge)
                if product in NEAR_ATM:
                    if excess >= sell_take_edge:
                        max_size = 3
                    elif excess >= 0.5 * sell_take_edge:
                        max_size = 2
                    else:
                        max_size = 1
                else:
                    max_size = limit
                if allow_aggressive_sell:
                    max_size = limit
                if not ENABLE_TAKER:
                    max_size = min(max_size, max(1, abs(position)))

                avail = abs(od.buy_orders[bp])
                q = min(avail, book.sell_room(), max_size)
                if q > 0 and add_order_safely(orders, book, bp, -q):
                    stats["take_sells"] = stats.get("take_sells", 0) + 1

        # ---- Makers ------------------------------------------------------ #
        if not ENABLE_MAKER:
            return orders

        # Step 13: optional target-inventory shift. HYDRO uses mean-reversion z;
        # VEV non-wings use residual z (rich -> target short, cheap -> target
        # long). Damped for VEVs so ATM/cheap targets stay small.
        target_pos = 0.0
        if ENABLE_TARGET_INVENTORY:
            if product == "HYDROGEL_PACK":
                std_mr = max(float(slow_var) ** 0.5, 1.0)
                z_mr = (mid - float(slow_mean)) / std_mr
                target_pos = -float(limit) * clamp(z_mr / 3.0, -1.0, 1.0) * TARGET_INV_STRENGTH
            elif is_vev_non_wing:
                target_pos = -float(limit) * clamp(rz / 3.0, -1.0, 1.0) * TARGET_INV_VEV_STRENGTH

        # Inventory-driven price skew (drag quotes toward target, default 0).
        skew = -INV_SKEW_K * ((position - target_pos) / max(1.0, float(limit))) * make_edge
        skew = clamp(skew, -1.35 * make_edge, 1.35 * make_edge)

        bid_edge = max(0.0, make_edge + side_edge_adj(product, "bid"))
        ask_edge = max(0.0, make_edge + side_edge_adj(product, "ask"))
        bid_score_bonus = 0.0
        ask_score_bonus = 0.0

        # Target inventory is an alpha pressure, not a forced trade. It shifts
        # maker edge on the side that moves position toward the target.
        if ENABLE_TARGET_INVENTORY and limit > 0:
            target_gap = clamp((target_pos - float(position)) / float(limit), -1.0, 1.0)
            if target_gap > 0.0:
                adj = TARGET_INV_EDGE_ADJ * abs(target_gap)
                bid_edge = max(0.0, bid_edge - adj)
                ask_edge += 0.5 * adj
                bid_score_bonus += adj
            elif target_gap < 0.0:
                adj = TARGET_INV_EDGE_ADJ * abs(target_gap)
                ask_edge = max(0.0, ask_edge - adj)
                bid_edge += 0.5 * adj
                ask_score_bonus += adj

        # Step 6: residual side gate (VEV non-wing).
        gate_disable_bid = False
        gate_disable_ask = False
        if is_vev_non_wing:
            r_bid_add, r_ask_add, gate_disable_bid, gate_disable_ask = residual_side_adjustments(rz)
            bid_edge = max(0.0, bid_edge + r_bid_add)
            ask_edge = max(0.0, ask_edge + r_ask_add)

        # Step 7: vertical sanity per-side edge nudge. Negative values reduce
        # the required edge on the favored side; positive values are not used.
        v_adj = vertical_sanity.get(product) if vertical_sanity else None
        if v_adj:
            bid_edge = max(0.0, bid_edge + float(v_adj.get("bid", 0.0)))
            ask_edge = max(0.0, ask_edge + float(v_adj.get("ask", 0.0)))

        bid_edge += strip_balance_edge(product, "bid", strip_pos)
        ask_edge += strip_balance_edge(product, "ask", strip_pos)

        bid_adapt = adaptive_side_controls(maker_stats, product, "bid", position, limit, tick)
        ask_adapt = adaptive_side_controls(maker_stats, product, "ask", position, limit, tick)
        bid_edge = max(0.0, bid_edge + float(bid_adapt["edge_adj"]))
        ask_edge = max(0.0, ask_edge + float(ask_adapt["edge_adj"]))
        if bid_adapt["state"] == "AGGRESSIVE":
            bid_score_bonus += 0.35
        elif bid_adapt["state"] == "DEFENSIVE":
            bid_score_bonus -= 0.25
        if ask_adapt["state"] == "AGGRESSIVE":
            ask_score_bonus += 0.35
        elif ask_adapt["state"] == "DEFENSIVE":
            ask_score_bonus -= 0.25

        confidence = product_confidence(product, spread, vol, last_return, rz, s_hat)
        conf_size_mult = 1.0
        if confidence > 0.75:
            bid_edge = max(0.0, bid_edge - 0.10)
            ask_edge = max(0.0, ask_edge - 0.10)
            bid_score_bonus += 0.10
            ask_score_bonus += 0.10
        elif confidence < CONF_LOW:
            penalty = CONF_EDGE_PENALTY * (CONF_LOW - confidence) / CONF_LOW
            bid_edge += penalty
            ask_edge += penalty
            conf_size_mult = max(CONF_SIZE_MIN, confidence / CONF_LOW)

        max_bid = expected + skew - bid_edge   # highest price we'd bid
        min_ask = expected + skew + ask_edge   # lowest price we'd ask

        if ENABLE_QUOTE_SCORING:
            mb, ma = quote_prices_scored(
                product, bb, ba, expected, skew, bid_edge, ask_edge,
                bid_score_bonus, ask_score_bonus,
            )
        else:
            mb, ma = quote_prices(product, bb, ba, max_bid, min_ask, spread)

        # Inventory-aware maker size.
        base_ms = maker_base_size(product)
        ms_buy = inventory_bucket_size(base_ms, "bid", position, limit)
        ms_sell = inventory_bucket_size(base_ms, "ask", position, limit)
        # Slightly larger when reducing inventory.
        if position > 0:
            ms_sell = base_ms + 1
        if position < 0:
            ms_buy = base_ms + 1
        ms_buy = max(1, int(round(ms_buy * float(bid_adapt["size_mult"]) * conf_size_mult)))
        ms_sell = max(1, int(round(ms_sell * float(ask_adapt["size_mult"]) * conf_size_mult)))
        # Smaller when near limit.
        room_b = book.buy_room()
        room_s = book.sell_room()
        if room_b > 0 and room_b <= base_ms:
            ms_buy = max(1, room_b)
        elif room_b == 0:
            ms_buy = 0
        if room_s > 0 and room_s <= base_ms:
            ms_sell = max(1, room_s)
        elif room_s == 0:
            ms_sell = 0

        # Adverse-selection: shrink or skip on the adverse side.
        if adverse_buy:
            ms_buy = max(0, ms_buy - 2)
        if adverse_sell:
            ms_sell = max(0, ms_sell - 2)

        # Cheap-VEV: only open maker shorts when the ask is clearly rich.
        cheap_blocks_maker_short = False
        if (
            ENABLE_CHEAP_VEV_RULE
            and product in CHEAP_VEVS_NON_WING
            and book.effective_position() <= 0
        ):
            cheap_short_edge = cheap_fair + max(CHEAP_SHORT_EDGE, CHEAP_SHORT_THRESHOLD_MULT * make_edge)
            cheap_blocks_maker_short = ma is None or ma < cheap_short_edge

        bid_reduces_inventory = position < 0
        ask_reduces_inventory = position > 0
        bid_active = (
            is_side_active(product, "bid")
            and bool(bid_adapt["active"])
            and not gate_disable_bid
            and not (confidence < CONF_SKIP and not bid_reduces_inventory)
        )
        ask_active = (
            is_side_active(product, "ask")
            and bool(ask_adapt["active"])
            and not gate_disable_ask
            and not (confidence < CONF_SKIP and not ask_reduces_inventory)
        )

        if (
            mb is not None and ms_buy > 0
            and bid_active
            and not (delta_block_buy or eg_block_buy)
        ):
            if add_order_safely(orders, book, mb, ms_buy):
                stats["maker_bids"] = stats.get("maker_bids", 0) + 1
                self.add_second_level_quote(
                    orders, book, product, "bid", mb, max_bid, min_ask,
                    bb, ba, ms_buy, endgame, stats,
                )

        if (
            ma is not None and ms_sell > 0
            and ask_active
            and not (delta_block_sell or eg_block_sell)
            and not cheap_blocks_maker_short
        ):
            if add_order_safely(orders, book, ma, -ms_sell):
                stats["maker_asks"] = stats.get("maker_asks", 0) + 1
                self.add_second_level_quote(
                    orders, book, product, "ask", ma, max_bid, min_ask,
                    bb, ba, ms_sell, endgame, stats,
                )

        # ---- Diagnostics ------------------------------------------------- #
        if delta_block_buy or delta_block_sell:
            stats["skipped_delta"] = stats.get("skipped_delta", 0) + 1
        if mb is None and ma is None:
            stats["skipped_no_edge"] = stats.get("skipped_no_edge", 0) + 1
        if adverse_buy or adverse_sell:
            stats["skipped_adverse"] = stats.get("skipped_adverse", 0) + 1

        return orders

    def add_second_level_quote(
        self,
        orders: List[Order],
        book: Book,
        product: str,
        side: str,
        first_price: int,
        max_bid: float,
        min_ask: float,
        bb: int,
        ba: int,
        first_size: int,
        endgame: bool,
        stats: Dict[str, int],
    ) -> None:
        if (
            not ENABLE_SECOND_LEVEL_QUOTES
            or endgame
            or product not in SECOND_LEVEL_PRODUCTS
            or first_size <= 1
            or book.limit <= 0
        ):
            return
        eff = book.effective_position()
        if abs(eff) / float(book.limit) > SECOND_LEVEL_MAX_ABS_INV_RATIO:
            return

        q = max(1, int(first_size * SECOND_LEVEL_SIZE_MULT))
        if side == "bid":
            if eff >= SECOND_LEVEL_MAX_ABS_INV_RATIO * book.limit:
                return
            price = first_price - 1
            if price <= 0 or price >= ba or price > max_bid:
                return
            if add_order_safely(orders, book, price, q):
                stats["maker_bids_l2"] = stats.get("maker_bids_l2", 0) + 1
        else:
            if eff <= -SECOND_LEVEL_MAX_ABS_INV_RATIO * book.limit:
                return
            price = first_price + 1
            if price <= bb or price < min_ask:
                return
            if add_order_safely(orders, book, price, -q):
                stats["maker_asks_l2"] = stats.get("maker_asks_l2", 0) + 1

    # ---- floor wing handler --------------------------------------------- #

    def trade_wing(
        self,
        product: str,
        od: OrderDepth,
        book: Book,
        endgame: bool,
    ) -> List[Order]:
        """VEV_6000 / VEV_6500: fair = 0.5. Bid 0 only; ask 1 only if long."""
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)

        # Scoop a 0-priced ask if any seller appears at price 0 (rare, free PnL).
        if not endgame and ba is not None and ba <= 0:
            avail = abs(od.sell_orders[ba])
            q = min(3, avail, book.buy_room())
            if q > 0:
                add_order_safely(orders, book, ba, q)

        # Resting maker bid at 0.
        if not endgame:
            add_order_safely(orders, book, 0, WING_BID_SIZE)

        # Ask 1 only to unwind longs.
        if book.position > 0:
            unwind = min(book.position, 2 * WING_UNWIND_SIZE if endgame else WING_UNWIND_SIZE)
            add_order_safely(orders, book, 1, -unwind)

        return orders
