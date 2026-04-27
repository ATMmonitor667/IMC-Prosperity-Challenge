"""Prosperity 4 Round 4 trader.

Hybrid implementation:

* Proven online EWMA fair value, Welford drift t-test, vol-scaled take/make.
* VEV option-ladder anchors from released data.
* Bounded overlays for the requested plan: VEV time-value fair, portfolio delta
  skew, participant-flow tilt, cheap-voucher guards, and wing floor quoting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, Trade, TradingState


DEFAULT_LIMIT = 20
_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 32,
    "HYDROGEL_PACK": 32,
    "VEV_4000": 20,
    "VEV_4500": 20,
    "VEV_5000": 20,
    "VEV_5100": 20,
    "VEV_5200": 20,
    "VEV_5300": 20,
    "VEV_5400": 20,
    "VEV_5500": 20,
    "VEV_6000": 20,
    "VEV_6500": 20,
}

_GLOBAL_KEY = "__g__"
HORIZON = 100
_ALPHA_FAIR_BASE = 2.0 / (60.0 + 1.0)
_ALPHA_FAIR_VELVET = 2.0 / (90.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
ALPHA_SLOW = 2.0 / (360.0 + 1.0)
ALPHA_TV = 0.0025
WARMUP_TICKS = 100

# Slow moving ratio VEV_5000 / VELVET, and per-strike ratios to VEV_5000.
ALPHA_KV = 0.0008
ALPHA_KR = 0.0025

# Existing ladder anchor weights. These are deliberately small; the local book
# fair is still the primary signal.
ANCHOR_VELVET = 0.028
ANCHOR_V5000 = 0.032
ANCHOR_LADDER = 0.014
ANCHOR_LADDER_NEAR: Dict[str, float] = {
    "VEV_5100": 0.32,
    "VEV_5200": 0.48,
    "VEV_5300": 0.55,
}
LADDER_VOL_SCALE_K = 34.0
LADDER_ANCHOR_CAP_MULT = 2.45

MICRO_TILT_BASE = 0.3
MICRO_TILT_VELVET = 0.42
MICRO_TILT_V5000 = 0.40

DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 1.4
INV_SKEW_K = 1.0
INV_SKEW_VELVET = 1.45
INV_SKEW_V5000 = 1.36
INV_SKEW_V51 = 1.32
INV_SKEW_V52 = 1.18
INV_SKEW_V53 = 1.15
ANTI_TREND_BARRIER_MULT = 3.0

WING_MID_MAX = 2.0
REF5000_MIN = 5.0
CHEAP_VEVS = {"VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}
WINGS = {"VEV_6000", "VEV_6500"}

VEV_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
VEV_ORDER = sorted(VEV_STRIKES, key=lambda p: VEV_STRIKES[p])
TV_SEED: Dict[str, float] = {
    "VEV_4000": 0.0,
    "VEV_4500": 0.0,
    "VEV_5000": 2.75,
    "VEV_5100": 11.5,
    "VEV_5200": 39.0,
    "VEV_5300": 34.0,
    "VEV_5400": 9.0,
    "VEV_5500": 2.7,
    "VEV_6000": 0.5,
    "VEV_6500": 0.5,
}
TV_BOUNDS: Dict[str, Tuple[float, float]] = {
    "VEV_4000": (-0.75, 1.25),
    "VEV_4500": (-0.75, 1.25),
    "VEV_5000": (0.0, 8.0),
    "VEV_5100": (4.0, 24.0),
    "VEV_5200": (24.0, 58.0),
    "VEV_5300": (16.0, 56.0),
    "VEV_5400": (3.0, 22.0),
    "VEV_5500": (0.5, 9.5),
    "VEV_6000": (0.5, 0.5),
    "VEV_6500": (0.5, 0.5),
}
DELTAS: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.0,
    "VEV_4000": 0.74,
    "VEV_4500": 0.67,
    "VEV_5000": 0.66,
    "VEV_5100": 0.59,
    "VEV_5200": 0.44,
    "VEV_5300": 0.25,
    "VEV_5400": 0.10,
    "VEV_5500": 0.04,
}
OPTION_FAIR_WEIGHT: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 0.020,
    "VEV_4000": 0.000,
    "VEV_4500": 0.000,
    "VEV_5000": 0.000,
    "VEV_5100": 0.012,
    "VEV_5200": 0.010,
    "VEV_5300": 0.000,
    "VEV_5400": 0.000,
    "VEV_5500": 0.000,
}
FLOW_DECAY = 0.92
FLOW_UNIT = 0.045
FLOW_PRICE_WEIGHT = 0.0
DELTA_LIMIT_SOFT = 42.0
DELTA_SKEW_K = 0.0
MR_ENTRY_Z = 1.5
MR_PULL = 0.12


@dataclass
class EdgeConfig:
    k_take: float
    k_make: float
    min_take: int
    min_make: int
    take_frac: float
    make_frac: float
    vol_floor: float


def _default_edge() -> EdgeConfig:
    return EdgeConfig(
        k_take=2.0,
        k_make=0.5,
        min_take=2,
        min_make=1,
        take_frac=0.25,
        make_frac=0.125,
        vol_floor=0.0,
    )


def _edge_config(product: str, mid: float) -> EdgeConfig:
    c = _default_edge()
    if product == "HYDROGEL_PACK":
        c.min_take = max(c.min_take, 6)
        c.min_make = max(c.min_make, 4)
        c.k_make = 0.42
        c.take_frac = 0.18
        c.make_frac = 0.09
        c.vol_floor = 1.0
    elif product == "VELVETFRUIT_EXTRACT":
        c.min_take = max(c.min_take, 2)
        c.min_make = max(c.min_make, 2)
        c.k_make = 0.45
        c.vol_floor = 0.5
    elif product.startswith("VEV_"):
        if product in ANCHOR_LADDER_NEAR:
            c.take_frac = 0.2
            c.make_frac = 0.11
            c.k_make = max(c.k_make, 0.48)
        if mid < 2.0:
            c.k_take, c.k_make = 1.2, 0.4
            c.min_take, c.min_make = 1, 1
            c.take_frac, c.make_frac = 0.2, 0.15
            c.vol_floor = 0.35
        elif mid < 30.0:
            c.k_take, c.k_make = 2.4, 0.5
            c.min_take, c.min_make = 1, 1
            c.vol_floor = 0.5
        elif mid < 2000.0:
            c.min_take = max(c.min_take, 2)
            c.min_make = max(c.min_make, 2)
        else:
            c.min_take = max(c.min_take, 4)
            c.min_make = max(c.min_make, 3)
            c.k_take, c.k_make = 2.0, 0.45
            c.take_frac, c.make_frac = 0.2, 0.1
    return c


def _load_memory(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        mem = json.loads(trader_data)
        return mem if isinstance(mem, dict) else {}
    except (ValueError, TypeError):
        return {}


def _best_bid_ask(od: OrderDepth) -> tuple[int | None, int | None]:
    best_bid = max(od.buy_orders) if od.buy_orders else None
    best_ask = min(od.sell_orders) if od.sell_orders else None
    return best_bid, best_ask


def _all_mids(order_depths: dict[str, OrderDepth]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, od in order_depths.items():
        bb, ba = _best_bid_ask(od)
        if bb is not None and ba is not None:
            out[name] = (bb + ba) / 2.0
    return out


def _microprice(od: OrderDepth) -> float | None:
    best_bid, best_ask = _best_bid_ask(od)
    if best_bid is None and best_ask is None:
        return None
    if best_bid is None:
        return float(best_ask)
    if best_ask is None:
        return float(best_bid)
    bid_sz = abs(od.buy_orders[best_bid])
    ask_sz = abs(od.sell_orders[best_ask])
    total = bid_sz + ask_sz
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return (ask_sz * best_bid + bid_sz * best_ask) / total


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _alpha_fair_for(product: str) -> float:
    if product == "VELVETFRUIT_EXTRACT":
        return _ALPHA_FAIR_VELVET
    return _ALPHA_FAIR_BASE


def _update_online_state(state: dict, mid: float, vol_floor: float, product: str) -> dict:
    af = _alpha_fair_for(product)
    prev_fair = state.get("fair")
    prev_mid = state.get("prev_mid")
    prev_vol = float(state.get("vol", 1.0))
    prev_slow = state.get("slow_mean")
    slow_var = float(state.get("slow_var", max(1.0, vol_floor * vol_floor)))

    if prev_fair is None:
        new_fair = mid
    else:
        new_fair = af * mid + (1.0 - af) * float(prev_fair)
    state["fair"] = new_fair
    state["prev_mid"] = mid

    if prev_slow is None:
        state["slow_mean"] = mid
    else:
        slow = (1.0 - ALPHA_SLOW) * float(prev_slow) + ALPHA_SLOW * mid
        resid = mid - slow
        state["slow_mean"] = slow
        state["slow_var"] = max(
            (1.0 - ALPHA_SLOW) * slow_var + ALPHA_SLOW * resid * resid,
            vol_floor * vol_floor,
            0.25,
        )

    if prev_mid is None:
        state.setdefault("ret_n", 0)
        state.setdefault("ret_mean", 0.0)
        state.setdefault("ret_M2", 0.0)
        state["vol"] = max(prev_vol, vol_floor)
        return state

    ret = mid - float(prev_mid)
    n = int(state.get("ret_n", 0)) + 1
    mean = float(state.get("ret_mean", 0.0))
    M2 = float(state.get("ret_M2", 0.0))
    delta = ret - mean
    mean += delta / n
    delta2 = ret - mean
    M2 += delta * delta2
    state["ret_n"] = n
    state["ret_mean"] = mean
    state["ret_M2"] = M2

    residual = abs(ret - mean)
    new_vol = ALPHA_VOL * residual + (1.0 - ALPHA_VOL) * prev_vol
    state["vol"] = max(new_vol, vol_floor)
    return state


def _drift_stats(state: dict) -> Tuple[float, float]:
    n = int(state.get("ret_n", 0))
    if n < 2:
        return 0.0, 0.0
    mean = float(state.get("ret_mean", 0.0))
    M2 = float(state.get("ret_M2", 0.0))
    var = M2 / (n - 1)
    if var <= 0.0:
        return mean, 0.0
    std_of_mean = (var / n) ** 0.5
    if std_of_mean <= 0.0:
        return mean, 0.0
    return mean, mean / std_of_mean


def _update_cross_section(
    product: str,
    mid: float,
    pstate: dict,
    g: dict[str, Any],
    mids: dict[str, float],
) -> None:
    m5000 = mids.get("VEV_5000")
    mvel = mids.get("VELVETFRUIT_EXTRACT")
    if m5000 is not None and mvel is not None and mvel > 1.0 and m5000 > 1.0:
        rkv = m5000 / mvel
        prev = float(g.get("kv", rkv))
        g["kv"] = (1.0 - ALPHA_KV) * prev + ALPHA_KV * rkv
    if (
        product.startswith("VEV_")
        and product != "VEV_5000"
        and mid > WING_MID_MAX
        and m5000 is not None
        and m5000 > REF5000_MIN
    ):
        kr_now = mid / m5000
        prev_r = pstate.get("kr")
        if prev_r is None:
            pstate["kr"] = kr_now
        else:
            pstate["kr"] = (1.0 - ALPHA_KR) * float(prev_r) + ALPHA_KR * kr_now


def _ladder_vol_scale(vol: float) -> float:
    v = max(float(vol), 0.1)
    return 1.0 / (1.0 + v / LADDER_VOL_SCALE_K)


def _anchor_offset(
    product: str,
    mid: float,
    pstate: dict,
    g: dict[str, Any],
    mids: dict[str, float],
    vol: float,
) -> float:
    m5000 = mids.get("VEV_5000")
    mvel = mids.get("VELVETFRUIT_EXTRACT")
    kv = g.get("kv")
    v_est = max(float(vol), 0.5)

    def _soft_cap(delta: float) -> float:
        m = LADDER_ANCHOR_CAP_MULT * v_est
        if delta > m:
            return m
        if delta < -m:
            return -m
        return delta

    if mid <= WING_MID_MAX:
        return 0.0
    if product == "VELVETFRUIT_EXTRACT" and m5000 is not None and kv and float(kv) > 0:
        anchor = m5000 / float(kv)
        return ANCHOR_VELVET * (anchor - mid)
    if product == "VEV_5000" and mvel is not None and mvel > 1.0 and kv and float(kv) > 0:
        anchor = mvel * float(kv)
        return ANCHOR_V5000 * (anchor - mid)
    if (
        product.startswith("VEV_")
        and product not in ("VEV_5000", "VEV_6000", "VEV_6500")
        and m5000 is not None
        and m5000 > REF5000_MIN
    ):
        kr = pstate.get("kr")
        if kr is None:
            return 0.0
        anchor = float(kr) * m5000
        w = ANCHOR_LADDER * ANCHOR_LADDER_NEAR.get(product, 1.0) * _ladder_vol_scale(v_est)
        return _soft_cap(w * (anchor - mid))
    return 0.0


def _micro_tilt_for(product: str) -> float:
    if product == "VELVETFRUIT_EXTRACT":
        return MICRO_TILT_VELVET
    if product == "VEV_5000":
        return MICRO_TILT_V5000
    return MICRO_TILT_BASE


def _inv_skew_k_for(product: str) -> float:
    if product == "VELVETFRUIT_EXTRACT":
        return INV_SKEW_VELVET
    if product == "VEV_5000":
        return INV_SKEW_V5000
    if product == "VEV_5100":
        return INV_SKEW_V51
    if product == "VEV_5200":
        return INV_SKEW_V52
    if product == "VEV_5300":
        return INV_SKEW_V53
    return INV_SKEW_K


def _init_tv(g: dict[str, Any]) -> dict[str, float]:
    tv = g.get("tv")
    if not isinstance(tv, dict):
        tv = {}
    for product, seed in TV_SEED.items():
        tv.setdefault(product, seed)
    g["tv"] = tv
    return tv  # type: ignore[return-value]


def _weighted_clipped(parts: list[tuple[float, float]], center: float | None, clip: float) -> float | None:
    if not parts:
        return None
    if center is None:
        sorted_vals = sorted(value for value, _weight in parts)
        center = sorted_vals[len(sorted_vals) // 2]
    num = 0.0
    den = 0.0
    for value, weight in parts:
        num += weight * _clamp(value, center - clip, center + clip)
        den += weight
    return num / den if den > 0 else None


def _estimate_underlying(mids: dict[str, float], g: dict[str, Any]) -> float | None:
    tv = _init_tv(g)
    center = mids.get("VELVETFRUIT_EXTRACT")
    parts: list[tuple[float, float]] = []
    if center is not None:
        parts.append((center, 4.0))
    if "VEV_4000" in mids:
        parts.append((mids["VEV_4000"] + 4000.0, 2.0))
    if "VEV_4500" in mids:
        parts.append((mids["VEV_4500"] + 4500.0, 2.0))
    if "VEV_5000" in mids:
        parts.append((mids["VEV_5000"] + 5000.0 - float(tv["VEV_5000"]), 1.0))
    return _weighted_clipped(parts, center, 8.0)


def _update_time_values(g: dict[str, Any], mids: dict[str, float], s_hat: float | None) -> None:
    if s_hat is None:
        return
    tv = _init_tv(g)
    for product in VEV_ORDER:
        if product in WINGS:
            tv[product] = 0.5
            continue
        if product not in mids:
            continue
        strike = VEV_STRIKES[product]
        observed = mids[product] - max(s_hat - strike, 0.0)
        lo, hi = TV_BOUNDS[product]
        observed = _clamp(observed, lo, hi)
        tv[product] = (1.0 - ALPHA_TV) * float(tv.get(product, TV_SEED[product])) + ALPHA_TV * observed


def _vev_fairs(g: dict[str, Any], s_hat: float | None) -> dict[str, float]:
    if s_hat is None:
        return {}
    tv = _init_tv(g)
    fairs: dict[str, float] = {}
    prev: float | None = None
    for product in VEV_ORDER:
        if product in WINGS:
            fair = 0.5
        else:
            fair = max(s_hat - VEV_STRIKES[product], 0.0) + float(tv.get(product, TV_SEED[product]))
        if prev is not None:
            fair = min(fair, prev)
        fairs[product] = max(0.5, fair)
        prev = fairs[product]
    return fairs


def _option_fair_offset(
    product: str,
    mid: float,
    vol: float,
    s_hat: float | None,
    fairs: dict[str, float],
) -> float:
    weight = OPTION_FAIR_WEIGHT.get(product, 0.0)
    if weight <= 0:
        return 0.0
    if product == "VELVETFRUIT_EXTRACT":
        if s_hat is None:
            return 0.0
        raw = s_hat - mid
    else:
        fair = fairs.get(product)
        if fair is None:
            return 0.0
        raw = fair - mid
    cap = max(0.75, 1.50 * max(vol, 0.5))
    return _clamp(weight * raw, -cap, cap)


def _portfolio_delta(position: Dict[str, int]) -> float:
    total = float(position.get("VELVETFRUIT_EXTRACT", 0))
    for product, delta in DELTAS.items():
        if product == "VELVETFRUIT_EXTRACT":
            continue
        total += float(position.get(product, 0)) * delta
    return total


def _flow_weight(product: str, trader: str) -> float:
    if product == "HYDROGEL_PACK":
        if trader == "Mark 14":
            return 1.0
        if trader == "Mark 38":
            return -1.0
        return 0.0
    if product == "VEV_4000":
        if trader == "Mark 14":
            return 0.9
        if trader == "Mark 38":
            return -0.9
        return 0.0
    if product == "VELVETFRUIT_EXTRACT":
        if trader == "Mark 01":
            return 0.52
        if trader == "Mark 67":
            return 0.38
        if trader == "Mark 14":
            return 0.20
        if trader == "Mark 55":
            return -0.62
        if trader == "Mark 49":
            return -0.45
    if product in CHEAP_VEVS:
        if trader == "Mark 01":
            return 0.25
        if trader == "Mark 22":
            return -0.25
    return 0.0


def _update_flow(memory: dict[str, Any], state: TradingState) -> None:
    for key, raw in list(memory.items()):
        if key == _GLOBAL_KEY or not isinstance(raw, dict):
            continue
        raw["flow_signal"] = FLOW_DECAY * float(raw.get("flow_signal", 0.0))
    for product, trades in state.market_trades.items():
        pstate = memory.get(product)
        if not isinstance(pstate, dict):
            pstate = {}
            memory[product] = pstate
        signal = float(pstate.get("flow_signal", 0.0))
        for trade in trades:
            if not isinstance(trade, Trade):
                continue
            qty = max(1, abs(int(trade.quantity)))
            signal += FLOW_UNIT * qty * (
                _flow_weight(product, str(trade.buyer))
                - _flow_weight(product, str(trade.seller))
            )
        pstate["flow_signal"] = _clamp(signal, -8.0, 8.0)


def _flow_offset(product: str, pstate: dict, vol: float) -> float:
    raw = FLOW_PRICE_WEIGHT * float(pstate.get("flow_signal", 0.0))
    cap = 1.5 * max(vol, 0.5)
    # Keep the participant signal targeted; broad VEV flow was weaker in sample.
    if product not in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_4000"} and product not in CHEAP_VEVS:
        raw *= 0.35
    return _clamp(raw, -cap, cap)


def _mr_offset(product: str, mid: float, pstate: dict, vol: float) -> tuple[float, int | None]:
    if product not in {"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"}:
        return 0.0, None
    slow = pstate.get("slow_mean")
    if slow is None:
        return 0.0, None
    std = max(float(pstate.get("slow_var", 1.0)) ** 0.5, 1.0)
    z = (mid - float(slow)) / std
    if abs(z) < MR_ENTRY_Z:
        return 0.0, None
    limit = _LIMITS.get(product, DEFAULT_LIMIT)
    target = int(round(-limit * _clamp(z / 3.0, -1.0, 1.0)))
    offset = _clamp(-MR_PULL * (mid - float(slow)), -2.0 * vol, 2.0 * vol)
    return offset, target


def _is_endgame(timestamp: int) -> bool:
    return int(timestamp % 1_000_000) > 990_000


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory: dict[str, Any] = _load_memory(state.traderData)
        _update_flow(memory, state)
        g = memory.get(_GLOBAL_KEY)
        if not isinstance(g, dict):
            g = {}
        result: Dict[str, List[Order]] = {}
        mids = _all_mids(state.order_depths)
        s_hat = _estimate_underlying(mids, g)
        _update_time_values(g, mids, s_hat)
        vev_fairs = _vev_fairs(g, s_hat)
        net_delta = _portfolio_delta(state.position)
        endgame = _is_endgame(state.timestamp)

        for product, od in state.order_depths.items():
            position = int(state.position.get(product, 0))
            product_state: dict = memory.get(product) or {}
            orders, product_state = self._adaptive(
                product,
                od,
                position,
                product_state,
                mids,
                g,
                s_hat,
                vev_fairs,
                net_delta,
                endgame,
            )
            result[product] = orders
            memory[product] = product_state

        memory[_GLOBAL_KEY] = g
        return result, 0, json.dumps(memory, separators=(",", ":"))

    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        mids: dict[str, float],
        g: dict[str, Any],
        s_hat: float | None,
        vev_fairs: dict[str, float],
        net_delta: float,
        endgame: bool,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, pstate

        mid = (best_bid + best_ask) / 2.0
        ec = _edge_config(product, mid)
        pstate = _update_online_state(pstate, float(mid), ec.vol_floor, product)
        _update_cross_section(product, float(mid), pstate, g, mids)

        if product in WINGS:
            return self._floor_wing(product, od, position, pstate, endgame)

        if int(pstate.get("ret_n", 0)) < WARMUP_TICKS:
            return orders, pstate

        fair = float(pstate["fair"])
        vol = float(pstate["vol"])
        drift_per_tick, t_stat = _drift_stats(pstate)
        micro = _microprice(od) or mid
        mtilt = _micro_tilt_for(product)

        if abs(t_stat) >= DRIFT_T_THRESHOLD:
            effective_drift = drift_per_tick * HORIZON
            target_frac = _clamp(effective_drift / DRIFT_TARGET_SCALE, -1.0, 1.0)
        else:
            effective_drift = 0.0
            target_frac = 0.0

        a_off = _anchor_offset(product, float(mid), pstate, g, mids, vol)
        o_off = _option_fair_offset(product, float(mid), vol, s_hat, vev_fairs)
        f_off = _flow_offset(product, pstate, vol)
        mr_off, mr_target = _mr_offset(product, float(mid), pstate, vol)
        expected = (fair + a_off + o_off + f_off + mr_off) * (1.0 - mtilt) + micro * mtilt

        base_take_edge = max(float(ec.min_take), ec.k_take * vol)
        make_edge = max(float(ec.min_make), ec.k_make * vol)
        if endgame:
            base_take_edge *= 1.5
            make_edge *= 1.25
        if effective_drift > 0:
            buy_edge = max(float(ec.min_take), base_take_edge - max(0.0, effective_drift))
            sell_edge = max(float(ec.min_take), base_take_edge + max(0.0, effective_drift))
        else:
            buy_edge = max(float(ec.min_take), base_take_edge - effective_drift)
            sell_edge = max(float(ec.min_take), base_take_edge + effective_drift)

        limit = int(_LIMITS.get(product, DEFAULT_LIMIT))
        target_pos = int(round(target_frac * limit))
        if mr_target is not None and abs(mr_target) > abs(target_pos):
            target_pos = mr_target
        cap_buy = limit - position
        cap_sell = limit + position
        take_cap = max(1, int(limit * ec.take_frac))
        make_cap = max(1, int(limit * ec.make_frac))

        rem = take_cap
        for ap in sorted(od.sell_orders):
            if cap_buy <= 0 or rem <= 0:
                break
            if ap > expected - buy_edge:
                break
            if endgame and position >= 0 and ap > expected - 2.0 * buy_edge:
                break
            sz = abs(od.sell_orders[ap])
            q = min(sz, cap_buy, rem)
            if q > 0:
                orders.append(Order(product, int(ap), int(q)))
                cap_buy -= q
                rem -= q

        rem = take_cap
        for bp in sorted(od.buy_orders, reverse=True):
            if cap_sell <= 0 or rem <= 0:
                break
            if bp < expected + sell_edge:
                break
            if product in CHEAP_VEVS and position <= 0 and bp < expected + max(2.0, 1.5 * sell_edge):
                break
            if endgame and position <= 0 and bp < expected + 2.0 * sell_edge:
                break
            sz = abs(od.buy_orders[bp])
            q = min(sz, cap_sell, rem)
            if q > 0:
                orders.append(Order(product, int(bp), -int(q)))
                cap_sell -= q
                rem -= q

        inv_e = position - target_pos
        isk = _inv_skew_k_for(product)
        skew = -isk * (inv_e / max(1.0, float(limit))) * make_edge
        delta_pressure = _clamp(net_delta / DELTA_LIMIT_SOFT, -1.5, 1.5)
        delta_skew = -DELTA_SKEW_K * delta_pressure * DELTAS.get(product, 0.0) * make_edge
        skew = _clamp(skew + delta_skew, -1.35 * make_edge, 1.35 * make_edge)
        s_bar = max(0.0, effective_drift) * ANTI_TREND_BARRIER_MULT
        b_bar = max(0.0, -effective_drift) * ANTI_TREND_BARRIER_MULT
        make_b = int(round(expected + skew - make_edge - b_bar))
        make_a = int(round(expected + skew + make_edge + s_bar))
        if b_bar == 0:
            make_b = min(make_b, best_bid)
        if s_bar == 0:
            make_a = max(make_a, best_ask)
        if make_b >= best_ask:
            make_b = best_ask - 1
        if make_a <= best_bid:
            make_a = best_bid + 1
        if make_b >= make_a:
            return orders, pstate

        if not (endgame and position >= 0):
            bq = min(make_cap, cap_buy)
            if bq > 0:
                orders.append(Order(product, make_b, int(bq)))
        if not (endgame and position <= 0):
            aq = min(make_cap, cap_sell)
            if product in CHEAP_VEVS and position <= 0:
                aq = 0
            if aq > 0:
                orders.append(Order(product, make_a, -int(aq)))
        return orders, pstate

    def _floor_wing(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        endgame: bool,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        limit = int(_LIMITS.get(product, DEFAULT_LIMIT))
        cap_buy = limit - position
        if not endgame and cap_buy > 0:
            orders.append(Order(product, 0, min(3, cap_buy)))
        if position > 0:
            orders.append(Order(product, 1, -min(position, 4 if not endgame else 8)))
        return orders, pstate
