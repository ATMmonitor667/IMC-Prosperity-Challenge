"""Round 3 trader — built from the structure of `prices_round_3_day_*.csv` / trades.

The historical books show 12 products:
  * VELVETFRUIT_EXTRACT, HYDROGEL_PACK  — ~5k / ~10k; HYDROGEL has a very wide top spread.
  * VEV_4000 … VEV_6500 — a strike ladder; mids and spreads differ a lot by level.

`analyze_r3_data.py` / `INSIGHTS.md` in this folder document correlations and mean spreads
used to set per-name edge and size multipliers. Final tuned version is a pure value/spread
maker: EWMA fair + vol-scaled take / make + tight inventory control, with a slow-anchor
residual inventory target on the spot-like books and the liquid VEV ladder. The drift leg
remains effectively disabled because it hurt out-of-sample backtests on the local round 3
days.

Tweak `LIMITS` to match the official IMC per-product cap table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

# Per-product max position; align with `run_bt.py` LIMITS.update (competition may override).
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

HORIZON = 100
ALPHA_FAIR = 2.0 / (60.0 + 1.0)
ALPHA_SLOW_FAIR = 2.0 / (20000.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 20

MICRO_TILT = 0.28
DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 10.0
INV_SKEW_K = 4.0
ANTI_TREND_BARRIER_MULT = 3.0
RESIDUAL_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.6,
    "HYDROGEL_PACK": 1.2,
    "VEV_4000": 0.6,
    "VEV_4500": 0.6,
    "VEV_5000": 1.0,
    "VEV_5100": 1.0,
    "VEV_5200": 1.0,
    "VEV_5300": 1.0,
    "VEV_5400": 0.6,
    "VEV_5500": 0.6,
}


@dataclass
class EdgeConfig:
    k_take: float
    k_make: float
    min_take: int
    min_make: int
    take_frac: float
    make_frac: float
    vol_floor: float


def _edge_config(product: str, mid: float) -> EdgeConfig:
    """Per-name + mid regime (from R3 CSV stats: spreads and scale)."""
    c = EdgeConfig(
        k_take=2.0,
        k_make=0.5,
        min_take=2,
        min_make=1,
        take_frac=0.25,
        make_frac=0.125,
        vol_floor=0.0,
    )
    if product == "HYDROGEL_PACK":
        c.min_take = max(c.min_take, 8)
        c.min_make = max(c.min_make, 5)
        c.k_make = 0.42
        c.take_frac = 0.18
        c.make_frac = 0.19
        c.vol_floor = 1.0
    elif product == "VELVETFRUIT_EXTRACT":
        c.k_take = 2.3
        c.min_take = max(c.min_take, 4)
        c.min_make = max(c.min_make, 2)
        c.k_make = 0.45
        c.take_frac = 0.04
        c.make_frac = 0.40
        c.vol_floor = 0.5
    elif product.startswith("VEV_"):
        if product in {"VEV_5000", "VEV_5100"}:
            c.k_take, c.k_make = 2.6, 0.55
            c.min_take, c.min_make = 3, 2
            c.take_frac, c.make_frac = 0.20, 0.10
            c.vol_floor = 0.5
            return c
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


def _update_online_state(state: dict, mid: float, vol_floor: float) -> dict:
    prev_fair = state.get("fair")
    prev_mid = state.get("prev_mid")
    prev_vol = float(state.get("vol", 1.0))

    if prev_fair is None:
        new_fair = mid
    else:
        new_fair = ALPHA_FAIR * mid + (1.0 - ALPHA_FAIR) * prev_fair
    state["fair"] = new_fair
    prev_slow = state.get("slow_fair")
    if prev_slow is None:
        state["slow_fair"] = mid
    else:
        state["slow_fair"] = ALPHA_SLOW_FAIR * mid + (1.0 - ALPHA_SLOW_FAIR) * prev_slow
    state["prev_mid"] = mid

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


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory: dict[str, Any] = _load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = int(state.position.get(product, 0))
            pstate: dict = memory.get(product) or {}
            orders, pstate = self._adaptive(product, od, position, pstate)
            result[product] = orders
            memory[product] = pstate

        return result, 0, json.dumps(memory)

    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, pstate

        mid = (best_bid + best_ask) / 2.0
        ec = _edge_config(product, mid)
        pstate = _update_online_state(pstate, float(mid), ec.vol_floor)

        if int(pstate.get("ret_n", 0)) < WARMUP_TICKS:
            return orders, pstate

        fair = float(pstate["fair"])
        vol = float(pstate["vol"])
        drift_per_tick, t_stat = _drift_stats(pstate)
        micro = _microprice(od) or mid

        if abs(t_stat) >= DRIFT_T_THRESHOLD:
            effective_drift = drift_per_tick * HORIZON
            target_frac = _clamp(effective_drift / DRIFT_TARGET_SCALE, -1.0, 1.0)
        else:
            effective_drift = 0.0
            target_frac = 0.0

        expected = (fair + effective_drift) * (1.0 - MICRO_TILT) + micro * MICRO_TILT

        base_take_edge = max(float(ec.min_take), ec.k_take * vol)
        residual_scale = float(RESIDUAL_TARGET_SCALES.get(product, 0.0))
        slow_fair = float(pstate.get("slow_fair", fair))
        residual_target = _clamp((slow_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * residual_scale
        target_frac = _clamp(target_frac + residual_target, -1.0, 1.0)
        make_edge = max(float(ec.min_make), ec.k_make * vol)
        if effective_drift > 0:
            buy_edge = max(float(ec.min_take), base_take_edge - max(0.0, effective_drift))
            sell_edge = max(float(ec.min_take), base_take_edge + max(0.0, effective_drift))
        else:
            buy_edge = max(float(ec.min_take), base_take_edge - effective_drift)
            sell_edge = max(float(ec.min_take), base_take_edge + effective_drift)

        limit = int(_LIMITS.get(product, DEFAULT_LIMIT))
        target_pos = int(round(target_frac * limit))
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
            sz = abs(od.buy_orders[bp])
            q = min(sz, cap_sell, rem)
            if q > 0:
                orders.append(Order(product, int(bp), -int(q)))
                cap_sell -= q
                rem -= q

        inv_e = position - target_pos
        skew = -INV_SKEW_K * (inv_e / max(1.0, float(limit))) * make_edge
        skew = _clamp(skew, -make_edge, make_edge)
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

        bq = min(make_cap, cap_buy)
        if bq > 0:
            orders.append(Order(product, make_b, int(bq)))
        aq = min(make_cap, cap_sell)
        if aq > 0:
            orders.append(Order(product, make_a, -int(aq)))
        return orders, pstate
