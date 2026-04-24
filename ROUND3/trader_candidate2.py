from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

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

VELVET = "VELVETFRUIT_EXTRACT"
OPTIONS = (
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
)
OPTION_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
}
CALIBRATION_OPTIONS = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")

HORIZON = 100
ALPHA_FAIR = 2.0 / (60.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 100

MICRO_TILT = 0.3
DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 1.0
INV_SKEW_K = 1.0
ANTI_TREND_BARRIER_MULT = 3.0

DEFAULT_TOTAL_VOL = 0.0322
SURFACE_ALPHA = 0.30
SURFACE_BIAS_ALPHA = 0.08
FLOW_WEIGHT = 0.62
OPTION_SIGNAL_SCALE = 1.75
OPTION_TARGET_FRAC = 0.80


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
        c.min_take = max(c.min_take, 6)
        c.min_make = max(c.min_make, 4)
        c.k_make = 0.42
        c.take_frac = 0.18
        c.make_frac = 0.09
        c.vol_floor = 1.0
    elif product == VELVET:
        c.min_take = max(c.min_take, 2)
        c.min_make = max(c.min_make, 2)
        c.k_make = 0.45
        c.vol_floor = 0.5
    elif product in OPTIONS:
        if mid < 20.0:
            c.k_take, c.k_make = 1.4, 0.45
            c.min_take, c.min_make = 1, 1
            c.take_frac, c.make_frac = 0.35, 0.18
            c.vol_floor = 0.2
        elif mid < 200.0:
            c.k_take, c.k_make = 1.8, 0.50
            c.min_take, c.min_make = 1, 1
            c.take_frac, c.make_frac = 0.30, 0.16
            c.vol_floor = 0.35
        else:
            c.k_take, c.k_make = 1.9, 0.45
            c.min_take, c.min_make = 2, 2
            c.take_frac, c.make_frac = 0.22, 0.12
            c.vol_floor = 0.5
    elif product.startswith("VEV_"):
        if mid < 2.0:
            c.k_take, c.k_make = 1.2, 0.4
            c.min_take, c.min_make = 1, 1
            c.take_frac, c.make_frac = 0.2, 0.15
            c.vol_floor = 0.35
        else:
            c.min_take = max(c.min_take, 2)
            c.min_make = max(c.min_make, 2)
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


def _mid(od: OrderDepth) -> float | None:
    best_bid, best_ask = _best_bid_ask(od)
    if best_bid is None and best_ask is None:
        return None
    if best_bid is None:
        return float(best_ask)
    if best_ask is None:
        return float(best_bid)
    return (best_bid + best_ask) / 2.0


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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _call_price(spot: float, strike: int, total_vol: float) -> float:
    if spot <= 0:
        return 0.0
    if total_vol <= 1e-9:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)


def _call_delta(spot: float, strike: int, total_vol: float) -> float:
    if spot <= 0:
        return 0.0
    if total_vol <= 1e-9:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * total_vol * total_vol) / total_vol
    return _norm_cdf(d1)


def _implied_total_vol(price: float, spot: float, strike: int) -> float | None:
    intrinsic = max(0.0, spot - strike)
    if spot <= 0 or price <= intrinsic + 1e-6:
        return None
    lo = 1e-5
    hi = 0.30
    for _ in range(40):
        mid = (lo + hi) / 2.0
        estimate = _call_price(spot, strike, mid)
        if estimate > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _estimate_surface_vol(order_depths: Dict[str, OrderDepth], fallback: float, spot_ref: float) -> float:
    values: list[float] = []
    for product in CALIBRATION_OPTIONS:
        od = order_depths.get(product)
        if od is None:
            continue
        mid = _mid(od)
        if mid is None:
            continue
        iv = _implied_total_vol(mid, spot_ref, OPTION_STRIKES[product])
        if iv is not None:
            values.append(iv)
    if len(values) >= 3:
        values.sort()
        raw = values[len(values) // 2]
        return (1.0 - SURFACE_ALPHA) * fallback + SURFACE_ALPHA * raw
    return fallback


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory: dict[str, Any] = _load_memory(state.traderData)
        surface_state: dict[str, Any] = memory.setdefault("surface", {})
        result: Dict[str, List[Order]] = {}

        spot_depth = state.order_depths.get(VELVET)
        spot_mid = _mid(spot_depth) if spot_depth is not None else None
        spot_micro = _microprice(spot_depth) if spot_depth is not None else None
        prev_spot = float(surface_state.get("prev_spot", spot_mid or 5250.0))
        spot_ref = float(spot_micro or spot_mid or prev_spot)
        prev_surface_vol = float(surface_state.get("nu", DEFAULT_TOTAL_VOL))
        surface_vol = _estimate_surface_vol(state.order_depths, prev_surface_vol, spot_ref)

        for product, od in state.order_depths.items():
            position = int(state.position.get(product, 0))
            pstate: dict = memory.get(product) or {}
            if product in OPTIONS and spot_mid is not None:
                orders, pstate = self._surface_option(
                    product,
                    od,
                    position,
                    pstate,
                    spot_ref,
                    prev_spot,
                    prev_surface_vol,
                    surface_vol,
                )
            else:
                orders, pstate = self._adaptive(product, od, position, pstate)
            result[product] = orders
            memory[product] = pstate

        surface_state["prev_spot"] = spot_ref
        surface_state["nu"] = surface_vol
        return result, 0, json.dumps(memory, separators=(",", ":"))

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
        return self._execute_quotes(product, od, position, expected, vol, target_frac, ec), pstate

    def _surface_option(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        spot_ref: float,
        prev_spot: float,
        prev_surface_vol: float,
        surface_vol: float,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, pstate

        mid = (best_bid + best_ask) / 2.0
        ec = _edge_config(product, mid)
        pstate = _update_online_state(pstate, float(mid), ec.vol_floor)

        predicted_surface = _call_price(spot_ref, OPTION_STRIKES[product], surface_vol)
        surface_bias = float(pstate.get("surface_bias", 0.0))
        surface_fair = predicted_surface + surface_bias

        flow_fair = surface_fair
        last_mid = pstate.get("last_mid")
        if last_mid is not None:
            delta = _call_delta(max(prev_spot, 1.0), OPTION_STRIKES[product], prev_surface_vol)
            flow_fair = float(last_mid) + delta * (spot_ref - prev_spot)

        fair = FLOW_WEIGHT * flow_fair + (1.0 - FLOW_WEIGHT) * surface_fair
        vol = max(float(pstate.get("vol", 1.0)), ec.vol_floor)
        micro = _microprice(od) or mid
        expected = fair * (1.0 - MICRO_TILT) + micro * MICRO_TILT

        signal = fair - mid
        target_frac = _clamp(signal / max(float(ec.min_take), OPTION_SIGNAL_SCALE * vol), -OPTION_TARGET_FRAC, OPTION_TARGET_FRAC)

        orders = self._execute_quotes(product, od, position, expected, vol, target_frac, ec)

        current_bias = mid - predicted_surface
        pstate["surface_bias"] = (1.0 - SURFACE_BIAS_ALPHA) * surface_bias + SURFACE_BIAS_ALPHA * current_bias
        pstate["last_mid"] = mid
        return orders, pstate

    def _execute_quotes(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        expected: float,
        vol: float,
        target_frac: float,
        ec: EdgeConfig,
    ) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        base_take_edge = max(float(ec.min_take), ec.k_take * vol)
        make_edge = max(float(ec.min_make), ec.k_make * vol)
        buy_edge = base_take_edge
        sell_edge = base_take_edge

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
        make_b = int(round(expected + skew - make_edge))
        make_a = int(round(expected + skew + make_edge))
        make_b = min(make_b, best_ask - 1)
        make_a = max(make_a, best_bid + 1)
        if make_b > best_bid:
            make_b = min(make_b, best_ask - 1)
        else:
            make_b = best_bid
        if make_a < best_ask:
            make_a = max(make_a, best_bid + 1)
        else:
            make_a = best_ask
        if make_b >= make_a:
            return orders

        bid_bias = max(0, target_pos - position)
        ask_bias = max(0, position - target_pos)
        bq = min(make_cap + bid_bias, cap_buy)
        if bq > 0:
            orders.append(Order(product, make_b, int(bq)))
        aq = min(make_cap + ask_bias, cap_sell)
        if aq > 0:
            orders.append(Order(product, make_a, -int(aq)))
        return orders
