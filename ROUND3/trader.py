"""Round 3 trader — built from the structure of `prices_round_3_day_*.csv` / trades.

The historical books show 12 products:
  * VELVETFRUIT_EXTRACT, HYDROGEL_PACK  — ~5k / ~10k; HYDROGEL has a very wide top spread.
  * VEV_4000 … VEV_6500 — a strike ladder; mids and spreads differ a lot by level.

`analyze_r3_data.py` / `INSIGHTS.md` in this folder document correlations and mean spreads
used to set per-name edge and size multipliers. Final tuned version keeps the robust
EWMA value/spread market maker, fits a live option surface for the VEV strike ladder, and
adds executable cointegration pair orders using beta hedge ratios. The drift leg remains
effectively disabled because it hurt out-of-sample backtests on the local round 3 days.

Tweak `LIMITS` to match the official IMC per-product cap table.
"""

from __future__ import annotations

import json
import math
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
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 100

MICRO_TILT = 0.3
DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 10.0
INV_SKEW_K = 1.0
ANTI_TREND_BARRIER_MULT = 3.0
RESIDUAL_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.05,
    "HYDROGEL_PACK": 0.45,
}

OPTION_STRIKES: Dict[str, int] = {
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

# Multipliers are a compact smile fit from the three Round 3 CSV days.
# A live median implied total vol is fitted every tick, then adjusted by strike.
STRIKE_VOL_MULT: Dict[int, float] = {
    4000: 1.00,
    4500: 1.00,
    5000: 1.00,
    5100: 1.00,
    5200: 1.00,
    5300: 1.01,
    5400: 0.95,
    5500: 1.03,
    6000: 1.05,
    6500: 1.05,
}
SURFACE_VOL_INIT = 0.032
SURFACE_VOL_ALPHA = 0.20
OPTION_MODEL_BLEND = 0.0
OPTION_TARGET_SCALE = 0.0
DELTA_HEDGE_DEADBAND = 4.0
DELTA_HEDGE_SCALE = 0.70

COINT_TRIGGER_Z = 1.25
COINT_ENTRY_Z = 2.5
COINT_MAX_PAIR_QTY = 2
COINT_MODEL_BLEND = 0.0
COINT_TARGET_SCALE = 0.0
COINT_PAIRS: Tuple[Tuple[str, str, float, float, float], ...] = (
    # y, x, alpha, beta, residual std for y = alpha + beta * x.
    ("VEV_4000", "VEV_4500", 499.906, 1.0001, 0.409),
    ("VELVETFRUIT_EXTRACT", "VEV_4500", 4501.328, 0.9982, 0.758),
    ("VEV_5000", "VEV_5100", 70.098, 1.1086, 2.663),
    ("VEV_5100", "VEV_5200", 42.692, 1.2990, 2.188),
    ("VEV_5200", "VEV_5300", 24.333, 1.5230, 1.850),
    ("VEV_5400", "VEV_5500", 3.625, 1.8560, 1.159),
)


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
        c.min_make = max(c.min_make, 4)
        c.k_make = 0.42
        c.take_frac = 0.18
        c.make_frac = 0.19
        c.vol_floor = 1.0
    elif product == "VELVETFRUIT_EXTRACT":
        c.k_take = 2.3
        c.min_take = max(c.min_take, 4)
        c.min_make = max(c.min_make, 2)
        c.k_make = 0.45
        c.take_frac = 0.14
        c.make_frac = 0.10
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


def _book_mid(od: OrderDepth) -> float | None:
    best_bid, best_ask = _best_bid_ask(od)
    if best_bid is None or best_ask is None:
        return None
    return 0.5 * (best_bid + best_ask)


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


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _call_price(spot: float, strike: float, total_vol: float) -> float:
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    intrinsic = max(0.0, spot - strike)
    if total_vol <= 1e-8:
        return intrinsic
    d1 = (math.log(spot / strike) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    return spot * _normal_cdf(d1) - strike * _normal_cdf(d2)


def _call_delta(spot: float, strike: float, total_vol: float) -> float:
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    if total_vol <= 1e-8:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * total_vol * total_vol) / total_vol
    return _normal_cdf(d1)


def _implied_total_vol(spot: float, strike: float, price: float) -> float | None:
    intrinsic = max(0.0, spot - strike)
    if spot <= 0.0 or strike <= 0.0 or price <= intrinsic + 0.05 or price >= spot:
        return None
    lo = 1e-5
    hi = 0.25
    while _call_price(spot, strike, hi) < price and hi < 2.0:
        hi *= 2.0
    if hi >= 2.0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _call_price(spot, strike, mid) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _median(values: List[float]) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _surface_from_books(
    order_depths: Dict[str, OrderDepth],
    memory: dict,
) -> tuple[Dict[str, float], Dict[str, float], dict]:
    surface_state = memory.get("_surface") or {}
    velvet_od = order_depths.get("VELVETFRUIT_EXTRACT")
    if velvet_od is None:
        return {}, {}, surface_state
    velvet_micro = _microprice(velvet_od)
    best_bid, best_ask = _best_bid_ask(velvet_od)
    if velvet_micro is None or best_bid is None or best_ask is None:
        return {}, {}, surface_state

    spot = float(velvet_micro)
    implied_bases: List[float] = []
    for product, strike in OPTION_STRIKES.items():
        if strike < 5000 or strike > 5500:
            continue
        od = order_depths.get(product)
        if od is None:
            continue
        bid, ask = _best_bid_ask(od)
        if bid is None or ask is None:
            continue
        mid = 0.5 * (bid + ask)
        implied = _implied_total_vol(spot, float(strike), mid)
        if implied is None:
            continue
        mult = STRIKE_VOL_MULT.get(strike, 1.0)
        base = implied / max(0.5, mult)
        if 0.015 <= base <= 0.060:
            implied_bases.append(base)

    previous = float(surface_state.get("base_vol", SURFACE_VOL_INIT))
    raw = _median(implied_bases)
    if raw is None:
        base_vol = previous
    else:
        base_vol = SURFACE_VOL_ALPHA * raw + (1.0 - SURFACE_VOL_ALPHA) * previous
        base_vol = _clamp(base_vol, 0.018, 0.055)
    surface_state["base_vol"] = base_vol

    fairs: Dict[str, float] = {}
    deltas: Dict[str, float] = {}
    for product, strike in OPTION_STRIKES.items():
        vol = base_vol * STRIKE_VOL_MULT.get(strike, 1.0)
        fairs[product] = _call_price(spot, float(strike), vol)
        deltas[product] = _call_delta(spot, float(strike), vol)
    return fairs, deltas, surface_state


def _add_weighted(acc: Dict[str, List[float]], product: str, fair: float, weight: float) -> None:
    if weight <= 0.0:
        return
    if product not in acc:
        acc[product] = [0.0, 0.0]
    acc[product][0] += fair * weight
    acc[product][1] += weight


def _cointegration_fairs(order_depths: Dict[str, OrderDepth]) -> Dict[str, float]:
    acc: Dict[str, List[float]] = {}
    for y, x, alpha, beta, sigma in COINT_PAIRS:
        y_od = order_depths.get(y)
        x_od = order_depths.get(x)
        if y_od is None or x_od is None or abs(beta) <= 1e-9:
            continue
        y_mid = _book_mid(y_od)
        x_mid = _book_mid(x_od)
        if y_mid is None or x_mid is None:
            continue

        y_fair = alpha + beta * x_mid
        residual = y_mid - y_fair
        z = residual / max(1e-6, sigma)
        if abs(z) < COINT_TRIGGER_Z:
            continue

        # Large residuals get more authority, but the cap avoids a single pair dominating.
        weight = min(3.0, abs(z)) / max(1.0, sigma)
        _add_weighted(acc, y, y_fair, weight)
        _add_weighted(acc, x, (y_mid - alpha) / beta, weight)

    return {product: total / weight for product, (total, weight) in acc.items() if weight > 0.0}


def _planned_positions(position: Dict[str, int], result: Dict[str, List[Order]]) -> Dict[str, int]:
    planned = {product: int(qty) for product, qty in position.items()}
    for product, orders in result.items():
        planned.setdefault(product, int(position.get(product, 0)))
        for order in orders:
            planned[product] += int(order.quantity)
    return planned


def _append_if_room(
    result: Dict[str, List[Order]],
    planned: Dict[str, int],
    product: str,
    price: int,
    quantity: int,
) -> bool:
    if quantity == 0:
        return False
    limit = int(_LIMITS.get(product, DEFAULT_LIMIT))
    current = int(planned.get(product, 0))
    if quantity > 0:
        quantity = min(quantity, limit - current)
    else:
        quantity = -min(-quantity, limit + current)
    if quantity == 0:
        return False
    result.setdefault(product, []).append(Order(product, int(price), int(quantity)))
    planned[product] = current + quantity
    return True


def _add_cointegration_pair_orders(
    result: Dict[str, List[Order]],
    order_depths: Dict[str, OrderDepth],
    position: Dict[str, int],
) -> None:
    planned = _planned_positions(position, result)
    for y, x, alpha, beta, sigma in COINT_PAIRS:
        y_od = order_depths.get(y)
        x_od = order_depths.get(x)
        if y_od is None or x_od is None or beta <= 0.0:
            continue
        y_bid, y_ask = _best_bid_ask(y_od)
        x_bid, x_ask = _best_bid_ask(x_od)
        if y_bid is None or y_ask is None or x_bid is None or x_ask is None:
            continue

        entry_edge = max(1.0, COINT_ENTRY_Z * sigma)
        rich_y_edge = y_bid - alpha - beta * x_ask
        cheap_y_edge = alpha + beta * x_bid - y_ask

        if rich_y_edge > entry_edge:
            y_room = min(abs(y_od.buy_orders[y_bid]), _LIMITS.get(y, DEFAULT_LIMIT) + planned.get(y, 0))
            x_room = min(abs(x_od.sell_orders[x_ask]), _LIMITS.get(x, DEFAULT_LIMIT) - planned.get(x, 0))
            y_qty = min(COINT_MAX_PAIR_QTY, y_room, int(x_room / max(1.0, beta)))
            if y_qty > 0:
                x_qty = min(x_room, max(1, int(round(beta * y_qty))))
                if _append_if_room(result, planned, y, y_bid, -y_qty):
                    _append_if_room(result, planned, x, x_ask, x_qty)

        elif cheap_y_edge > entry_edge:
            y_room = min(abs(y_od.sell_orders[y_ask]), _LIMITS.get(y, DEFAULT_LIMIT) - planned.get(y, 0))
            x_room = min(abs(x_od.buy_orders[x_bid]), _LIMITS.get(x, DEFAULT_LIMIT) + planned.get(x, 0))
            y_qty = min(COINT_MAX_PAIR_QTY, y_room, int(x_room / max(1.0, beta)))
            if y_qty > 0:
                x_qty = min(x_room, max(1, int(round(beta * y_qty))))
                if _append_if_room(result, planned, y, y_ask, y_qty):
                    _append_if_room(result, planned, x, x_bid, -x_qty)


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
        option_fairs, option_deltas, surface_state = _surface_from_books(state.order_depths, memory)
        memory["_surface"] = surface_state
        option_delta_exposure = 0.0
        for product, delta in option_deltas.items():
            option_delta_exposure += float(state.position.get(product, 0)) * delta

        for product, od in state.order_depths.items():
            position = int(state.position.get(product, 0))
            pstate: dict = memory.get(product) or {}
            orders, pstate = self._adaptive(
                product,
                od,
                position,
                pstate,
                option_fairs.get(product),
                option_deltas.get(product),
                None,
                option_delta_exposure,
            )
            result[product] = orders
            memory[product] = pstate

        _add_cointegration_pair_orders(result, state.order_depths, state.position)
        return result, 0, json.dumps(memory)

    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        option_fair: float | None,
        option_delta: float | None,
        coint_fair: float | None,
        option_delta_exposure: float,
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

        book_expected = (fair + effective_drift) * (1.0 - MICRO_TILT) + micro * MICRO_TILT
        if option_fair is not None:
            expected = OPTION_MODEL_BLEND * option_fair + (1.0 - OPTION_MODEL_BLEND) * book_expected
        else:
            expected = book_expected
        if coint_fair is not None:
            expected = COINT_MODEL_BLEND * coint_fair + (1.0 - COINT_MODEL_BLEND) * expected

        base_take_edge = max(float(ec.min_take), ec.k_take * vol)
        residual_scale = float(RESIDUAL_TARGET_SCALES.get(product, 0.0))
        if coint_fair is not None:
            residual_target = _clamp((coint_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * COINT_TARGET_SCALE
        elif option_fair is not None:
            residual_target = _clamp((option_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * OPTION_TARGET_SCALE
        else:
            residual_target = _clamp((fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * residual_scale
        target_frac = _clamp(target_frac + residual_target, -1.0, 1.0)
        make_edge = max(float(ec.min_make), ec.k_make * vol)
        if effective_drift > 0:
            buy_edge = max(float(ec.min_take), base_take_edge - max(0.0, effective_drift))
            sell_edge = max(float(ec.min_take), base_take_edge + max(0.0, effective_drift))
        else:
            buy_edge = max(float(ec.min_take), base_take_edge - effective_drift)
            sell_edge = max(float(ec.min_take), base_take_edge + effective_drift)

        limit = int(_LIMITS.get(product, DEFAULT_LIMIT))
        if product == "VELVETFRUIT_EXTRACT":
            net_delta = option_delta_exposure + position
            if abs(net_delta) > DELTA_HEDGE_DEADBAND:
                hedge_shift = _clamp(-DELTA_HEDGE_SCALE * net_delta / max(1.0, float(limit)), -0.75, 0.75)
                target_frac = _clamp(target_frac + hedge_shift, -1.0, 1.0)
        target_pos = int(round(target_frac * limit))
        cap_buy = limit - position
        cap_sell = limit + position
        take_cap = max(1, int(limit * ec.take_frac))
        make_cap = max(1, int(limit * ec.make_frac))

        rem = take_cap
        for ap in sorted(od.sell_orders):
            if cap_buy <= 0 or rem <= 0:
                break
            book_take = ap <= expected - buy_edge
            model_take = option_fair is not None and ap <= option_fair - base_take_edge
            coint_take = coint_fair is not None and ap <= coint_fair - base_take_edge
            if not (book_take or model_take or coint_take):
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
            book_take = bp >= expected + sell_edge
            model_take = option_fair is not None and bp >= option_fair + base_take_edge
            coint_take = coint_fair is not None and bp >= coint_fair + base_take_edge
            if not (book_take or model_take or coint_take):
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
