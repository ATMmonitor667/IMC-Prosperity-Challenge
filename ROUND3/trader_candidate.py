from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState

DEFAULT_LIMIT = 20
LIMITS: Dict[str, int] = {
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
HYDROGEL = "HYDROGEL_PACK"
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
STATIC_WINGS = ("VEV_6000", "VEV_6500")
CALIBRATION_OPTIONS = ("VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500")

DEFAULT_TOTAL_VOL = 0.0322
SURFACE_ALPHA = 0.25
OPTION_RESID_ALPHA = 0.08
RET_VOL_ALPHA = 0.18
SPOT_FAIR_ALPHA = 0.06
FLOW_WEIGHT = 0.58
HEDGE_RATIO = 0.70


@dataclass(frozen=True)
class QuoteConfig:
    take_spread: float
    take_vol: float
    take_min: int
    make_spread: float
    make_vol: float
    make_min: int
    take_size: int
    make_size: int
    signal_scale: float
    max_target_frac: float
    inventory_skew: float
    quote_slack: float = 0.0


QUOTE_CONFIGS: Dict[str, QuoteConfig] = {
    HYDROGEL: QuoteConfig(0.44, 0.90, 5, 0.28, 0.60, 3, 7, 5, 3.0, 0.55, 2.0, 0.3),
    VELVET: QuoteConfig(0.42, 1.00, 2, 0.24, 0.55, 1, 9, 6, 3.5, 0.65, 2.2, 0.3),
    "VEV_4000": QuoteConfig(0.20, 1.60, 4, 0.14, 1.10, 3, 5, 3, 3.2, 0.65, 1.6, 0.2),
    "VEV_4500": QuoteConfig(0.20, 1.55, 3, 0.14, 1.05, 2, 5, 3, 3.4, 0.65, 1.5, 0.2),
    "VEV_5000": QuoteConfig(0.26, 1.70, 2, 0.17, 1.10, 1, 6, 4, 4.2, 0.70, 1.3, 0.2),
    "VEV_5100": QuoteConfig(0.26, 1.70, 2, 0.17, 1.10, 1, 6, 4, 4.0, 0.70, 1.3, 0.2),
    "VEV_5200": QuoteConfig(0.30, 1.85, 1, 0.20, 1.20, 1, 7, 4, 4.0, 0.75, 1.2, 0.2),
    "VEV_5300": QuoteConfig(0.32, 1.90, 1, 0.22, 1.25, 1, 7, 4, 4.2, 0.75, 1.2, 0.2),
    "VEV_5400": QuoteConfig(0.45, 2.10, 1, 0.30, 1.30, 1, 6, 3, 4.8, 0.80, 1.0, 0.2),
    "VEV_5500": QuoteConfig(0.48, 2.10, 1, 0.34, 1.30, 1, 6, 3, 5.2, 0.80, 0.9, 0.2),
}


def _limit(product: str) -> int:
    return int(LIMITS.get(product, DEFAULT_LIMIT))


def _load_memory(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}


def _best_bid_ask(depth: OrderDepth) -> tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def _mid_from_book(depth: OrderDepth) -> float | None:
    best_bid, best_ask = _best_bid_ask(depth)
    if best_bid is None and best_ask is None:
        return None
    if best_bid is None:
        return float(best_ask)
    if best_ask is None:
        return float(best_bid)
    return (best_bid + best_ask) / 2.0


def _microprice(depth: OrderDepth) -> float | None:
    best_bid, best_ask = _best_bid_ask(depth)
    if best_bid is None and best_ask is None:
        return None
    if best_bid is None:
        return float(best_ask)
    if best_ask is None:
        return float(best_bid)
    bid_sz = abs(depth.buy_orders[best_bid])
    ask_sz = abs(depth.sell_orders[best_ask])
    denom = bid_sz + ask_sz
    if denom <= 0:
        return (best_bid + best_ask) / 2.0
    return (ask_sz * best_bid + bid_sz * best_ask) / denom


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
    vols: list[float] = []
    for product in CALIBRATION_OPTIONS:
        depth = order_depths.get(product)
        if depth is None:
            continue
        mid = _mid_from_book(depth)
        if mid is None:
            continue
        iv = _implied_total_vol(mid, spot_ref, OPTION_STRIKES[product])
        if iv is not None:
            vols.append(iv)
    if len(vols) >= 3:
        vols.sort()
        raw = vols[len(vols) // 2]
        return (1.0 - SURFACE_ALPHA) * fallback + SURFACE_ALPHA * raw
    return fallback


def _update_ret_vol(pstate: dict[str, Any], mid: float) -> float:
    prev_mid = pstate.get("prev_mid")
    prev_vol = float(pstate.get("ret_vol", 1.0))
    if prev_mid is None:
        pstate["prev_mid"] = mid
        pstate["ret_vol"] = prev_vol
        return prev_vol
    diff = abs(mid - float(prev_mid))
    new_vol = (1.0 - RET_VOL_ALPHA) * prev_vol + RET_VOL_ALPHA * diff
    pstate["prev_mid"] = mid
    pstate["ret_vol"] = max(0.05, new_vol)
    return float(pstate["ret_vol"])


def _quote_product(
    product: str,
    depth: OrderDepth,
    fair: float,
    position: int,
    target_position: int,
    ret_vol: float,
) -> list[Order]:
    config = QUOTE_CONFIGS[product]
    limit = _limit(product)
    best_bid, best_ask = _best_bid_ask(depth)
    if best_bid is None or best_ask is None:
        return []

    spread = float(best_ask - best_bid)
    take_edge = max(float(config.take_min), config.take_spread * spread, config.take_vol * ret_vol)
    make_edge = max(float(config.make_min), config.make_spread * spread, config.make_vol * ret_vol)

    buy_capacity = limit - position
    sell_capacity = limit + position
    orders: list[Order] = []

    signal = fair - (best_bid + best_ask) / 2.0
    buy_take_cap = min(
        buy_capacity,
        config.take_size + max(0, target_position - position),
    )
    sell_take_cap = min(
        sell_capacity,
        config.take_size + max(0, position - target_position),
    )

    for ask in sorted(depth.sell_orders):
        if buy_take_cap <= 0:
            break
        if ask > fair - take_edge:
            break
        size = min(abs(depth.sell_orders[ask]), buy_take_cap)
        if size > 0:
            orders.append(Order(product, int(ask), int(size)))
            buy_capacity -= size
            buy_take_cap -= size

    for bid in sorted(depth.buy_orders, reverse=True):
        if sell_take_cap <= 0:
            break
        if bid < fair + take_edge:
            break
        size = min(abs(depth.buy_orders[bid]), sell_take_cap)
        if size > 0:
            orders.append(Order(product, int(bid), -int(size)))
            sell_capacity -= size
            sell_take_cap -= size

    if buy_capacity <= 0 and sell_capacity <= 0:
        return orders

    reservation = fair - config.inventory_skew * (position - target_position) / max(1.0, float(limit)) * make_edge

    buy_ok = reservation - make_edge >= best_bid - config.quote_slack and buy_capacity > 0
    sell_ok = reservation + make_edge <= best_ask + config.quote_slack and sell_capacity > 0

    if buy_ok:
        buy_quote = int(math.floor(reservation - make_edge))
        buy_quote = min(best_ask - 1, max(best_bid, buy_quote))
        buy_size = min(buy_capacity, config.make_size + max(0, target_position - position))
        if buy_size > 0 and buy_quote < best_ask:
            orders.append(Order(product, buy_quote, int(buy_size)))

    if sell_ok:
        sell_quote = int(math.ceil(reservation + make_edge))
        sell_quote = max(best_bid + 1, min(best_ask, sell_quote))
        sell_size = min(sell_capacity, config.make_size + max(0, position - target_position))
        if sell_size > 0 and sell_quote > best_bid:
            orders.append(Order(product, sell_quote, -int(sell_size)))

    return orders


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load_memory(state.traderData)
        memory.setdefault("spot", {})
        memory.setdefault("option", {})
        memory.setdefault("surface", {})

        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        spot_depth = state.order_depths.get(VELVET)
        spot_mid = _mid_from_book(spot_depth) if spot_depth else None
        spot_micro = _microprice(spot_depth) if spot_depth else None
        prev_spot = float(memory["surface"].get("prev_spot", spot_mid or 5250.0))
        spot_ref = float(spot_micro or spot_mid or prev_spot)

        surface_prev = float(memory["surface"].get("nu", DEFAULT_TOTAL_VOL))
        surface_vol = _estimate_surface_vol(state.order_depths, surface_prev, spot_ref)
        memory["surface"]["nu"] = surface_vol

        option_deltas: Dict[str, float] = {}
        option_targets: Dict[str, int] = {}

        for product in OPTIONS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            mid = _mid_from_book(depth)
            if mid is None:
                continue

            position = int(state.position.get(product, 0))
            pstate = memory["option"].setdefault(product, {})
            residual_anchor = float(pstate.get("resid_ema", 0.0))
            predicted_surface = _call_price(spot_ref, OPTION_STRIKES[product], surface_vol)
            surface_fair = predicted_surface + residual_anchor

            flow_fair = surface_fair
            if "last_mid" in pstate:
                delta = _call_delta(max(prev_spot, 1.0), OPTION_STRIKES[product], surface_prev)
                flow_fair = float(pstate["last_mid"]) + delta * (spot_ref - prev_spot)

            fair = FLOW_WEIGHT * flow_fair + (1.0 - FLOW_WEIGHT) * surface_fair
            fair = max(0.0, fair)

            ret_vol = _update_ret_vol(pstate, mid)
            config = QUOTE_CONFIGS[product]
            edge_unit = max(1.0, float(config.take_min), ret_vol, (depth and (_mid_from_book(depth) or 0.0)) * 0.0)
            signal = fair - mid
            raw_target = signal / max(1.0, config.signal_scale * edge_unit)
            target = int(round(_clamp(raw_target, -config.max_target_frac, config.max_target_frac) * _limit(product)))
            option_targets[product] = target

            result[product] = _quote_product(product, depth, fair, position, target, ret_vol)

            current_residual = mid - predicted_surface
            pstate["resid_ema"] = (1.0 - OPTION_RESID_ALPHA) * residual_anchor + OPTION_RESID_ALPHA * current_residual
            pstate["last_mid"] = mid

            option_deltas[product] = _call_delta(max(spot_ref, 1.0), OPTION_STRIKES[product], surface_vol)

        net_option_delta = 0.0
        for product, delta in option_deltas.items():
            net_option_delta += delta * int(state.position.get(product, 0))
            net_option_delta += 0.30 * delta * option_targets.get(product, 0)

        for product in STATIC_WINGS:
            result.setdefault(product, [])

        for product in (VELVET, HYDROGEL):
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            mid = _mid_from_book(depth)
            if mid is None:
                continue
            position = int(state.position.get(product, 0))
            pstate = memory["spot"].setdefault(product, {})
            fair_prev = float(pstate.get("fair", mid))
            fair = (1.0 - SPOT_FAIR_ALPHA) * fair_prev + SPOT_FAIR_ALPHA * mid
            ret_vol = _update_ret_vol(pstate, mid)

            target = 0
            if product == VELVET:
                hedge_target = int(round(_clamp(-HEDGE_RATIO * net_option_delta, -_limit(product), _limit(product))))
                signal_target = int(round(_clamp((fair_prev - mid) / max(1.5, ret_vol * 1.8), -0.35, 0.35) * _limit(product)))
                target = hedge_target + signal_target
            else:
                target = int(round(_clamp((fair_prev - mid) / max(3.0, ret_vol * 1.4), -0.45, 0.45) * _limit(product)))
            target = int(_clamp(target, -_limit(product), _limit(product)))

            result[product] = _quote_product(product, depth, fair_prev, position, target, ret_vol)
            pstate["fair"] = fair

        memory["surface"]["prev_spot"] = spot_ref

        return result, 0, json.dumps(memory, separators=(",", ":"))
