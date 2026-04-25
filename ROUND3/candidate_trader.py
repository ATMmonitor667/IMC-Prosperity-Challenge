"""Experimental Round 3 hybrid trader.

This file is intentionally small: it imports the current `final_trader` helpers and
tests a slow-anchor spot inventory target plus soft VELVET delta hedge.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import final_trader as base
from datamodel import Order, OrderDepth, TradingState


ALPHA_SLOW_FAIR = float(os.environ.get("ALPHA_SLOW_FAIR", str(2.0 / (10000.0 + 1.0))))
SLOW_TARGET_PRODUCTS = frozenset({"VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"})
SLOW_TARGET_SCALES = {
    "VELVETFRUIT_EXTRACT": float(os.environ.get("VELVET_SLOW_SCALE", "1.8")),
    "HYDROGEL_PACK": float(os.environ.get("HYDRO_SLOW_SCALE", "1.5")),
}
DELTA_HEDGE_DEADBAND = float(os.environ.get("DELTA_HEDGE_DEADBAND", "4.0"))
DELTA_HEDGE_SCALE = float(os.environ.get("DELTA_HEDGE_SCALE", "0.70"))
USE_DELTA_HEDGE = os.environ.get("USE_DELTA_HEDGE", "1") != "0"
base.WARMUP_TICKS = int(os.environ.get("WARMUP_TICKS", str(base.WARMUP_TICKS)))
base.INV_SKEW_K = float(os.environ.get("INV_SKEW_K", str(base.INV_SKEW_K)))
base.NEIGHBOR_RESIDUAL_SCALE = float(os.environ.get("NEIGHBOR_RESIDUAL_SCALE", str(base.NEIGHBOR_RESIDUAL_SCALE)))
base.BLOCK_RISK_SKEW_K = float(os.environ.get("BLOCK_RISK_SKEW_K", str(base.BLOCK_RISK_SKEW_K)))
base.USE_WING_THROTTLE = os.environ.get("USE_WING_THROTTLE", "1" if base.USE_WING_THROTTLE else "0") != "0"
HYDRO_MIN_TAKE = int(os.environ.get("HYDRO_MIN_TAKE", "8"))
HYDRO_MIN_MAKE = int(os.environ.get("HYDRO_MIN_MAKE", "4"))


def _update_state_with_slow_anchor(pstate: dict, mid: float, vol_floor: float) -> dict:
    pstate = base._update_online_state(pstate, mid, vol_floor)
    prev_slow = pstate.get("slow_fair")
    if prev_slow is None:
        pstate["slow_fair"] = mid
    else:
        pstate["slow_fair"] = ALPHA_SLOW_FAIR * mid + (1.0 - ALPHA_SLOW_FAIR) * float(prev_slow)
    return pstate


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        memory: dict[str, Any] = base._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}
        option_fairs, option_deltas, surface_state = base._surface_from_books(state.order_depths, memory)
        memory["_surface"] = surface_state
        block_risk = base._vev_block_risk(state.position)
        option_delta_exposure = 0.0
        for product, delta in option_deltas.items():
            option_delta_exposure += float(state.position.get(product, 0)) * float(delta)

        for product, od in state.order_depths.items():
            position = int(state.position.get(product, 0))
            pstate: dict = memory.get(product) or {}
            nbr = base._vev_neighbor_predicted_mid(product, state.order_depths)
            orders, pstate = self._adaptive(
                product,
                od,
                position,
                pstate,
                option_fairs.get(product),
                None,
                block_risk=block_risk,
                neighbor_pred=nbr,
                option_delta_exposure=option_delta_exposure,
            )
            result[product] = orders
            memory[product] = pstate

        base._add_cointegration_pair_orders(result, state.order_depths, state.position)
        return result, 0, json.dumps(memory)

    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        option_fair: float | None,
        coint_fair: float | None,
        *,
        block_risk: float = 0.0,
        neighbor_pred: float | None = None,
        option_delta_exposure: float = 0.0,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []
        best_bid, best_ask = base._best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, pstate

        mid = (best_bid + best_ask) / 2.0
        ec = base._edge_config(product, mid)
        if product == "HYDROGEL_PACK":
            ec.min_take = max(ec.min_take, HYDRO_MIN_TAKE)
            ec.min_make = max(ec.min_make, HYDRO_MIN_MAKE)
        pstate = _update_state_with_slow_anchor(pstate, float(mid), ec.vol_floor)

        if int(pstate.get("ret_n", 0)) < base.WARMUP_TICKS:
            return orders, pstate

        fair = float(pstate["fair"])
        vol = float(pstate["vol"])
        drift_per_tick, t_stat = base._drift_stats(pstate)
        micro = base._microprice(od) or mid

        if abs(t_stat) >= base.DRIFT_T_THRESHOLD:
            effective_drift = drift_per_tick * base.HORIZON
            target_frac = base._clamp(effective_drift / base.DRIFT_TARGET_SCALE, -1.0, 1.0)
        else:
            effective_drift = 0.0
            target_frac = 0.0

        book_expected = (fair + effective_drift) * (1.0 - base.MICRO_TILT) + micro * base.MICRO_TILT
        if option_fair is not None:
            expected = base.OPTION_MODEL_BLEND * option_fair + (1.0 - base.OPTION_MODEL_BLEND) * book_expected
        else:
            expected = book_expected
        if coint_fair is not None:
            expected = base.COINT_MODEL_BLEND * coint_fair + (1.0 - base.COINT_MODEL_BLEND) * expected

        base_take_edge = max(float(ec.min_take), ec.k_take * vol)
        residual_scale = float(base.RESIDUAL_TARGET_SCALES.get(product, 0.0))
        if product in SLOW_TARGET_PRODUCTS:
            slow_fair = float(pstate.get("slow_fair", fair))
            slow_scale = float(SLOW_TARGET_SCALES.get(product, residual_scale))
            residual_target = base._clamp((slow_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * slow_scale
        elif coint_fair is not None:
            residual_target = base._clamp((coint_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * base.COINT_TARGET_SCALE
        elif option_fair is not None:
            residual_target = base._clamp((option_fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * base.OPTION_TARGET_SCALE
        else:
            residual_target = base._clamp((fair - mid) / max(1.0, base_take_edge), -1.0, 1.0) * residual_scale
        target_frac = base._clamp(target_frac + residual_target, -1.0, 1.0)
        if product.startswith("VEV_") and neighbor_pred is not None:
            nr = base._clamp((neighbor_pred - mid) / max(1.0, base_take_edge), -1.0, 1.0) * base.NEIGHBOR_RESIDUAL_SCALE
            target_frac = base._clamp(target_frac + nr, -1.0, 1.0)

        make_edge = max(float(ec.min_make), ec.k_make * vol)
        if base.USE_WING_THROTTLE and product in base.WING_VEV:
            make_edge *= base.WING_MAKE_EDGE_MULT
        if effective_drift > 0:
            buy_edge = max(float(ec.min_take), base_take_edge - max(0.0, effective_drift))
            sell_edge = max(float(ec.min_take), base_take_edge + max(0.0, effective_drift))
        else:
            buy_edge = max(float(ec.min_take), base_take_edge - effective_drift)
            sell_edge = max(float(ec.min_take), base_take_edge + effective_drift)

        limit = int(base._LIMITS.get(product, base.DEFAULT_LIMIT))
        if USE_DELTA_HEDGE and product == "VELVETFRUIT_EXTRACT":
            net_delta = option_delta_exposure + position
            if abs(net_delta) > DELTA_HEDGE_DEADBAND:
                hedge_shift = base._clamp(-DELTA_HEDGE_SCALE * net_delta / max(1.0, float(limit)), -0.75, 0.75)
                target_frac = base._clamp(target_frac + hedge_shift, -1.0, 1.0)

        target_pos = int(round(target_frac * limit))
        cap_buy = limit - position
        cap_sell = limit + position
        take_cap = max(1, int(limit * ec.take_frac))
        make_cap = max(1, int(limit * ec.make_frac))
        if base.USE_WING_THROTTLE and product in base.WING_VEV:
            take_cap = max(1, int(take_cap * base.WING_TAKE_FRAC))
            make_cap = max(1, int(make_cap * base.WING_MAKE_FRAC))

        rem = take_cap
        for ap in sorted(od.sell_orders):
            if cap_buy <= 0 or rem <= 0:
                break
            book_take = ap <= expected - buy_edge
            model_take = option_fair is not None and ap <= option_fair - base_take_edge
            coint_take = coint_fair is not None and ap <= coint_fair - base_take_edge
            if not (book_take or model_take or coint_take):
                break
            q = min(abs(od.sell_orders[ap]), cap_buy, rem)
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
            q = min(abs(od.buy_orders[bp]), cap_sell, rem)
            if q > 0:
                orders.append(Order(product, int(bp), -int(q)))
                cap_sell -= q
                rem -= q

        inv_e = position - target_pos
        skew = -base.INV_SKEW_K * (inv_e / max(1.0, float(limit))) * make_edge
        if product.startswith("VEV_") and base.BLOCK_RISK_SKEW_K > 0.0:
            skew -= base.BLOCK_RISK_SKEW_K * block_risk * make_edge
        skew = base._clamp(skew, -make_edge, make_edge)
        s_bar = max(0.0, effective_drift) * base.ANTI_TREND_BARRIER_MULT
        b_bar = max(0.0, -effective_drift) * base.ANTI_TREND_BARRIER_MULT
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
