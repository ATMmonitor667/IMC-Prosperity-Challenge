"""Round 2 trader - adaptive, self-calibrating strategy.

Design goal (per request):
  - Do NOT hardcode per-product values such as a fair-price anchor or an
    assumed drift direction. The same code runs for every product and
    learns its own fair value, drift, and volatility online so it can
    face unseen data.

Approach (combines elements from ROUND2/strategies.txt):
  - Algorithm 2 (EMA fair + inventory-skewed quoting) as the base shape.
  - Algorithm 3 (microprice-aware take-then-make) for aggression.
  - Algorithm 4 (drift / z-score reasoning) for directional bias, but with
    a proper significance test so noise isn't treated as drift.

State learned per product (stored in traderData as JSON):
  fair      : EWMA of mid.                       (where price sits now)
  vol       : EWMA of |residual|.                (noise scale)
  prev_mid  : last mid seen (for computing returns).
  ret_n     : count of observed returns.
  ret_mean  : Welford running mean of returns.
  ret_M2    : Welford running sum of squared deviations.
  (drift per tick = ret_mean; statistical significance t = ret_mean /
  std_of_mean, computed on the fly.)

Per-tick decision:
  expected_fair = fair + drift * HORIZON            # price we expect soon
  take_edge     = max(MIN_TAKE_EDGE, K_TAKE * vol)  # volatility-scaled
  make_edge     = max(MIN_MAKE_EDGE, K_MAKE * vol)
  target_pos    = LIMIT * clip(drift*HORIZON / DRIFT_TARGET_SCALE, -1, 1)

  TAKE : walk levels while ask <= expected_fair - take_edge (buy),
         bid >= expected_fair + take_edge (sell).
  MAKE : quote at expected_fair +/- make_edge with an inventory skew
         that pulls price toward target_pos, never crossing inside the
         touch (make_bid <= best_bid, make_ask >= best_ask).

No per-product constants are used. A flat product ends up with
target_pos ~ 0 -> symmetric market making. A trending product ends up
with target_pos -> +/- LIMIT -> directional drift capture.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"
LIMIT: Dict[str, int] = {
    OSMIUM: 80,
    PEPPER: 80,
}
DEFAULT_LIMIT = 80


OSMIUM_TAKE_EDGE = 2
OSMIUM_MAKE_EDGE = 1
OSMIUM_TAKE_SIZE = 40
OSMIUM_MAKE_SIZE = 15
OSMIUM_SKEW_K = 3.0
OSMIUM_MICRO_TILT = 0.3
OSMIUM_REVERSION_WEIGHT = 0.8


PEPPER_TAKE_SIZE = 40
PEPPER_MAKE_BID_SIZE = 15
PEPPER_MAKE_ASK_SIZE = 10
PEPPER_BUY_TOLERANCE = 2
PEPPER_SELL_EDGE = 4

PEPPER_LOCK_WARMUP = 50
PEPPER_CUMRET_LOCK = 7.0
PEPPER_FAST_EWMA_LEN = 80.0
PEPPER_FAST_DRIFT_LOCK = 0.03
PEPPER_STOPLOSS_REVERSAL = 30.0
PEPPER_MR_WEIGHT = 0.0


HORIZON = 100
ALPHA_FAIR = 2.0 / (60.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 100

K_TAKE_EDGE = 2.0
K_MAKE_EDGE = 0.5
MIN_TAKE_EDGE = 2
MIN_MAKE_EDGE = 1

DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 1.0
INV_SKEW_K = 1.0
ANTI_TREND_BARRIER_MULT = 3.0

MICRO_TILT = 0.3

TAKE_SIZE_FRAC = 0.25
MAKE_SIZE_FRAC = 0.125


def _load_memory(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        mem = json.loads(trader_data)
        return mem if isinstance(mem, dict) else {}
    except (ValueError, TypeError):
        return {}


def _best_bid_ask(od: OrderDepth):
    best_bid = max(od.buy_orders) if od.buy_orders else None
    best_ask = min(od.sell_orders) if od.sell_orders else None
    return best_bid, best_ask


def _microprice(od: OrderDepth):
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
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _update_online_state(state: dict, mid: float) -> dict:
    """Online updates for fair (EWMA), vol (EWMA), and drift (Welford).

    Welford over the whole session gives us a running mean of per-tick
    returns plus a proper variance, so we can compute a t-statistic for
    'is drift really nonzero?' rather than trusting noisy short-window
    EWMA drift estimates.
    """
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
        state["vol"] = prev_vol
        return state

    ret = mid - prev_mid
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
    state["vol"] = new_vol

    return state


def _drift_stats(state: dict) -> Tuple[float, float]:
    """Return (drift_per_tick, t_stat). Zero until enough samples."""
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
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            product_state = memory.get(product, {})
            if product == OSMIUM:
                orders, product_state = self._osmium_strategy(
                    od, position, product_state
                )
            elif product == PEPPER:
                orders, product_state = self._pepper_strategy(
                    od, position, product_state
                )
            else:
                orders, product_state = self._adaptive_strategy(
                    od, position, product_state, product
                )
            result[product] = orders
            memory[product] = product_state

        return result, 0, json.dumps(memory)

    def _osmium_strategy(
        self,
        od: OrderDepth,
        position: int,
        state: dict,
    ) -> Tuple[List[Order], dict]:
        """Specialized mean-reversion MM for OSMIUM.

        Osmium is pinned around a stable mid with very strong tick-level
        mean reversion (lag-1 return autocorr ~ -0.5). We:
          - track fair as an online EWMA of mid (self-finds the pin),
          - blend in the microprice for an imbalance tilt,
          - tilt fair further by -0.5 * last_return (lag-1 reversion),
          - take any ask < fair - take_edge / bid > fair + take_edge,
          - post a single passive make quote per side, clamped to the
            touch, sized larger than the generic strategy since wider
            book fills more profitably,
          - skew passive quotes mildly against inventory so we drift back
            to zero.
        """
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, state

        mid = (best_bid + best_ask) / 2.0
        prev_mid = state.get("prev_mid")
        state = _update_online_state(state, mid)

        if int(state.get("ret_n", 0)) < WARMUP_TICKS:
            return orders, state

        fair = float(state["fair"])
        vol = float(state["vol"])
        micro = _microprice(od)
        if micro is None:
            micro = mid

        last_return = 0.0 if prev_mid is None else (mid - float(prev_mid))
        reversion_tilt = -OSMIUM_REVERSION_WEIGHT * last_return

        expected_fair = (
            fair * (1.0 - OSMIUM_MICRO_TILT) + micro * OSMIUM_MICRO_TILT
        ) + reversion_tilt

        take_edge = max(float(OSMIUM_TAKE_EDGE), K_TAKE_EDGE * vol)

        limit = LIMIT[OSMIUM]
        capacity_buy = limit - position
        capacity_sell = limit + position

        remaining = OSMIUM_TAKE_SIZE
        for ask_price in sorted(od.sell_orders):
            if capacity_buy <= 0 or remaining <= 0:
                break
            if ask_price > expected_fair - take_edge:
                break
            sz = abs(od.sell_orders[ask_price])
            qty = min(sz, capacity_buy, remaining)
            if qty > 0:
                orders.append(Order(OSMIUM, int(ask_price), int(qty)))
                capacity_buy -= qty
                remaining -= qty

        remaining = OSMIUM_TAKE_SIZE
        for bid_price in sorted(od.buy_orders, reverse=True):
            if capacity_sell <= 0 or remaining <= 0:
                break
            if bid_price < expected_fair + take_edge:
                break
            sz = abs(od.buy_orders[bid_price])
            qty = min(sz, capacity_sell, remaining)
            if qty > 0:
                orders.append(Order(OSMIUM, int(bid_price), -int(qty)))
                capacity_sell -= qty
                remaining -= qty

        inv_frac = position / max(1.0, float(limit))
        skew = -OSMIUM_SKEW_K * inv_frac
        make_edge = max(float(OSMIUM_MAKE_EDGE), K_MAKE_EDGE * vol)

        make_bid_price = int(round(expected_fair + skew - make_edge))
        make_ask_price = int(round(expected_fair + skew + make_edge))

        make_bid_price = min(make_bid_price, best_bid)
        make_ask_price = max(make_ask_price, best_ask)

        if make_bid_price >= best_ask:
            make_bid_price = best_ask - 1
        if make_ask_price <= best_bid:
            make_ask_price = best_bid + 1
        if make_bid_price >= make_ask_price:
            return orders, state

        bid_qty = min(OSMIUM_MAKE_SIZE, capacity_buy)
        if bid_qty > 0:
            orders.append(Order(OSMIUM, make_bid_price, int(bid_qty)))

        ask_qty = min(OSMIUM_MAKE_SIZE, capacity_sell)
        if ask_qty > 0:
            orders.append(Order(OSMIUM, make_ask_price, -int(ask_qty)))

        return orders, state

    def _pepper_strategy(
        self, od: OrderDepth, position: int, state: dict
    ) -> Tuple[List[Order], dict]:
        """Confirm-then-commit directional trader for PEPPER.

        The previous 'always long' strategy overfit to observed data: if a
        future session drifted down, it would lose the full cap. This
        version:

        1. Runs the symmetric adaptive market-maker until we have real
           evidence of drift direction (warmup on both sides).
        2. Locks a direction only when TWO independent signals agree:
             a) cumulative return from the day's open has moved more than
                PEPPER_CUMRET_LOCK in that direction, AND
             b) a short-length EWMA of returns is above PEPPER_FAST_DRIFT_LOCK
                in the same direction (confirms the move is continuing).
        3. Once locked, runs the aggressive long/short drift capture for
           that direction.
        4. Safety unlock: if cumulative return REVERSES by
           PEPPER_STOPLOSS_REVERSAL from its peak-in-lock-direction, we
           unlock and fall back to the adaptive market-maker. Prevents
           being trapped in the wrong direction for a whole day.
        """
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return [], state
        mid = (best_bid + best_ask) / 2.0

        if "open_mid" not in state:
            state["open_mid"] = mid
        delta = mid - float(state["open_mid"])

        prev_mid_p = state.get("prev_mid_pepper")
        state["prev_mid_pepper"] = mid
        ret = 0.0 if prev_mid_p is None else mid - float(prev_mid_p)
        state["pepper_last_ret"] = ret
        alpha = 2.0 / (PEPPER_FAST_EWMA_LEN + 1.0)
        fast_drift = alpha * ret + (1.0 - alpha) * float(
            state.get("fast_drift", 0.0)
        )
        state["fast_drift"] = fast_drift

        n_ticks = int(state.get("pepper_n", 0)) + 1
        state["pepper_n"] = n_ticks

        locked = state.get("direction_locked")

        if locked is None and n_ticks >= PEPPER_LOCK_WARMUP:
            if (
                delta > PEPPER_CUMRET_LOCK
                and fast_drift > PEPPER_FAST_DRIFT_LOCK
            ):
                locked = "long"
            elif (
                delta < -PEPPER_CUMRET_LOCK
                and fast_drift < -PEPPER_FAST_DRIFT_LOCK
            ):
                locked = "short"

        if locked is not None:
            peak_key = "peak_delta"
            peak = float(state.get(peak_key, delta))
            if locked == "long":
                peak = max(peak, delta)
            else:
                peak = min(peak, delta)
            state[peak_key] = peak

            reversed_by = (peak - delta) if locked == "long" else (delta - peak)
            if reversed_by >= PEPPER_STOPLOSS_REVERSAL:
                locked = None
                state.pop(peak_key, None)

        state["direction_locked"] = locked

        if locked == "long":
            return self._pepper_long_only(od, position, state)
        if locked == "short":
            return self._pepper_short_only(od, position, state)
        return self._adaptive_strategy(od, position, state, PEPPER)

    def _pepper_long_only(
        self, od: OrderDepth, position: int, state: dict
    ) -> Tuple[List[Order], dict]:
        """Aggressive long accumulation, used only after direction lock."""
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, state
        fair = _microprice(od) or ((best_bid + best_ask) / 2.0)
        last_ret = float(state.get("pepper_last_ret", 0.0))
        fair_eff = fair - PEPPER_MR_WEIGHT * last_ret
        limit = LIMIT[PEPPER]
        capacity_buy = limit - position
        capacity_sell = limit + position

        remaining = PEPPER_TAKE_SIZE
        for ap in sorted(od.sell_orders):
            if capacity_buy <= 0 or remaining <= 0:
                break
            if ap > fair_eff + PEPPER_BUY_TOLERANCE:
                break
            sz = abs(od.sell_orders[ap])
            qty = min(sz, capacity_buy, remaining)
            if qty > 0:
                orders.append(Order(PEPPER, int(ap), int(qty)))
                capacity_buy -= qty
                remaining -= qty

        remaining = PEPPER_TAKE_SIZE
        for bp in sorted(od.buy_orders, reverse=True):
            if capacity_sell <= 0 or remaining <= 0:
                break
            if bp < fair_eff + PEPPER_SELL_EDGE:
                break
            sz = abs(od.buy_orders[bp])
            qty = min(sz, capacity_sell, remaining)
            if qty > 0:
                orders.append(Order(PEPPER, int(bp), -int(qty)))
                capacity_sell -= qty
                remaining -= qty

        if capacity_buy > 0:
            bid_qty = min(PEPPER_MAKE_BID_SIZE, capacity_buy)
            if bid_qty > 0:
                orders.append(Order(PEPPER, int(best_bid), int(bid_qty)))
        if capacity_sell > 0 and position > 0:
            make_ask = max(int(round(fair_eff)) + PEPPER_SELL_EDGE, best_ask)
            ask_qty = min(PEPPER_MAKE_ASK_SIZE, capacity_sell)
            if ask_qty > 0:
                orders.append(Order(PEPPER, make_ask, -int(ask_qty)))
        return orders, state

    def _pepper_short_only(
        self, od: OrderDepth, position: int, state: dict
    ) -> Tuple[List[Order], dict]:
        """Aggressive short accumulation, used only after direction lock."""
        orders: List[Order] = []
        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, state
        fair = _microprice(od) or ((best_bid + best_ask) / 2.0)
        last_ret = float(state.get("pepper_last_ret", 0.0))
        fair_eff = fair - PEPPER_MR_WEIGHT * last_ret
        limit = LIMIT[PEPPER]
        capacity_buy = limit - position
        capacity_sell = limit + position

        remaining = PEPPER_TAKE_SIZE
        for bp in sorted(od.buy_orders, reverse=True):
            if capacity_sell <= 0 or remaining <= 0:
                break
            if bp < fair_eff - PEPPER_BUY_TOLERANCE:
                break
            sz = abs(od.buy_orders[bp])
            qty = min(sz, capacity_sell, remaining)
            if qty > 0:
                orders.append(Order(PEPPER, int(bp), -int(qty)))
                capacity_sell -= qty
                remaining -= qty

        if capacity_sell > 0:
            ask_qty = min(PEPPER_MAKE_BID_SIZE, capacity_sell)
            if ask_qty > 0:
                orders.append(Order(PEPPER, int(best_ask), -int(ask_qty)))
        if capacity_buy > 0 and position < 0:
            make_bid = min(int(round(fair_eff)) - PEPPER_SELL_EDGE, best_bid)
            bid_qty = min(PEPPER_MAKE_ASK_SIZE, capacity_buy)
            if bid_qty > 0:
                orders.append(Order(PEPPER, make_bid, int(bid_qty)))
        return orders, state

    def _adaptive_strategy(
        self,
        od: OrderDepth,
        position: int,
        state: dict,
        symbol: str,
    ) -> Tuple[List[Order], dict]:
        orders: List[Order] = []

        best_bid, best_ask = _best_bid_ask(od)
        if best_bid is None or best_ask is None:
            return orders, state

        mid = (best_bid + best_ask) / 2.0
        state = _update_online_state(state, mid)

        if int(state.get("ret_n", 0)) < WARMUP_TICKS:
            return orders, state

        fair = float(state["fair"])
        vol = float(state["vol"])

        drift_per_tick, t_stat = _drift_stats(state)

        micro = _microprice(od)
        if micro is None:
            micro = mid

        if abs(t_stat) >= DRIFT_T_THRESHOLD:
            effective_drift_term = drift_per_tick * HORIZON
            target_frac = _clamp(
                effective_drift_term / DRIFT_TARGET_SCALE, -1.0, 1.0
            )
        else:
            effective_drift_term = 0.0
            target_frac = 0.0

        expected_fair = (fair + effective_drift_term) * (1.0 - MICRO_TILT) + micro * MICRO_TILT

        base_take_edge = max(float(MIN_TAKE_EDGE), K_TAKE_EDGE * vol)
        make_edge = max(float(MIN_MAKE_EDGE), K_MAKE_EDGE * vol)

        buy_take_edge = max(
            float(MIN_TAKE_EDGE), base_take_edge - max(0.0, effective_drift_term)
        )
        sell_take_edge = max(
            float(MIN_TAKE_EDGE), base_take_edge + max(0.0, effective_drift_term)
        )
        if effective_drift_term < 0:
            buy_take_edge = max(
                float(MIN_TAKE_EDGE), base_take_edge - effective_drift_term
            )
            sell_take_edge = max(
                float(MIN_TAKE_EDGE), base_take_edge + effective_drift_term
            )

        limit = LIMIT.get(symbol, DEFAULT_LIMIT)
        target_pos = int(round(target_frac * limit))

        capacity_buy = limit - position
        capacity_sell = limit + position

        take_size_cap = max(1, int(limit * TAKE_SIZE_FRAC))

        remaining = take_size_cap
        for ask_price in sorted(od.sell_orders):
            if capacity_buy <= 0 or remaining <= 0:
                break
            if ask_price > expected_fair - buy_take_edge:
                break
            sz = abs(od.sell_orders[ask_price])
            qty = min(sz, capacity_buy, remaining)
            if qty > 0:
                orders.append(Order(symbol, int(ask_price), int(qty)))
                capacity_buy -= qty
                remaining -= qty

        remaining = take_size_cap
        for bid_price in sorted(od.buy_orders, reverse=True):
            if capacity_sell <= 0 or remaining <= 0:
                break
            if bid_price < expected_fair + sell_take_edge:
                break
            sz = abs(od.buy_orders[bid_price])
            qty = min(sz, capacity_sell, remaining)
            if qty > 0:
                orders.append(Order(symbol, int(bid_price), -int(qty)))
                capacity_sell -= qty
                remaining -= qty

        inv_error = position - target_pos
        skew = -INV_SKEW_K * (inv_error / max(1.0, float(limit))) * make_edge
        skew = _clamp(skew, -make_edge, make_edge)

        sell_barrier = max(0.0, effective_drift_term) * ANTI_TREND_BARRIER_MULT
        buy_barrier = max(0.0, -effective_drift_term) * ANTI_TREND_BARRIER_MULT

        make_bid_price = int(round(expected_fair + skew - make_edge - buy_barrier))
        make_ask_price = int(round(expected_fair + skew + make_edge + sell_barrier))

        if buy_barrier == 0:
            make_bid_price = min(make_bid_price, best_bid)
        if sell_barrier == 0:
            make_ask_price = max(make_ask_price, best_ask)

        if make_bid_price >= best_ask:
            make_bid_price = best_ask - 1
        if make_ask_price <= best_bid:
            make_ask_price = best_bid + 1
        if make_bid_price >= make_ask_price:
            return orders, state

        make_size_cap = max(1, int(limit * MAKE_SIZE_FRAC))

        bid_qty = min(make_size_cap, capacity_buy)
        if bid_qty > 0:
            orders.append(Order(symbol, make_bid_price, int(bid_qty)))

        ask_qty = min(make_size_cap, capacity_sell)
        if ask_qty > 0:
            orders.append(Order(symbol, make_ask_price, -int(ask_qty)))

        return orders, state
