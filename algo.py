"""finalTrader.py — Round 1 finale.

Synthesis of three expert analyses (claude/gemini/chat) plus direct data audit.

ASH_COATED_OSMIUM  (stationary, mean 10,000, std ~5, spread modal=16)
    Strategy: fixed-fair aggressive market maker.
    Anchor 10_000.2 (empirical long-run mean), take any ask < fair /
    bid > fair, then post two-sided passive quotes at fair±edge with
    inventory skew.  Empirically edge=7 (ONE tick inside the 16-wide
    book) beats edge=1 in this matching engine because it captures
    the full spread on aggressor flow rather than trading constantly
    for 1-tick edge.  A subtle microprice tilt (±0.5) improves queue
    priority.  Add a second LADDER layer further out to pick up the
    occasional 18-19 wide-spread print.

INTARIAN_PEPPER_ROOT (deterministic +0.001/timestamp linear drift,
                      residual std ~2, spread modal 11-14)
    Strategy: slope-projected fair + long-biased asymmetric market maker.
    Within a day the trend is effectively deterministic: open→close
    climbs ~1,000.  Core changes over traderPrime:

      (1) Fair value = fast_EMA + slope * forward_horizon, where the
          slope itself is a slow EMA of mid-differences per unit
          timestamp.  This REMOVES the systematic lag that pure EMA
          (alpha=0.7) has against a monotonic drift.

      (2) Inventory is biased LONG (target ≈ +30) because every 100
          ticks the price rises ~10, so a long book is structurally
          profitable.  Skew is applied relative to this target.

      (3) Asymmetric edges: bid tight (eager to accumulate),
          ask wide (reluctant seller while trend keeps lifting).
          Gets more aggressive when measured slope is strongly positive.

      (4) Uses microprice (not mid) to drive the EMA so we don't
          chase bid/ask bounce.

State is persisted compactly as JSON:  pepper EMA, pepper slope
EMA, pepper anchor timestamp/price.

Position limit is 80 per product.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


# ══════════════════════════════════════════════════════════════
#   PRODUCTS  /  GLOBALS
# ══════════════════════════════════════════════════════════════

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# ── Osmium parameters ─────────────────────────────────────────
OSMIUM_FAIR        = 10_000.2   # empirical long-run mean
OSMIUM_EDGE = 7          # 1 tick inside typical 16-wide book
OSMIUM_EDGE_FAR = 11         # second-layer ladder (rarely filled, large edge)
OSMIUM_SIZE_NEAR = 60         # size at near quote
OSMIUM_SIZE_FAR = 20         # size at far quote
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP   = 0.5

# ── Pepper parameters ─────────────────────────────────────────
PEPPER_FAST_ALPHA = 0.6     # fast EMA of microprice
PEPPER_SLOPE_ALPHA = 0.02    # slow EMA of mid-change per ts unit
PEPPER_PRIOR_SLOPE = 0.001   # prior on slope from training data (+1000/day)
PEPPER_FORECAST_TS = 100     # project fair 300 timestamp units forward (~3 ticks)
PEPPER_BID_EDGE = 2       # tight bid: eager to accumulate longs
PEPPER_ASK_EDGE = 7       # wider ask: hold longs during uptrend
PEPPER_SKEW_DENOM = 40      # mild skew so we don't flatten against the trend
PEPPER_LONG_TARGET = 40      # structural long bias (inventory target)
PEPPER_SLOPE_ASYM_K = 800     # slope strength → extra asymmetry (edge tweak)


# ══════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════

class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if not od.buy_orders and not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid = float(best_bid)
            else:
                mid = float(best_ask)

            if product == OSMIUM:
                result[product] = _trade_osmium(
                    od, position, best_bid, best_ask, mid
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid, state.timestamp, memory
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


# ══════════════════════════════════════════════════════════════
#   OSMIUM — fixed-fair aggressive MM with two-layer ladder
# ══════════════════════════════════════════════════════════════

def _trade_osmium(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
) -> List[Order]:
    fair = OSMIUM_FAIR

    # Microprice tilt (very mild)
    tot_b = sum(abs(v) for v in od.buy_orders.values())
    tot_a = sum(abs(v) for v in od.sell_orders.values())
    denom = tot_b + tot_a
    if denom > 0:
        imbalance = (tot_b - tot_a) / denom
        fair += imbalance * OSMIUM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    # Phase 1: sweep mispriced levels
    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # Phase 2: passive ladder with inventory skew
    skew = position // OSMIUM_SKEW_DENOM
    fair_i = int(round(fair))

    post_bid_near = fair_i - OSMIUM_EDGE - skew
    post_ask_near = fair_i + OSMIUM_EDGE - skew
    post_bid_far  = fair_i - OSMIUM_EDGE_FAR - skew
    post_ask_far  = fair_i + OSMIUM_EDGE_FAR - skew

    # sanitise near quotes against the book
    if best_ask is not None and post_bid_near >= best_ask:
        post_bid_near = best_ask - 1
    if best_bid is not None and post_ask_near <= best_bid:
        post_ask_near = best_bid + 1
    if post_bid_near >= post_ask_near:
        post_bid_near = fair_i - 1
        post_ask_near = fair_i + 1

    # Near layer
    near_buy = min(OSMIUM_SIZE_NEAR, buy_cap)
    near_sell = min(OSMIUM_SIZE_NEAR, sell_cap)
    if near_buy > 0:
        orders.append(Order(OSMIUM, post_bid_near, near_buy))
        buy_cap -= near_buy
    if near_sell > 0:
        orders.append(Order(OSMIUM, post_ask_near, -near_sell))
        sell_cap -= near_sell

    # Far layer — catches wide-spread prints (18-19 tick spreads)
    far_buy = min(OSMIUM_SIZE_FAR, buy_cap)
    far_sell = min(OSMIUM_SIZE_FAR, sell_cap)
    if far_buy > 0 and post_bid_far < post_bid_near:
        orders.append(Order(OSMIUM, post_bid_far, far_buy))
    if far_sell > 0 and post_ask_far > post_ask_near:
        orders.append(Order(OSMIUM, post_ask_far, -far_sell))

    return orders


# ══════════════════════════════════════════════════════════════
#   PEPPER — slope-projected fair + long-biased asymmetric MM
# ══════════════════════════════════════════════════════════════

def _trade_pepper(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
    timestamp: int, memory: dict,
) -> List[Order]:

    # ── Microprice input (robust to bid/ask bounce) ──────────
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        price_in = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) else mid
    else:
        price_in = mid

    # ── Fast EMA of microprice ───────────────────────────────
    ema_prev = memory.get("pema")
    ts_prev = memory.get("pts")

    if ema_prev is None:
        ema = price_in
        slope = PEPPER_PRIOR_SLOPE  # start with training prior
    else:
        ema = PEPPER_FAST_ALPHA * price_in + (1 - PEPPER_FAST_ALPHA) * ema_prev
        slope_prev = memory.get("pslope", PEPPER_PRIOR_SLOPE)
        dt = max(1, timestamp - ts_prev) if ts_prev is not None else 100
        inst_slope = (price_in - ema_prev) / dt
        # Constrain instantaneous slope to avoid outliers
        inst_slope = max(-0.02, min(0.02, inst_slope))
        slope = PEPPER_SLOPE_ALPHA * inst_slope + (1 - PEPPER_SLOPE_ALPHA) * slope_prev

    memory["pema"] = ema
    memory["pslope"] = slope
    memory["pts"] = timestamp

    # ── Slope-projected fair ────────────────────────────────
    fair = ema + slope * PEPPER_FORECAST_TS

    # Slope sign for asymmetry/bias
    trend_up = slope > 0

    # ── Sweep aggressively vs projected fair ────────────────
    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(PEPPER, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(PEPPER, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # ── Long-biased skew ────────────────────────────────────
    # We want to be long when trending up. Skew is distance from target.
    target = PEPPER_LONG_TARGET if trend_up else 0
    skew = (position - target) // PEPPER_SKEW_DENOM

    # ── Asymmetric edges scaled by slope strength ────────────
    slope_str = max(0.0, slope) * PEPPER_SLOPE_ASYM_K
    extra = int(min(2, max(0, round(slope_str))))
    bid_edge = max(1, PEPPER_BID_EDGE - extra)      # tighter bid in strong uptrend
    ask_edge = PEPPER_ASK_EDGE + extra              # wider ask in strong uptrend

    fair_i = int(round(fair))
    post_bid = fair_i - bid_edge - skew
    post_ask = fair_i + ask_edge - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = fair_i - 1
        post_ask = fair_i + 1

    if buy_cap > 0:
        orders.append(Order(PEPPER, post_bid, buy_cap))
    if sell_cap > 0:
        orders.append(Order(PEPPER, post_ask, -sell_cap))

    return orders


# ══════════════════════════════════════════════════════════════
#   Utility
# ══════════════════════════════════════════════════════════════

def _load(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
``````````````````````````````````````````````````````````````````````````
"""finalTrader.py — Round 1 finale.

Synthesis of three expert analyses (claude/gemini/chat) plus direct data audit.

ASH_COATED_OSMIUM  (stationary, mean 10,000, std ~5, spread modal=16)
    Strategy: fixed-fair aggressive market maker.
    Anchor 10_000.2 (empirical long-run mean), take any ask < fair /
    bid > fair, then post two-sided passive quotes at fair±edge with
    inventory skew.  Empirically edge=7 (ONE tick inside the 16-wide
    book) beats edge=1 in this matching engine because it captures
    the full spread on aggressor flow rather than trading constantly
    for 1-tick edge.  A subtle microprice tilt (±0.5) improves queue
    priority.  Add a second LADDER layer further out to pick up the
    occasional 18-19 wide-spread print.

INTARIAN_PEPPER_ROOT (deterministic +0.001/timestamp linear drift,
                      residual std ~2, spread modal 11-14)
    Strategy: slope-projected fair + long-biased asymmetric market maker.
    Within a day the trend is effectively deterministic: open→close
    climbs ~1,000.  Core changes over traderPrime:

      (1) Fair value = fast_EMA + slope * forward_horizon, where the
          slope itself is a slow EMA of mid-differences per unit
          timestamp.  This REMOVES the systematic lag that pure EMA
          (alpha=0.7) has against a monotonic drift.

      (2) Inventory is biased LONG (target ≈ +30) because every 100
          ticks the price rises ~10, so a long book is structurally
          profitable.  Skew is applied relative to this target.

      (3) Asymmetric edges: bid tight (eager to accumulate),
          ask wide (reluctant seller while trend keeps lifting).
          Gets more aggressive when measured slope is strongly positive.

      (4) Uses microprice (not mid) to drive the EMA so we don't
          chase bid/ask bounce.

State is persisted compactly as JSON:  pepper EMA, pepper slope
EMA, pepper anchor timestamp/price.

Position limit is 80 per product.
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


# ══════════════════════════════════════════════════════════════
#   PRODUCTS  /  GLOBALS
# ══════════════════════════════════════════════════════════════

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# ── Osmium parameters ─────────────────────────────────────────
OSMIUM_FAIR        = 10_000.2   # empirical long-run mean
OSMIUM_EDGE = 7
OSMIUM_EDGE_FAR = 12
OSMIUM_SIZE_NEAR = 60
OSMIUM_SIZE_FAR = 20
OSMIUM_SKEW_DENOM = 40
OSMIUM_MICRO_CAP   = 0.5
OSMIUM_TAKE_SLOP = 1

# ── Pepper parameters ─────────────────────────────────────────
PEPPER_FAST_ALPHA = 0.6
PEPPER_SLOPE_ALPHA = 0.02
PEPPER_PRIOR_SLOPE = 0.001
PEPPER_FORECAST_TS = 100
PEPPER_BID_EDGE = 2
PEPPER_ASK_EDGE = 8
PEPPER_SKEW_DENOM = 20
PEPPER_LONG_TARGET = 80
PEPPER_SLOPE_ASYM_K = 800


# ══════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════

class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if not od.buy_orders and not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid = float(best_bid)
            else:
                mid = float(best_ask)

            if product == OSMIUM:
                result[product] = _trade_osmium(
                    od, position, best_bid, best_ask, mid
                )
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid, state.timestamp, memory
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


# ══════════════════════════════════════════════════════════════
#   OSMIUM — fixed-fair aggressive MM with two-layer ladder
# ══════════════════════════════════════════════════════════════

def _trade_osmium(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
) -> List[Order]:
    fair = OSMIUM_FAIR

    # Microprice tilt (very mild)
    tot_b = sum(abs(v) for v in od.buy_orders.values())
    tot_a = sum(abs(v) for v in od.sell_orders.values())
    denom = tot_b + tot_a
    if denom > 0:
        imbalance = (tot_b - tot_a) / denom
        fair += imbalance * OSMIUM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    # Phase 1: AGGRESSIVE sweep. Take asks up to fair + OSMIUM_TAKE_SLOP
    # and bids down to fair - OSMIUM_TAKE_SLOP. Lag-1 autocorr = -0.5,
    # so any print away from 10,000 has strong reversion expectancy.
    for ask_price in sorted(od.sell_orders):
        if ask_price > fair + OSMIUM_TAKE_SLOP or buy_cap <= 0:
            break
        # Discount: only take price-levels that are cheaper than fair
        # or where we need inventory (position still short of target).
        if ask_price >= fair and position >= 0:
            # neutral or long already; paying premium only makes sense
            # if price is clearly below 10000 — skip otherwise.
            continue
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price < fair - OSMIUM_TAKE_SLOP or sell_cap <= 0:
            break
        if bid_price <= fair and position <= 0:
            continue
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # Phase 2: passive ladder with inventory skew
    skew = position // OSMIUM_SKEW_DENOM
    fair_i = int(round(fair))

    post_bid_near = fair_i - OSMIUM_EDGE - skew
    post_ask_near = fair_i + OSMIUM_EDGE - skew
    post_bid_far  = fair_i - OSMIUM_EDGE_FAR - skew
    post_ask_far  = fair_i + OSMIUM_EDGE_FAR - skew

    if best_ask is not None and post_bid_near >= best_ask:
        post_bid_near = best_ask - 1
    if best_bid is not None and post_ask_near <= best_bid:
        post_ask_near = best_bid + 1
    if post_bid_near >= post_ask_near:
        post_bid_near = fair_i - 1
        post_ask_near = fair_i + 1

    near_buy = min(OSMIUM_SIZE_NEAR, buy_cap)
    near_sell = min(OSMIUM_SIZE_NEAR, sell_cap)
    if near_buy > 0:
        orders.append(Order(OSMIUM, post_bid_near, near_buy))
        buy_cap -= near_buy
    if near_sell > 0:
        orders.append(Order(OSMIUM, post_ask_near, -near_sell))
        sell_cap -= near_sell

    far_buy = min(OSMIUM_SIZE_FAR, buy_cap)
    far_sell = min(OSMIUM_SIZE_FAR, sell_cap)
    if far_buy > 0 and post_bid_far < post_bid_near:
        orders.append(Order(OSMIUM, post_bid_far, far_buy))
    if far_sell > 0 and post_ask_far > post_ask_near:
        orders.append(Order(OSMIUM, post_ask_far, -far_sell))

    return orders


# ══════════════════════════════════════════════════════════════
#   PEPPER — slope-projected fair + long-biased asymmetric MM
# ══════════════════════════════════════════════════════════════

def _trade_pepper(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
    timestamp: int, memory: dict,
) -> List[Order]:

    # ── Microprice input (robust to bid/ask bounce) ──────────
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        price_in = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) else mid
    else:
        price_in = mid

    # ── Fast EMA of microprice ───────────────────────────────
    ema_prev = memory.get("pema")
    ts_prev = memory.get("pts")

    if ema_prev is None:
        ema = price_in
        slope = PEPPER_PRIOR_SLOPE  # start with training prior
    else:
        ema = PEPPER_FAST_ALPHA * price_in + (1 - PEPPER_FAST_ALPHA) * ema_prev
        slope_prev = memory.get("pslope", PEPPER_PRIOR_SLOPE)
        dt = max(1, timestamp - ts_prev) if ts_prev is not None else 100
        inst_slope = (price_in - ema_prev) / dt
        # Constrain instantaneous slope to avoid outliers
        inst_slope = max(-0.02, min(0.02, inst_slope))
        slope = PEPPER_SLOPE_ALPHA * inst_slope + (1 - PEPPER_SLOPE_ALPHA) * slope_prev

    memory["pema"] = ema
    memory["pslope"] = slope
    memory["pts"] = timestamp

    # ── Slope-projected fair ────────────────────────────────
    fair = ema + slope * PEPPER_FORECAST_TS

    # Slope sign for asymmetry/bias
    trend_up = slope > 0

    # ── Sweep aggressively vs projected fair ────────────────
    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(PEPPER, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair or sell_cap <= 0:
            break
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(PEPPER, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # ── Long-biased skew ────────────────────────────────────
    # We want to be long when trending up. Skew is distance from target.
    target = PEPPER_LONG_TARGET if trend_up else 0
    skew = (position - target) // PEPPER_SKEW_DENOM

    # ── Asymmetric edges scaled by slope strength ────────────
    slope_str = max(0.0, slope) * PEPPER_SLOPE_ASYM_K
    extra = int(min(2, max(0, round(slope_str))))
    bid_edge = max(1, PEPPER_BID_EDGE - extra)      # tighter bid in strong uptrend
    ask_edge = PEPPER_ASK_EDGE + extra              # wider ask in strong uptrend

    fair_i = int(round(fair))
    post_bid = fair_i - bid_edge - skew
    post_ask = fair_i + ask_edge - skew

    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = fair_i - 1
        post_ask = fair_i + 1

    # Ladder bids: we accumulate more aggressively on dips
    if buy_cap > 0:
        near_buy = min(50, buy_cap)
        far_buy = buy_cap - near_buy
        orders.append(Order(PEPPER, post_bid, near_buy))
        if far_buy > 0:
            orders.append(Order(PEPPER, post_bid - 3, far_buy))

    # Only post asks when we have significant inventory. During a
    # strong uptrend, a passive ask that fills costs us the trend PnL
    # we could have earned by holding. Only sell when we are meaningfully
    # above the long target — otherwise let the long ride.
    if sell_cap > 0:
        if trend_up and position < PEPPER_LONG_TARGET + 5:
            pass  # skip passive ask — keep accumulating / holding
        else:
            # ladder sell: first chunk near, rest further out
            near_sell = min(40, sell_cap)
            far_sell = sell_cap - near_sell
            orders.append(Order(PEPPER, post_ask, -near_sell))
            if far_sell > 0:
                orders.append(Order(PEPPER, post_ask + 3, -far_sell))

    return orders


# ══════════════════════════════════════════════════════════════
#   Utility
# ══════════════════════════════════════════════════════════════

def _load(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
``````````````````````````````````````````````````````````````````````````````````
"""finalTrader.py — Round 1 finale.

Synthesis of three expert analyses (claude/gemini/chat) plus direct data audit.

ASH_COATED_OSMIUM (stationary, mean 10,000, std ~5, spread modal=16)
    Strategy: fixed-fair aggressive market-maker with two-layer ladder.
    Anchor 10_000.2 (empirical mean), take asks below fair, bids above
    fair (with mild slop for mean-reversion expectancy).  Post passive
    near-quotes inside the 16-wide book plus a wider outer tier that
    only fills when spreads briefly widen to 18-19.  Microprice tilt
    (±0.5) improves queue priority.

INTARIAN_PEPPER_ROOT (deterministic +0.001/timestamp drift, ~±5 noise)
    Empirical finding: a pure "buy every ask and hold" baseline earns
    ~238k across 3 days — 97% of trend theoretical max.  Posting a
    passive ASK during an uptrend is a NEGATIVE-EV action because the
    guaranteed drift keeps lifting price out from under us; every time
    our ask fills we've forfeited ~10 ticks of future drift.

    Strategy:
      1. Maintain fast EMA of microprice + slow EMA of slope.
      2. fair = ema + slope * forecast_horizon  (slope-projected).
      3. Take every ask below fair (aggressive accumulation).
      4. Post a tight passive bid to catch seller-aggressor flow.
      5. Post a passive ask ONLY when position is above the long target
         (ie. we have inventory we're happy to shed on a spike).
      6. If a bid > fair + large_margin appears (rare), sell INTO it.

Position limit = 80 per product (exchange rule).
"""

import json
from typing import Dict, List, Tuple

from datamodel import Order, OrderDepth, TradingState


# ══════════════════════════════════════════════════════════════
#   PRODUCTS  /  GLOBALS
# ══════════════════════════════════════════════════════════════

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = 80

# ── Osmium parameters ─────────────────────────────────────────
OSMIUM_FAIR        = 10_000.2
OSMIUM_EDGE        = 7
OSMIUM_EDGE_FAR    = 12
OSMIUM_SIZE_NEAR   = 60
OSMIUM_SIZE_FAR    = 20
OSMIUM_SKEW_DENOM  = 40
OSMIUM_MICRO_CAP   = 0.5
OSMIUM_TAKE_SLOP   = 1   # how far above fair we'll take; leverages lag-1 autocorr=-0.5

# ── Pepper parameters ─────────────────────────────────────────
PEPPER_FAST_ALPHA        = 0.6
PEPPER_SLOPE_ALPHA       = 0.02
PEPPER_PRIOR_SLOPE       = 0.001    # training prior: +1000 per day / 1M ts
PEPPER_FORECAST_TS = 1000
PEPPER_BID_EDGE          = 2        # tight bid: eager to accumulate
PEPPER_ASK_EDGE          = 7        # wide ask: reluctant seller in uptrend
PEPPER_LONG_TARGET = 75
PEPPER_ASK_TRIGGER_POS = 78
PEPPER_SELL_TAKE_MARGIN = 3


# ══════════════════════════════════════════════════════════════
#   ENTRY POINT
# ══════════════════════════════════════════════════════════════

class Trader:
    def run(
        self, state: TradingState
    ) -> Tuple[Dict[str, List[Order]], int, str]:
        memory = _load(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            position = state.position.get(product, 0)
            if not od.buy_orders and not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None
            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2.0
            elif best_bid is not None:
                mid = float(best_bid)
            else:
                mid = float(best_ask)

            if product == OSMIUM:
                result[product] = _trade_osmium(od, position, best_bid, best_ask, mid)
            elif product == PEPPER:
                result[product] = _trade_pepper(
                    od, position, best_bid, best_ask, mid, state.timestamp, memory
                )
            else:
                result[product] = []

        return result, 0, json.dumps(memory)


# ══════════════════════════════════════════════════════════════
#   OSMIUM — fixed-fair aggressive MM with two-layer ladder
# ══════════════════════════════════════════════════════════════

def _trade_osmium(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
) -> List[Order]:
    fair = OSMIUM_FAIR

    tot_b = sum(abs(v) for v in od.buy_orders.values())
    tot_a = sum(abs(v) for v in od.sell_orders.values())
    denom = tot_b + tot_a
    if denom > 0:
        imbalance = (tot_b - tot_a) / denom
        fair += imbalance * OSMIUM_MICRO_CAP

    orders: List[Order] = []
    buy_cap = LIMIT - position
    sell_cap = LIMIT + position

    # Phase 1 — sweep mispriced levels.  Lag-1 autocorr = -0.5, so
    # any print a couple of ticks away from fair has strong reversion
    # expectancy; allow `OSMIUM_TAKE_SLOP` ticks of tolerance on top of fair.
    for ask_price in sorted(od.sell_orders):
        if ask_price > fair + OSMIUM_TAKE_SLOP or buy_cap <= 0:
            break
        if ask_price >= fair and position >= 0:
            # already long / flat; don't pay premium
            continue
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, ask_price, vol))
            buy_cap -= vol
            position += vol

    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price < fair - OSMIUM_TAKE_SLOP or sell_cap <= 0:
            break
        if bid_price <= fair and position <= 0:
            continue
        vol = min(abs(od.buy_orders[bid_price]), sell_cap)
        if vol > 0:
            orders.append(Order(OSMIUM, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # Phase 2 — passive ladder with inventory skew
    skew = position // OSMIUM_SKEW_DENOM
    fair_i = int(round(fair))

    post_bid_near = fair_i - OSMIUM_EDGE - skew
    post_ask_near = fair_i + OSMIUM_EDGE - skew
    post_bid_far  = fair_i - OSMIUM_EDGE_FAR - skew
    post_ask_far  = fair_i + OSMIUM_EDGE_FAR - skew

    if best_ask is not None and post_bid_near >= best_ask:
        post_bid_near = best_ask - 1
    if best_bid is not None and post_ask_near <= best_bid:
        post_ask_near = best_bid + 1
    if post_bid_near >= post_ask_near:
        post_bid_near = fair_i - 1
        post_ask_near = fair_i + 1

    near_buy = min(OSMIUM_SIZE_NEAR, buy_cap)
    near_sell = min(OSMIUM_SIZE_NEAR, sell_cap)
    if near_buy > 0:
        orders.append(Order(OSMIUM, post_bid_near, near_buy))
        buy_cap -= near_buy
    if near_sell > 0:
        orders.append(Order(OSMIUM, post_ask_near, -near_sell))
        sell_cap -= near_sell

    far_buy = min(OSMIUM_SIZE_FAR, buy_cap)
    far_sell = min(OSMIUM_SIZE_FAR, sell_cap)
    if far_buy > 0 and post_bid_far < post_bid_near:
        orders.append(Order(OSMIUM, post_bid_far, far_buy))
    if far_sell > 0 and post_ask_far > post_ask_near:
        orders.append(Order(OSMIUM, post_ask_far, -far_sell))

    return orders


# ══════════════════════════════════════════════════════════════
#   PEPPER — slope-projected fair, trend-priority long-only MM
# ══════════════════════════════════════════════════════════════

def _trade_pepper(
    od: OrderDepth, position: int,
    best_bid, best_ask, mid: float,
    timestamp: int, memory: dict,
) -> List[Order]:
    # Microprice as robust price input (resists bid/ask bounce)
    if best_bid is not None and best_ask is not None:
        bv = abs(od.buy_orders[best_bid])
        av = abs(od.sell_orders[best_ask])
        price_in = (best_bid * av + best_ask * bv) / (bv + av) if (bv + av) else mid
    else:
        price_in = mid

    # Fast EMA + slow slope EMA
    ema_prev = memory.get("pema")
    ts_prev  = memory.get("pts")
    if ema_prev is None:
        ema = price_in
        slope = PEPPER_PRIOR_SLOPE
    else:
        ema = PEPPER_FAST_ALPHA * price_in + (1 - PEPPER_FAST_ALPHA) * ema_prev
        slope_prev = memory.get("pslope", PEPPER_PRIOR_SLOPE)
        dt = max(1, timestamp - ts_prev) if ts_prev is not None else 100
        inst = max(-0.02, min(0.02, (price_in - ema_prev) / dt))
        slope = PEPPER_SLOPE_ALPHA * inst + (1 - PEPPER_SLOPE_ALPHA) * slope_prev
    memory["pema"]   = ema
    memory["pslope"] = slope
    memory["pts"]    = timestamp

    # Forward-projected fair (removes EMA lag against monotonic drift)
    fair = ema + slope * PEPPER_FORECAST_TS

    orders: List[Order] = []
    buy_cap  = LIMIT - position
    sell_cap = LIMIT + position

    # TAKE: sweep every ask below projected fair (aggressive accumulation)
    for ask_price in sorted(od.sell_orders):
        if ask_price >= fair or buy_cap <= 0:
            break
        vol = min(abs(od.sell_orders[ask_price]), buy_cap)
        if vol > 0:
            orders.append(Order(PEPPER, ask_price, vol))
            buy_cap  -= vol
            position += vol

    # TAKE (rare): if a bid is clearly above fair + margin, fade the spike
    for bid_price in sorted(od.buy_orders, reverse=True):
        if bid_price <= fair + PEPPER_SELL_TAKE_MARGIN or sell_cap <= 0:
            break
        # Only sell out of our long stack — never go short into an uptrend
        if position <= 0:
            break
        max_sell = min(position, sell_cap)
        vol = min(abs(od.buy_orders[bid_price]), max_sell)
        if vol > 0:
            orders.append(Order(PEPPER, bid_price, -vol))
            sell_cap -= vol
            position -= vol

    # MAKE side —— tight bid (always), conditional ask
    fair_i = int(round(fair))

    post_bid = fair_i - PEPPER_BID_EDGE
    post_ask = fair_i + PEPPER_ASK_EDGE
    if best_ask is not None and post_bid >= best_ask:
        post_bid = best_ask - 1
    if best_bid is not None and post_ask <= best_bid:
        post_ask = best_bid + 1
    if post_bid >= post_ask:
        post_bid = fair_i - 1
        post_ask = fair_i + 1

    if buy_cap > 0:
        orders.append(Order(PEPPER, post_bid, buy_cap))

    # Only post a passive ask when we have EXCESS inventory above the
    # long target.  Posting an ask while just at/under target would cost
    # us the trend PnL we're trying to harvest.
    if sell_cap > 0 and position >= PEPPER_ASK_TRIGGER_POS:
        excess = min(position - PEPPER_LONG_TARGET, sell_cap)
        if excess > 0:
            orders.append(Order(PEPPER, post_ask, -excess))

    return orders


# ══════════════════════════════════════════════════════════════
#   Utility
# ══════════════════════════════════════════════════════════════

def _load(raw: str) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}
