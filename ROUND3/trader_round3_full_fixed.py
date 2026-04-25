"""
Round 3 — `Trader` (IMC Prosperity-style API).

Products (edit at top for other rounds, e.g. VOLCANIC_ROCK + vouchers):
  UNDERLYING_SYMBOL = VELVETFRUIT_EXTRACT, INDEPENDENT_SPOT = HYDROGEL_PACK, VEV_* ladder.

What this file does
  * Standard Black–Scholes call (r≈0), Gaussian N via erf, delta, implied vol by bisection.
  * Per-tick IV series, quadratic least-squares smile in log-moneyness ``m = log(K/S)``, IV
    residuals vs smile, Welford IV z-scores — persisted in ``traderData`` JSON.
  * **Execution**: by default uses the **median-implied one-factor surface** (proven on local
    BT) for ``option_fair`` / delta passed into the MM engine; set ``USE_STABLE_TRADE_SURFACE``
    False to trade directly on the smile (often worse in this simulator).
  * EWMA + vol + inventory MM, neighbor/residual heuristics, optional cointegration *pairs*,
    light VELVET delta hedge when underlying spread is not huge.

Tuning (see constants below): ``YEAR_FRACTION_BS``, ``USE_STABLE_TRADE_SURFACE``, ``ENABLE_*``,
``TAU`` via ``YEAR_FRACTION_BS``, ``_LIMITS``.

Backtest checklist (local): ``python run_bt.py trader.py 3-0 3-1 3-2``; stress
``--match-trades worse``; verify limits vs PDF.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datamodel import Order, OrderDepth, TradingState

# ---------------------------------------------------------------------------
#  A) Product & strike configuration (change here for other rounds)
# ---------------------------------------------------------------------------

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

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
INDEPENDENT_SPOT = "HYDROGEL_PACK"  # modeled separately (low corr with VEV ladder in data notes)

# Strikes: keys must match order book product symbols
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
VEV_LIST = tuple(OPTION_STRIKES.keys())

# Black–Scholes: r = 0; T in *years* (tune: one full trading day is common for intraday games)
RISK_FREE_RATE = 0.0
# BS time: use 1.0 *year* as the internal unit so total-vol and IV magnitudes match the
# earlier competition surface (σ≈0.02–0.06). A tiny T (e.g. 1/252) breaks inversion in sim.
YEAR_FRACTION_BS = 1.0


def _T_years() -> float:
    return max(1.0e-8, float(YEAR_FRACTION_BS))

# Module switches (N: modular)
ENABLE_IV_SMILE = True
ENABLE_IVR_Z = True

# Improved version:
# 1) keep the stable median-IV surface as a low-noise anchor,
# 2) blend in the quadratic smile only when the smile fit is healthy,
# 3) use residual z-scores for inventory targeting and take decisions.
ENABLE_PAIR_ARB_BS = True
USE_STABLE_TRADE_SURFACE = True
HYBRID_SMILE_WEIGHT = 0.35          # 0 = pure stable surface, 1 = pure smile
SMILE_MSE_CUTOFF = 0.0016           # if smile fit is noisy, fall back toward stable surface
ENABLE_COINT_PAIRS = False          # old hard-coded pairs are easy to overfit; leave off by default
ENABLE_UMR_UNDERLYING = True

# Moderate model influence. The old file set most of these to 0, so the option model mostly
# became an analysis tool instead of a trading signal.
BS_FAIR_LEVEL_BLEND = 0.10
OPTION_MODEL_BLEND = 0.10
OPTION_TARGET_SCALE = 0.32
RESIDUAL_Z_TARGET_SCALE = 0.24
MAX_OPTION_TARGET_FRAC = 0.65
# Strengthen/soften how smile vs IV z-score can disagree (0 = ignore disagreement)
SIGNAL_CONFLICT_DAMP = 0.55

# IV inversion & smile
IV_SOLVER_LO, IV_SOLVER_HI = 0.01, 2.5
IV_SOLVER_IT = 55
IV_MIN, IV_MAX = 0.02, 1.8
IV_TRAIL_MAX = 64
SMEAR_MIN_PTS = 3

# Pair arb (BS): adjacent strikes only, conservative
PAIR_ARB_MIN_EDGE = 6.0
PAIR_ARB_QTY = 1

# Underlying EMA mean reversion (light)
UMR_FAST = 0.12
UMR_SLOW = 0.004
UMR_DEADBAND_FRAC = 0.0
UMR_STRENGTH = 0.12

# Delta / hedge on underlying
DELTA_HEDGE_DEADBAND = 4.0
DELTA_HEDGE_SCALE = 0.42
WIDE_HEDGE_SPREAD = 7  # if velvet spread this wide, down-weight hedge nudge (ticks)

# ---------------------------------------------------------------------------
#  Legacy MM + coint (unchanged idea; compact)
# ---------------------------------------------------------------------------

HORIZON = 100
ALPHA_FAIR = 2.0 / (60.0 + 1.0)
ALPHA_SLOW_FAIR = 2.0 / (10000.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 20
MICRO_TILT = 0.3
DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 10.0
INV_SKEW_K = 2.0
ANTI_TREND_BARRIER_MULT = 3.0
RESIDUAL_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.05,
    "HYDROGEL_PACK": 0.47,
}
SLOW_TARGET_PRODUCTS = frozenset({UNDERLYING_SYMBOL, INDEPENDENT_SPOT})
SLOW_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.20,
    "HYDROGEL_PACK": 1.50,
}

# Old smile-style multipliers (used only as fallback if IV smile disabled)
STRIKE_VOL_MULT: Dict[int, float] = {
    4000: 1.0,
    4500: 1.0,
    5000: 1.0,
    5100: 1.0,
    5200: 1.0,
    5300: 1.01,
    5400: 0.95,
    5500: 1.03,
    6000: 1.05,
    6500: 1.05,
}
SURFACE_VOL_INIT = 0.032
SURFACE_VOL_ALPHA = 0.20
OPTION_TARGET_SCALES: Dict[str, float] = {}

COINT_TRIGGER_Z = 1.25
COINT_ENTRY_Z = 2.5
COINT_MAX_PAIR_QTY = 2
COINT_MODEL_BLEND = 0.0
COINT_TARGET_SCALE = 0.0
COINT_PAIRS: Tuple[Tuple[str, str, float, float, float], ...] = (
    ("VEV_4000", "VEV_4500", 499.906, 1.0001, 0.409),
    ("VELVETFRUIT_EXTRACT", "VEV_4500", 4501.328, 0.9982, 0.758),
    ("VEV_5000", "VEV_5100", 70.098, 1.1086, 2.663),
    ("VEV_5100", "VEV_5200", 42.692, 1.2990, 2.188),
    ("VEV_5200", "VEV_5300", 24.333, 1.5230, 1.850),
    ("VEV_5400", "VEV_5500", 3.625, 1.8560, 1.159),
)

_VEV_LADDER: Tuple[str, ...] = VEV_LIST
NEIGHBOR_RESIDUAL_SCALE = 0.10
_VEV_DELTA_W: Dict[str, float] = {
    "VEV_4000": 0.745,
    "VEV_4500": 0.662,
    "VEV_5000": 0.654,
    "VEV_5100": 0.577,
    "VEV_5200": 0.437,
    "VEV_5300": 0.273,
    "VEV_5400": 0.129,
    "VEV_5500": 0.055,
    "VEV_6000": 0.02,
    "VEV_6500": 0.02,
}
_BLOCK_RISK_DIV = 80.0
BLOCK_RISK_SKEW_K = 0.12
USE_WING_THROTTLE = False
WING_VEV: frozenset[str] = frozenset({"VEV_6000", "VEV_6500"})
WING_MAKE_FRAC = 0.55
WING_TAKE_FRAC = 0.6
WING_MAKE_EDGE_MULT = 1.05


# ---------------------------------------------------------------------------
#  Book helpers
# ---------------------------------------------------------------------------


def _load_memory(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        m = json.loads(trader_data)
        return m if isinstance(m, dict) else {}
    except (TypeError, ValueError):
        return {}


def _best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    if not od.buy_orders or not od.sell_orders:
        return None, None
    return max(od.buy_orders), min(od.sell_orders)


def _book_mid(od: OrderDepth) -> Optional[float]:
    b, a = _best_bid_ask(od)
    if b is None or a is None:
        return None
    return 0.5 * (b + a)


def _microprice(od: OrderDepth) -> Optional[float]:
    b, a = _best_bid_ask(od)
    if b is None and a is None:
        return None
    if b is None:
        return float(a)
    if a is None:
        return float(b)
    bs = abs(od.buy_orders[b])
    ax = abs(od.sell_orders[a])
    t = bs + ax
    if t <= 0:
        return (b + a) / 2.0
    return (ax * b + bs * a) / t


def _spread(od: OrderDepth) -> float:
    b, a = _best_bid_ask(od)
    if b is None or a is None:
        return 0.0
    return float(a - b)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
#  B) Black–Scholes (call), CDF, delta, implied vol
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(
    s: float,
    k: float,
    t: float,
    r: float,
    vol: float,
) -> float:
    """Black–Scholes call, continuous div/riskless r."""
    if s <= 0.0 or k <= 0.0:
        return 0.0
    intrinsic = max(0.0, s - k)
    if t <= 1e-12 or vol <= 1e-12:
        return intrinsic
    vsqrt = vol * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / vsqrt
    d2 = d1 - vsqrt
    return math.exp(-r * t) * (s * _norm_cdf(d1) - k * _norm_cdf(d2))


def bs_call_delta(
    s: float,
    k: float,
    t: float,
    r: float,
    vol: float,
) -> float:
    if s <= 0.0 or k <= 0.0 or t <= 1e-12 or vol <= 1e-12:
        return 1.0 if s > k else 0.0
    vsqrt = vol * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / vsqrt
    return math.exp(-r * t) * _norm_cdf(d1)


def implied_vol_bisect(
    s: float,
    k: float,
    t: float,
    r: float,
    price: float,
) -> Optional[float]:
    """Return annualized vol or None if price not in (intrinsic, S)."""
    if s <= 0.0 or k <= 0.0 or t <= 1e-12:
        return None
    intrinsic = max(0.0, s - k) * math.exp(r * t)
    disc_s = s
    if price <= intrinsic - 0.2 or price >= disc_s + 0.2:
        return None
    lo, hi = IV_SOLVER_LO, IV_SOLVER_HI
    f_lo = bs_call_price(s, k, t, r, lo) - price
    f_hi = bs_call_price(s, k, t, r, hi) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(IV_SOLVER_IT):
        mid = 0.5 * (lo + hi)
        fm = bs_call_price(s, k, t, r, mid) - price
        if abs(fm) < 0.01:
            return mid
        if f_lo * fm <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
#  E) Quadratic smile fit: IV = a m^2 + b m + c, m = log(K/S)
# ---------------------------------------------------------------------------


def _solve_3x3(
    a11: float,
    a12: float,
    a13: float,
    a21: float,
    a22: float,
    a23: float,
    a31: float,
    a32: float,
    a33: float,
    b1: float,
    b2: float,
    b3: float,
) -> Optional[Tuple[float, float, float]]:
    """Cramer's rule for 3x3 (columns: a,b,c for a*m^2 + b*m + c)."""
    def det3(
        x11: float,
        x12: float,
        x13: float,
        x21: float,
        x22: float,
        x23: float,
        x31: float,
        x32: float,
        x33: float,
    ) -> float:
        return (
            x11 * (x22 * x33 - x23 * x32)
            - x12 * (x21 * x33 - x23 * x31)
            + x13 * (x21 * x32 - x22 * x31)
        )

    d = det3(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    if abs(d) < 1e-12:
        return None
    d1 = det3(b1, a12, a13, b2, a22, a23, b3, a32, a33)
    d2 = det3(a11, b1, a13, a21, b2, a23, a31, b3, a33)
    d3 = det3(a11, a12, b1, a21, a22, b2, a31, a32, b3)
    return d1 / d, d2 / d, d3 / d


def fit_quadratic_smile(
    m_iv: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float, float, float]]:  # a,b,c, loss
    """Least squares: IV ≈ a*m^2 + b*m + c. Returns (a,b,c, mse) or None."""
    if len(m_iv) < SMEAR_MIN_PTS:
        return None
    s11 = s12 = s13 = s22 = s23 = s33 = 0.0
    t1 = t2 = t3 = 0.0
    for m, iv in m_iv:
        m2, m1 = m * m, m
        s11 += m2 * m2
        s12 += m2 * m1
        s13 += m2
        s22 += m1 * m1
        s23 += m1
        s33 += 1.0
        t1 += m2 * iv
        t2 += m1 * iv
        t3 += iv
    sol = _solve_3x3(s11, s12, s13, s12, s22, s23, s13, s23, s33, t1, t2, t3)
    if sol is None:
        return None
    a, b, c = sol
    # MSE
    se = 0.0
    for m, iv in m_iv:
        p = a * m * m + b * m + c
        d = p - iv
        se += d * d
    n = max(1, len(m_iv))
    return a, b, c, se / n


# ---------------------------------------------------------------------------
#  IV trail + z
# ---------------------------------------------------------------------------


def _iv_trail_update(mem: dict, product: str, iv: float) -> None:
    tr = mem.setdefault("_iv_trails", {})
    lst = tr.get(product) or []
    lst.append(float(iv))
    if len(lst) > IV_TRAIL_MAX:
        lst = lst[-IV_TRAIL_MAX:]
    tr[product] = lst
    w = mem.setdefault("_iv_welford", {})
    s = w.get(product) or [0, 0.0, 0.0]  # n, mean, M2
    n = int(s[0]) + 1
    mean = float(s[1])
    m2 = float(s[2])
    delta = iv - mean
    mean += delta / n
    m2 += delta * (iv - mean)
    w[product] = [n, mean, m2]


def _iv_z(mem: dict, product: str, current_iv: float) -> float:
    w = (mem.get("_iv_welford") or {}).get(product)
    if w is None or int(w[0]) < 3:
        return 0.0
    n, mean, m2 = int(w[0]), float(w[1]), float(w[2])
    var = m2 / max(1, n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-6:
        return 0.0
    return (current_iv - mean) / std


def _residual_z_update(mem: dict, product: str, residual: float) -> float:
    """Online z-score for smile residual. Positive = observed IV above fitted smile."""
    w = mem.setdefault("_res_welford", {})
    s = w.get(product) or [0, 0.0, 0.0]
    n0, mean0, m20 = int(s[0]), float(s[1]), float(s[2])
    z = 0.0
    if n0 >= 4:
        var0 = m20 / max(1, n0 - 1)
        std0 = math.sqrt(var0) if var0 > 0 else 0.0
        if std0 > 1e-6:
            z = (float(residual) - mean0) / std0
    n = n0 + 1
    mean = mean0
    m2 = m20
    delta = float(residual) - mean
    mean += delta / n
    m2 += delta * (float(residual) - mean)
    w[product] = [n, mean, m2]
    return _clamp(z, -4.0, 4.0)



# ---------------------------------------------------------------------------
#  Fallback “median IV” surface (if smile fails)
# ---------------------------------------------------------------------------


def _surface_from_median(
    s: float,
    order_depths: Dict[str, OrderDepth],
    mem: dict,
) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    st = mem.get("_surface") or {}
    implied_bases: List[float] = []
    t = _T_years()
    for prod, k in OPTION_STRIKES.items():
        if 5000 <= k <= 5500:
            od = order_depths.get(prod)
            if od is None:
                continue
            m = _book_mid(od)
            if m is None:
                continue
            iv0 = implied_vol_bisect(s, float(k), t, RISK_FREE_RATE, m)
            if iv0 is None:
                continue
            mult = STRIKE_VOL_MULT.get(k, 1.0)
            base = iv0 / max(0.5, mult)
            if 0.015 < base < 0.7:
                implied_bases.append(base)
    prev = float(st.get("base_vol", SURFACE_VOL_INIT))
    if not implied_bases:
        bvol = prev
    else:
        implied_bases.sort()
        med = implied_bases[len(implied_bases) // 2]
        bvol = SURFACE_VOL_ALPHA * med + (1.0 - SURFACE_VOL_ALPHA) * prev
        bvol = _clamp(bvol, 0.016, 0.6)
    st["base_vol"] = bvol
    fairs, dels = {}, {}
    for pr, k in OPTION_STRIKES.items():
        v = bvol * STRIKE_VOL_MULT.get(k, 1.0)
        fairs[pr] = bs_call_price(s, float(k), t, RISK_FREE_RATE, v)
        dels[pr] = bs_call_delta(s, float(k), t, RISK_FREE_RATE, v)
    return fairs, dels, st


# ---------------------------------------------------------------------------
#  Edge config + online state
# ---------------------------------------------------------------------------


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
    c = EdgeConfig(2.0, 0.5, 2, 1, 0.25, 0.125, 0.0)
    if product == INDEPENDENT_SPOT:
        c.min_take, c.min_make = max(c.min_take, 8), max(c.min_make, 4)
        c.k_make, c.take_frac, c.make_frac, c.vol_floor = 0.42, 0.18, 0.19, 1.0
    elif product == UNDERLYING_SYMBOL:
        c.k_take = 2.3
        c.min_take, c.min_make = max(4, c.min_take), max(2, c.min_make)
        c.k_make, c.take_frac, c.make_frac, c.vol_floor = 0.45, 0.14, 0.1, 0.5
    elif product.startswith("VEV_"):
        if product in {"VEV_5000", "VEV_5100"}:
            c.k_take, c.k_make = 2.6, 0.55
            c.min_take, c.min_make = 3, 2
            c.take_frac, c.make_frac = 0.20, 0.10
            c.vol_floor = 0.5
            return c
        if mid < 2.0:
            c.k_take, c.k_make, c.min_take, c.min_make = 1.2, 0.4, 1, 1
            c.take_frac, c.make_frac, c.vol_floor = 0.2, 0.15, 0.35
        elif mid < 30.0:
            c.k_take, c.k_make = 2.4, 0.5
            c.min_take, c.min_make = 1, 1
            c.vol_floor = 0.5
        elif mid < 2000.0:
            c.min_take, c.min_make = max(2, c.min_take), max(2, c.min_make)
        else:
            c.min_take, c.min_make = max(4, c.min_take), max(3, c.min_make)
            c.k_take, c.k_make, c.take_frac, c.make_frac = 2.0, 0.45, 0.2, 0.1
    return c


def _update_online_state(pstate: dict, mid: float, vol_floor: float) -> dict:
    prev_fair = pstate.get("fair")
    prev_vol = float(pstate.get("vol", 1.0))
    if prev_fair is None:
        pstate["fair"] = mid
    else:
        pstate["fair"] = ALPHA_FAIR * mid + (1.0 - ALPHA_FAIR) * float(prev_fair)
    p_slow = pstate.get("slow_fair")
    if p_slow is None:
        pstate["slow_fair"] = mid
    else:
        pstate["slow_fair"] = ALPHA_SLOW_FAIR * mid + (1.0 - ALPHA_SLOW_FAIR) * float(p_slow)
    p_mid0 = pstate.get("prev_mid")
    pstate["prev_mid"] = mid
    if p_mid0 is None:
        pstate.setdefault("ret_n", 0)
        pstate.setdefault("ret_mean", 0.0)
        pstate.setdefault("ret_M2", 0.0)
        pstate["vol"] = max(prev_vol, vol_floor)
        return pstate
    ret = float(mid) - float(p_mid0)
    n = int(pstate.get("ret_n", 0)) + 1
    mean = float(pstate.get("ret_mean", 0.0))
    m2 = float(pstate.get("ret_M2", 0.0))
    delta = ret - mean
    mean += delta / n
    m2 += delta * (ret - mean)
    pstate["ret_n"] = n
    pstate["ret_mean"] = mean
    pstate["ret_M2"] = m2
    pstate["vol"] = max(ALPHA_VOL * abs(ret - mean) + (1.0 - ALPHA_VOL) * prev_vol, vol_floor)
    return pstate


def _umr_state(mem: dict, mid: float) -> Tuple[float, float, float]:
    g = mem.setdefault("_umr", {"fast": None, "slow": None})
    fe = g.get("fast")
    sl = g.get("slow")
    if fe is None:
        g["fast"] = g["slow"] = mid
        return 0.0, float(mid), float(mid)
    g["fast"] = (1.0 - UMR_FAST) * fe + UMR_FAST * mid
    g["slow"] = (1.0 - UMR_SLOW) * (sl or mid) + UMR_SLOW * mid
    fe2, sl2 = float(g["fast"]), float(g["slow"])
    return (fe2 - sl2) / max(1.0, abs(sl2) * UMR_DEADBAND_FRAC + 1.0), fe2, sl2


def _drift_stats(pstate: dict) -> Tuple[float, float]:
    n = int(pstate.get("ret_n", 0))
    if n < 2:
        return 0.0, 0.0
    mean, m2 = float(pstate.get("ret_mean", 0.0)), float(pstate.get("ret_M2", 0.0))
    var = m2 / (n - 1)
    if var <= 0.0:
        return mean, 0.0
    sm = (var / n) ** 0.5
    return mean, mean / sm if sm > 0 else 0.0


def _vev_neighbor_predicted_mid(product: str, ods: Dict[str, OrderDepth]) -> Optional[float]:
    if product not in _VEV_LADDER:
        return None
    i = _VEV_LADDER.index(product)
    acc: List[float] = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(_VEV_LADDER):
            od2 = ods.get(_VEV_LADDER[j])
            if od2 and _book_mid(od2) is not None:
                acc.append(_book_mid(od2) or 0.0)  # type: ignore
    if not acc:
        return None
    return float(sum(acc) / len(acc))


def _vev_block_risk(pos: Dict[str, int]) -> float:
    s = 0.0
    for p, w in _VEV_DELTA_W.items():
        s += int(pos.get(p, 0)) * w
    return _clamp(s / _BLOCK_RISK_DIV, -1.0, 1.0)


# ---------------------------------------------------------------------------
#  Pricing pipeline: build smile fairs
# ---------------------------------------------------------------------------


def _build_bsmile(
    s: float,
    order_depths: Dict[str, OrderDepth],
    mem: dict,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], dict]:
    """
    Returns smile-based fairs/deltas, observed IVs, and IV residuals.
    Improvement over original:
      * weights the quadratic smile by liquidity / extrinsic-value reliability
      * rejects unstable smile fits instead of forcing a noisy parabola
      * stores fit quality for the execution layer to blend stable vs smile surface
    """
    t = _T_years()
    m_iv_w: List[Tuple[float, float, float]] = []
    obs_iv: Dict[str, float] = {}

    for prod, k in OPTION_STRIKES.items():
        od = order_depths.get(prod)
        if od is None:
            continue
        b, a = _best_bid_ask(od)
        if b is None or a is None or s <= 0:
            continue
        mpx = 0.5 * (b + a)
        spr = max(1.0, float(a - b))
        intrinsic = max(0.0, s - float(k))
        extrinsic = mpx - intrinsic

        # Deep ITM quotes have tiny extrinsic value; a 1 tick quote move can explode IV.
        # Keep them if needed, but down-weight heavily.
        iv0 = implied_vol_bisect(s, float(k), t, RISK_FREE_RATE, mpx)
        if iv0 is None or not (IV_MIN <= iv0 <= IV_MAX):
            continue

        mone = math.log(float(k) / s)
        # Higher weight for tight/liquid and non-tiny extrinsic options.
        extrinsic_w = math.sqrt(max(0.20, extrinsic + 0.50))
        mone_w = 1.0 / (1.0 + 18.0 * mone * mone)
        weight = _clamp((extrinsic_w * mone_w) / spr, 0.05, 3.0)

        m_iv_w.append((mone, float(iv0), weight))
        obs_iv[prod] = float(iv0)
        _iv_trail_update(mem, prod, float(iv0))

    fairs: Dict[str, float] = {}
    dels: Dict[str, float] = {}
    iv_res: Dict[str, float] = {}

    fq: Optional[Tuple[float, float, float, float]] = None
    if ENABLE_IV_SMILE and len(m_iv_w) >= SMEAR_MIN_PTS:
        # Weighted least squares for IV = a*m^2 + b*m + c.
        s11 = s12 = s13 = s22 = s23 = s33 = 0.0
        t1 = t2 = t3 = 0.0
        w_sum = 0.0
        for m, iv, w in m_iv_w:
            x1 = m * m
            x2 = m
            x3 = 1.0
            s11 += w * x1 * x1
            s12 += w * x1 * x2
            s13 += w * x1 * x3
            s22 += w * x2 * x2
            s23 += w * x2 * x3
            s33 += w
            t1 += w * x1 * iv
            t2 += w * x2 * iv
            t3 += w * iv
            w_sum += w
        sol = _solve_3x3(s11, s12, s13, s12, s22, s23, s13, s23, s33, t1, t2, t3)
        if sol is not None:
            aa, bb, cc = sol
            se = 0.0
            for m, iv, w in m_iv_w:
                pred = aa * m * m + bb * m + cc
                se += w * (pred - iv) * (pred - iv)
            mse = se / max(1.0, w_sum)
            # Guard against wild curvature from noisy wings.
            center_iv = _clamp(cc, IV_MIN, IV_MAX)
            curvature_ok = abs(aa) < 25.0
            center_ok = IV_MIN <= center_iv <= IV_MAX
            if mse <= SMILE_MSE_CUTOFF and curvature_ok and center_ok:
                fq = (aa, bb, cc, mse)

    if fq is not None:
        aa, bb, cc, mse = fq
        mem["_smile_abc"] = [aa, bb, cc, len(m_iv_w)]
        mem["_smile_fit_mse"] = float(mse)
        mem["_smile_valid"] = True
        for prod, k in OPTION_STRIKES.items():
            mone = math.log(float(k) / s)
            iv_fit = _clamp(aa * mone * mone + bb * mone + cc, IV_MIN, IV_MAX)
            obs = obs_iv.get(prod)
            if obs is not None:
                iv_res[prod] = float(obs) - float(iv_fit)
            fairs[prod] = bs_call_price(s, float(k), t, RISK_FREE_RATE, iv_fit)
            dels[prod] = bs_call_delta(s, float(k), t, RISK_FREE_RATE, iv_fit)
    else:
        # Fallback to median-implied one-factor vol surface.
        ff, dd, st = _surface_from_median(s, order_depths, mem)
        mem["_surface"] = st
        fairs, dels = ff, dd
        mem["_smile_abc"] = [0, 0, 0, 0]
        mem["_smile_fit_mse"] = 999.0
        mem["_smile_valid"] = False

    return fairs, dels, obs_iv, iv_res, mem


# ---------------------------------------------------------------------------
#  Coint, pairs, plan helpers (legacy)
# ---------------------------------------------------------------------------


def _planned_positions(
    pos: Dict[str, int], result: Dict[str, List[Order]]
) -> Dict[str, int]:
    out = {p: int(q) for p, q in pos.items()}
    for p, olist in result.items():
        out.setdefault(p, int(pos.get(p, 0)))
        for o in olist:
            out[p] = out.get(p, 0) + int(o.quantity)
    return out


def _append_if_room(
    result: Dict[str, List[Order]],
    plan: Dict[str, int],
    product: str,
    price: int,
    qty: int,
) -> bool:
    if qty == 0:
        return False
    lim = int(_LIMITS.get(product, DEFAULT_LIMIT))
    cur = int(plan.get(product, 0))
    if qty > 0:
        qty = min(qty, lim - cur)
    else:
        qty = -min(-qty, lim + cur)
    if qty == 0:
        return False
    result.setdefault(product, []).append(Order(product, int(price), int(qty)))
    plan[product] = cur + qty
    return True


def _add_cointegration_pair_orders(
    result: Dict[str, List[Order]],
    ods: Dict[str, OrderDepth],
    pos: Dict[str, int],
) -> None:
    if not ENABLE_COINT_PAIRS:
        return
    plan = _planned_positions(pos, result)
    for y, x, al, be, sg in COINT_PAIRS:
        yod, xod = ods.get(y), ods.get(x)
        if yod is None or xod is None or be <= 0.0:
            continue
        yb, ya = _best_bid_ask(yod)
        xb, xa = _best_bid_ask(xod)
        if yb is None or ya is None or xb is None or xa is None:
            continue
        ent = max(1.0, COINT_ENTRY_Z * sg)
        re = yb - al - be * xa
        ch = al + be * xb - ya
        if re > ent:
            yr = min(abs(yod.buy_orders[yb]), _LIMITS.get(y, DEFAULT_LIMIT) + plan.get(y, 0))
            xr = min(abs(xod.sell_orders[xa]), _LIMITS.get(x, DEFAULT_LIMIT) - plan.get(x, 0))
            yq = min(COINT_MAX_PAIR_QTY, yr, int(xr / max(1.0, be)))
            if yq > 0:
                xq = min(xr, max(1, int(round(be * yq))))
                if _append_if_room(result, plan, y, yb, -yq):
                    _append_if_room(result, plan, x, xa, xq)
        elif ch > ent:
            yr = min(abs(yod.sell_orders[ya]), _LIMITS.get(y, DEFAULT_LIMIT) - plan.get(y, 0))
            xr = min(abs(xod.buy_orders[xb]), _LIMITS.get(x, DEFAULT_LIMIT) + plan.get(x, 0))
            yq = min(COINT_MAX_PAIR_QTY, yr, int(xr / max(1.0, be)))
            if yq > 0:
                xq = min(xr, max(1, int(round(be * yq))))
                if _append_if_room(result, plan, y, ya, yq):
                    _append_if_room(result, plan, x, xb, -xq)


def _add_bs_pair_arb(
    result: Dict[str, List[Order]],
    ods: Dict[str, OrderDepth],
    pos: Dict[str, int],
    fairs: Dict[str, float],
) -> None:
    """
    Conservative adjacent-strike relative value.
    Improvement over original: both legs are sized atomically and edge must exceed
    the two crossed spreads, so we do not accidentally put on a naked option leg.
    """
    if not ENABLE_PAIR_ARB_BS or not fairs:
        return
    plan = _planned_positions(pos, result)

    def buy_room(p: str) -> int:
        return int(_LIMITS.get(p, DEFAULT_LIMIT)) - int(plan.get(p, 0))

    def sell_room(p: str) -> int:
        return int(_LIMITS.get(p, DEFAULT_LIMIT)) + int(plan.get(p, 0))

    for j in range(len(_VEV_LADDER) - 1):
        p0, p1 = _VEV_LADDER[j], _VEV_LADDER[j + 1]
        f0, f1 = fairs.get(p0), fairs.get(p1)
        o0, o1 = ods.get(p0), ods.get(p1)
        if f0 is None or f1 is None or o0 is None or o1 is None:
            continue
        b0, a0 = _best_bid_ask(o0)
        b1, a1 = _best_bid_ask(o1)
        if b0 is None or a0 is None or b1 is None or a1 is None:
            continue

        m0, m1 = 0.5 * (b0 + a0), 0.5 * (b1 + a1)
        theo = float(f1) - float(f0)
        mkt = float(m1) - float(m0)
        edge = theo - mkt
        crossing_cost = 0.5 * float(a0 - b0) + 0.5 * float(a1 - b1) + 1.0
        if abs(edge) < PAIR_ARB_MIN_EDGE + crossing_cost:
            continue

        if edge > 0.0:
            # p0 cheap relative to p1: buy p0 at ask, sell p1 at bid.
            qty = min(
                PAIR_ARB_QTY,
                abs(o0.sell_orders[a0]),
                abs(o1.buy_orders[b1]),
                buy_room(p0),
                sell_room(p1),
            )
            if qty > 0:
                _append_if_room(result, plan, p0, a0, int(qty))
                _append_if_room(result, plan, p1, b1, -int(qty))
        else:
            # p0 rich relative to p1: sell p0 at bid, buy p1 at ask.
            qty = min(
                PAIR_ARB_QTY,
                abs(o0.buy_orders[b0]),
                abs(o1.sell_orders[a1]),
                sell_room(p0),
                buy_room(p1),
            )
            if qty > 0:
                _append_if_room(result, plan, p0, b0, -int(qty))
                _append_if_room(result, plan, p1, a1, int(qty))


# ---------------------------------------------------------------------------
#  Trader
# ---------------------------------------------------------------------------


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        mem = _load_memory(state.traderData)
        ods = state.order_depths
        u_od = ods.get(UNDERLYING_SYMBOL)
        s = _book_mid(u_od) if u_od else None
        if s is None or s <= 0:
            s = max(50.0, float(mem.get("_last_good_S", 5000.0)))
        mem = dict(mem)
        if u_od:
            sbm = _book_mid(u_od)
            if sbm is not None and sbm > 0:
                mem["_last_good_S"] = float(sbm)

        smile_f, smile_d, obs_iv, iv_res, mem = _build_bsmile(float(s), ods, mem)
        stable_f, stable_d, st = _surface_from_median(float(s), ods, mem)
        mem["_surface"] = st

        # Hybrid execution surface: stable surface is the anchor; valid smile contributes edge.
        op_f: Dict[str, float] = {}
        op_d: Dict[str, float] = {}
        smile_valid = bool(mem.get("_smile_valid", False))
        w_smile = HYBRID_SMILE_WEIGHT if smile_valid else 0.0
        if not USE_STABLE_TRADE_SURFACE:
            w_smile = 1.0 if smile_valid else 0.0
        for pr in OPTION_STRIKES:
            sf = stable_f.get(pr)
            sd = stable_d.get(pr)
            mf = smile_f.get(pr, sf)
            md = smile_d.get(pr, sd)
            if sf is None or sd is None:
                continue
            op_f[pr] = (1.0 - w_smile) * float(sf) + w_smile * float(mf)
            op_d[pr] = (1.0 - w_smile) * float(sd) + w_smile * float(md)

        resid_z_map: Dict[str, float] = {}
        for pr, rr in iv_res.items():
            resid_z_map[pr] = _residual_z_update(mem, pr, float(rr))

        v_hedge = 1.0
        uu = ods.get(UNDERLYING_SYMBOL)
        if uu and _spread(uu) >= WIDE_HEDGE_SPREAD:
            v_hedge = 0.45

        # Aggregate option delta for the light underlying hedge.
        opt_exp = 0.0
        for pr, d in op_d.items():
            opt_exp += float(state.position.get(pr, 0)) * float(d)

        block = _vev_block_risk(state.position)
        res: Dict[str, List[Order]] = {}
        for product, od in ods.items():
            pos = int(state.position.get(product, 0))
            pstate: dict = mem.get(product) or {}
            of = op_f.get(product) if op_f else None
            odv = op_d.get(product) if op_d else None
            nbr = _vev_neighbor_predicted_mid(product, ods)
            oiv = obs_iv.get(product) if product in obs_iv else None
            ziv = _iv_z(mem, product, float(oiv)) if (ENABLE_IVR_Z and oiv is not None) else 0.0
            orders, pstate = self._adaptive(
                product,
                od,
                pos,
                pstate,
                of,
                odv,
                None,
                mem,
                block_risk=block,
                neighbor_pred=nbr,
                option_delta_exposure=opt_exp,
                iv_z=ziv,
                iv_res=float(iv_res.get(product, 0.0)),
                resid_z=float(resid_z_map.get(product, 0.0)),
                hedge_scale=v_hedge,
            )
            res[product] = orders
            mem[product] = pstate

        _add_cointegration_pair_orders(res, ods, state.position)
        _add_bs_pair_arb(res, ods, state.position, op_f)
        return res, 0, json.dumps(mem, separators=(",", ":"))


    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        option_fair: Optional[float],
        option_delta: Optional[float],
        coint_fair: Optional[float],
        mem: dict,
        *,
        block_risk: float = 0.0,
        neighbor_pred: Optional[float] = None,
        option_delta_exposure: float = 0.0,
        iv_z: float = 0.0,
        iv_res: float = 0.0,
        resid_z: float = 0.0,
        hedge_scale: float = 1.0,
    ) -> Tuple[List[Order], dict]:
        _ = option_delta
        out: List[Order] = []
        bb, ba = _best_bid_ask(od)
        if bb is None or ba is None:
            return out, pstate
        mid = (bb + ba) / 2.0
        ec = _edge_config(product, mid)
        pstate = _update_online_state(pstate, float(mid), ec.vol_floor)
        if int(pstate.get("ret_n", 0)) < WARMUP_TICKS:
            return out, pstate

        fair = float(pstate["fair"])
        vol = float(pstate["vol"])
        dpt, tstat = _drift_stats(pstate)
        micro = _microprice(od) or mid
        if abs(tstat) >= DRIFT_T_THRESHOLD:
            ed = dpt * HORIZON
            target_frac = _clamp(ed / DRIFT_TARGET_SCALE, -1.0, 1.0)
        else:
            ed, target_frac = 0.0, 0.0
        bexp = (fair + ed) * (1.0 - MICRO_TILT) + micro * MICRO_TILT
        if option_fair is not None and BS_FAIR_LEVEL_BLEND > 0.0:
            bexp = BS_FAIR_LEVEL_BLEND * option_fair + (1.0 - BS_FAIR_LEVEL_BLEND) * bexp
        if option_fair is not None:
            ex = OPTION_MODEL_BLEND * option_fair + (1.0 - OPTION_MODEL_BLEND) * bexp
        else:
            ex = bexp
        if coint_fair is not None:
            ex = COINT_MODEL_BLEND * coint_fair + (1.0 - COINT_MODEL_BLEND) * ex
        bte = max(float(ec.min_take), ec.k_take * vol)
        rscale = float(RESIDUAL_TARGET_SCALES.get(product, 0.0))
        if product in SLOW_TARGET_PRODUCTS:
            sf = float(pstate.get("slow_fair", fair))
            ss = float(SLOW_TARGET_SCALES.get(product, rscale))
            rtarget = _clamp((sf - mid) / max(1.0, bte), -1.0, 1.0) * ss
        elif coint_fair is not None:
            rtarget = _clamp((coint_fair - mid) / max(1.0, bte), -1.0, 1.0) * COINT_TARGET_SCALE
        elif option_fair is not None:
            ots = float(OPTION_TARGET_SCALES.get(product, OPTION_TARGET_SCALE))
            model_edge = _clamp((option_fair - mid) / max(1.0, bte), -1.0, 1.0)
            rtarget = model_edge * ots
        else:
            rtarget = _clamp((fair - mid) / max(1.0, bte), -1.0, 1.0) * rscale

        # G: IV confirmation layer.
        # Positive residual/z means observed IV is high vs smile => option likely rich => lean short.
        if product.startswith("VEV_") and option_fair is not None and ENABLE_IVR_Z:
            sm_edge = (option_fair - mid) / max(1.0, bte)
            if sm_edge * iv_z < -0.5:
                rtarget *= 1.0 - SIGNAL_CONFLICT_DAMP
            rtarget = _clamp(
                rtarget - RESIDUAL_Z_TARGET_SCALE * _clamp(resid_z / 2.5, -1.0, 1.0),
                -MAX_OPTION_TARGET_FRAC,
                MAX_OPTION_TARGET_FRAC,
            )
        if ENABLE_UMR_UNDERLYING and product == UNDERLYING_SYMBOL:
            umr, _, _ = _umr_state(mem, float(mid))
            rtarget = _clamp(rtarget - UMR_STRENGTH * _clamp(umr / 50.0, -1.0, 1.0), -1.0, 1.0)
        target_frac = _clamp(target_frac + rtarget, -1.0, 1.0)
        if product.startswith("VEV_") and neighbor_pred is not None:
            nrr = _clamp(
                (neighbor_pred - mid) / max(1.0, bte), -1.0, 1.0
            ) * NEIGHBOR_RESIDUAL_SCALE
            target_frac = _clamp(target_frac + nrr, -1.0, 1.0)
        me = max(float(ec.min_make), ec.k_make * vol)
        if USE_WING_THROTTLE and product in WING_VEV:
            me *= WING_MAKE_EDGE_MULT
        if ed > 0.0:
            b_edge = max(float(ec.min_take), bte - max(0.0, ed))
            s_edge = max(float(ec.min_take), bte + max(0.0, ed))
        else:
            b_edge, s_edge = max(float(ec.min_take), bte - ed), max(float(ec.min_take), bte + ed)
        lim = int(_LIMITS.get(product, DEFAULT_LIMIT))
        if product == UNDERLYING_SYMBOL:
            nde = option_delta_exposure + position
            if abs(nde) > DELTA_HEDGE_DEADBAND:
                hs = _clamp(
                    -DELTA_HEDGE_SCALE
                    * hedge_scale
                    * nde
                    / max(1.0, float(lim)),
                    -0.75,
                    0.75,
                )
                target_frac = _clamp(target_frac + hs, -1.0, 1.0)
        tpos = int(round(target_frac * lim))
        c_buy, c_sell = lim - position, lim + position
        tc, mc = max(1, int(lim * ec.take_frac)), max(1, int(lim * ec.make_frac))
        if USE_WING_THROTTLE and product in WING_VEV:
            tc, mc = max(1, int(tc * WING_TAKE_FRAC)), max(1, int(mc * WING_MAKE_FRAC))
        r = tc
        for ap in sorted(od.sell_orders):
            if c_buy <= 0 or r <= 0:
                break
            bt = ap <= ex - b_edge
            iv_ok_buy = (not product.startswith("VEV_")) or resid_z <= 1.25
            mt = option_fair is not None and iv_ok_buy and ap <= option_fair - bte
            ct = coint_fair is not None and ap <= coint_fair - bte
            if not (bt or mt or ct):
                break
            sz = abs(od.sell_orders[ap])
            q = min(sz, c_buy, r)
            if q:
                out.append(Order(product, int(ap), int(q)))
                c_buy, r = c_buy - q, r - q
        r = tc
        for bp in sorted(od.buy_orders, reverse=True):
            if c_sell <= 0 or r <= 0:
                break
            bt2 = bp >= ex + s_edge
            iv_ok_sell = (not product.startswith("VEV_")) or resid_z >= -1.25
            mt2 = option_fair is not None and iv_ok_sell and bp >= option_fair + bte
            ct2 = coint_fair is not None and bp >= coint_fair + bte
            if not (bt2 or mt2 or ct2):
                break
            sz = abs(od.buy_orders[bp])
            q = min(sz, c_sell, r)
            if q:
                out.append(Order(product, int(bp), -int(q)))
                c_sell, r = c_sell - q, r - q
        inv_e = position - tpos
        sk = -INV_SKEW_K * (inv_e / max(1.0, float(lim))) * me
        if product.startswith("VEV_") and BLOCK_RISK_SKEW_K > 0.0:
            sk -= BLOCK_RISK_SKEW_K * block_risk * me
        sk = _clamp(sk, -me, me)
        sb, bb_ = max(0.0, ed) * ANTI_TREND_BARRIER_MULT, max(0.0, -ed) * ANTI_TREND_BARRIER_MULT
        mb, ma = int(round(ex + sk - me - bb_)), int(round(ex + sk + me + sb))
        if bb_ == 0:
            mb = min(mb, bb)
        if sb == 0:
            ma = max(ma, ba)
        if mb >= ba:
            mb = ba - 1
        if ma <= bb:
            ma = bb + 1
        if mb >= ma:
            return out, pstate
        bq, aq = min(mc, c_buy), min(mc, c_sell)
        if bq > 0:
            out.append(Order(product, mb, int(bq)))
        if aq > 0:
            out.append(Order(product, ma, -int(aq)))
        return out, pstate
