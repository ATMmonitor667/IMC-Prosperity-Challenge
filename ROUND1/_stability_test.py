"""Stress-test: verify pepper cap shrinks and unwinds on adverse slope.

Does not hit the real backtester — directly exercises _trade_pepper with
synthetic order depths and memory states reflecting different slope
regimes. Asserts the new defensive behavior on a weakening / negative
slope without breaking the healthy-slope behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
import prosperity4bt, os
sys.path.insert(0, os.path.dirname(prosperity4bt.__file__))

import finalTrader as ft


def make_od(best_bid: int, best_ask: int, depth_each: int = 14):
    od = SimpleNamespace()
    od.buy_orders = {best_bid: depth_each, best_bid - 1: depth_each}
    od.sell_orders = {best_ask: -depth_each, best_ask + 1: -depth_each}
    return od


def mk_mem(ema: float, slope: float, ts: int) -> dict:
    return {"pema": ema, "pslope": slope, "pts": ts}


def summarize(label: str, orders):
    net_buy = sum(o.quantity for o in orders if o.quantity > 0)
    net_sell = sum(-o.quantity for o in orders if o.quantity < 0)
    print(f"{label:40s} buy_orders={net_buy:>4}  sell_orders={net_sell:>4}  n={len(orders)}")


def run_case(label, position, slope, ts):
    od = make_od(best_bid=11_000, best_ask=11_002, depth_each=14)
    mem = mk_mem(ema=11_001.0, slope=slope, ts=ts - 100)
    orders = ft._trade_pepper(
        od, position, 11_000, 11_002, 11_001.0, ts, mem
    )
    summarize(label, orders)
    return orders


print("=== HEALTHY slope (full long OK) ===")
run_case("pos=0  slope=+PRIOR  ts=5000",   0,  ft.PEPPER_PRIOR_SLOPE,        5000)
run_case("pos=70 slope=+PRIOR  ts=5000",  70,  ft.PEPPER_PRIOR_SLOPE,        5000)

print("\n=== WEAKENING slope (cap shrinks) ===")
run_case("pos=70 slope=+PRIOR/4 ts=5000", 70,  ft.PEPPER_PRIOR_SLOPE * 0.25, 5000)
run_case("pos=40 slope=+PRIOR/4 ts=5000", 40,  ft.PEPPER_PRIOR_SLOPE * 0.25, 5000)

print("\n=== ADVERSE slope (forced unwind) ===")
run_case("pos=60 slope=-PRIOR  ts=5000",  60, -ft.PEPPER_PRIOR_SLOPE,        5000)
run_case("pos=0  slope=-PRIOR  ts=5000",   0, -ft.PEPPER_PRIOR_SLOPE,        5000)

print("\n=== WARMUP (no cap yet — trusts prior) ===")
run_case("pos=70 slope=0        ts=100",  70,  0.0,                          100)
