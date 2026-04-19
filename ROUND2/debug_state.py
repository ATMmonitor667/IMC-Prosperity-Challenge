"""Inspect TradingState: types, positions, and order books.

Run from repo root:
  python ROUND2/debug_state.py

Or from ROUND2 (so `datamodel` resolves):
  cd ROUND2 && python debug_state.py

This builds a *fake* TradingState (same shapes as the real game) so you can
print and experiment without the exchange. For live backtester output, use
prosperity3bt with a print flag or add logging inside Trader.run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

# Allow `python ROUND2/debug_state.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datamodel import Listing, Observation, OrderDepth, Trade, TradingState

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"


def explain_types() -> None:
    print("=== How to see types in Python ===\n")
    print("1. Runtime:  type(x)     -> e.g. <class 'datamodel.OrderDepth'>")
    print("2. Runtime:  isinstance(x, OrderDepth)")
    print("3. Contents:  pprint(vars(x))  or  x.__dict__  (for simple objects)")
    print("4. Dicts:     list(d.keys()), list(d.values())")
    print("5. IDE:       hover over variables in VS Code/Cursor (Pylance/Pyright)")
    print("6. REPL:      In notebook:  state.order_depths?   (IPython)")
    print()


def make_sample_state() -> TradingState:
    """Minimal state matching what Trader.run receives."""
    od_osm = OrderDepth()
    # bids: price -> quantity (positive)
    od_osm.buy_orders = {9998: 10, 9995: 25}
    # asks: price -> quantity (simulator may use negative for sells — use abs() if needed)
    od_osm.sell_orders = {10002: -15, 10005: -20}

    od_pep = OrderDepth()
    od_pep.buy_orders = {12000: 8}
    od_pep.sell_orders = {12014: -12}

    order_depths = {OSMIUM: od_osm, PEPPER: od_pep}

    listings = {
        OSMIUM: Listing(OSMIUM, OSMIUM, OSMIUM),
        PEPPER: Listing(PEPPER, PEPPER, PEPPER),
    }

    position = {OSMIUM: 5, PEPPER: -12}

    return TradingState(
        traderData=json.dumps({"note": "example memory"}),
        timestamp=1000,
        listings=listings,
        order_depths=order_depths,
        own_trades={OSMIUM: [], PEPPER: []},
        market_trades={OSMIUM: [], PEPPER: []},
        position=position,
        observations=Observation({}, {}),
    )


def print_order_depth(label: str, od: OrderDepth) -> None:
    print(f"\n--- {label} (OrderDepth) ---")
    print(f"type: {type(od)}")
    print("buy_orders:  Dict[price:int, qty:int]  (bid side)")
    pprint(dict(od.buy_orders))
    print("sell_orders: Dict[price:int, qty:int] (ask side; qty may be negative in sim)")
    pprint(dict(od.sell_orders))
    if od.buy_orders:
        bb = max(od.buy_orders.keys())
        print(f"best_bid = {bb}, size = {od.buy_orders[bb]}")
    if od.sell_orders:
        ba = min(od.sell_orders.keys())
        print(f"best_ask = {ba}, size = {od.sell_orders[ba]} (use abs() if comparing volume)")


def print_trading_state(state: TradingState) -> None:
    print("\n=== TradingState snapshot ===\n")
    print(f"type(state) = {type(state)}")
    print(f"timestamp   = {state.timestamp!r}")
    print(f"traderData  = {state.traderData[:80]}..." if len(state.traderData) > 80 else f"traderData  = {state.traderData!r}")

    print("\n--- position: Dict[product, int] ---")
    pprint(dict(state.position))
    for sym, pos in state.position.items():
        print(f"  {sym!r}: position = {pos:+d}  (long>0, short<0)")

    print("\n--- order_depths: Dict[symbol, OrderDepth] ---")
    print(f"products in book: {list(state.order_depths.keys())}")
    for sym, od in state.order_depths.items():
        print_order_depth(sym, od)

    print("\n--- own_trades / market_trades (often empty until you simulate fills) ---")
    print("own_trades:", {k: len(v) for k, v in state.own_trades.items()})
    print("market_trades:", {k: len(v) for k, v in state.market_trades.items()})

    print("\n--- Full state as JSON (shallow; OrderDepth becomes empty in naive dump) ---")
    print("Tip: use print_trading_state() instead of raw JSON for OrderDepth.")
    d: dict[str, Any] = {
        "timestamp": state.timestamp,
        "position": dict(state.position),
        "traderData": state.traderData,
    }
    print(json.dumps(d, indent=2))


def main() -> None:
    explain_types()
    state = make_sample_state()
    print_trading_state(state)


if __name__ == "__main__":
    main()
