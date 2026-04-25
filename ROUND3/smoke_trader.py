"""Quick import/run smoke test — does not need the backtester."""

from __future__ import annotations

from datamodel import OrderDepth, TradingState, Observation, ConversionObservation, Listing
from trader import Trader, bs_call_price, implied_vol_bisect, _T_years

# BS sanity
S, K, T, r = 5250.0, 5000.0, _T_years(), 0.0
sig = 0.04
c = bs_call_price(S, K, T, r, sig)
iv = implied_vol_bisect(S, K, T, r, c)
assert c > 0 and iv is not None
print("bs_call, implied vol roundtrip OK:", round(c, 4), round(iv, 6))

od = OrderDepth()
od.buy_orders[100] = 10
od.sell_orders[102] = 10
st = TradingState(
    "",
    0,
    {"X": Listing("X", "VEV_5000", "X")},
    {"VEV_5000": od},
    {},
    {},
    {"VEV_5000": 0},
    Observation({}, {}),
)
# Empty underlying — trader uses memory fallback; should not raise
T = Trader()
out, c0, data = T.run(st)
print("empty-book run OK, conversions=", c0, "len data=", len(data))
