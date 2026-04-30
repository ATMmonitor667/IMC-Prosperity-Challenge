# 571500 failure analysis

## What failed

`571500` finished at `480.15` PnL versus `29845.66` for `570532`. The failure was not mainly bad adverse selection; it was inactivity.

Key log symptoms:

| run | profit | graph max drawdown | requested-family open positions | active requested families |
| --- | ---: | ---: | ---: | ---: |
| 570532 | 29845.66 | 4071.52 | 22 | 7 |
| 571500 | 480.15 | 893.46 | 1 | 4 |

Largest missing contributors versus `570532`:

| product | 570532 PnL | 571500 PnL | delta |
| --- | ---: | ---: | ---: |
| PANEL_4X4 | 3158.95 | 161.00 | -2997.95 |
| TRANSLATOR_ECLIPSE_CHARCOAL | 2996.02 | 0.00 | -2996.02 |
| SLEEP_POD_NYLON | 1706.42 | -148.00 | -1854.42 |
| GALAXY_SOUNDS_BLACK_HOLES | 1574.77 | 0.00 | -1574.77 |
| UV_VISOR_RED | 1574.60 | 0.00 | -1574.60 |
| TRANSLATOR_GRAPHITE_MIST | 1183.59 | 0.00 | -1183.59 |
| PANEL_2X4 | 1081.66 | 0.00 | -1081.66 |
| GALAXY_SOUNDS_SOLAR_FLAMES | 1032.77 | -2.00 | -1034.77 |

## Root causes

1. The previous trader over-gated neutral market making. When `alpha` was small and position was zero, `_state_adjust_target` set `repair_only=True`, and `_make_orders` then suppressed both sides. That removed the neutral two-sided quoting that produced much of `570532`'s PnL.
2. Avoid-list behavior was diluted. Products with zero intended scale could still quote because `_make_orders` did not return early when `cap <= 0`.
3. The branch parameters were from the first scan, not the final robustness pass:
   - `PANEL_2X2` used edge `0.25` / skew `0.75`; final research selected edge `1.00` / skew `1.00`.
   - `UV_VISOR_MAGENTA` used rolling regression; final research selected day-3 static leave-one-out regression.
   - pair lookbacks were all `90`; final research supported longer lookbacks, but only as small overlays.

## Fix implemented

`final_trader.py` now:

- restores neutral two-sided core quoting when flat;
- blocks generic trading for avoided products unless explicitly needed by a pair branch;
- returns early from `_make_orders` when a product has no active cap;
- uses researched passive configs for `PANEL_2X2`, `UV_VISOR_MAGENTA`, `SNACKPACK_PISTACHIO`, and `GALAXY_SOUNDS_SOLAR_FLAMES`;
- hardcodes the selected day-3 static fair-value coefficients for UV and Galaxy static branches;
- keeps residual pairs as smaller overlays using practical `180`-bar histories;
- uses compact JSON traderData serialization to keep runtime and state size under control.

Validation:

| check | result |
| --- | --- |
| `python -m py_compile ROUND5/final_trader.py` | passed |
| synthetic day-4 10,000-tick order generation | passed |
| synthetic nonempty ticks | 10,000 / 10,000 |
| synthetic max traderData length | 282,246 chars |
| researched passive branch order products | all active |
