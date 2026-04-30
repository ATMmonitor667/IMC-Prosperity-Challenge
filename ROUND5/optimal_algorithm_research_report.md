# Round 5 optimal algorithm research

This report is a targeted robustness pass after the broad `family_strategy_scan.py` scan. It tests the strongest candidate branches across walk-forward folds, parameter neighborhoods, fill stress, markouts, cap sensitivity, slippage, and current-bot log diagnostics from `LOGS/570532.json`.

## Executive decision matrix
| branch | family | decision | best config | worst fold | mean fold | stress worst | + quarters | robust share | 570532 pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| passive:UV_VISOR_MAGENTA | UV_VISORS | implement | static_loo e=0.5 s=1.0 cap=5 | 4186.00 | 4468.50 | 0.00 | 7 | 0.50 | 441.78 |
| passive:PANEL_2X2 | PANELS | implement | panel_geometry e=1.0 s=1.0 cap=6 | 1789.00 | 3516.50 | 0.00 | 5 | 1.00 | 0.00 |
| passive:SNACKPACK_PISTACHIO | SNACKPACKS | implement | rolling_loo_90 e=0.25 s=0.75 cap=4 | 591.00 | 593.50 | 0.00 | 7 | 0.72 | 392.00 |
| pair:SLEEP_POD_POLYESTER|SLEEP_POD_NYLON | SLEEP_PODS | small_size | lb=180 entry=2.5 exit=0.5 | 3174.00 | 6046.00 | 2424.00 | 6 | 0.23 | nan |
| pair:MICROCHIP_CIRCLE|MICROCHIP_RECTANGLE | MICROCHIPS | small_size | lb=300 entry=2.5 exit=0.5 | 930.00 | 3187.50 | 553.00 | 5 | 0.07 | nan |
| passive:GALAXY_SOUNDS_SOLAR_FLAMES | GALAXY_SOUNDS | small_size | static_loo e=0.25 s=0.0 cap=5 | 865.00 | 955.00 | 0.00 | 3 | 0.53 | 1032.77 |
| pair:TRANSLATOR_ECLIPSE_CHARCOAL|TRANSLATOR_VOID_BLUE | TRANSLATORS | small_size | lb=300 entry=2.5 exit=1.0 | 478.00 | 626.00 | 82.00 | 4 | 0.10 | nan |

Decision rule: implement only if both walk-forward folds are positive, the surrounding parameter neighborhood is not fragile, and intraday quarter stability is acceptable. `small_size` means the branch has positive evidence but should start as a capped overlay rather than a full-size engine.

Passive `stress worst` uses a strict fill-through requirement. The Round 5 public tape mostly fills at the touch, so fill-through values above zero often produce no fills. Treat that column as queue/fill-availability sensitivity, not as a direct rejection test for passive quoting.

## Candidate portfolio approximation
| fold | portfolio pnl | full-size only | branches |
| --- | --- | --- | --- |
| fit_day2_test_day3 | 12987.5 | 10026.0 | 7 |
| fit_day3_test_day4 | 14984.0 | 7131.0 | 7 |

This approximation sums non-overlapping branch simulations. It is not a replacement for a full trader backtest because live branches can still compete for shared product limits and alpha overlays.

## Passive fair-value branches
| target | model | edge | skew | cap | d2->d3 | d3->d4 | worst | fills | + quarters | robust cfgs | robust share | neighbor median | m20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UV_VISOR_MAGENTA | static_loo | 0.50 | 1.00 | 5 | 4186.00 | 4751.00 | 4186.00 | 93.00 | 7 | 20 | 0.50 | 3982.50 | 7.76 |
| PANEL_2X2 | panel_geometry | 1.00 | 1.00 | 6 | 5244.00 | 1789.00 | 1789.00 | 26.00 | 5 | 20 | 1.00 | 1785.50 | -8.98 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 0.25 | 0.00 | 5 | 1045.00 | 865.00 | 865.00 | 56.00 | 3 | 21 | 0.53 | 865.00 | 2.20 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 0.25 | 0.75 | 4 | 596.00 | 591.00 | 591.00 | 155.00 | 7 | 29 | 0.72 | 371.50 | 6.99 |

### Passive fill stress
| target | model | fill-through | worst | mean | fills | + quarters |
| --- | --- | --- | --- | --- | --- | --- |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 0.00 | 865.00 | 955.00 | 56.00 | 3 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 0.25 | 0.00 | 0.00 | 0.00 | 0 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 0.50 | 0.00 | 0.00 | 0.00 | 0 |
| PANEL_2X2 | panel_geometry | 0.00 | 1789.00 | 3516.50 | 26.00 | 5 |
| PANEL_2X2 | panel_geometry | 0.25 | 0.00 | 0.00 | 0.00 | 0 |
| PANEL_2X2 | panel_geometry | 0.50 | 0.00 | 0.00 | 0.00 | 0 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 0.00 | 591.00 | 593.50 | 155.00 | 7 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 0.25 | 0.00 | 0.00 | 0.00 | 0 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 0.50 | 0.00 | 0.00 | 0.00 | 0 |
| UV_VISOR_MAGENTA | static_loo | 0.00 | 4186.00 | 4468.50 | 93.00 | 7 |
| UV_VISOR_MAGENTA | static_loo | 0.25 | 0.00 | 0.00 | 0.00 | 0 |
| UV_VISOR_MAGENTA | static_loo | 0.50 | 0.00 | 0.00 | 0.00 | 0 |

### Passive cap sensitivity
| target | cap | worst | mean | fills | mean dd |
| --- | --- | --- | --- | --- | --- |
| GALAXY_SOUNDS_SOLAR_FLAMES | 2 | -91.00 | 49.50 | 26.00 | 1854.50 |
| GALAXY_SOUNDS_SOLAR_FLAMES | 5 | 865.00 | 955.00 | 56.00 | 3831.50 |
| GALAXY_SOUNDS_SOLAR_FLAMES | 10 | 2425.00 | 5146.00 | 95.00 | 6916.25 |
| PANEL_2X2 | 2 | 363.00 | 1088.00 | 10.00 | 2018.00 |
| PANEL_2X2 | 6 | 1789.00 | 3516.50 | 26.00 | 5871.00 |
| PANEL_2X2 | 10 | 3348.00 | 5922.00 | 34.00 | 9475.50 |
| SNACKPACK_PISTACHIO | 2 | 591.00 | 593.50 | 155.00 | 252.50 |
| SNACKPACK_PISTACHIO | 4 | 591.00 | 593.50 | 155.00 | 252.50 |
| SNACKPACK_PISTACHIO | 10 | 591.00 | 593.50 | 155.00 | 252.50 |
| UV_VISOR_MAGENTA | 2 | 1532.00 | 1856.00 | 52.00 | 1814.50 |
| UV_VISOR_MAGENTA | 5 | 4186.00 | 4468.50 | 93.00 | 3421.75 |
| UV_VISOR_MAGENTA | 10 | 6847.00 | 8840.00 | 136.00 | 5972.00 |

### Passive signed markouts
| target | model | bars | avg signed markout |
| --- | --- | --- | --- |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 1 | 6.718 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 5 | 5.819 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 10 | 3.064 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 20 | 2.203 |
| GALAXY_SOUNDS_SOLAR_FLAMES | static_loo | 50 | -5.263 |
| PANEL_2X2 | panel_geometry | 1 | 4.865 |
| PANEL_2X2 | panel_geometry | 5 | 4.556 |
| PANEL_2X2 | panel_geometry | 10 | -0.278 |
| PANEL_2X2 | panel_geometry | 20 | -8.979 |
| PANEL_2X2 | panel_geometry | 50 | -6.840 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 1 | 7.175 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 5 | 7.269 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 10 | 8.528 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 20 | 6.988 |
| SNACKPACK_PISTACHIO | rolling_loo_90 | 50 | 7.778 |
| UV_VISOR_MAGENTA | static_loo | 1 | 8.304 |
| UV_VISOR_MAGENTA | static_loo | 5 | 9.139 |
| UV_VISOR_MAGENTA | static_loo | 10 | 11.849 |
| UV_VISOR_MAGENTA | static_loo | 20 | 7.755 |
| UV_VISOR_MAGENTA | static_loo | 50 | -0.347 |

## Residual pair branches
| pair | lookback | entry | exit | d2->d3 | d3->d4 | worst | trades | mean dd | + quarters | robust cfgs | robust share | neighbor median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SLEEP_POD_POLYESTER vs SLEEP_POD_NYLON | 180 | 2.50 | 0.50 | 3174.00 | 8918.00 | 3174.00 | 206.00 | 5640.25 | 6 | 7 | 0.23 | -1046.50 |
| MICROCHIP_CIRCLE vs MICROCHIP_RECTANGLE | 300 | 2.50 | 0.50 | 930.00 | 5445.00 | 930.00 | 115.00 | 6716.25 | 5 | 2 | 0.07 | -221.00 |
| TRANSLATOR_ECLIPSE_CHARCOAL vs TRANSLATOR_VOID_BLUE | 300 | 2.50 | 1.00 | 774.00 | 478.00 | 478.00 | 140.00 | 7324.50 | 4 | 3 | 0.10 | -1634.00 |

### Pair slippage stress
| pair | slippage | worst | mean | trades | mean dd | + quarters |
| --- | --- | --- | --- | --- | --- | --- |
| MICROCHIP_CIRCLE vs MICROCHIP_RECTANGLE | 0.00 | 930.00 | 3187.50 | 115.00 | 6716.25 | 5 |
| MICROCHIP_CIRCLE vs MICROCHIP_RECTANGLE | 0.50 | 553.00 | 2810.50 | 115.00 | 6807.25 | 5 |
| MICROCHIP_CIRCLE vs MICROCHIP_RECTANGLE | 1.00 | 176.00 | 2433.50 | 115.00 | 6898.25 | 5 |
| SLEEP_POD_POLYESTER vs SLEEP_POD_NYLON | 0.00 | 3174.00 | 6046.00 | 206.00 | 5640.25 | 6 |
| SLEEP_POD_POLYESTER vs SLEEP_POD_NYLON | 0.50 | 2424.00 | 5239.00 | 206.00 | 5670.75 | 6 |
| SLEEP_POD_POLYESTER vs SLEEP_POD_NYLON | 1.00 | 1674.00 | 4432.00 | 206.00 | 5701.25 | 6 |
| TRANSLATOR_ECLIPSE_CHARCOAL vs TRANSLATOR_VOID_BLUE | 0.00 | 478.00 | 626.00 | 140.00 | 7324.50 | 4 |
| TRANSLATOR_ECLIPSE_CHARCOAL vs TRANSLATOR_VOID_BLUE | 0.50 | 82.00 | 122.00 | 140.00 | 7366.00 | 4 |
| TRANSLATOR_ECLIPSE_CHARCOAL vs TRANSLATOR_VOID_BLUE | 1.00 | -450.00 | -382.00 | 140.00 | 7407.50 | 4 |

## Current bot log diagnostics: 570532
- Status: `FINISHED`.
- Reported submission profit: `29845.66`.
- Activities log days present: `[4]`.
- Total graph max drawdown: `4071.52`.
- Requested-family open positions at the end: `22`.

### 570532 family PnL
| family | log final pnl | negative products | flat products |
| --- | --- | --- | --- |
| TRANSLATORS | 5780.16 | 0 | 1 |
| PANELS | 4240.60 | 0 | 3 |
| GALAXY_SOUNDS | 3074.61 | 0 | 2 |
| SLEEP_PODS | 2866.48 | 0 | 2 |
| MICROCHIPS | 2242.83 | 0 | 2 |
| UV_VISORS | 2042.20 | 0 | 2 |
| SNACKPACKS | 1361.30 | 0 | 1 |

### 570532 weakest requested products
| family | product | log final pnl | log max dd |
| --- | --- | --- | --- |
| GALAXY_SOUNDS | GALAXY_SOUNDS_PLANETARY_RINGS | 0.00 | 0.00 |
| SLEEP_PODS | SLEEP_POD_LAMB_WOOL | 0.00 | 0.00 |
| TRANSLATORS | TRANSLATOR_SPACE_GRAY | 0.00 | 0.00 |
| PANELS | PANEL_2X2 | 0.00 | 0.00 |
| MICROCHIPS | MICROCHIP_RECTANGLE | 0.00 | 0.00 |
| SLEEP_PODS | SLEEP_POD_COTTON | 0.00 | 0.00 |
| PANELS | PANEL_1X2 | 0.00 | 0.00 |
| PANELS | PANEL_1X4 | 0.00 | 0.00 |
| GALAXY_SOUNDS | GALAXY_SOUNDS_DARK_MATTER | 0.00 | 0.00 |
| UV_VISORS | UV_VISOR_ORANGE | 0.00 | 0.00 |
| UV_VISORS | UV_VISOR_YELLOW | 0.00 | 0.00 |
| MICROCHIPS | MICROCHIP_TRIANGLE | 0.00 | 0.00 |

### 570532 strongest requested products
| family | product | log final pnl | log max dd |
| --- | --- | --- | --- |
| PANELS | PANEL_4X4 | 3158.95 | 1112.55 |
| TRANSLATORS | TRANSLATOR_ECLIPSE_CHARCOAL | 2996.02 | 920.66 |
| SLEEP_PODS | SLEEP_POD_NYLON | 1706.42 | 1436.94 |
| GALAXY_SOUNDS | GALAXY_SOUNDS_BLACK_HOLES | 1574.77 | 1285.76 |
| UV_VISORS | UV_VISOR_RED | 1574.60 | 1131.82 |
| MICROCHIPS | MICROCHIP_SQUARE | 1204.34 | 2063.55 |
| TRANSLATORS | TRANSLATOR_GRAPHITE_MIST | 1183.59 | 966.25 |
| PANELS | PANEL_2X4 | 1081.66 | 1273.98 |
| GALAXY_SOUNDS | GALAXY_SOUNDS_SOLAR_FLAMES | 1032.77 | 912.71 |
| TRANSLATORS | TRANSLATOR_ASTRO_BLACK | 932.11 | 529.84 |
| SLEEP_PODS | SLEEP_POD_POLYESTER | 876.66 | 1822.00 |
| MICROCHIPS | MICROCHIP_OVAL | 780.44 | 588.70 |

## Implementation blueprint for the next trader

1. Keep the profitable existing core, but separate hard-avoid products from branch-specific allowlists. `PANEL_2X2` should stay blocked from the generic engine while the dedicated panel-geometry passive branch is allowed to quote it.
2. Add only the `implement` branches at normal size. Add `small_size` branches at half-size or with stricter live markout throttles.
3. For passive branches, track product-specific live fill markouts and realized branch PnL. Use 20-bar markouts for UV/SNACK/GALAXY, but use shorter 5-10 bar markouts for `PANEL_2X2` because the panel branch earns the spread before the longer-horizon reversal in this scan.
4. For pair branches, use the researched lookback/entry/exit values but cap overlay strength. Pair signals should alter reservation prices or target inventory, not force broad aggressive basket trades.
5. Add branch-level PnL and markout accounting in `traderData` so the next log can tell us which branch made each trade.

## Research caveats

- Passive fills use public trade-through checks, which are conservative but not identical to exchange queue priority.
- Pair branches cross the next bar in simulation; live implementation as alpha overlays will usually realize smaller but smoother edge.
- The combined portfolio table is additive because the selected branches do not intentionally share product limits. Full validation still requires `prosperity4bt` once runtime is acceptable.