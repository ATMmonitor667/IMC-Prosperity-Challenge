"""Wrapper for prosperity3bt: register Prosperity 4 position limits (Round 4).

  python run_bt.py trader.py 4-0 4-1  [...]

If `--data` is absent, data come from `bt_data/` under this directory when you add it
(e.g. copy from official bundle). Until then, point `--data` at a folder that contains
`prices_round_*` / `trades_round_*` for the round.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sync keys with `trader._LIMITS` and official IMC limit table when published.
from prosperity3bt.data import LIMITS  # type: ignore[import-not-found]

_LIMITS_BUNDLE = {
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

if __name__ == "__main__":
    LIMITS.update(_LIMITS_BUNDLE)
    from prosperity3bt.__main__ import main  # type: ignore[import-not-found]

    if not any(a.startswith("--data") for a in sys.argv):
        here = Path(__file__).parent / "bt_data"
        if here.is_dir():
            sys.argv += ["--data", str(here)]

    main()
