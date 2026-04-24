"""prosperity3bt entry point for IMC Round 3 — position limits for all 12 book names.

  python run_bt.py trader.py 3-0 3-1 3-2  [prosperity3bt options]

If `--data` is omitted, uses `./bt_data` when that directory exists. Otherwise set
`--data` to the folder that contains the backtester’s packaged CSV layout (e.g. your
`bt_data` copy from the official bundle with `prices_round_3_*.csv`).
"""

from __future__ import annotations

import sys
from pathlib import Path

from prosperity3bt.data import LIMITS  # type: ignore[import-not-found]

# Must match all symbols the trader and simulator use; update from the competition PDF if needed.
_LIMITS_R3 = {
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
    LIMITS.update(_LIMITS_R3)
    from prosperity3bt.__main__ import main  # type: ignore[import-not-found]

    if not any(a.startswith("--data") for a in sys.argv):
        here = Path(__file__).parent / "bt_data"
        if here.is_dir():
            sys.argv += ["--data", str(here)]

    main()
