"""IMC **game** round 3 (this folder) using the **Prosperity 4** backtester (`prosperity4bt`).

That is: competition **round 3** data under ``round3/``, *not* the old
``prosperity3bt`` package.

  python run_bt.py trader.py 3-0 3-1 3-2  [prosperity4bt options]
  # same as:  python -m prosperity4bt trader.py 3-0 ...  [--data ...]

The simulator looks for::

  <--data>/round3/prices_round_3_day_*.csv

``--data`` = directory that **contains** ``round3/``. If you omit it, this
script defaults to this ``ROUND3/`` directory when ``./round3/`` exists, or to
``./bt_data/`` when that contains ``round3/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from prosperity4bt.data import LIMITS  # type: ignore[import-not-found]

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
    from prosperity4bt.__main__ import main  # type: ignore[import-not-found]

    if not any(a.startswith("--data") for a in sys.argv):
        root = Path(__file__).resolve().parent
        bt_data = root / "bt_data"
        # Prefer official-style tree: <bt_data>/round3/...
        if bt_data.is_dir() and (bt_data / "round3").is_dir():
            sys.argv += ["--data", str(bt_data)]
        # Else use this repo: <ROUND3>/round3/...  (parent of the round3 folder = --data)
        elif (root / "round3").is_dir():
            sys.argv += ["--data", str(root)]
        # Only bt_data without nested round3 (user layout); still let backtester try
        elif bt_data.is_dir():
            sys.argv += ["--data", str(bt_data)]

    main()
