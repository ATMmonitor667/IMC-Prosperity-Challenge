"""Prosperity 4 backtester wrapper (`prosperity4bt`).

  cd ROUND4
  python run_bt.py trader.py 4-0 4-1
  # or:  python -m prosperity4bt trader.py 4-0 --data <parent-of-round4>

`prosperity4bt` loads data from::

    <data_root>/round4/prices_round_4_day_<d>.csv
    <data_root>/round4/trades_round_4_day_<d>.csv

So ``--data`` must be the directory that **contains** a ``round4/`` subfolder, not
``round4`` itself. If you omit ``--data``:

* use ``./bt_data`` when ``bt_data/round4/`` exists, else
* use this folder (``ROUND4``) when ``./round4/`` exists, else
* use ``./bt_data`` if it exists (may still warn if it has no ``round4`` — add data).

Add your official bundle under ``ROUND4/round4/`` (or set ``--data`` to the
folder that already contains ``round4/`` with P4 CSVs).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Sync with `trader._LIMITS` and the IMC P4 spec when published.
from prosperity4bt.data import LIMITS  # type: ignore[import-not-found]

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
    from prosperity4bt.__main__ import main  # type: ignore[import-not-found]

    if not any(a.startswith("--data") for a in sys.argv):
        root = Path(__file__).resolve().parent
        bt_data = root / "bt_data"
        if bt_data.is_dir() and (bt_data / "round4").is_dir():
            sys.argv += ["--data", str(bt_data)]
        elif (root / "round4").is_dir():
            sys.argv += ["--data", str(root)]
        elif bt_data.is_dir():
            sys.argv += ["--data", str(bt_data)]

    main()
