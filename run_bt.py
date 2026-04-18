"""Wrapper around prosperity3bt that registers Prosperity 4 Round 2 limits.

Usage:
  python run_bt.py ROUND2/trader.py 2-0 2-1 2--1 [any other prosperity3bt flags]
"""

from __future__ import annotations

import sys

from prosperity3bt.data import LIMITS

LIMITS.update({
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
})


if __name__ == "__main__":
    from prosperity3bt.__main__ import main

    if not any(a.startswith("--data") for a in sys.argv):
        import pathlib

        data_dir = pathlib.Path(__file__).parent / "bt_data"
        sys.argv += ["--data", str(data_dir)]

    main()
