"""Prosperity 4 backtester wrapper (`prosperity4bt`) for **game round 5**.

Usage (from this directory):

  python run_bt.py final_trader.py 5-0 5-1 5-2   [prosperity4bt options]

Data layout expected by `prosperity4bt`:

  <data_root>/round5/prices_round_5_day_<d>.csv
  <data_root>/round5/trades_round_5_day_<d>.csv

So `--data` must be the directory that **contains** the `round5/` folder.

If you omit `--data`, this script will try:
  - `./bt_data` if `bt_data/round5/` exists, else
  - this folder (`ROUND5/`) if `./round5/` exists, else
  - `./bt_data` if it exists (even if it doesn't yet contain round5/)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Optional, Set

from prosperity4bt.data import LIMITS  # type: ignore[import-not-found]


# Round 5 hard-limits are 10 per product in the official spec.
# If your installed `prosperity4bt` already has correct limits, you can leave this empty.
_LIMITS_R5: dict[str, int] = {}


def _arg_value(argv: list[str], name: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def _default_data_root(root: Path, argv: list[str]) -> Optional[Path]:
    data = _arg_value(argv, "--data")
    if data:
        return Path(data)

    bt_data = root / "bt_data"
    if bt_data.is_dir() and (bt_data / "round5").is_dir():
        return bt_data
    if (root / "round5").is_dir():
        return root
    if bt_data.is_dir():
        return bt_data
    return None


def _iter_price_files(round5_dir: Path) -> Iterable[Path]:
    for p in sorted(round5_dir.glob("prices_round_5_day_*.csv")):
        if p.is_file():
            yield p


def _discover_products(round5_dir: Path, max_files: int = 5) -> Set[str]:
    products: Set[str] = set()
    for i, p in enumerate(_iter_price_files(round5_dir)):
        if i >= max_files:
            break
        with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
            # Official IMC price CSVs are `;` delimited.
            reader = csv.DictReader(f, delimiter=";")
            # Official files use "product"; fall back to "symbol" if needed.
            for row in reader:
                prod = (row.get("product") or row.get("symbol") or "").strip()
                if prod:
                    products.add(prod)
    return products


def _install_round5_limits(data_root: Path) -> None:
    round5_dir = data_root / "round5"
    if not round5_dir.is_dir():
        return
    products = _discover_products(round5_dir)
    for p in products:
        LIMITS.setdefault(p, 10)


if __name__ == "__main__":
    from prosperity4bt.__main__ import main  # type: ignore[import-not-found]

    root = Path(__file__).resolve().parent
    data_root = _default_data_root(root, sys.argv)
    if data_root is not None and not any(a.startswith("--data") for a in sys.argv):
        sys.argv += ["--data", str(data_root)]

    # Ensure Round 5 products have limits (default 10) to avoid KeyError inside prosperity4bt.
    if data_root is not None:
        _install_round5_limits(data_root)

    # Optional user overrides (wins over auto-discovery/defaults).
    LIMITS.update(_LIMITS_R5)

    main()

