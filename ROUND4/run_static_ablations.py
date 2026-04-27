"""Temporary trader variants for static maker ablations. Does not modify trader.py."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = (ROOT / "trader.py").read_text(encoding="utf-8")
OUT = ROOT / "_ablate_trader_tmp.py"


def apply_replacements(text: str, pairs: list[tuple[str, str]]) -> str:
    out = text
    for old, new in pairs:
        if old not in out:
            raise SystemExit(f"patch anchor not found: {old!r}")
        out = out.replace(old, new, 1)
    return out


def run_bt(path: Path) -> tuple[list[int], int]:
    r = subprocess.run(
        [sys.executable, str(ROOT / "run_bt.py"), str(path), "4-1", "4-2", "4-3"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(r.returncode)
    summary = r.stdout.rsplit("Profit summary:", 1)[-1]
    days = [int(m.group(1).replace(",", "")) for m in re.finditer(r"Round 4 day \d: ([\d,]+)", summary)]
    matches = list(re.finditer(r"Total profit: ([\d,]+)", summary))
    if not matches:
        print(r.stdout[-2000:])
        raise SystemExit("no Total profit in summary")
    total = int(matches[-1].group(1).replace(",", ""))
    return days, total


def main() -> None:
    variants: list[tuple[str, list[tuple[str, str]]]] = [
        ("A_current_trader_py", []),  # use file as-is
        ("B_per_product_edge_off", [("ENABLE_PER_PRODUCT_MAKER_EDGE = True", "ENABLE_PER_PRODUCT_MAKER_EDGE = False")]),
        ("C_atm_vev_size2", [
            (
                'SIZE_BY_PRODUCT: Dict[str, int] = {\n    "HYDROGEL_PACK": 5,\n    "VELVETFRUIT_EXTRACT": 4,\n    "VEV_4000": 2,\n    "VEV_4500": 2,\n    "VEV_5000": 2,\n    "VEV_5100": 2,\n    "VEV_5200": 1,\n    "VEV_5300": 1,',
                'SIZE_BY_PRODUCT: Dict[str, int] = {\n    "HYDROGEL_PACK": 5,\n    "VELVETFRUIT_EXTRACT": 4,\n    "VEV_4000": 2,\n    "VEV_4500": 2,\n    "VEV_5000": 2,\n    "VEV_5100": 2,\n    "VEV_5200": 2,\n    "VEV_5300": 2,',
            ),
        ]),
        ("D_resid_edge_add_1", [("RESID_OPPOSITE_EDGE_ADD = 0.5", "RESID_OPPOSITE_EDGE_ADD = 1.0")]),
        ("E_vertical_off", [("ENABLE_VERTICAL_SANITY = True", "ENABLE_VERTICAL_SANITY = False")]),
        ("F_vertical_050", [("VERTICAL_EDGE_ADJ = 0.25", "VERTICAL_EDGE_ADJ = 0.50")]),
        ("G_tv_fixed", [('TV_MODE = "ema"', 'TV_MODE = "fixed"')]),
        ("H_tv_hybrid7030", [('TV_MODE = "ema"', 'TV_MODE = "hybrid_70_30"')]),
        ("I_emergency_taker_on", [("ENABLE_INVENTORY_REDUCING_TAKER = False", "ENABLE_INVENTORY_REDUCING_TAKER = True")]),
        ("K_hybrid7030_plus_vert_050", [
            ('TV_MODE = "ema"', 'TV_MODE = "hybrid_70_30"'),
            ("VERTICAL_EDGE_ADJ = 0.25", "VERTICAL_EDGE_ADJ = 0.50"),
        ]),
    ]

    rows = []
    for name, patches in variants:
        if name == "A_current_trader_py":
            path = ROOT / "trader.py"
        else:
            text = SRC
            text = apply_replacements(text, patches)
            OUT.write_text(text, encoding="utf-8")
            path = OUT
        days, total = run_bt(path)
        rows.append((name, days, total))
        print(f"{name}: days={days} total={total}")

    print("\n=== sorted by total ===")
    for name, days, total in sorted(rows, key=lambda x: -x[2]):
        print(f"{total:6d}  {name}  {days}")


if __name__ == "__main__":
    main()
