"""Two-stage parameter sweep: first OSMIUM, then PEPPER.

Stage 1: freeze PEPPER at baseline (alpha=0.7, edge=5, skew=60), sweep OSMIUM.
Stage 2: freeze OSMIUM at its best, sweep PEPPER.
"""
from __future__ import annotations

import itertools
import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).parent


def run_backtest(filename: str = "traderPrime.py") -> tuple[int, int, int]:
    """Return (total, osmium, pepper)."""
    out = subprocess.run(
        [sys.executable, "-m", "prosperity4bt", filename, "1", "--no-progress"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout
    # Sum each product across all days
    osm = sum(int(x.replace(",", "")) for x in re.findall(r"ASH_COATED_OSMIUM:\s*([-\d,]+)", out))
    pep = sum(int(x.replace(",", "")) for x in re.findall(r"INTARIAN_PEPPER_ROOT:\s*([-\d,]+)", out))
    m = re.search(r"Total profit:\s*([-\d,]+)\s*\n\nRisk metrics", out)
    tot = int(m.group(1).replace(",", "")) if m else osm + pep
    return tot, osm, pep


def patch(name: str, value) -> None:
    p = ROOT / "traderPrime.py"
    text = p.read_text()
    text = re.sub(rf"{name} = [\d.]+", f"{name} = {value}", text)
    p.write_text(text)


def patch_params(**kwargs) -> None:
    for k, v in kwargs.items():
        patch(k, v)


def main() -> None:
    print("=== STAGE 1: OSMIUM sweep ===")
    patch_params(PEPPER_ALPHA=0.7, PEPPER_EDGE=5, PEPPER_SKEW_DENOM=60)
    osm_results = []
    for oe, osk in itertools.product([3, 5, 7, 9, 11], [20, 40, 80]):
        patch_params(OSMIUM_EDGE=oe, OSMIUM_SKEW_DENOM=osk)
        tot, osm, pep = run_backtest()
        osm_results.append((osm, oe, osk, tot))
        print(f"OE={oe:2d} OSK={osk:3d} -> OSM={osm:,}  PEP={pep:,}  TOT={tot:,}")
    osm_results.sort(reverse=True)
    best = osm_results[0]
    print(f"\nBest OSMIUM: OE={best[1]} OSK={best[2]}  OSM={best[0]:,}  TOT={best[3]:,}")
    best_oe, best_osk = best[1], best[2]

    print("\n=== STAGE 2: PEPPER sweep (freezing OSMIUM best) ===")
    patch_params(OSMIUM_EDGE=best_oe, OSMIUM_SKEW_DENOM=best_osk)
    pep_results = []
    for pe, psk, pa in itertools.product([2, 3, 5, 7, 9], [15, 30, 60, 120], [0.3, 0.5, 0.7, 0.9]):
        patch_params(PEPPER_EDGE=pe, PEPPER_SKEW_DENOM=psk, PEPPER_ALPHA=pa)
        tot, osm, pep = run_backtest()
        pep_results.append((tot, pe, psk, pa, osm, pep))
        print(f"PE={pe:2d} PSK={psk:3d} PA={pa} -> OSM={osm:,} PEP={pep:,} TOT={tot:,}")
    pep_results.sort(reverse=True)
    print("\n=== TOP 10 combined ===")
    for r in pep_results[:10]:
        tot, pe, psk, pa, osm, pep = r
        print(f"TOT={tot:,}  PE={pe} PSK={psk} PA={pa}  OSM={osm:,} PEP={pep:,}")


if __name__ == "__main__":
    main()
