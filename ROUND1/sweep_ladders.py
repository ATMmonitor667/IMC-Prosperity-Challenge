"""Ladder configuration sweep for OSMIUM and PEPPER."""
from __future__ import annotations

import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).parent


def run() -> tuple[int, int, int]:
    out = subprocess.run(
        [sys.executable, "-m", "prosperity4bt", "traderPrime.py", "1", "--no-progress"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout
    osm = sum(int(x.replace(",", "")) for x in re.findall(r"ASH_COATED_OSMIUM:\s*([-\d,]+)", out))
    pep = sum(int(x.replace(",", "")) for x in re.findall(r"INTARIAN_PEPPER_ROOT:\s*([-\d,]+)", out))
    m = re.search(r"Total profit:\s*([-\d,]+)\s*\n\nRisk metrics", out)
    tot = int(m.group(1).replace(",", "")) if m else osm + pep
    return tot, osm, pep


def patch_ladders(osm_edges, osm_sizes, pep_edges, pep_sizes) -> None:
    p = ROOT / "traderPrime.py"
    text = p.read_text()
    text = re.sub(r"OSMIUM_LADDER_EDGES = \([^)]*\)", f"OSMIUM_LADDER_EDGES = {tuple(osm_edges)}", text)
    text = re.sub(r"OSMIUM_LADDER_SIZES = \([^)]*\)", f"OSMIUM_LADDER_SIZES = {tuple(osm_sizes)}", text)
    text = re.sub(r"PEPPER_LADDER_EDGES = \([^)]*\)", f"PEPPER_LADDER_EDGES = {tuple(pep_edges)}", text)
    text = re.sub(r"PEPPER_LADDER_SIZES = \([^)]*\)", f"PEPPER_LADDER_SIZES = {tuple(pep_sizes)}", text)
    p.write_text(text)


# ---- Stage 1: find best OSMIUM ladder (pepper = single-layer edge=5, size=80) ----
print("=== OSMIUM LADDER SWEEP ===")
osm_configs = [
    ([7], [80]),                # baseline single-layer
    ([7, 10], [50, 30]),
    ([7, 10], [60, 20]),
    ([7, 11], [50, 30]),
    ([7, 11], [60, 20]),
    ([7, 12], [50, 30]),
    ([7, 12], [60, 20]),
    ([5, 8], [50, 30]),
    ([6, 9], [50, 30]),
    ([7, 10, 13], [40, 25, 15]),
    ([7, 11, 15], [45, 20, 15]),
    ([6, 9, 12], [40, 25, 15]),
]
best_osm_cfg = None
best_osm = -1
for edges, sizes in osm_configs:
    patch_ladders(edges, sizes, [5], [80])
    tot, osm, pep = run()
    print(f"OSM edges={edges} sizes={sizes} -> OSM={osm:,}  TOT={tot:,}")
    if osm > best_osm:
        best_osm = osm
        best_osm_cfg = (edges, sizes)

print(f"\nBest OSMIUM ladder: {best_osm_cfg} -> OSM={best_osm:,}\n")

# ---- Stage 2: find best PEPPER ladder with best OSMIUM ----
print("=== PEPPER LADDER SWEEP (OSMIUM frozen) ===")
patch_ladders(best_osm_cfg[0], best_osm_cfg[1], [5], [80])
pep_configs = [
    ([5], [80]),                # baseline
    ([5, 8], [50, 30]),
    ([5, 8], [60, 20]),
    ([5, 9], [50, 30]),
    ([5, 10], [50, 30]),
    ([5, 10], [60, 20]),
    ([4, 8], [50, 30]),
    ([3, 7], [50, 30]),
    ([5, 8, 12], [40, 25, 15]),
    ([5, 9, 13], [45, 20, 15]),
]
best_pep_cfg = None
best_pep = -1
best_tot = -1
for edges, sizes in pep_configs:
    patch_ladders(best_osm_cfg[0], best_osm_cfg[1], edges, sizes)
    tot, osm, pep = run()
    print(f"PEP edges={edges} sizes={sizes} -> PEP={pep:,}  TOT={tot:,}")
    if tot > best_tot:
        best_tot = tot
        best_pep = pep
        best_pep_cfg = (edges, sizes)

print(f"\nBest combined: OSM={best_osm_cfg} PEP={best_pep_cfg} -> TOT={best_tot:,}")

# Apply the best overall
patch_ladders(best_osm_cfg[0], best_osm_cfg[1], best_pep_cfg[0], best_pep_cfg[1])
print("Applied best config to traderPrime.py")
