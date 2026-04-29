"""Round 4 PnL ablation runner.

This script generates temporary trader variants from ``trader.py``, runs the
local Round 4 backtester day-by-day in one command, parses the summary plus the
saved log, and writes a machine-readable results table.

Typical usage from this directory:

    python run_ablation.py --suite baselines
    python run_ablation.py --suite ablations

The local checkout has data for days 4-1, 4-2, and 4-3. If a 4-0 CSV is added
later it will be picked up automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROOT = Path(__file__).resolve().parent
TRADER = ROOT / "trader.py"
RUN_BT = ROOT / "run_bt.py"
VARIANT_DIR = ROOT / ".dist" / "ablation"
RESULT_DIR = ROOT / "ablation_results"
CONSOLIDATED_CSV = ROOT / "round4_optimization_results.csv"

PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

ALL_FLAGS = [
    "ENABLE_VEV_LADDER",
    "ENABLE_DELTA_CONTROL",
    "ENABLE_FLOW",
    "ENABLE_RESIDUAL_Z",
    "ENABLE_MR",
    "ENABLE_CHEAP_VEV_RULE",
    "ENABLE_ENDGAME",
    "ENABLE_ADVERSE_SELECTION_FILTER",
    "ENABLE_DYNAMIC_EDGES",
    "ENABLE_VELVET_HEDGE_PRESSURE",
]


@dataclass(frozen=True)
class Variant:
    name: str
    flags: Dict[str, bool] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    dict_constants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_products: Optional[Set[str]] = None
    simple_s_hat: bool = False
    monotone_vev: bool = True
    freeze_tv: bool = False
    simple_vev_fair: bool = False
    enable_taker: Optional[bool] = None
    enable_maker: Optional[bool] = None
    tv_alpha_default: Optional[float] = None
    tv_alpha_atm: Optional[float] = None
    notes: str = ""


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def available_days() -> List[str]:
    days: List[str] = []
    for d in range(4):
        flat = ROOT / f"prices_round_4_day_{d}.csv"
        nested = ROOT / "round4" / f"prices_round_4_day_{d}.csv"
        if flat.exists() or nested.exists():
            days.append(f"4-{d}")
    return days


def set_flag(src: str, name: str, value: bool) -> str:
    pattern = rf"^{name}\s*=\s*(?:True|False)\s*$"
    repl = f"{name} = {value}"
    out, count = re.subn(pattern, repl, src, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"could not set flag {name}")
    return out


def set_constant(src: str, name: str, value: Any) -> str:
    if isinstance(value, bool):
        value_text = "True" if value else "False"
    elif isinstance(value, str):
        value_text = repr(value)
    else:
        value_text = repr(value)
    pattern = rf"^{name}(?:\s*:\s*[^=]+)?\s*=\s*[^#\r\n]*(?P<comment>\s*#.*)?$"

    def repl(match: re.Match[str]) -> str:
        comment = match.group("comment") or ""
        return f"{name} = {value_text}{comment}"

    out, count = re.subn(pattern, repl, src, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"could not set constant {name}")
    return out


def literal(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def set_dict_entry(src: str, dict_name: str, key: str, value: Any) -> str:
    """Patch one simple literal value inside a top-level dict assignment."""
    m = re.search(rf"^{re.escape(dict_name)}(?:\s*:\s*[^=]+)?\s*=\s*\{{", src, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"could not find dict {dict_name}")

    start = m.end() - 1
    depth = 0
    end = None
    for i in range(start, len(src)):
        ch = src[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError(f"could not parse dict {dict_name}")

    block = src[start:end]
    key_pat = re.escape(str(key))
    pattern = rf"(?P<prefix>[\"']{key_pat}[\"']\s*:\s*)(?P<value>[^,\n}}]+)"
    repl = rf"\g<prefix>{literal(value)}"
    new_block, count = re.subn(pattern, repl, block, count=1)
    if count != 1:
        raise ValueError(f"could not set {dict_name}[{key!r}]")
    return src[:start] + new_block + src[end:]


def inject_extra_switches(src: str) -> str:
    marker = "ENABLE_VELVET_HEDGE_PRESSURE = True\n"
    extra = """ENABLE_TAKER = True
ENABLE_MAKER = True
SIMPLE_S_HAT = False
ACTIVE_PRODUCTS = None
TAKE_EDGE_MULT = 1.0
MAKE_EDGE_MULT = 1.0
CHEAP_SHORT_THRESHOLD_MULT = 1.5
"""
    if "ENABLE_TAKER = " in src:
        return src
    if marker not in src:
        raise ValueError("could not inject switches after feature flags")
    return src.replace(marker, marker + extra, 1)


def patch_runtime_switches(src: str) -> str:
    src = inject_extra_switches(src)

    src = src.replace(
        "    if not parts:\n"
        "        return velvet if velvet is not None else prev_s_hat\n\n"
        "    # Pass 1: weighted mean.\n",
        "    if not parts:\n"
        "        return velvet if velvet is not None else prev_s_hat\n\n"
        "    if SIMPLE_S_HAT:\n"
        "        s = sum(w * v for v, w in parts)\n"
        "        w_tot = sum(w for _, w in parts)\n"
        "        return s / w_tot if w_tot > 0 else (velvet if velvet is not None else prev_s_hat)\n\n"
        "    # Pass 1: weighted mean.\n",
        1,
    )

    src = src.replace(
        "    base_take = float(BASE_TAKE_EDGE.get(product, 2))\n"
        "    base_make = float(BASE_MAKE_EDGE.get(product, 1))\n",
        "    base_take = float(BASE_TAKE_EDGE.get(product, 2)) * TAKE_EDGE_MULT\n"
        "    base_make = float(BASE_MAKE_EDGE.get(product, 1)) * MAKE_EDGE_MULT\n",
        1,
    )

    src = src.replace(
        "        result: Dict[str, List[Order]] = {}\n"
        "        for product, od in state.order_depths.items():\n",
        "        result: Dict[str, List[Order]] = {}\n"
        "        for product, od in state.order_depths.items():\n"
        "            if ACTIVE_PRODUCTS is not None and product not in ACTIVE_PRODUCTS:\n"
        "                result[product] = []\n"
        "                continue\n",
        1,
    )

    src = src.replace(
        "        if not (delta_block_buy or eg_block_buy):\n"
        "            for ap in sorted(od.sell_orders):\n",
        "        if ENABLE_TAKER and not (delta_block_buy or eg_block_buy):\n"
        "            for ap in sorted(od.sell_orders):\n",
        1,
    )
    src = src.replace(
        "        if not (delta_block_sell or eg_block_sell):\n"
        "            for bp in sorted(od.buy_orders, reverse=True):\n",
        "        if ENABLE_TAKER and not (delta_block_sell or eg_block_sell):\n"
        "            for bp in sorted(od.buy_orders, reverse=True):\n",
        1,
    )
    src = src.replace(
        "        # ---- Makers ------------------------------------------------------ #\n"
        "        # Inventory-driven price skew (drag quotes toward reducing inventory).\n",
        "        # ---- Makers ------------------------------------------------------ #\n"
        "        if not ENABLE_MAKER:\n"
        "            return orders\n\n"
        "        # Inventory-driven price skew (drag quotes toward reducing inventory).\n",
        1,
    )
    src = src.replace(
        "                    short_threshold = cheap_fair + max(2.0, 1.5 * take_edge)\n",
        "                    short_threshold = cheap_fair + max(2.0, CHEAP_SHORT_THRESHOLD_MULT * take_edge)\n",
        1,
    )
    return src


def apply_variant(src: str, variant: Variant) -> str:
    src = patch_runtime_switches(src)
    src = f"# Generated by run_ablation.py for variant: {variant.name}\n" + src
    preset_overrides: Dict[str, Any] = {}

    def add_override(section: str, key: str, value: Any) -> None:
        preset_overrides.setdefault(section, {})[key] = value

    def set_preset_scalar(key: str, value: Any) -> None:
        preset_overrides[key] = value

    for flag, value in variant.flags.items():
        src = set_flag(src, flag, value)
        add_override("flags", flag, value)

    if variant.active_products is None:
        src = set_constant(src, "ACTIVE_PRODUCTS", None)
    else:
        products = "{" + ", ".join(repr(p) for p in sorted(variant.active_products)) + "}"
        src = set_constant(src, "ACTIVE_PRODUCTS", products)
        src = src.replace("ACTIVE_PRODUCTS = '" + products + "'", f"ACTIVE_PRODUCTS = {products}")
        preset_overrides["active_products"] = set(variant.active_products)

    src = set_constant(src, "SIMPLE_S_HAT", variant.simple_s_hat)
    if variant.enable_taker is not None:
        src = set_constant(src, "ENABLE_TAKER", variant.enable_taker)
    if variant.enable_maker is not None:
        src = set_constant(src, "ENABLE_MAKER", variant.enable_maker)

    if variant.freeze_tv or variant.simple_vev_fair:
        default_alpha = 0.0
        atm_alpha = 0.0
    else:
        default_alpha = variant.tv_alpha_default
        atm_alpha = variant.tv_alpha_atm
    if default_alpha is not None or atm_alpha is not None:
        d = 0.03 if default_alpha is None else float(default_alpha)
        a = 0.015 if atm_alpha is None else float(atm_alpha)
        tv_block = (
            f"TV_ALPHA: Dict[int, float] = {{K: {d!r} for K in STRIKES}}\n"
            f"TV_ALPHA[5200] = {a!r}\n"
            f"TV_ALPHA[5300] = {a!r}\n"
        )
        src, count = re.subn(
            r"TV_ALPHA: Dict\[int, float\] = \{K: .*? for K in STRIKES\}\n"
            r"TV_ALPHA\[5200\] = .*?\n"
            r"TV_ALPHA\[5300\] = .*?\n",
            tv_block,
            src,
            count=1,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError("could not patch TV_ALPHA block")

    if not variant.monotone_vev or variant.simple_vev_fair:
        src = src.replace(
            "            if prev is not None:\n"
            "                fair = min(fair, prev)\n",
            "            if False and prev is not None:\n"
            "                fair = min(fair, prev)\n",
            1,
        )

    for name, value in variant.constants.items():
        if name == "SPREAD_EDGE_MULT":
            name = "EDGE_SPREAD_MULT"
        if name == "PRESET_OVERRIDES" and isinstance(value, dict):
            preset_overrides.update(value)
            continue
        if name == "S_HAT_PROFILE":
            set_preset_scalar("s_hat_profile", value)
        elif name == "TV_MODE":
            set_preset_scalar("tv_mode", value)
        elif name == "RESID_SIDE_GATE":
            add_override("resid", "side_gate", value)
        elif name == "RESID_SIDE_THRESHOLD":
            add_override("resid", "side_threshold", value)
        elif name == "RESID_OPPOSITE_EDGE_ADD":
            add_override("resid", "opposite_edge_add", value)
        elif name == "RESID_HARD_THRESHOLD":
            add_override("resid", "hard_threshold", value)
        elif name == "ENABLE_VERTICAL_SANITY":
            add_override("vertical", "enabled", value)
        elif name == "VERTICAL_EDGE_ADJ":
            add_override("vertical", "edge_adj", value)
        elif name == "VERTICAL_SIGNAL_THRESHOLD":
            add_override("vertical", "signal_threshold", value)
        elif name == "HYDRO_MR_ENTRY_Z":
            add_override("mr", "hydro_entry_z", value)
        elif name == "HYDRO_MR_PULL":
            add_override("mr", "hydro_pull", value)
        elif name == "VELVET_MR_ENTRY_Z":
            add_override("mr", "velvet_entry_z", value)
        elif name == "VELVET_MR_PULL":
            add_override("mr", "velvet_pull", value)
        src = set_constant(src, name, value)

    for dict_name, updates in variant.dict_constants.items():
        override_section = {
            "SIZE_BY_PRODUCT": "sizes",
            "MAKE_EDGE_MULT_BY_PRODUCT": "maker_edges",
            "QUOTE_STYLE_BY_PRODUCT": "quote_styles",
            "ACTIVE_SIDES": "active_sides",
            "SIDE_EDGE_ADJ": "side_edge_adj",
        }.get(dict_name)
        if override_section:
            preset_overrides.setdefault(override_section, {}).update(updates)
            continue
        for key, value in updates.items():
            src = set_dict_entry(src, dict_name, key, value)

    if preset_overrides:
        src = set_constant(src, "PRESET_OVERRIDES", preset_overrides)

    return src


def parse_number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_stdout(stdout: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, Optional[Path]]:
    day_product: Dict[str, Dict[str, float]] = {}
    day_totals: Dict[str, float] = {}
    current_day: Optional[str] = None
    log_path: Optional[Path] = None

    for raw in stdout.splitlines():
        line = raw.strip()
        if line == "Profit summary:":
            current_day = None
            continue

        day_match = re.search(r"Backtesting .* round 4 day (-?\d+)", line)
        if day_match:
            current_day = f"4-{day_match.group(1)}"
            day_product.setdefault(current_day, {})
            continue

        if current_day:
            prod_match = re.match(r"^([A-Z][A-Z0-9_]+):\s*(-?[\d,]+(?:\.\d+)?)$", line)
            if prod_match and prod_match.group(1) in PRODUCTS:
                day_product[current_day][prod_match.group(1)] = parse_number(prod_match.group(2))
                continue

            total_match = re.match(r"^Total profit:\s*(-?[\d,]+(?:\.\d+)?)$", line)
            if total_match:
                day_totals[current_day] = parse_number(total_match.group(1))
                continue

        log_match = re.search(r"Successfully saved backtest results to\s+(.+?\.log)\s*$", line)
        if log_match:
            raw_path = log_match.group(1).strip().strip('"')
            p = Path(raw_path)
            log_path = p if p.is_absolute() else (ROOT / p)

    total = sum(day_totals.values())
    return day_product, day_totals, total, log_path


def parse_log(log_path: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "trade_count": None,
        "final_positions": {},
        "max_drawdown": None,
        "traderdata_max_size": None,
    }
    if log_path is None or not log_path.exists():
        return out

    text = log_path.read_text(encoding="utf-8", errors="replace")

    trade_idx = text.find("Trade History:")
    trade_count = 0
    positions: Dict[str, int] = {}
    if trade_idx >= 0:
        trade_text = text[trade_idx:]
        for obj in re.finditer(r"\{([^{}]*)\}", trade_text, flags=re.DOTALL):
            body = obj.group(1)
            sym = re.search(r'"symbol"\s*:\s*"([^"]+)"', body)
            qty = re.search(r'"quantity"\s*:\s*(-?\d+)', body)
            buyer = re.search(r'"buyer"\s*:\s*"([^"]*)"', body)
            seller = re.search(r'"seller"\s*:\s*"([^"]*)"', body)
            if not sym or not qty:
                continue
            symbol = sym.group(1)
            q = abs(int(qty.group(1)))
            if buyer and buyer.group(1) == "SUBMISSION":
                trade_count += 1
                positions[symbol] = positions.get(symbol, 0) + q
            if seller and seller.group(1) == "SUBMISSION":
                trade_count += 1
                positions[symbol] = positions.get(symbol, 0) - q
    out["trade_count"] = trade_count
    out["final_positions"] = positions

    header = "day;timestamp;product;"
    header_idx = text.find(header)
    if header_idx >= 0:
        end_idx = text.find("Trade History:", header_idx)
        activity_text = text[header_idx:end_idx if end_idx >= 0 else len(text)]
        per_tick: Dict[Tuple[int, int], float] = {}
        reader = csv.DictReader(StringIO(activity_text), delimiter=";")
        for row in reader:
            try:
                day = int(row["day"])
                ts = int(row["timestamp"])
                pnl = float(row.get("profit_and_loss") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            per_tick[(day, ts)] = per_tick.get((day, ts), 0.0) + pnl

        max_dd = 0.0
        by_day: Dict[int, List[Tuple[int, float]]] = {}
        for (day, ts), pnl in per_tick.items():
            by_day.setdefault(day, []).append((ts, pnl))
        for points in by_day.values():
            peak: Optional[float] = None
            for _, pnl in sorted(points):
                peak = pnl if peak is None else max(peak, pnl)
                max_dd = max(max_dd, peak - pnl)
        out["max_drawdown"] = max_dd

    sizes = [len(m.group(1)) for m in re.finditer(r'"traderData"\s*:\s*"([^"]*)"', text)]
    if sizes:
        out["traderdata_max_size"] = max(sizes)

    return out


def run_variant(variant: Variant, days: Sequence[str]) -> Dict[str, Any]:
    src = TRADER.read_text(encoding="utf-8")
    variant_src = apply_variant(src, variant)
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    variant_file = VARIANT_DIR / f"{safe_name(variant.name)}.py"
    variant_file.write_text(variant_src, encoding="utf-8", newline="\n")

    cmd = [sys.executable, str(RUN_BT), str(variant_file), *days]
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - start
    day_product, day_totals, total, log_path = parse_stdout(proc.stdout)
    log_stats = parse_log(log_path)

    row: Dict[str, Any] = {
        "variant": variant.name,
        "notes": variant.notes,
        "command": " ".join(str(x) for x in cmd),
        "variant_file": str(variant_file),
        "returncode": proc.returncode,
        "elapsed_sec": round(elapsed, 3),
        "days": list(days),
        "day_totals": day_totals,
        "product_pnl_by_day": day_product,
        "total_pnl": total,
        "log_path": str(log_path) if log_path else None,
        **log_stats,
    }
    if proc.returncode != 0:
        row["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-80:])
    return row


def base_off_flags() -> Dict[str, bool]:
    return {
        "ENABLE_RESIDUAL_Z": False,
        "ENABLE_DYNAMIC_EDGES": False,
        "ENABLE_ADVERSE_SELECTION_FILTER": False,
        "ENABLE_VELVET_HEDGE_PRESSURE": False,
        "ENABLE_FLOW": False,
        "ENABLE_MR": False,
        "ENABLE_CHEAP_VEV_RULE": False,
        "ENABLE_ENDGAME": False,
        "ENABLE_DELTA_CONTROL": False,
    }


def intended_flags() -> Dict[str, bool]:
    return {
        "ENABLE_VEV_LADDER": True,
        "ENABLE_DELTA_CONTROL": True,
        "ENABLE_FLOW": True,
        "ENABLE_MR": True,
        "ENABLE_CHEAP_VEV_RULE": True,
        "ENABLE_ENDGAME": True,
        "ENABLE_RESIDUAL_Z": False,
        "ENABLE_DYNAMIC_EDGES": False,
        "ENABLE_ADVERSE_SELECTION_FILTER": False,
        "ENABLE_VELVET_HEDGE_PRESSURE": False,
    }


def product_variant(name: str, products: Iterable[str]) -> Variant:
    flags = intended_flags()
    flags.update({"ENABLE_FLOW": False, "ENABLE_MR": False, "ENABLE_CHEAP_VEV_RULE": False, "ENABLE_ENDGAME": False})
    return Variant(name=name, flags=flags, active_products=set(products), notes="product isolation")


def baselines() -> List[Variant]:
    return [
        Variant("A_current_all_features", notes="current trader.py, all latest flags"),
        Variant("B_simple_base", flags=base_off_flags(), notes="basic fair/taker/maker/inventory only"),
        Variant("C_original_intended_hybrid", flags=intended_flags(), notes="original intended hybrid feature set"),
        Variant(
            "D_vev_only_ladder",
            flags={
                "ENABLE_VEV_LADDER": True,
                "ENABLE_DELTA_CONTROL": True,
                "ENABLE_FLOW": False,
                "ENABLE_RESIDUAL_Z": False,
                "ENABLE_MR": False,
                "ENABLE_CHEAP_VEV_RULE": False,
                "ENABLE_ENDGAME": False,
                "ENABLE_ADVERSE_SELECTION_FILTER": False,
                "ENABLE_DYNAMIC_EDGES": False,
                "ENABLE_VELVET_HEDGE_PRESSURE": False,
            },
            active_products={
                "VELVETFRUIT_EXTRACT",
                "VEV_4000",
                "VEV_4500",
                "VEV_5000",
                "VEV_5100",
                "VEV_5200",
                "VEV_5300",
                "VEV_5400",
                "VEV_5500",
                "VEV_6000",
                "VEV_6500",
            },
            notes="VELVET + VEV ladder, simple execution",
        ),
        product_variant("E_only_HYDROGEL_PACK", ["HYDROGEL_PACK"]),
        product_variant("E_only_VELVETFRUIT_EXTRACT", ["VELVETFRUIT_EXTRACT"]),
        product_variant("E_only_VEV_4000_4500", ["VEV_4000", "VEV_4500"]),
        product_variant("E_only_VEV_5000_5300", ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]),
        product_variant("E_only_VEV_5400_5500", ["VEV_5400", "VEV_5500"]),
        product_variant("E_only_VEV_6000_6500", ["VEV_6000", "VEV_6500"]),
    ]


def ablations() -> List[Variant]:
    return [
        Variant("01_current_all_features", notes="current trader.py"),
        Variant("02_disable_dynamic_edges", flags={"ENABLE_DYNAMIC_EDGES": False}),
        Variant("03_disable_adverse_selection_filter", flags={"ENABLE_ADVERSE_SELECTION_FILTER": False}),
        Variant("04_disable_velvet_hedge_pressure", flags={"ENABLE_VELVET_HEDGE_PRESSURE": False}),
        Variant("05_disable_residual_z", flags={"ENABLE_RESIDUAL_Z": False}),
        Variant("06_disable_flow", flags={"ENABLE_FLOW": False}),
        Variant("07_disable_cheap_voucher_rule", flags={"ENABLE_CHEAP_VEV_RULE": False}),
        Variant("08_disable_endgame", flags={"ENABLE_ENDGAME": False}),
        Variant("09_disable_mr", flags={"ENABLE_MR": False}),
        Variant("10_disable_delta_control", flags={"ENABLE_DELTA_CONTROL": False}),
        Variant("11_simple_s_hat_no_anchor_rejection", simple_s_hat=True),
        Variant("12_disable_monotonic_vev_sweep", monotone_vev=False),
        Variant("13_freeze_tv_to_seeds", freeze_tv=True),
        Variant("14_simple_vev_fair_fixed_tv_no_monotone", simple_vev_fair=True),
        Variant("15_taker_only", enable_maker=False),
        Variant("16_maker_only", enable_taker=False),
    ]


def selected_base() -> Dict[str, bool]:
    """Starting point for targeted parameter sweeps after feature ablation."""
    flags = intended_flags()
    return flags


def best_feature_flags() -> Dict[str, bool]:
    return {
        "ENABLE_VEV_LADDER": True,
        "ENABLE_DELTA_CONTROL": False,
        "ENABLE_FLOW": False,
        "ENABLE_RESIDUAL_Z": True,
        "ENABLE_MR": True,
        "ENABLE_CHEAP_VEV_RULE": True,
        "ENABLE_ENDGAME": True,
        "ENABLE_ADVERSE_SELECTION_FILTER": False,
        "ENABLE_DYNAMIC_EDGES": True,
        "ENABLE_VELVET_HEDGE_PRESSURE": False,
    }


def param_sweeps() -> List[Variant]:
    # These are one-at-a-time probes around the intended hybrid. The grid is not
    # crossed blindly; use the results to select a follow-up neighborhood.
    variants: List[Variant] = []
    flags = selected_base()

    for v in [0.0, 0.03, 0.05, 0.08]:
        variants.append(Variant(f"P_delta_skew_{v}", flags=flags, constants={"DELTA_SKEW_STRENGTH": v}))
    for v in [30, 40, 60, 999]:
        variants.append(Variant(f"P_delta_hard_{v}", flags=flags, constants={"DELTA_HARD": float(v)}))
    for v in [0.75, 1.0, 1.25]:
        variants.append(Variant(f"P_take_edge_mult_{v}", flags=flags, constants={"TAKE_EDGE_MULT": v}))
    for v in [0.5, 0.75, 1.0]:
        variants.append(Variant(f"P_make_edge_mult_{v}", flags=flags, constants={"MAKE_EDGE_MULT": v}))
    for v in [0.0, 0.1, 0.2]:
        variants.append(Variant(f"P_edge_vol_mult_{v}", flags=flags, constants={"EDGE_VOL_MULT": v}))
    for v in [0.0, 0.1, 0.25]:
        variants.append(Variant(f"P_spread_edge_mult_{v}", flags=flags, constants={"SPREAD_EDGE_MULT": v}))
    for d, a in [(0.0, 0.0), (0.01, 0.01), (0.03, 0.015), (0.05, 0.03)]:
        variants.append(Variant(f"P_tv_alpha_{d}_atm_{a}", flags=flags, tv_alpha_default=d, tv_alpha_atm=a))
    for v in [1.0, 1.5, 2.0]:
        variants.append(Variant(f"P_mr_entry_z_{v}", flags=flags, constants={"MR_ENTRY_Z": v}))
    for v in [0.0, 0.05, 0.10]:
        variants.append(Variant(f"P_mr_pull_{v}", flags=flags, constants={"MR_PULL": v}))
    for v in [0.0, 0.02, 0.05]:
        variants.append(Variant(f"P_flow_unit_{v}", flags=flags, constants={"FLOW_UNIT": v}))
    for v in [0.88, 0.92, 0.96]:
        variants.append(Variant(f"P_flow_decay_{v}", flags=flags, constants={"FLOW_DECAY": v}))
    for v in [1.0, 1.5, 2.0]:
        variants.append(Variant(f"P_cheap_short_threshold_mult_{v}", flags=flags, constants={"CHEAP_SHORT_THRESHOLD_MULT": v}))
    return variants


def params_best() -> List[Variant]:
    flags = best_feature_flags()
    variants: List[Variant] = [Variant("PB_base_maker_no_adverse_no_delta", flags=flags, enable_taker=False)]

    for v in [0.75, 0.9, 1.0, 1.1, 1.25]:
        variants.append(Variant(f"PB_take_edge_mult_{v}", flags=flags, enable_taker=False, constants={"TAKE_EDGE_MULT": v}))
    for v in [0.5, 0.65, 0.75, 0.9, 1.0, 1.15]:
        variants.append(Variant(f"PB_make_edge_mult_{v}", flags=flags, enable_taker=False, constants={"MAKE_EDGE_MULT": v}))
    for v in [0.0, 0.1, 0.2, 0.35]:
        variants.append(Variant(f"PB_edge_vol_mult_{v}", flags=flags, enable_taker=False, constants={"EDGE_VOL_MULT": v}))
    for v in [0.0, 0.1, 0.25, 0.4]:
        variants.append(Variant(f"PB_spread_edge_mult_{v}", flags=flags, enable_taker=False, constants={"SPREAD_EDGE_MULT": v}))
    for d, a in [(0.0, 0.0), (0.01, 0.01), (0.03, 0.015), (0.05, 0.03)]:
        variants.append(Variant(f"PB_tv_alpha_{d}_atm_{a}", flags=flags, enable_taker=False, tv_alpha_default=d, tv_alpha_atm=a))
    for v in [1.0, 1.25, 1.5, 1.75, 2.0]:
        variants.append(Variant(f"PB_mr_entry_z_{v}", flags=flags, enable_taker=False, constants={"MR_ENTRY_Z": v}))
    for v in [0.0, 0.05, 0.10, 0.15, 0.20]:
        variants.append(Variant(f"PB_mr_pull_{v}", flags=flags, enable_taker=False, constants={"MR_PULL": v}))
    for v in [0.0, 0.02, 0.05, 0.08]:
        variants.append(Variant(f"PB_flow_unit_{v}", flags=flags, enable_taker=False, constants={"FLOW_UNIT": v}))
    for v in [0.88, 0.92, 0.96]:
        variants.append(Variant(f"PB_flow_decay_{v}", flags=flags, enable_taker=False, constants={"FLOW_DECAY": v}))
    for v in [1.0, 1.5, 2.0]:
        variants.append(Variant(f"PB_cheap_short_threshold_mult_{v}", flags=flags, enable_taker=False, constants={"CHEAP_SHORT_THRESHOLD_MULT": v}))
    for skew, hard in [(0.0, 999), (0.03, 60), (0.05, 60), (0.08, 40)]:
        f = dict(flags)
        f["ENABLE_DELTA_CONTROL"] = True
        variants.append(
            Variant(
                f"PB_delta_enabled_skew_{skew}_hard_{hard}",
                flags=f,
                enable_taker=False,
                constants={"DELTA_SKEW_STRENGTH": skew, "DELTA_HARD": float(hard)},
            )
        )
    return variants


def params_best_make075() -> List[Variant]:
    flags = best_feature_flags()
    base = {"MAKE_EDGE_MULT": 0.75}
    variants: List[Variant] = [
        Variant("PB2_base_make075", flags=flags, enable_taker=False, constants=base),
    ]

    for v in [0.65, 0.70, 0.75, 0.80, 0.85]:
        variants.append(Variant(f"PB2_make_edge_mult_{v}", flags=flags, enable_taker=False, constants={"MAKE_EDGE_MULT": v}))
    for v in [0.0, 0.1, 0.2, 0.35]:
        c = dict(base)
        c["EDGE_VOL_MULT"] = v
        variants.append(Variant(f"PB2_edge_vol_mult_{v}", flags=flags, enable_taker=False, constants=c))
    for v in [0.0, 0.1, 0.25, 0.4]:
        c = dict(base)
        c["SPREAD_EDGE_MULT"] = v
        variants.append(Variant(f"PB2_spread_edge_mult_{v}", flags=flags, enable_taker=False, constants=c))
    for d, a in [(0.0, 0.0), (0.01, 0.01), (0.03, 0.015), (0.05, 0.03)]:
        variants.append(Variant(f"PB2_tv_alpha_{d}_atm_{a}", flags=flags, enable_taker=False, constants=base, tv_alpha_default=d, tv_alpha_atm=a))
    for v in [1.0, 1.25, 1.5, 1.75, 2.0]:
        c = dict(base)
        c["MR_ENTRY_Z"] = v
        variants.append(Variant(f"PB2_mr_entry_z_{v}", flags=flags, enable_taker=False, constants=c))
    for v in [0.0, 0.05, 0.10, 0.15, 0.20]:
        c = dict(base)
        c["MR_PULL"] = v
        variants.append(Variant(f"PB2_mr_pull_{v}", flags=flags, enable_taker=False, constants=c))
    for v in [0.0, 0.02, 0.05, 0.08]:
        c = dict(base)
        c["FLOW_UNIT"] = v
        variants.append(Variant(f"PB2_flow_unit_{v}", flags=flags, enable_taker=False, constants=c))
    for v in [0.88, 0.92, 0.96]:
        c = dict(base)
        c["FLOW_DECAY"] = v
        variants.append(Variant(f"PB2_flow_decay_{v}", flags=flags, enable_taker=False, constants=c))
    return variants


def maker_refine() -> List[Variant]:
    flags = best_feature_flags()
    variants: List[Variant] = [
        Variant("M_current_file", flags=flags, notes="uses constants from trader.py"),
    ]

    for v in [0.45, 0.55, 0.65, 0.75, 0.85, 1.0]:
        variants.append(Variant(f"M_make_edge_{v}", flags=flags, constants={"MAKE_EDGE_MULT": v}))

    for enabled in [True, False]:
        variants.append(Variant(f"M_dynamic_edges_{enabled}", flags={**flags, "ENABLE_DYNAMIC_EDGES": enabled}))
    for v in [0.0, 0.05, 0.10, 0.20]:
        variants.append(Variant(f"M_make_edge_vol_{v}", flags=flags, constants={"MAKE_EDGE_VOL_MULT": v}))
    for style in ["improve", "join", "center", "hybrid"]:
        variants.append(Variant(f"M_quote_style_{style}", flags=flags, constants={"QUOTE_STYLE": style}))

    for v in [2, 3, 4, 5]:
        variants.append(Variant(f"M_hydro_size_{v}", flags=flags, constants={"HYDRO_SIZE": v}))
    for v in [2, 3, 4, 5]:
        variants.append(Variant(f"M_velvet_size_{v}", flags=flags, constants={"VELVET_SIZE": v}))
    for v in [1, 2, 3]:
        variants.append(Variant(f"M_deep_vev_size_{v}", flags=flags, constants={"DEEP_VEV_SIZE": v}))
    for v in [1, 2, 3]:
        variants.append(Variant(f"M_atm_vev_size_{v}", flags=flags, constants={"ATM_VEV_SIZE": v}))
    for v in [1, 2]:
        variants.append(Variant(f"M_cheap_vev_size_{v}", flags=flags, constants={"CHEAP_VEV_SIZE": v}))
    return variants


def vev_refine() -> List[Variant]:
    flags = best_feature_flags()
    variants: List[Variant] = []
    for mode in ["fixed", "ema", "hybrid"]:
        variants.append(Variant(f"V_tv_mode_{mode}", flags=flags, constants={"TV_MODE": mode}))
    for enabled in [True, False]:
        variants.append(Variant(f"V_monotone_{enabled}", flags={**flags, "ENABLE_MONOTONIC_VEV_FAIR": enabled}))
    for name, constants in {
        "S_HAT_VELVET_HEAVY": {"S_WEIGHT_VELVET": 6.0, "S_WEIGHT_VEV4000": 1.0, "S_WEIGHT_VEV4500": 1.0, "S_WEIGHT_VEV5000": 0.0},
        "S_HAT_BALANCED": {"S_WEIGHT_VELVET": 4.0, "S_WEIGHT_VEV4000": 2.0, "S_WEIGHT_VEV4500": 2.0, "S_WEIGHT_VEV5000": 1.0},
        "S_HAT_DEEP_ITM": {"S_WEIGHT_VELVET": 3.0, "S_WEIGHT_VEV4000": 3.0, "S_WEIGHT_VEV4500": 3.0, "S_WEIGHT_VEV5000": 0.0},
        "S_HAT_SIMPLE": {"SIMPLE_S_HAT": True},
    }.items():
        variants.append(Variant(f"V_{name}", flags=flags, constants=constants))
    for v in [0.0, 0.05, 0.10, 0.20, 0.30]:
        variants.append(Variant(f"V_resid_tilt_{v}", flags=flags, constants={"RESID_Z_TILT": v}))
    return variants


def structure_refine() -> List[Variant]:
    flags = best_feature_flags()
    return [
        Variant("S_current_candidate", flags=flags, notes="current trader.py constants"),
        Variant("S_no_second_level", flags={**flags, "ENABLE_SECOND_LEVEL_QUOTES": False}),
        Variant("S_second_level_on", flags={**flags, "ENABLE_SECOND_LEVEL_QUOTES": True}),
        Variant("S_inventory_bucket_on", flags={**flags, "ENABLE_INVENTORY_BUCKET_SIZING": True}),
        Variant("S_resid_side_gate_off", flags=flags, constants={"RESID_SIDE_GATE": False}),
        Variant("S_resid_side_gate_050", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_SIDE_THRESHOLD": 0.50}),
        Variant("S_resid_side_gate_075", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_SIDE_THRESHOLD": 0.75}),
        Variant("S_resid_side_gate_100", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_SIDE_THRESHOLD": 1.00}),
        Variant("S_resid_opp_edge_05", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_OPPOSITE_EDGE_ADD": 0.50}),
        Variant("S_resid_opp_edge_10", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_OPPOSITE_EDGE_ADD": 1.00}),
        Variant("S_resid_opp_edge_15", flags=flags, constants={"RESID_SIDE_GATE": True, "RESID_OPPOSITE_EDGE_ADD": 1.50}),
        Variant("S_vertical_off", flags={**flags, "ENABLE_VERTICAL_TILT": False}),
        Variant("S_vertical_sanity_off", flags={**flags, "ENABLE_VERTICAL_SANITY": False}),
        Variant("S_vertical_sanity_on", flags={**flags, "ENABLE_VERTICAL_SANITY": True, "ENABLE_VERTICAL_TILT": False}),
        Variant("S_vertical_005", flags={**flags, "ENABLE_VERTICAL_TILT": True}, constants={"VERTICAL_TILT_STRENGTH": 0.05}),
        Variant("S_vertical_010", flags={**flags, "ENABLE_VERTICAL_TILT": True}, constants={"VERTICAL_TILT_STRENGTH": 0.10}),
        Variant("S_per_product_edge_off", flags={**flags, "ENABLE_PER_PRODUCT_MAKER_EDGE": False}),
        Variant("S_per_product_edge_on", flags={**flags, "ENABLE_PER_PRODUCT_MAKER_EDGE": True}),
        Variant("S_quote_all_hybrid", flags=flags, constants={"QUOTE_STYLE": "hybrid"}),
        Variant("S_quote_all_join", flags=flags, constants={"QUOTE_STYLE": "join"}),
        Variant("S_quote_all_improve", flags=flags, constants={"QUOTE_STYLE": "improve"}),
        Variant("S_s_hat_global", flags=flags, constants={"S_HAT_MODE": "global"}),
        Variant("S_s_hat_strike_local", flags=flags, constants={"S_HAT_MODE": "strike_local"}),
        Variant("S_tv_ema", flags=flags, constants={"TV_MODE": "ema"}),
        Variant("S_tv_fixed", flags=flags, constants={"TV_MODE": "fixed"}),
        Variant("S_tv_hybrid_70_30", flags=flags, constants={"TV_MODE": "hybrid", "TV_HYBRID_SEED_WEIGHT": 0.70}),
        Variant("S_tv_hybrid_50_50", flags=flags, constants={"TV_MODE": "hybrid", "TV_HYBRID_SEED_WEIGHT": 0.50}),
        Variant("S_endgame_off", flags={**flags, "ENABLE_ENDGAME": False}),
        Variant("S_endgame_last_5k", flags=flags, constants={"ENDGAME_MODE": "normal_until_last_5k"}),
        Variant("S_inventory_reducing_taker", flags={**flags, "ENABLE_INVENTORY_REDUCING_TAKER": True}),
    ]


def adaptive_refine() -> List[Variant]:
    flags = best_feature_flags()
    off = {
        "ENABLE_ADAPTIVE_MARKOUT": False,
        "ENABLE_ADAPTIVE_CONFIDENCE": False,
        "ENABLE_ADAPTIVE_STATE_MACHINE": False,
        "ENABLE_QUOTE_SCORING": False,
        "ENABLE_TARGET_INVENTORY": False,
        "ENABLE_STRIP_BALANCE": False,
        "ENABLE_ADAPTIVE_SECOND_LEVEL": False,
    }
    return [
        Variant("AM_baseline_current", flags=flags, notes="current trader.py constants"),
        Variant("AM_static_no_adaptive", flags={**flags, **off}),
        Variant("AM_adaptive_markout_only", flags={**flags, **off, "ENABLE_ADAPTIVE_MARKOUT": True}),
        Variant("AM_adaptive_confidence_only", flags={**flags, **off, "ENABLE_ADAPTIVE_CONFIDENCE": True}),
        Variant("AM_target_inventory_only", flags={**flags, **off, "ENABLE_TARGET_INVENTORY": True}),
        Variant("AM_quote_scoring_only", flags={**flags, **off, "ENABLE_QUOTE_SCORING": True}),
        Variant("AM_state_machine_only", flags={**flags, **off, "ENABLE_ADAPTIVE_MARKOUT": True, "ENABLE_ADAPTIVE_STATE_MACHINE": True}),
        Variant("AM_all_adaptive", flags={
            **flags,
            "ENABLE_ADAPTIVE_MARKOUT": True,
            "ENABLE_ADAPTIVE_CONFIDENCE": True,
            "ENABLE_ADAPTIVE_STATE_MACHINE": True,
            "ENABLE_QUOTE_SCORING": True,
            "ENABLE_TARGET_INVENTORY": True,
            "ENABLE_STRIP_BALANCE": True,
            "ENABLE_ADAPTIVE_SECOND_LEVEL": False,
        }),
    ]


def adaptive_horizon() -> List[Variant]:
    flags = best_feature_flags()
    return [
        Variant(f"AH_markout_horizon_{h}", flags=flags, constants={"MARKOUT_HORIZON": h})
        for h in [20, 50, 100]
    ]


def adaptive_strength() -> List[Variant]:
    flags = best_feature_flags()
    variants: List[Variant] = []
    for v in [1.0, 1.5, 2.0]:
        variants.append(Variant(f"AS_edge_max_{v}", flags=flags, constants={"ADAPT_EDGE_MAX": v}))
    for v in [-0.25, -0.5, -0.75]:
        variants.append(Variant(f"AS_edge_min_{v}", flags=flags, constants={"ADAPT_EDGE_MIN": v}))
    for v in [1.25, 1.5]:
        variants.append(Variant(f"AS_size_max_{v}", flags=flags, constants={"ADAPT_SIZE_MAX": v}))
    for v in [0.0, 0.25, 0.5, 0.75]:
        variants.append(Variant(f"AS_score_threshold_{v}", flags=flags, constants={"SCORE_THRESHOLD": v}))
    return variants


def adaptive_target() -> List[Variant]:
    flags = best_feature_flags()
    variants: List[Variant] = []
    for v in [0.25, 0.5, 0.75]:
        variants.append(Variant(f"AT_target_strength_{v}", flags=flags, constants={"TARGET_INV_STRENGTH": v}))
    for v in [0.3, 0.5, 0.6]:
        variants.append(Variant(f"AT_vev_target_strength_{v}", flags=flags, constants={"TARGET_INV_VEV_STRENGTH": v}))
    return variants


def static_calibration() -> List[Variant]:
    """Small coordinate-search suite around the restored static maker bot."""
    return [
        Variant("SC_current_calibrated", notes="current trader.py constants"),
        Variant("SC_per_product_edge_off", flags={"ENABLE_PER_PRODUCT_MAKER_EDGE": False}),
        Variant(
            "SC_size_5200_5300_2",
            dict_constants={"SIZE_BY_PRODUCT": {"VEV_5200": 2, "VEV_5300": 2}},
        ),
        Variant("SC_resid_opp_edge_10", constants={"RESID_OPPOSITE_EDGE_ADD": 1.0}),
        Variant("SC_vertical_off", flags={"ENABLE_VERTICAL_SANITY": False}),
        Variant("SC_vertical_edge_025", constants={"VERTICAL_EDGE_ADJ": 0.25}),
        Variant("SC_vertical_edge_05", constants={"VERTICAL_EDGE_ADJ": 0.5}),
        Variant("SC_tv_ema", constants={"TV_MODE": "ema"}),
        Variant("SC_tv_fixed", constants={"TV_MODE": "fixed"}),
        Variant("SC_tv_hybrid_70_30", constants={"TV_MODE": "hybrid_70_30"}),
        Variant("SC_inventory_reducing_taker", flags={"ENABLE_INVENTORY_REDUCING_TAKER": True}),
    ]


def preset_suite() -> List[Variant]:
    names = [
        "minrisk_live",
        "robust_live",
        "current_best",
        "core_three_only",
        "hydro_only",
        "velvet_only",
        "vev4000_only",
        "hydro_velvet_vev4000",
        "core_plus_deep_vev",
        "core_plus_atm_vev",
        "full_static_ladder",
        "conservative_core",
        "aggressive_core",
        "no_small_vevs",
        "no_velvet_standalone",
        "vev_ladder_only",
        "aggressive_hydro",
        "aggressive_vev4000",
        "velvet_profit_engine",
    ]
    return [Variant(f"PRESET_{name}", constants={"STRATEGY_PRESET": name}) for name in names]


def _ov(name: str, overrides: Dict[str, Any], preset: str = "current_best") -> Variant:
    return Variant(name, constants={"STRATEGY_PRESET": preset, "PRESET_OVERRIDES": overrides})


def hydro_suite() -> List[Variant]:
    variants = [Variant("HY_current_best", constants={"STRATEGY_PRESET": "current_best"})]
    for size in [6, 7, 8, 10]:
        variants.append(_ov(f"HY_size_{size}", {"sizes": {"HYDROGEL_PACK": size}}))
    for edge in [0.45, 0.50, 0.55, 0.60]:
        variants.append(_ov(f"HY_edge_{edge}", {"maker_edges": {"HYDROGEL_PACK": edge}}))
    for entry in [1.0, 1.15, 1.25, 1.4]:
        variants.append(_ov(f"HY_mr_entry_{entry}", {"mr": {"hydro_entry_z": entry}}))
    for pull in [0.10, 0.12, 0.14, 0.16]:
        variants.append(_ov(f"HY_mr_pull_{pull}", {"mr": {"hydro_pull": pull}}))
    for style in ["hybrid", "improve", "join"]:
        variants.append(_ov(f"HY_quote_{style}", {"quote_styles": {"HYDROGEL_PACK": style}}))
    return variants


def velvet_suite() -> List[Variant]:
    variants = [Variant("VEL_current_best", constants={"STRATEGY_PRESET": "current_best"})]
    for size in [4, 5, 6]:
        variants.append(_ov(f"VEL_size_{size}", {"sizes": {"VELVETFRUIT_EXTRACT": size}}))
    for edge in [0.55, 0.60, 0.65, 0.75]:
        variants.append(_ov(f"VEL_edge_{edge}", {"maker_edges": {"VELVETFRUIT_EXTRACT": edge}}))
    for entry in [1.5, 1.75, 2.0, 999]:
        variants.append(_ov(f"VEL_mr_entry_{entry}", {"mr": {"velvet_entry_z": entry}}))
    for pull in [0.0, 0.03, 0.05, 0.08]:
        variants.append(_ov(f"VEL_mr_pull_{pull}", {"mr": {"velvet_pull": pull}}))
    for style in ["hybrid", "improve", "join"]:
        variants.append(_ov(f"VEL_quote_{style}", {"quote_styles": {"VELVETFRUIT_EXTRACT": style}}))
    return variants


def vev4000_suite() -> List[Variant]:
    variants = [Variant("V4000_current_best", constants={"STRATEGY_PRESET": "current_best"})]
    for size in [3, 4, 5]:
        variants.append(_ov(f"V4000_size_{size}", {"sizes": {"VEV_4000": size}}))
    for edge in [0.40, 0.45, 0.50, 0.55, 0.60]:
        variants.append(_ov(f"V4000_edge_{edge}", {"maker_edges": {"VEV_4000": edge}}))
    for style in ["hybrid", "improve", "join"]:
        variants.append(_ov(f"V4000_quote_{style}", {"quote_styles": {"VEV_4000": style}}))
    for bid in [-0.25, 0.0, 0.25]:
        variants.append(_ov(f"V4000_bid_adj_{bid}", {"side_edge_adj": {"VEV_4000": {"bid": bid}}}))
    for ask in [-0.25, 0.0, 0.25]:
        variants.append(_ov(f"V4000_ask_adj_{ask}", {"side_edge_adj": {"VEV_4000": {"ask": ask}}}))
    return variants


def s_hat_suite() -> List[Variant]:
    return [
        _ov(f"SH_{profile}", {"s_hat_profile": profile})
        for profile in ["balanced", "no_5000", "deep_itm", "velvet_heavy"]
    ]


def tv_suite() -> List[Variant]:
    return [
        _ov(f"TV_{mode}", {"tv_mode": mode})
        for mode in ["hybrid_70_30", "fixed", "ema", "hybrid_50_50"]
    ]


def sides_suite() -> List[Variant]:
    return [
        Variant("SIDE_current_best", constants={"STRATEGY_PRESET": "current_best"}),
        _ov("SIDE_5000_5100_both", {
            "active_sides": {
                "VEV_5000": {"bid": True, "ask": True},
                "VEV_5100": {"bid": True, "ask": True},
            }
        }),
        _ov("SIDE_5000_5100_bid_only", {
            "active_sides": {
                "VEV_5000": {"bid": True, "ask": False},
                "VEV_5100": {"bid": True, "ask": False},
            }
        }),
        _ov("SIDE_5000_5100_ask_only", {
            "active_sides": {
                "VEV_5000": {"bid": False, "ask": True},
                "VEV_5100": {"bid": False, "ask": True},
            }
        }),
        _ov("SIDE_velvet_off", {
            "active_sides": {"VELVETFRUIT_EXTRACT": {"bid": False, "ask": False}}
        }),
    ]


def emergency_suite() -> List[Variant]:
    return [
        Variant("EMERGENCY_off", constants={"STRATEGY_PRESET": "current_best"}),
        _ov("EMERGENCY_on", {
            "flags": {
                "ENABLE_EMERGENCY_REDUCE_TAKER": True,
                "ENABLE_INVENTORY_REDUCING_TAKER": True,
            }
        }),
    ]


def combos() -> List[Variant]:
    return [
        Variant("C1_maker_only_current", enable_maker=True, enable_taker=False),
        Variant("C2_maker_only_no_residual", flags={"ENABLE_RESIDUAL_Z": False}, enable_taker=False),
        Variant("C3_maker_only_no_adverse", flags={"ENABLE_ADVERSE_SELECTION_FILTER": False}, enable_taker=False),
        Variant("C4_maker_only_no_delta", flags={"ENABLE_DELTA_CONTROL": False}, enable_taker=False),
        Variant("C5_maker_only_no_residual_no_adverse", flags={"ENABLE_RESIDUAL_Z": False, "ENABLE_ADVERSE_SELECTION_FILTER": False}, enable_taker=False),
        Variant("C6_maker_only_no_residual_no_delta", flags={"ENABLE_RESIDUAL_Z": False, "ENABLE_DELTA_CONTROL": False}, enable_taker=False),
        Variant("C7_maker_only_no_adverse_no_delta", flags={"ENABLE_ADVERSE_SELECTION_FILTER": False, "ENABLE_DELTA_CONTROL": False}, enable_taker=False),
        Variant("C8_maker_only_no_residual_no_adverse_no_delta", flags={"ENABLE_RESIDUAL_Z": False, "ENABLE_ADVERSE_SELECTION_FILTER": False, "ENABLE_DELTA_CONTROL": False}, enable_taker=False),
        Variant("C9_maker_only_no_cheap_rule", flags={"ENABLE_CHEAP_VEV_RULE": False}, enable_taker=False),
        Variant("C10_maker_only_no_endgame", flags={"ENABLE_ENDGAME": False}, enable_taker=False),
        Variant("C11_maker_only_no_hedge_pressure", flags={"ENABLE_VELVET_HEDGE_PRESSURE": False}, enable_taker=False),
        Variant("C12_maker_only_intended_hybrid", flags=intended_flags(), enable_taker=False),
        Variant("C13_maker_only_intended_no_residual_no_adverse", flags={**intended_flags(), "ENABLE_RESIDUAL_Z": False, "ENABLE_ADVERSE_SELECTION_FILTER": False}, enable_taker=False),
        Variant("C14_maker_only_no_near_atm_vev", active_products={"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_4000", "VEV_4500", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}, enable_taker=False),
        Variant("C15_maker_only_hydro_velvet_deep_vev", active_products={"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_4000", "VEV_4500"}, enable_taker=False),
    ]


def build_suite(name: str) -> List[Variant]:
    if name == "presets":
        return preset_suite()
    if name == "hydro":
        return hydro_suite()
    if name == "velvet":
        return velvet_suite()
    if name == "vev4000":
        return vev4000_suite()
    if name == "s_hat":
        return s_hat_suite()
    if name == "tv":
        return tv_suite()
    if name == "sides":
        return sides_suite()
    if name == "emergency":
        return emergency_suite()
    if name == "baselines":
        return baselines()
    if name == "ablations":
        return ablations()
    if name == "params":
        return param_sweeps()
    if name == "params_best":
        return params_best()
    if name == "params_best_make075":
        return params_best_make075()
    if name == "maker_refine":
        return maker_refine()
    if name == "vev_refine":
        return vev_refine()
    if name == "structure_refine":
        return structure_refine()
    if name == "adaptive_refine":
        return adaptive_refine()
    if name == "adaptive_horizon":
        return adaptive_horizon()
    if name == "adaptive_strength":
        return adaptive_strength()
    if name == "adaptive_target":
        return adaptive_target()
    if name == "static_calibration":
        return static_calibration()
    if name == "combos":
        return combos()
    if name == "all":
        return (
            preset_suite() + hydro_suite() + velvet_suite() + vev4000_suite()
            + s_hat_suite() + tv_suite() + sides_suite() + emergency_suite()
            + baselines() + ablations() + combos() + maker_refine() + vev_refine()
            + structure_refine() + adaptive_refine() + adaptive_horizon()
            + adaptive_strength() + adaptive_target() + static_calibration()
            + param_sweeps()
        )
    raise ValueError(f"unknown suite {name}")


def all_named_variants() -> List[Variant]:
    seen = set()
    out: List[Variant] = []
    for suite in [
        "presets", "hydro", "velvet", "vev4000", "s_hat", "tv", "sides",
        "emergency", "baselines", "ablations", "combos", "maker_refine", "vev_refine",
        "structure_refine", "adaptive_refine", "adaptive_horizon",
        "adaptive_strength", "adaptive_target", "static_calibration",
        "params", "params_best", "params_best_make075",
    ]:
        for variant in build_suite(suite):
            if variant.name not in seen:
                seen.add(variant.name)
                out.append(variant)
    return out


def write_results(rows: List[Dict[str, Any]], suite: str) -> Tuple[Path, Path]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"{suite}_{stamp}.json"
    csv_path = RESULT_DIR / f"{suite}_{stamp}.csv"

    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    day_labels = sorted({d for row in rows for d in row.get("day_totals", {})})
    fields = [
        "variant",
        *day_labels,
        "total_pnl",
        "trade_count",
        "max_drawdown",
        "final_positions",
        "product_pnl_by_day",
        "log_path",
        "variant_file",
        "returncode",
        "elapsed_sec",
        "notes",
    ]
    def write_csv(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = {
                    "variant": row["variant"],
                    "total_pnl": row["total_pnl"],
                    "trade_count": row["trade_count"],
                    "max_drawdown": row["max_drawdown"],
                    "final_positions": json.dumps(row["final_positions"], sort_keys=True),
                    "product_pnl_by_day": json.dumps(row["product_pnl_by_day"], sort_keys=True),
                    "log_path": row["log_path"],
                    "variant_file": row.get("variant_file"),
                    "returncode": row["returncode"],
                    "elapsed_sec": row["elapsed_sec"],
                    "notes": row["notes"],
                }
                for d in day_labels:
                    flat[d] = row.get("day_totals", {}).get(d)
                writer.writerow(flat)

    write_csv(csv_path)
    write_csv(CONSOLIDATED_CSV)
    return json_path, csv_path


def product_totals(row: Dict[str, Any]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for products in row.get("product_pnl_by_day", {}).values():
        for product, pnl in products.items():
            totals[product] = totals.get(product, 0.0) + float(pnl)
    return totals


def print_product_regressions(rows: List[Dict[str, Any]]) -> None:
    if len(rows) < 2:
        return
    baseline = rows[0]
    base = product_totals(baseline)
    print(f"product regressions vs {baseline['variant']}:")
    for row in sorted(rows[1:], key=lambda r: float(r.get("total_pnl") or 0.0), reverse=True):
        current = product_totals(row)
        regressions = []
        for product in PRODUCTS:
            delta = current.get(product, 0.0) - base.get(product, 0.0)
            if delta < 0:
                regressions.append(f"{product}:{delta:.1f}")
        if regressions:
            print(f"{row['variant']}: " + ", ".join(regressions))
        else:
            print(f"{row['variant']}: no product regressions")


def print_table(rows: List[Dict[str, Any]]) -> None:
    day_labels = sorted({d for row in rows for d in row.get("day_totals", {})})
    header = ["variant", *day_labels, "total", "trades", "dd"]
    print(",".join(header))
    for row in sorted(rows, key=lambda r: float(r.get("total_pnl") or 0.0), reverse=True):
        vals = [row["variant"]]
        for d in day_labels:
            v = row.get("day_totals", {}).get(d)
            vals.append("" if v is None else f"{v:.1f}")
        vals.extend([
            f"{float(row.get('total_pnl') or 0.0):.1f}",
            "" if row.get("trade_count") is None else str(row["trade_count"]),
            "" if row.get("max_drawdown") is None else f"{float(row['max_drawdown']):.1f}",
        ])
        print(",".join(vals))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "presets", "hydro", "velvet", "vev4000", "s_hat", "tv",
            "sides", "emergency",
            "baselines", "ablations", "combos", "maker_refine", "vev_refine",
            "structure_refine", "adaptive_refine", "adaptive_horizon",
            "adaptive_strength", "adaptive_target", "static_calibration",
            "params", "params_best", "params_best_make075", "all",
        ],
        default="baselines",
    )
    parser.add_argument("--variant", help="run one exact named variant from any suite")
    parser.add_argument("--only", help="run variants whose name contains this substring")
    parser.add_argument("--days", nargs="*", help="override day list, e.g. 4-1 4-2 4-3")
    args = parser.parse_args()

    days = args.days or available_days()
    if not days:
        raise SystemExit("no Round 4 price CSVs found")
    variants = all_named_variants() if args.variant else build_suite(args.suite)
    if args.variant:
        variants = [v for v in variants if v.name == args.variant]
    if args.only:
        variants = [v for v in variants if args.only.lower() in v.name.lower()]
    if not variants:
        raise SystemExit("no variants selected")

    print(f"days={days}")
    print(f"variants={len(variants)}")

    rows: List[Dict[str, Any]] = []
    for i, variant in enumerate(variants, 1):
        print(f"[{i}/{len(variants)}] {variant.name}", flush=True)
        row = run_variant(variant, days)
        rows.append(row)
        totals = row.get("day_totals", {})
        day_bits = " ".join(f"{d}={totals.get(d)}" for d in days)
        print(f"  total={row['total_pnl']} {day_bits} trades={row.get('trade_count')} dd={row.get('max_drawdown')}", flush=True)
        if row.get("returncode") != 0:
            print(row.get("stdout_tail", ""), flush=True)

    json_path, csv_path = write_results(rows, args.suite)
    successful = [r for r in rows if r.get("returncode") == 0 and r.get("variant_file")]
    best_copy: Optional[Path] = None
    if successful:
        best = max(successful, key=lambda r: float(r.get("total_pnl") or 0.0))
        best_src = Path(str(best["variant_file"]))
        if best_src.exists():
            best_copy = RESULT_DIR / f"best_{args.suite}_{safe_name(best['variant'])}.py"
            shutil.copyfile(best_src, best_copy)
    print_table(rows)
    print_product_regressions(rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {CONSOLIDATED_CSV}")
    if best_copy is not None:
        print(f"preserved best variant {best_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
