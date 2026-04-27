"""Approximate Round 4 maker fill diagnostics from a prosperity4bt log.

This does not run a backtest. It reads an existing log and writes
``round4_fill_diagnostics.csv`` with product/side fill and position summaries.

Usage:
    python analyze_round4_fills.py backtests/2026-04-27_03-07-23.log
    python analyze_round4_fills.py

If no log is passed, the newest file under ``ROUND4/backtests`` is used.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "round4_fill_diagnostics.csv"
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


def latest_log() -> Path:
    logs = sorted((ROOT / "backtests").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        raise SystemExit("no log path supplied and no logs found under ROUND4/backtests")
    return logs[0]


def parse_field(body: str, name: str) -> Optional[str]:
    m = re.search(rf'"{re.escape(name)}"\s*:\s*("([^"]*)"|-?\d+(?:\.\d+)?)', body)
    if not m:
        return None
    return m.group(2) if m.group(2) is not None else m.group(1)


def iter_trade_objects(text: str) -> Iterable[Dict[str, Any]]:
    trade_idx = text.find("Trade History:")
    if trade_idx < 0:
        return
    for obj in re.finditer(r"\{([^{}]*)\}", text[trade_idx:], flags=re.DOTALL):
        body = obj.group(1)
        symbol = parse_field(body, "symbol")
        price = parse_field(body, "price")
        quantity = parse_field(body, "quantity")
        buyer = parse_field(body, "buyer")
        seller = parse_field(body, "seller")
        timestamp = parse_field(body, "timestamp")
        if not symbol or not price or not quantity:
            continue
        yield {
            "symbol": symbol,
            "price": int(float(price)),
            "quantity": abs(int(float(quantity))),
            "buyer": buyer or "",
            "seller": seller or "",
            "timestamp": int(float(timestamp or 0)),
        }


def parse_product_pnl(text: str) -> Dict[str, float]:
    header = "day;timestamp;product;"
    header_idx = text.find(header)
    if header_idx < 0:
        return {}
    end_idx = text.find("Trade History:", header_idx)
    activity_text = text[header_idx:end_idx if end_idx >= 0 else len(text)]
    last: Dict[Tuple[int, str], float] = {}
    reader = csv.DictReader(StringIO(activity_text), delimiter=";")
    for row in reader:
        try:
            day = int(row["day"])
            product = row["product"]
            pnl = float(row.get("profit_and_loss") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        last[(day, product)] = pnl
    out: Dict[str, float] = defaultdict(float)
    for (_, product), pnl in last.items():
        out[product] += pnl
    return dict(out)


def analyze(log_path: Path) -> List[Dict[str, Any]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pnl_by_product = parse_product_pnl(text)

    side_stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    positions: Dict[str, int] = defaultdict(int)
    max_abs_position: Dict[str, int] = defaultdict(int)

    trades = sorted(iter_trade_objects(text), key=lambda t: t["timestamp"])
    for t in trades:
        product = t["symbol"]
        qty = int(t["quantity"])
        price = int(t["price"])
        side = None
        signed_qty = 0
        if t["buyer"] == "SUBMISSION":
            side = "bid"
            signed_qty = qty
        elif t["seller"] == "SUBMISSION":
            side = "ask"
            signed_qty = -qty
        if side is None:
            continue

        positions[product] += signed_qty
        max_abs_position[product] = max(max_abs_position[product], abs(positions[product]))
        stats = side_stats[(product, side)]
        stats["maker_fills"] += 1
        stats["fill_qty"] += qty
        stats["notional"] += qty * price
        stats["signed_notional"] += signed_qty * price

    rows: List[Dict[str, Any]] = []
    for product in PRODUCTS:
        for side in ["bid", "ask"]:
            stats = side_stats[(product, side)]
            qty = stats.get("fill_qty", 0.0)
            rows.append({
                "product": product,
                "side": side,
                "maker_orders_submitted": "",
                "maker_fills": int(stats.get("maker_fills", 0)),
                "fill_rate": "",
                "fill_qty": int(qty),
                "avg_fill_price": "" if qty <= 0 else round(stats.get("notional", 0.0) / qty, 4),
                "avg_quoted_edge": "",
                "avg_realized_edge_if_available": "",
                "gross_pnl_by_product": round(pnl_by_product.get(product, 0.0), 4),
                "final_position": int(positions.get(product, 0)),
                "max_abs_position": int(max_abs_position.get(product, 0)),
                "signed_notional": round(stats.get("signed_notional", 0.0), 4),
            })
    return rows


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    rows = analyze(log_path)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"read {log_path}")
    print(f"wrote {OUT}")
    for row in rows:
        if row["maker_fills"] or row["gross_pnl_by_product"]:
            print(
                f"{row['product']} {row['side']}: fills={row['maker_fills']} "
                f"qty={row['fill_qty']} pnl={row['gross_pnl_by_product']} "
                f"final={row['final_position']} max_abs={row['max_abs_position']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
