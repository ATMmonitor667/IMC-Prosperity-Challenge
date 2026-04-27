"""Approximate Round 4 maker fill diagnostics from a prosperity4bt log.

This does not run a backtest. It reads an existing log and writes:

* ``round4_fill_diagnostics.csv`` with product/side fill summaries.
* ``round4_segment_diagnostics.csv`` with time-segment PnL/fill/position
  summaries.

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
SEGMENT_OUT = ROOT / "round4_segment_diagnostics.csv"
SEGMENT_SIZE = 20_000
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
        day = parse_field(body, "day")
        if not symbol or not price or not quantity:
            continue
        yield {
            "symbol": symbol,
            "price": int(float(price)),
            "quantity": abs(int(float(quantity))),
            "buyer": buyer or "",
            "seller": seller or "",
            "timestamp": int(float(timestamp or 0)),
            "day": int(float(day)) if day not in (None, "") else None,
        }


def parse_activity_rows(text: str) -> List[Dict[str, Any]]:
    header = "day;timestamp;product;"
    header_idx = text.find(header)
    if header_idx < 0:
        return []
    end_idx = text.find("Trade History:", header_idx)
    activity_text = text[header_idx:end_idx if end_idx >= 0 else len(text)]
    rows: List[Dict[str, Any]] = []
    reader = csv.DictReader(StringIO(activity_text), delimiter=";")
    for row in reader:
        try:
            rows.append({
                "day": int(row["day"]),
                "timestamp": int(row["timestamp"]),
                "product": row["product"],
                "mid_price": None if row.get("mid_price") in (None, "") else float(row["mid_price"]),
                "profit_and_loss": float(row.get("profit_and_loss") or 0.0),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def parse_product_pnl(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    last: Dict[Tuple[int, str], float] = {}
    for row in rows:
        day = int(row["day"])
        product = str(row["product"])
        pnl = float(row["profit_and_loss"])
        last[(day, product)] = pnl
    out: Dict[str, float] = defaultdict(float)
    for (_, product), pnl in last.items():
        out[product] += pnl
    return dict(out)


def activity_maps(rows: List[Dict[str, Any]]) -> Tuple[Dict[Tuple[int, int, str], float], Dict[Tuple[int, str], float]]:
    mid_by_exact: Dict[Tuple[int, int, str], float] = {}
    final_mid: Dict[Tuple[int, str], float] = {}
    for row in rows:
        mid = row.get("mid_price")
        if mid is None:
            continue
        key = (int(row["day"]), int(row["timestamp"]), str(row["product"]))
        mid_by_exact[key] = float(mid)
        final_mid[(int(row["day"]), str(row["product"]))] = float(mid)
    return mid_by_exact, final_mid


def quote_style(product: str) -> str:
    try:
        import trader  # type: ignore

        by_product = getattr(trader, "QUOTE_STYLE_BY_PRODUCT", {})
        return str(by_product.get(product, getattr(trader, "QUOTE_STYLE", "")))
    except Exception:
        return ""


def analyze(log_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    activity = parse_activity_rows(text)
    pnl_by_product = parse_product_pnl(activity)
    mid_by_exact, final_mid = activity_maps(activity)

    side_stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    positions: Dict[str, int] = defaultdict(int)
    max_abs_position: Dict[str, int] = defaultdict(int)
    segment_stats: Dict[Tuple[int, int, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    trades = list(iter_trade_objects(text))
    for i, t in enumerate(trades):
        product = t["symbol"]
        qty = int(t["quantity"])
        price = int(t["price"])
        day = 0 if t.get("day") is None else int(t["day"])
        timestamp = int(t["timestamp"])
        segment = timestamp // SEGMENT_SIZE
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
        mid = mid_by_exact.get((day, timestamp, product))
        if mid is None and t.get("day") is None:
            # Combined logs may not carry day in trade history; use any exact timestamp match.
            candidates = [v for (d, ts, p), v in mid_by_exact.items() if ts == timestamp and p == product]
            mid = candidates[-1] if candidates else None
        mark = final_mid.get((day, product))
        if mark is None:
            marks = [v for (d, p), v in final_mid.items() if p == product]
            mark = marks[-1] if marks else None

        stats["maker_fills"] += 1
        stats["fill_qty"] += qty
        stats["notional"] += qty * price
        stats["signed_notional"] += signed_qty * price
        if mid is not None:
            stats["mid_at_fill_sum"] += qty * mid
            if side == "bid":
                stats["realized_proxy"] += qty * (mid - price)
            else:
                stats["realized_proxy"] += qty * (price - mid)
        if mark is not None:
            if side == "bid":
                stats["markout_to_final"] += qty * (mark - price)
            else:
                stats["markout_to_final"] += qty * (price - mark)

        seg = segment_stats[(day, segment, product)]
        seg[f"{side}_fills"] += 1
        seg[f"{side}_qty"] += qty
        seg["net_position_end"] = positions[product]
        seg["max_abs_position"] = max(seg.get("max_abs_position", 0.0), abs(positions[product]))

    rows: List[Dict[str, Any]] = []
    for product in PRODUCTS:
        for side in ["bid", "ask"]:
            stats = side_stats[(product, side)]
            qty = stats.get("fill_qty", 0.0)
            rows.append({
                "product": product,
                "side": side,
                "quote_style": quote_style(product),
                "submitted_orders": "",
                "maker_fills": int(stats.get("maker_fills", 0)),
                "fill_rate": "",
                "fill_qty": int(qty),
                "avg_fill_price": "" if qty <= 0 else round(stats.get("notional", 0.0) / qty, 4),
                "avg_mid_at_fill": "" if qty <= 0 or stats.get("mid_at_fill_sum", 0.0) == 0.0 else round(stats["mid_at_fill_sum"] / qty, 4),
                "avg_expected_at_fill": "",
                "avg_quoted_edge": "",
                "avg_realized_edge_if_available": "" if qty <= 0 or stats.get("realized_proxy", 0.0) == 0.0 else round(stats["realized_proxy"] / qty, 4),
                "realized_pnl_proxy": round(stats.get("realized_proxy", 0.0), 4),
                "markout_to_final": round(stats.get("markout_to_final", 0.0), 4),
                "gross_pnl_by_product": round(pnl_by_product.get(product, 0.0), 4),
                "final_position": int(positions.get(product, 0)),
                "max_abs_position": int(max_abs_position.get(product, 0)),
                "signed_notional": round(stats.get("signed_notional", 0.0), 4),
            })

    segment_rows = build_segment_rows(activity, segment_stats)
    return rows, segment_rows


def build_segment_rows(
    activity: List[Dict[str, Any]],
    segment_stats: Dict[Tuple[int, int, str], Dict[str, float]],
) -> List[Dict[str, Any]]:
    pnl_last: Dict[Tuple[int, str, int], float] = {}
    for row in activity:
        day = int(row["day"])
        product = str(row["product"])
        segment = int(row["timestamp"]) // SEGMENT_SIZE
        pnl_last[(day, product, segment)] = float(row["profit_and_loss"])

    rows: List[Dict[str, Any]] = []
    keys = sorted(pnl_last)
    previous: Dict[Tuple[int, str], float] = defaultdict(float)
    for day, product, segment in keys:
        end_pnl = pnl_last[(day, product, segment)]
        start_pnl = previous[(day, product)]
        pnl_delta = end_pnl - start_pnl
        previous[(day, product)] = end_pnl
        stats = segment_stats[(day, segment, product)]
        rows.append({
            "day": day,
            "segment_start": segment * SEGMENT_SIZE,
            "segment_end": (segment + 1) * SEGMENT_SIZE,
            "product": product,
            "pnl_delta": round(pnl_delta, 4),
            "segment_end_pnl": round(end_pnl, 4),
            "bid_fills": int(stats.get("bid_fills", 0)),
            "ask_fills": int(stats.get("ask_fills", 0)),
            "bid_qty": int(stats.get("bid_qty", 0)),
            "ask_qty": int(stats.get("ask_qty", 0)),
            "net_position_end": int(stats.get("net_position_end", 0)),
            "max_abs_position": int(stats.get("max_abs_position", 0)),
        })
    return rows


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_log()
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    rows, segment_rows = analyze(log_path)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if segment_rows:
        with SEGMENT_OUT.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(segment_rows[0].keys()))
            writer.writeheader()
            writer.writerows(segment_rows)

    print(f"read {log_path}")
    print(f"wrote {OUT}")
    if segment_rows:
        print(f"wrote {SEGMENT_OUT}")
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
