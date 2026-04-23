"""Local dashboard server for IMC Prosperity algorithmic-trend analysis.

Run:  python dashboard/server.py [--port 8765]
Open: http://localhost:8765
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import urllib.parse
import webbrowser
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"

ROUND_DIRS = {
    "1": ROOT / "ROUND1",
    "2": ROOT / "ROUND2",
}
BACKTEST_DIRS = [ROOT / "backtests", ROOT / "ROUND1" / "backtests", ROOT / "ROUND2" / "backtests"]


# --------------------------------------------------------------------------- helpers

def _discover_round_days(round_id: str) -> list[int]:
    directory = ROUND_DIRS.get(round_id)
    if not directory or not directory.is_dir():
        return []
    days: set[int] = set()
    for path in directory.glob(f"prices_round_{round_id}_day_*.csv"):
        match = re.search(r"day_(-?\d+)\.csv$", path.name)
        if match:
            days.add(int(match.group(1)))
    return sorted(days)


def _discover_backtests() -> list[dict]:
    out = []
    for directory in BACKTEST_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.log"), reverse=True):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            out.append({
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "name": path.name,
                "folder": str(directory.relative_to(ROOT)).replace("\\", "/"),
                "size_kb": round(size / 1024, 1),
            })
    return out


def _round_csv(round_id: str, day: int, kind: str) -> Path | None:
    directory = ROUND_DIRS.get(round_id)
    if not directory:
        return None
    path = directory / f"{kind}_round_{round_id}_day_{day}.csv"
    return path if path.is_file() else None


@lru_cache(maxsize=8)
def _load_prices_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, sep=";")
    return df


@lru_cache(maxsize=8)
def _load_trades_csv(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, sep=";")
    return df


def _sanitize_json_trailing_commas(text: str) -> str:
    # Backtest logs emit trailing commas before '}' / ']' — strip them for json.loads
    return re.sub(r",(\s*[}\]])", r"\1", text)


@lru_cache(maxsize=4)
def _parse_backtest(path_str: str) -> dict:
    path = Path(path_str)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Sections are delimited by their header lines at column 0.
    sandbox_idx = text.find("Sandbox logs:")
    activities_idx = text.find("Activities log:")
    trade_idx = text.find("Trade History:")

    if activities_idx == -1 or trade_idx == -1:
        raise ValueError(f"Malformed backtest log: {path.name}")

    sandbox_block = text[sandbox_idx + len("Sandbox logs:"):activities_idx].strip()
    activities_block = text[activities_idx + len("Activities log:"):trade_idx].strip()
    trade_block = text[trade_idx + len("Trade History:"):].strip()

    # Sandbox logs are a stream of JSON objects, one after another (not a JSON array).
    sandbox_entries: list[dict] = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(sandbox_block)
    while i < n:
        while i < n and sandbox_block[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(sandbox_block, i)
        except json.JSONDecodeError:
            break
        sandbox_entries.append(obj)
        i = end

    # Activities log is a semicolon-CSV identical to round prices CSVs.
    from io import StringIO
    activities_df = pd.read_csv(StringIO(activities_block), sep=";")

    # Trade history is a JSON array (with trailing commas).
    trade_list = json.loads(_sanitize_json_trailing_commas(trade_block))

    return {
        "sandbox": sandbox_entries,
        "activities": activities_df,
        "trades": trade_list,
    }


# --------------------------------------------------------------------------- payload builders

def _nan_safe(series: pd.Series) -> list:
    return [None if pd.isna(v) else float(v) for v in series]


def _prices_payload(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(int)
    # When a log spans multiple days we offset timestamps by day index so the x-axis is monotonic.
    if "day" in df.columns and df["day"].nunique() > 1:
        days = sorted(df["day"].unique())
        base = {d: i * 1_100_000 for i, d in enumerate(days)}
        df["t"] = df["day"].map(base) + df["timestamp"]
    else:
        df["t"] = df["timestamp"]

    products: dict[str, dict] = {}
    for product, group in df.groupby("product", sort=True):
        group = group.sort_values("t")
        bid1 = group["bid_price_1"]
        ask1 = group["ask_price_1"]
        spread = ask1 - bid1
        bid_depth = group[["bid_volume_1", "bid_volume_2", "bid_volume_3"]].fillna(0).sum(axis=1)
        ask_depth = group[["ask_volume_1", "ask_volume_2", "ask_volume_3"]].fillna(0).sum(axis=1)
        pnl_col = group["profit_and_loss"] if "profit_and_loss" in group.columns else pd.Series([0] * len(group))
        products[product] = {
            "t": group["t"].astype(int).tolist(),
            "mid": _nan_safe(group["mid_price"]),
            "bid1": _nan_safe(bid1),
            "ask1": _nan_safe(ask1),
            "bid2": _nan_safe(group["bid_price_2"]),
            "ask2": _nan_safe(group["ask_price_2"]),
            "bid3": _nan_safe(group["bid_price_3"]),
            "ask3": _nan_safe(group["ask_price_3"]),
            "spread": _nan_safe(spread),
            "bid_depth": bid_depth.astype(int).tolist(),
            "ask_depth": ask_depth.astype(int).tolist(),
            "pnl": _nan_safe(pnl_col),
        }
    return {
        "products": sorted(products.keys()),
        "data": products,
    }


def _trades_payload(trades: list[dict] | pd.DataFrame) -> list[dict]:
    if isinstance(trades, pd.DataFrame):
        df = trades.astype(object).where(trades.notna(), None)
        return df.to_dict(orient="records")
    # list[dict] from backtest log — already clean JSON.
    return trades


# --------------------------------------------------------------------------- HTTP handler

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        # Keep console tidy; show only errors.
        if args and isinstance(args[1], str) and args[1].startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, "Not found")
            return
        ctype, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._send_file(STATIC / "index.html")
                return
            if path.startswith("/static/"):
                self._send_file(STATIC / path[len("/static/"):])
                return

            if path == "/api/sources":
                rounds = []
                for rid in sorted(ROUND_DIRS):
                    days = _discover_round_days(rid)
                    if days:
                        rounds.append({"round": rid, "days": days})
                self._send_json({
                    "rounds": rounds,
                    "backtests": _discover_backtests(),
                })
                return

            if path == "/api/round":
                rid = qs.get("round", ["1"])[0]
                day = int(qs.get("day", ["0"])[0])
                prices_path = _round_csv(rid, day, "prices")
                trades_path = _round_csv(rid, day, "trades")
                if not prices_path:
                    self._send_json({"error": f"No prices CSV for round {rid} day {day}"}, 404)
                    return
                prices = _load_prices_csv(str(prices_path))
                trades = _load_trades_csv(str(trades_path)) if trades_path else pd.DataFrame()
                self._send_json({
                    "prices": _prices_payload(prices),
                    "trades": _trades_payload(trades) if not trades.empty else [],
                    "has_submission": False,
                })
                return

            if path == "/api/backtest":
                rel = qs.get("file", [""])[0]
                if not rel:
                    self._send_json({"error": "Missing file parameter"}, 400)
                    return
                abs_path = (ROOT / rel).resolve()
                if not str(abs_path).startswith(str(ROOT)):
                    self._send_json({"error": "Path escape"}, 400)
                    return
                if not abs_path.is_file():
                    self._send_json({"error": "File not found"}, 404)
                    return
                parsed_log = _parse_backtest(str(abs_path))
                self._send_json({
                    "prices": _prices_payload(parsed_log["activities"]),
                    "trades": _trades_payload(parsed_log["trades"]),
                    "has_submission": True,
                    "sandbox_count": len(parsed_log["sandbox"]),
                })
                return

            self.send_error(404, "Not found")
        except Exception as exc:  # surface errors to the UI
            import traceback
            traceback.print_exc()
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"IMC dashboard serving at {url}")
    print(f"  root:      {ROOT}")
    print(f"  rounds:    {[k for k, v in ROUND_DIRS.items() if v.is_dir()]}")
    print(f"  backtests: {sum(1 for d in BACKTEST_DIRS if d.is_dir())} dirs")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        srv.server_close()


if __name__ == "__main__":
    main()
