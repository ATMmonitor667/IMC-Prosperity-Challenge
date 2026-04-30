from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FAMILIES: dict[str, list[str]] = {
    "SNACKPACKS": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
    "PANELS": [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "UV_VISORS": [
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",
    ],
    "TRANSLATORS": [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "SLEEP_PODS": [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ],
    "MICROCHIPS": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",
    ],
}


DAYS = (2, 3, 4)
WALKFORWARD_FOLDS = (
    ("fit_day2_test_day3", 2, 3),
    ("fit_day3_test_day4", 3, 4),
)


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir)
        if (path / "prices_round_5_day_2.csv").exists():
            return path

    here = Path(__file__).resolve().parent
    candidates = [
        here / "round5",
        here,
        Path.cwd() / "round5",
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "prices_round_5_day_2.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find Round 5 price files.")


def _std(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _t_stat_like(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return float("nan")
    arr = np.asarray(pnls, dtype=float)
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(arr) / sd * math.sqrt(len(arr)))


def _pair_unit_sizes(beta: float, position_limit: int = 10) -> tuple[int, int]:
    beta_abs = abs(beta) if math.isfinite(beta) and abs(beta) > 1e-9 else 1.0
    left_qty = max(1, int(math.floor(position_limit / max(1.0, beta_abs))))
    right_qty = max(1, int(round(beta_abs * left_qty)))
    return left_qty, min(position_limit, right_qty)


def _cashflow(delta: int, bid: float, ask: float) -> float:
    if delta > 0:
        return -delta * ask
    if delta < 0:
        return (-delta) * bid
    return 0.0


def load_round5_data(data_dir: str | Path | None = None) -> dict[str, Any]:
    data_path = _resolve_data_dir(data_dir)
    products = {product for names in FAMILIES.values() for product in names}
    price_lists: dict[str, dict[int, dict[str, list[float]]]] = {
        product: {
            day: {
                "timestamp": [],
                "global_timestamp": [],
                "bid": [],
                "ask": [],
                "mid": [],
                "spread": [],
                "depth": [],
            }
            for day in DAYS
        }
        for product in products
    }
    trade_tape: dict[tuple[int, str], dict[str, float]] = {}
    trade_counts: dict[str, int] = defaultdict(int)
    trade_volume: dict[str, int] = defaultdict(int)

    for day in DAYS:
        offset = (day - 2) * 1_000_000
        price_path = data_path / f"prices_round_5_day_{day}.csv"
        with price_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                product = row["product"]
                if product not in products:
                    continue
                timestamp = int(row["timestamp"])
                bid = float(row["bid_price_1"])
                ask = float(row["ask_price_1"])
                mid = float(row["mid_price"])
                bid_volume = abs(int(float(row.get("bid_volume_1") or 0)))
                ask_volume = abs(int(float(row.get("ask_volume_1") or 0)))

                bucket = price_lists[product][day]
                bucket["timestamp"].append(timestamp)
                bucket["global_timestamp"].append(timestamp + offset)
                bucket["bid"].append(bid)
                bucket["ask"].append(ask)
                bucket["mid"].append(mid)
                bucket["spread"].append(max(1.0, ask - bid))
                bucket["depth"].append(bid_volume + ask_volume)

        trade_path = data_path / f"trades_round_5_day_{day}.csv"
        with trade_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                symbol = row["symbol"]
                if symbol not in products:
                    continue
                timestamp = int(row["timestamp"]) + offset
                price = float(row["price"])
                quantity = int(float(row["quantity"]))
                key = (timestamp, symbol)
                current = trade_tape.get(key)
                if current is None:
                    trade_tape[key] = {
                        "trade_price_min": price,
                        "trade_price_max": price,
                        "trade_qty": quantity,
                    }
                else:
                    current["trade_price_min"] = min(current["trade_price_min"], price)
                    current["trade_price_max"] = max(current["trade_price_max"], price)
                    current["trade_qty"] += quantity
                trade_counts[symbol] += 1
                trade_volume[symbol] += quantity

    price_data: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for product, day_map in price_lists.items():
        price_data[product] = {}
        for day, fields in day_map.items():
            price_data[product][day] = {
                field: np.asarray(values, dtype=float)
                for field, values in fields.items()
            }

    return {
        "data_dir": str(data_path),
        "price_data": price_data,
        "trade_tape": trade_tape,
        "trade_counts": dict(trade_counts),
        "trade_volume": dict(trade_volume),
    }


def _family_matrix(
    price_data: dict[str, dict[int, dict[str, np.ndarray]]],
    group: str,
    day: int,
    field: str,
) -> np.ndarray:
    return np.column_stack([price_data[product][day][field] for product in FAMILIES[group]])


def _family_timestamps(
    price_data: dict[str, dict[int, dict[str, np.ndarray]]],
    group: str,
    day: int,
) -> np.ndarray:
    return price_data[FAMILIES[group][0]][day]["global_timestamp"]


def microstructure_scan(data: dict[str, Any]) -> list[dict[str, Any]]:
    price_data = data["price_data"]
    rows = []
    for group, products in FAMILIES.items():
        spreads = []
        depths = []
        return_vols = []
        for product in products:
            product_returns = []
            for day in DAYS:
                day_data = price_data[product][day]
                spreads.extend(day_data["spread"].tolist())
                depths.extend(day_data["depth"].tolist())
                diffs = np.diff(day_data["mid"])
                product_returns.extend(diffs.tolist())
            return_vols.append(_std(np.asarray(product_returns, dtype=float)))

        rows.append(
            {
                "family": group,
                "avg_spread": float(np.mean(spreads)),
                "avg_depth": float(np.mean(depths)),
                "avg_return_vol": float(np.mean(return_vols)),
                "total_trades": sum(data["trade_counts"].get(product, 0) for product in products),
                "total_trade_volume": sum(data["trade_volume"].get(product, 0) for product in products),
            }
        )
    return rows


def cross_section_signal_scan(data: dict[str, Any]) -> list[dict[str, Any]]:
    price_data = data["price_data"]
    rows = []
    lookbacks = (100, 200)

    for group in FAMILIES:
        best_rows = []
        for lookback in lookbacks:
            min_periods = max(20, lookback // 3)
            mr_pnls: list[float] = []
            mom_pnls: list[float] = []
            for day in DAYS:
                mids = _family_matrix(price_data, group, day, "mid")
                relative = mids - mids.mean(axis=1, keepdims=True)
                returns = mids[1:] - mids[:-1]
                for t in range(min_periods, len(mids) - 1):
                    start = max(0, t - lookback)
                    if t - start < min_periods:
                        continue
                    hist = relative[start:t]
                    mu = hist.mean(axis=0)
                    sd = hist.std(axis=0, ddof=1)
                    sd = np.where(sd <= 1e-9, np.nan, sd)
                    zscore = np.clip((relative[t] - mu) / sd, -2.0, 2.0)
                    if np.isnan(zscore).any():
                        continue

                    for sign, pnl_bucket in ((-1.0, mr_pnls), (1.0, mom_pnls)):
                        raw_signal = sign * zscore
                        neutral = raw_signal - raw_signal.mean()
                        gross = np.abs(neutral).sum()
                        if gross <= 1e-9:
                            continue
                        signal = neutral / gross
                        pnl_bucket.append(float(signal @ returns[t]))

            for strategy, pnls in (("mean_reversion", mr_pnls), ("momentum", mom_pnls)):
                best_rows.append(
                    {
                        "family": group,
                        "strategy": strategy,
                        "lookback": lookback,
                        "observations": len(pnls),
                        "cum_pnl_units": float(sum(pnls)),
                        "avg_step_pnl": float(np.mean(pnls)) if pnls else float("nan"),
                        "t_stat_like": _t_stat_like(pnls),
                    }
                )
        rows.extend(best_rows)
    return rows


def _fit_beta(left: np.ndarray, right: np.ndarray) -> float:
    right_var = float(np.var(right, ddof=1))
    if right_var <= 1e-9:
        return 1.0
    beta = float(np.cov(left, right, ddof=1)[0, 1] / right_var)
    if not math.isfinite(beta):
        return 1.0
    return beta


def _run_pair_config(
    train_left: np.ndarray,
    train_right: np.ndarray,
    test_left: dict[str, np.ndarray],
    test_right: dict[str, np.ndarray],
    beta: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
    position_limit: int = 10,
) -> dict[str, float]:
    left_qty, right_qty = _pair_unit_sizes(beta, position_limit)
    hedge_sign = -1 if beta >= 0 else 1
    train_residual = train_left - beta * train_right
    history = list(train_residual[-lookback:])

    left_pos = 0
    right_pos = 0
    cash = 0.0
    trade_events = 0
    equity_path: list[float] = []
    current_side = 0

    mid_left = test_left["mid"]
    mid_right = test_right["mid"]
    bid_left = test_left["bid"]
    ask_left = test_left["ask"]
    bid_right = test_right["bid"]
    ask_right = test_right["ask"]

    for i in range(len(mid_left) - 1):
        if len(history) >= max(20, lookback // 3):
            hist_arr = np.asarray(history[-lookback:], dtype=float)
            sd = float(np.std(hist_arr, ddof=1)) if len(hist_arr) > 1 else 0.0
            if sd > 1e-9:
                zscore = (mid_left[i] - beta * mid_right[i] - float(np.mean(hist_arr))) / sd
                if zscore >= entry_z:
                    current_side = -1
                elif zscore <= -entry_z:
                    current_side = 1
                elif abs(zscore) <= exit_z:
                    current_side = 0

        target_left = current_side * left_qty
        target_right = current_side * hedge_sign * right_qty
        delta_left = target_left - left_pos
        delta_right = target_right - right_pos
        j = i + 1
        if delta_left:
            cash += _cashflow(delta_left, bid_left[j], ask_left[j])
            left_pos += delta_left
            trade_events += 1
        if delta_right:
            cash += _cashflow(delta_right, bid_right[j], ask_right[j])
            right_pos += delta_right
            trade_events += 1

        equity_path.append(cash + left_pos * mid_left[j] + right_pos * mid_right[j])
        history.append(float(mid_left[i] - beta * mid_right[i]))
        if len(history) > lookback:
            history = history[-lookback:]

    final_idx = len(mid_left) - 1
    if left_pos > 0:
        cash += left_pos * bid_left[final_idx]
    elif left_pos < 0:
        cash += left_pos * ask_left[final_idx]
    if right_pos > 0:
        cash += right_pos * bid_right[final_idx]
    elif right_pos < 0:
        cash += right_pos * ask_right[final_idx]

    peak = -1e18
    max_drawdown = 0.0
    for equity in equity_path:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "final_pnl": float(cash),
        "max_drawdown": float(max_drawdown),
        "trade_events": float(trade_events),
        "t_stat_like": _t_stat_like(equity_path),
    }


def pair_residual_scan(data: dict[str, Any]) -> list[dict[str, Any]]:
    price_data = data["price_data"]
    lookbacks = (150, 300)
    entry_grid = (2.0, 2.5)
    exit_grid = (0.5, 1.0)
    rows = []

    for group, products in FAMILIES.items():
        for left_index, left in enumerate(products):
            for right in products[left_index + 1 :]:
                fold_cache = {}
                for fold_label, train_day, test_day in WALKFORWARD_FOLDS:
                    beta = _fit_beta(price_data[left][train_day]["mid"], price_data[right][train_day]["mid"])
                    fold_cache[fold_label] = {
                        "beta": beta,
                        "train_left": price_data[left][train_day]["mid"],
                        "train_right": price_data[right][train_day]["mid"],
                        "test_left": price_data[left][test_day],
                        "test_right": price_data[right][test_day],
                    }

                for lookback in lookbacks:
                    for entry_z in entry_grid:
                        for exit_z in exit_grid:
                            fold_results = {}
                            for fold_label, _, _ in WALKFORWARD_FOLDS:
                                cached = fold_cache[fold_label]
                                fold_results[fold_label] = _run_pair_config(
                                    cached["train_left"],
                                    cached["train_right"],
                                    cached["test_left"],
                                    cached["test_right"],
                                    beta=cached["beta"],
                                    lookback=lookback,
                                    entry_z=entry_z,
                                    exit_z=exit_z,
                                )
                            pnls = [fold_results[label]["final_pnl"] for label, _, _ in WALKFORWARD_FOLDS]
                            rows.append(
                                {
                                    "family": group,
                                    "pair": f"{left} vs {right}",
                                    "lookback": lookback,
                                    "entry_z": entry_z,
                                    "exit_z": exit_z,
                                    "fit_day2_test_day3_pnl": pnls[0],
                                    "fit_day3_test_day4_pnl": pnls[1],
                                    "worst_fold_pnl": min(pnls),
                                    "mean_fold_pnl": float(np.mean(pnls)),
                                    "positive_folds": int(sum(pnl > 0 for pnl in pnls)),
                                    "mean_trade_events": float(np.mean([fold_results[label]["trade_events"] for label, _, _ in WALKFORWARD_FOLDS])),
                                }
                            )
    return rows


def _run_leader_follower_config(
    train_leader: np.ndarray,
    train_follower: np.ndarray,
    test_leader: dict[str, np.ndarray],
    test_follower: dict[str, np.ndarray],
    hold_bars: int,
    edge_threshold_spreads: float,
) -> dict[str, float]:
    leader_returns = train_leader[1 : -(hold_bars + 1)] - train_leader[: -(hold_bars + 2)]
    future_moves = train_follower[(hold_bars + 2) :] - train_follower[1 : -(hold_bars + 1)]
    if len(leader_returns) < 100 or np.std(leader_returns) <= 1e-9:
        return {"final_pnl": 0.0, "trade_events": 0.0, "t_stat_like": float("nan")}

    design = np.column_stack([np.ones(len(leader_returns)), leader_returns])
    intercept, beta = np.linalg.lstsq(design, future_moves, rcond=None)[0]

    leader_mid = test_leader["mid"]
    follower_bid = test_follower["bid"]
    follower_ask = test_follower["ask"]
    if len(leader_mid) <= hold_bars + 2:
        return {"final_pnl": 0.0, "trade_events": 0.0, "t_stat_like": float("nan")}

    leader_move = leader_mid[1 : -(hold_bars + 1)] - leader_mid[: -(hold_bars + 2)]
    signal = intercept + beta * leader_move
    entry_bid = follower_bid[2:-hold_bars]
    entry_ask = follower_ask[2:-hold_bars]
    exit_bid = follower_bid[(hold_bars + 2) :]
    exit_ask = follower_ask[(hold_bars + 2) :]
    spread = np.maximum(1.0, entry_ask - entry_bid)
    active = np.abs(signal) >= edge_threshold_spreads * spread
    long_pnl = exit_bid - entry_ask
    short_pnl = entry_bid - exit_ask
    cash_pnls = np.where(signal > 0, long_pnl, short_pnl)[active].astype(float).tolist()

    return {
        "final_pnl": float(sum(cash_pnls)),
        "trade_events": float(len(cash_pnls)),
        "t_stat_like": _t_stat_like(cash_pnls),
    }


def leader_follower_scan(data: dict[str, Any]) -> list[dict[str, Any]]:
    price_data = data["price_data"]
    rows = []
    hold_grid = (1, 2)
    edge_grid = (1.0, 1.5)

    for group, products in FAMILIES.items():
        for leader in products:
            for follower in products:
                if leader == follower:
                    continue
                for hold_bars in hold_grid:
                    for edge_threshold in edge_grid:
                        fold_results = {}
                        for fold_label, train_day, test_day in WALKFORWARD_FOLDS:
                            fold_results[fold_label] = _run_leader_follower_config(
                                price_data[leader][train_day]["mid"],
                                price_data[follower][train_day]["mid"],
                                price_data[leader][test_day],
                                price_data[follower][test_day],
                                hold_bars,
                                edge_threshold,
                            )
                        pnls = [fold_results[label]["final_pnl"] for label, _, _ in WALKFORWARD_FOLDS]
                        rows.append(
                            {
                                "family": group,
                                "leader": leader,
                                "follower": follower,
                                "hold_bars": hold_bars,
                                "edge_threshold_spreads": edge_threshold,
                                "fit_day2_test_day3_pnl": pnls[0],
                                "fit_day3_test_day4_pnl": pnls[1],
                                "worst_fold_pnl": min(pnls),
                                "mean_fold_pnl": float(np.mean(pnls)),
                                "positive_folds": int(sum(pnl > 0 for pnl in pnls)),
                                "mean_trade_events": float(np.mean([fold_results[label]["trade_events"] for label, _, _ in WALKFORWARD_FOLDS])),
                            }
                        )
    return rows


def _fit_leave_one_out_model(
    price_data: dict[str, dict[int, dict[str, np.ndarray]]],
    target: str,
    hedges: list[str],
    train_day: int,
) -> np.ndarray:
    y = price_data[target][train_day]["mid"]
    x = np.column_stack([np.ones(len(y))] + [price_data[hedge][train_day]["mid"] for hedge in hedges])
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _run_passive_config(
    target_data: dict[str, np.ndarray],
    hedge_mids: np.ndarray,
    coefficients: np.ndarray,
    trade_tape: dict[tuple[int, str], dict[str, float]],
    target: str,
    edge_threshold_spreads: float,
    inventory_skew_spreads: float,
    position_limit: int = 10,
) -> dict[str, float]:
    timestamps = target_data["global_timestamp"]
    bid = target_data["bid"]
    ask = target_data["ask"]
    mid = target_data["mid"]
    spread = target_data["spread"]
    fair = np.column_stack([np.ones(len(mid)), hedge_mids]) @ coefficients

    position = 0
    cash = 0.0
    quote_count = 0
    fill_count = 0
    buy_fills = 0
    sell_fills = 0
    equity_path: list[float] = []

    for i in range(len(mid)):
        row_spread = max(1.0, spread[i])
        reservation = fair[i] - inventory_skew_spreads * row_spread * position
        buy_signal = reservation - mid[i] >= edge_threshold_spreads * row_spread
        sell_signal = mid[i] - reservation >= edge_threshold_spreads * row_spread
        quoted_buy = bool(buy_signal and position < position_limit)
        quoted_sell = bool(sell_signal and position > -position_limit)
        quote_count += int(quoted_buy) + int(quoted_sell)

        tape = trade_tape.get((int(timestamps[i]), target))
        if tape:
            if quoted_buy and tape["trade_price_min"] <= bid[i]:
                cash -= bid[i]
                position += 1
                fill_count += 1
                buy_fills += 1
            if quoted_sell and tape["trade_price_max"] >= ask[i]:
                cash += ask[i]
                position -= 1
                fill_count += 1
                sell_fills += 1
        equity_path.append(float(cash + position * mid[i]))

    if position > 0:
        cash += position * bid[-1]
    elif position < 0:
        cash += position * ask[-1]

    peak = -1e18
    max_drawdown = 0.0
    for equity in equity_path:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    return {
        "final_pnl": float(cash),
        "max_drawdown": float(max_drawdown),
        "quote_count": float(quote_count),
        "fill_count": float(fill_count),
        "buy_fills": float(buy_fills),
        "sell_fills": float(sell_fills),
        "t_stat_like": _t_stat_like(equity_path),
    }


def passive_fair_value_scan(data: dict[str, Any]) -> list[dict[str, Any]]:
    price_data = data["price_data"]
    trade_tape = data["trade_tape"]
    rows = []
    edge_grid = (0.25, 0.5)
    inventory_grid = (0.25, 0.75)

    for group, products in FAMILIES.items():
        for target in products:
            hedges = [product for product in products if product != target]
            for edge_threshold in edge_grid:
                for inventory_skew in inventory_grid:
                    fold_results = {}
                    for fold_label, train_day, test_day in WALKFORWARD_FOLDS:
                        coefficients = _fit_leave_one_out_model(price_data, target, hedges, train_day)
                        hedge_mids = np.column_stack([price_data[hedge][test_day]["mid"] for hedge in hedges])
                        fold_results[fold_label] = _run_passive_config(
                            price_data[target][test_day],
                            hedge_mids,
                            coefficients,
                            trade_tape,
                            target,
                            edge_threshold_spreads=edge_threshold,
                            inventory_skew_spreads=inventory_skew,
                        )
                    pnls = [fold_results[label]["final_pnl"] for label, _, _ in WALKFORWARD_FOLDS]
                    fills = [fold_results[label]["fill_count"] for label, _, _ in WALKFORWARD_FOLDS]
                    rows.append(
                        {
                            "family": group,
                            "target": target,
                            "edge_threshold_spreads": edge_threshold,
                            "inventory_skew_spreads": inventory_skew,
                            "fit_day2_test_day3_pnl": pnls[0],
                            "fit_day3_test_day4_pnl": pnls[1],
                            "worst_fold_pnl": min(pnls),
                            "mean_fold_pnl": float(np.mean(pnls)),
                            "positive_folds": int(sum(pnl > 0 for pnl in pnls)),
                            "min_fill_count": float(min(fills)),
                            "total_fill_count": float(sum(fills)),
                            "mean_quote_count": float(np.mean([fold_results[label]["quote_count"] for label, _, _ in WALKFORWARD_FOLDS])),
                        }
                    )
    return rows


def _best_by_family(rows: list[dict[str, Any]], key: str = "worst_fold_pnl") -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        current = best.get(family)
        if current is None or (row.get(key, -1e18), row.get("mean_fold_pnl", -1e18)) > (
            current.get(key, -1e18),
            current.get("mean_fold_pnl", -1e18),
        ):
            best[family] = row
    return best


def _best_cross_section(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        current = best.get(family)
        if current is None or row["t_stat_like"] > current["t_stat_like"]:
            best[family] = row
    return best


def run_family_strategy_scan(data_dir: str | Path | None = None) -> dict[str, Any]:
    data = load_round5_data(data_dir)
    micro = microstructure_scan(data)
    cross_section = cross_section_signal_scan(data)
    pairs = pair_residual_scan(data)
    leader_follower = leader_follower_scan(data)
    passive = passive_fair_value_scan(data)

    best_cross = _best_cross_section(cross_section)
    best_pair = _best_by_family(pairs)
    best_leader = _best_by_family(leader_follower)
    best_passive = _best_by_family(passive)

    overview = []
    for micro_row in micro:
        family = micro_row["family"]
        candidates = [
            ("pair_residual", best_pair.get(family, {})),
            ("leader_follower", best_leader.get(family, {})),
            ("passive_fair_value", best_passive.get(family, {})),
        ]
        candidates = [item for item in candidates if item[1]]
        selected_name, selected_row = max(
            candidates,
            key=lambda item: (item[1].get("worst_fold_pnl", -1e18), item[1].get("mean_fold_pnl", -1e18)),
        )
        overview.append(
            {
                **micro_row,
                "cross_section_strategy": best_cross[family]["strategy"],
                "cross_section_lookback": best_cross[family]["lookback"],
                "cross_section_t_stat": best_cross[family]["t_stat_like"],
                "recommended_branch": selected_name,
                "recommended_worst_fold_pnl": selected_row.get("worst_fold_pnl"),
                "recommended_mean_fold_pnl": selected_row.get("mean_fold_pnl"),
                "recommended_positive_folds": selected_row.get("positive_folds"),
                "best_pair": best_pair.get(family),
                "best_leader_follower": best_leader.get(family),
                "best_passive": best_passive.get(family),
            }
        )

    return {
        "data_dir": data["data_dir"],
        "microstructure": micro,
        "cross_section": cross_section,
        "pair_residual": pairs,
        "leader_follower": leader_follower,
        "passive_fair_value": passive,
        "overview": overview,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], float_digits: int = 1) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(key, ""), float_digits) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def format_family_strategy_markdown(results: dict[str, Any]) -> str:
    overview = sorted(results["overview"], key=lambda row: row["recommended_worst_fold_pnl"], reverse=True)
    lines = [
        "## Automated family strategy scan",
        "",
        f"Data directory: `{results['data_dir']}`",
        "",
        "This scan compares four strategy families for the requested products:",
        "",
        "- cross-sectional mean reversion versus momentum, measured in normalized next-step units;",
        "- delayed residual pair trading with day2->day3 and day3->day4 walk-forward folds;",
        "- leader-follower one-lot diagnostics with next-bar entry and fixed short holding windows;",
        "- passive leave-one-out fair-value quoting using public trades for conservative fill checks.",
        "",
        "### Family overview",
        _markdown_table(
            overview,
            [
                ("family", "family"),
                ("avg_spread", "avg spread"),
                ("avg_return_vol", "return vol"),
                ("total_trades", "trades"),
                ("cross_section_strategy", "xs bias"),
                ("cross_section_t_stat", "xs t"),
                ("recommended_branch", "best branch"),
                ("recommended_worst_fold_pnl", "worst fold"),
                ("recommended_mean_fold_pnl", "mean fold"),
                ("recommended_positive_folds", "+ folds"),
            ],
            float_digits=2,
        ),
        "",
        "### Best branch detail by family",
    ]

    for row in sorted(results["overview"], key=lambda item: item["family"]):
        family = row["family"]
        lines.append("")
        lines.append(f"#### {family}")
        lines.append(
            f"- Cross-section bias: `{row['cross_section_strategy']}` at lookback `{row['cross_section_lookback']}` "
            f"(t-stat-like `{row['cross_section_t_stat']:.2f}`)."
        )
        pair = row["best_pair"]
        lines.append(
            f"- Best residual pair: `{pair['pair']}`, lookback `{pair['lookback']}`, entry `{pair['entry_z']}`, "
            f"exit `{pair['exit_z']}`, folds `{pair['fit_day2_test_day3_pnl']:.0f}` / "
            f"`{pair['fit_day3_test_day4_pnl']:.0f}`."
        )
        leader = row["best_leader_follower"]
        lines.append(
            f"- Best leader-follower: `{leader['leader']} -> {leader['follower']}`, hold `{leader['hold_bars']}`, "
            f"edge `{leader['edge_threshold_spreads']}`, folds `{leader['fit_day2_test_day3_pnl']:.0f}` / "
            f"`{leader['fit_day3_test_day4_pnl']:.0f}`."
        )
        passive = row["best_passive"]
        lines.append(
            f"- Best passive fair value: target `{passive['target']}`, edge `{passive['edge_threshold_spreads']}`, "
            f"inventory skew `{passive['inventory_skew_spreads']}`, folds `{passive['fit_day2_test_day3_pnl']:.0f}` / "
            f"`{passive['fit_day3_test_day4_pnl']:.0f}`, fills `{passive['total_fill_count']:.0f}`."
        )
        lines.append(
            f"- Implementation direction: prioritize `{row['recommended_branch']}` only if the live fill model and "
            "markouts remain consistent; otherwise use the cross-section bias as a low-size overlay."
        )

    return "\n".join(lines)


if __name__ == "__main__":
    scan_results = run_family_strategy_scan()
    print(format_family_strategy_markdown(scan_results))
