from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from family_strategy_scan import (
    FAMILIES,
    WALKFORWARD_FOLDS,
    _fit_beta,
    _pair_unit_sizes,
    _t_stat_like,
    load_round5_data,
)


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "optimal_algorithm_research_report.md"
RESULTS_PATH = ROOT / "optimal_algorithm_research_results.json"

EDGE_GRID = (0.25, 0.50, 0.75, 1.00)
SKEW_GRID = (0.00, 0.25, 0.50, 0.75, 1.00)
FILL_THROUGH_GRID = (0.00, 0.25, 0.50)
PAIR_LOOKBACK_GRID = (60, 90, 120, 180, 300)
PAIR_ENTRY_GRID = (2.0, 2.5, 3.0)
PAIR_EXIT_GRID = (0.5, 1.0)
PAIR_SLIPPAGE_GRID = (0.0, 0.5, 1.0)
MARKOUT_HORIZONS = (1, 5, 10, 20, 50)


PANEL_FEATURES = {
    "PANEL_1X2": (1.0, 3.0, 2.0),
    "PANEL_2X2": (1.0, 4.0, 4.0),
    "PANEL_1X4": (1.0, 5.0, 4.0),
    "PANEL_2X4": (1.0, 6.0, 8.0),
    "PANEL_4X4": (1.0, 8.0, 16.0),
}


PASSIVE_CANDIDATES = (
    {
        "family": "PANELS",
        "target": "PANEL_2X2",
        "models": ("panel_geometry",),
        "cap": 6,
    },
    {
        "family": "UV_VISORS",
        "target": "UV_VISOR_MAGENTA",
        "models": ("static_loo", "rolling_loo_90"),
        "cap": 5,
    },
    {
        "family": "GALAXY_SOUNDS",
        "target": "GALAXY_SOUNDS_SOLAR_FLAMES",
        "models": ("static_loo", "rolling_loo_90"),
        "cap": 5,
    },
    {
        "family": "SNACKPACKS",
        "target": "SNACKPACK_PISTACHIO",
        "models": ("static_loo", "rolling_loo_90"),
        "cap": 4,
    },
)


PAIR_CANDIDATES = (
    {
        "family": "SLEEP_PODS",
        "left": "SLEEP_POD_POLYESTER",
        "right": "SLEEP_POD_NYLON",
    },
    {
        "family": "MICROCHIPS",
        "left": "MICROCHIP_CIRCLE",
        "right": "MICROCHIP_RECTANGLE",
    },
    {
        "family": "TRANSLATORS",
        "left": "TRANSLATOR_ECLIPSE_CHARCOAL",
        "right": "TRANSLATOR_VOID_BLUE",
    },
)


def _fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 1) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(key, ""), digits) for key, _ in columns) + " |")
    return "\n".join([header, sep, *body])


def _ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    xtx = design.T @ design
    xty = design.T @ target
    return np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), xty)


def _max_drawdown(equity_path: list[float]) -> float:
    peak = -1e18
    drawdown = 0.0
    for equity in equity_path:
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return float(drawdown)


def _quarter_pnls(equity_path: list[float], final_equity: float) -> list[float]:
    if not equity_path:
        return [0.0, 0.0, 0.0, 0.0]
    path = np.asarray(equity_path, dtype=float).copy()
    path[-1] = final_equity
    cuts = np.linspace(0, len(path) - 1, 5, dtype=int)
    out = []
    previous = 0.0
    for idx in cuts[1:]:
        value = float(path[idx])
        out.append(value - previous)
        previous = value
    return out


def _family_clip(fair: np.ndarray, family_matrix: np.ndarray) -> np.ndarray:
    row_std = np.std(family_matrix, axis=1, ddof=1)
    width = 2.0 * np.maximum(1.0, row_std)
    lower = np.min(family_matrix, axis=1) - width
    upper = np.max(family_matrix, axis=1) + width
    return np.clip(fair, lower, upper)


def _panel_geometry_fair_series(price_data: dict[str, Any], target: str, day: int) -> np.ndarray:
    others = [product for product in PANEL_FEATURES if product != target]
    feature_matrix = np.asarray([PANEL_FEATURES[product] for product in others], dtype=float)
    transform = np.linalg.solve(
        feature_matrix.T @ feature_matrix + 1e-4 * np.eye(feature_matrix.shape[1]),
        feature_matrix.T,
    )
    y_matrix = np.vstack([price_data[product][day]["mid"] for product in others])
    coeffs_by_time = transform @ y_matrix
    return np.asarray(PANEL_FEATURES[target], dtype=float) @ coeffs_by_time


def _static_loo_fair_series(
    price_data: dict[str, Any],
    family: str,
    target: str,
    train_day: int,
    test_day: int,
) -> np.ndarray:
    hedges = [product for product in FAMILIES[family] if product != target]
    train_y = price_data[target][train_day]["mid"]
    train_x = np.column_stack([np.ones(len(train_y))] + [price_data[hedge][train_day]["mid"] for hedge in hedges])
    coeffs = _ridge_fit(train_x, train_y, ridge=1e-5)

    test_x = np.column_stack(
        [np.ones(len(price_data[target][test_day]["mid"]))] + [price_data[hedge][test_day]["mid"] for hedge in hedges]
    )
    fair = test_x @ coeffs
    family_matrix = np.column_stack([price_data[product][test_day]["mid"] for product in FAMILIES[family]])
    return _family_clip(fair, family_matrix)


def _rolling_loo_fair_series(
    price_data: dict[str, Any],
    family: str,
    target: str,
    train_day: int,
    test_day: int,
    lookback: int = 90,
    warmup: int = 45,
) -> np.ndarray:
    hedges = [product for product in FAMILIES[family] if product != target]
    train_y = price_data[target][train_day]["mid"][-lookback:]
    train_x = np.column_stack([np.ones(len(train_y))] + [price_data[hedge][train_day]["mid"][-lookback:] for hedge in hedges])

    n = len(price_data[target][test_day]["mid"])
    test_y = price_data[target][test_day]["mid"]
    test_x = np.column_stack([np.ones(n)] + [price_data[hedge][test_day]["mid"] for hedge in hedges])

    all_y = np.concatenate([train_y, test_y])
    all_x = np.vstack([train_x, test_x])
    seed_len = len(train_y)
    fair = np.full(n, np.nan, dtype=float)

    for i in range(n):
        end = seed_len + i
        start = max(0, end - lookback)
        if end - start < warmup:
            continue
        coeffs = _ridge_fit(all_x[start:end], all_y[start:end], ridge=0.05)
        fair[i] = float(test_x[i] @ coeffs)

    family_matrix = np.column_stack([price_data[product][test_day]["mid"] for product in FAMILIES[family]])
    valid = np.isfinite(fair)
    if np.any(valid):
        fair[valid] = _family_clip(fair[valid], family_matrix[valid])
    return fair


def _build_passive_fair_series(
    data: dict[str, Any],
    spec: dict[str, Any],
    model: str,
    train_day: int,
    test_day: int,
) -> np.ndarray:
    price_data = data["price_data"]
    family = str(spec["family"])
    target = str(spec["target"])
    if model == "panel_geometry":
        return _panel_geometry_fair_series(price_data, target, test_day)
    if model == "static_loo":
        return _static_loo_fair_series(price_data, family, target, train_day, test_day)
    if model == "rolling_loo_90":
        return _rolling_loo_fair_series(price_data, family, target, train_day, test_day)
    raise ValueError(f"Unknown passive model: {model}")


def _run_passive_fold(
    target_data: dict[str, np.ndarray],
    fair: np.ndarray,
    trade_tape: dict[tuple[int, str], dict[str, float]],
    target: str,
    edge: float,
    skew: float,
    cap: int,
    fill_through: float = 0.0,
) -> dict[str, Any]:
    timestamps = target_data["global_timestamp"]
    bid = target_data["bid"]
    ask = target_data["ask"]
    mid = target_data["mid"]
    spread = target_data["spread"]

    position = 0
    cash = 0.0
    quote_count = 0
    fill_events: list[dict[str, float]] = []
    equity_path: list[float] = []

    for i in range(len(mid)):
        if not math.isfinite(float(fair[i])):
            equity_path.append(float(cash + position * mid[i]))
            continue

        row_spread = max(1.0, float(spread[i]))
        reservation = float(fair[i]) - skew * row_spread * position
        buy_signal = reservation - mid[i] >= edge * row_spread
        sell_signal = mid[i] - reservation >= edge * row_spread
        quoted_buy = bool(buy_signal and position < cap)
        quoted_sell = bool(sell_signal and position > -cap)
        quote_count += int(quoted_buy) + int(quoted_sell)

        tape = trade_tape.get((int(timestamps[i]), target))
        if tape:
            buy_trigger = bid[i] - fill_through * row_spread
            sell_trigger = ask[i] + fill_through * row_spread
            if quoted_buy and tape["trade_price_min"] <= buy_trigger:
                cash -= bid[i]
                position += 1
                fill_events.append({"idx": float(i), "side": 1.0, "price": float(bid[i])})
            if quoted_sell and tape["trade_price_max"] >= sell_trigger:
                cash += ask[i]
                position -= 1
                fill_events.append({"idx": float(i), "side": -1.0, "price": float(ask[i])})

        equity_path.append(float(cash + position * mid[i]))

    final_idx = len(mid) - 1
    if position > 0:
        cash += position * bid[final_idx]
    elif position < 0:
        cash += position * ask[final_idx]

    markouts: dict[str, float] = {}
    for horizon in MARKOUT_HORIZONS:
        values = []
        for event in fill_events:
            idx = int(event["idx"])
            j = min(idx + horizon, len(mid) - 1)
            values.append(float(event["side"] * (mid[j] - event["price"])))
        markouts[f"m{horizon}"] = float(np.mean(values)) if values else float("nan")

    quarters = _quarter_pnls(equity_path, float(cash))
    return {
        "final_pnl": float(cash),
        "max_drawdown": _max_drawdown(equity_path),
        "quote_count": float(quote_count),
        "fill_count": float(len(fill_events)),
        "buy_fills": float(sum(1 for event in fill_events if event["side"] > 0)),
        "sell_fills": float(sum(1 for event in fill_events if event["side"] < 0)),
        "positive_quarters": int(sum(value > 0 for value in quarters)),
        "quarter_pnls": quarters,
        "markouts": markouts,
    }


def _passive_row_from_folds(
    spec: dict[str, Any],
    model: str,
    edge: float,
    skew: float,
    cap: int,
    fill_through: float,
    fold_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fold_labels = [label for label, _, _ in WALKFORWARD_FOLDS]
    pnls = [fold_results[label]["final_pnl"] for label in fold_labels]
    fills = [fold_results[label]["fill_count"] for label in fold_labels]
    drawdowns = [fold_results[label]["max_drawdown"] for label in fold_labels]
    positive_quarters = sum(int(fold_results[label]["positive_quarters"]) for label in fold_labels)
    markout_20 = [
        fold_results[label]["markouts"]["m20"]
        for label in fold_labels
        if math.isfinite(float(fold_results[label]["markouts"]["m20"]))
    ]
    return {
        "family": spec["family"],
        "target": spec["target"],
        "branch": f"passive:{spec['target']}",
        "model": model,
        "edge": float(edge),
        "skew": float(skew),
        "cap": int(cap),
        "fill_through": float(fill_through),
        "fit_day2_test_day3_pnl": float(pnls[0]),
        "fit_day3_test_day4_pnl": float(pnls[1]),
        "worst_fold_pnl": float(min(pnls)),
        "mean_fold_pnl": float(np.mean(pnls)),
        "positive_folds": int(sum(pnl > 0 for pnl in pnls)),
        "total_fills": float(sum(fills)),
        "min_fills": float(min(fills)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "positive_quarters": int(positive_quarters),
        "markout_20": float(np.mean(markout_20)) if markout_20 else float("nan"),
    }


def research_passive_branches(data: dict[str, Any]) -> dict[str, Any]:
    price_data = data["price_data"]
    trade_tape = data["trade_tape"]
    fair_cache: dict[tuple[str, str, str], np.ndarray] = {}
    grid_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    markout_rows: list[dict[str, Any]] = []

    for spec in PASSIVE_CANDIDATES:
        target = str(spec["target"])
        for model in spec["models"]:
            for fold_label, train_day, test_day in WALKFORWARD_FOLDS:
                fair_cache[(target, model, fold_label)] = _build_passive_fair_series(data, spec, model, train_day, test_day)

            for edge in EDGE_GRID:
                for skew in SKEW_GRID:
                    fold_results = {}
                    for fold_label, _, test_day in WALKFORWARD_FOLDS:
                        fold_results[fold_label] = _run_passive_fold(
                            price_data[target][test_day],
                            fair_cache[(target, model, fold_label)],
                            trade_tape,
                            target,
                            edge=edge,
                            skew=skew,
                            cap=int(spec["cap"]),
                            fill_through=0.0,
                        )
                    grid_rows.append(
                        _passive_row_from_folds(spec, model, edge, skew, int(spec["cap"]), 0.0, fold_results)
                    )

    summaries = []
    for spec in PASSIVE_CANDIDATES:
        target = str(spec["target"])
        candidate_rows = [row for row in grid_rows if row["target"] == target]
        best = max(
            candidate_rows,
            key=lambda row: (
                row["positive_folds"],
                row["worst_fold_pnl"],
                row["mean_fold_pnl"],
                row["total_fills"],
            ),
        )
        robust_rows = [row for row in candidate_rows if row["positive_folds"] == 2 and row["worst_fold_pnl"] > 0]
        neighbor_rows = [
            row
            for row in candidate_rows
            if row["model"] == best["model"]
            and abs(row["edge"] - best["edge"]) <= 0.25
            and abs(row["skew"] - best["skew"]) <= 0.25
        ]
        summaries.append(
            {
                **best,
                "robust_config_count": int(len(robust_rows)),
                "robust_config_share": float(len(robust_rows) / max(1, len(candidate_rows))),
                "neighbor_worst_median": float(np.median([row["worst_fold_pnl"] for row in neighbor_rows]))
                if neighbor_rows
                else float("nan"),
                "tested_configs": int(len(candidate_rows)),
            }
        )

        for fill_through in FILL_THROUGH_GRID:
            fold_results = {}
            for fold_label, _, test_day in WALKFORWARD_FOLDS:
                fold_results[fold_label] = _run_passive_fold(
                    price_data[target][test_day],
                    fair_cache[(target, best["model"], fold_label)],
                    trade_tape,
                    target,
                    edge=float(best["edge"]),
                    skew=float(best["skew"]),
                    cap=int(best["cap"]),
                    fill_through=fill_through,
                )
            stress_rows.append(
                _passive_row_from_folds(
                    spec,
                    str(best["model"]),
                    float(best["edge"]),
                    float(best["skew"]),
                    int(best["cap"]),
                    fill_through,
                    fold_results,
                )
            )
            if fill_through == 0.0:
                for horizon in MARKOUT_HORIZONS:
                    values = []
                    for fold_result in fold_results.values():
                        value = fold_result["markouts"][f"m{horizon}"]
                        if math.isfinite(float(value)):
                            values.append(float(value))
                    markout_rows.append(
                        {
                            "family": spec["family"],
                            "target": target,
                            "model": best["model"],
                            "horizon": horizon,
                            "avg_signed_markout": float(np.mean(values)) if values else float("nan"),
                        }
                    )

        for cap in sorted({2, int(spec["cap"]), 10}):
            fold_results = {}
            for fold_label, _, test_day in WALKFORWARD_FOLDS:
                fold_results[fold_label] = _run_passive_fold(
                    price_data[target][test_day],
                    fair_cache[(target, best["model"], fold_label)],
                    trade_tape,
                    target,
                    edge=float(best["edge"]),
                    skew=float(best["skew"]),
                    cap=cap,
                    fill_through=0.0,
                )
            cap_rows.append(
                _passive_row_from_folds(
                    spec,
                    str(best["model"]),
                    float(best["edge"]),
                    float(best["skew"]),
                    cap,
                    0.0,
                    fold_results,
                )
            )

    return {
        "grid_rows": grid_rows,
        "summary": summaries,
        "stress_rows": stress_rows,
        "cap_rows": cap_rows,
        "markout_rows": markout_rows,
    }


def _cashflow_with_slippage(delta: int, bid: float, ask: float, slippage: float) -> float:
    if delta > 0:
        return -delta * (ask + slippage)
    if delta < 0:
        return (-delta) * (bid - slippage)
    return 0.0


def _run_pair_fold(
    train_left: np.ndarray,
    train_right: np.ndarray,
    test_left: dict[str, np.ndarray],
    test_right: dict[str, np.ndarray],
    beta: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
    slippage: float = 0.0,
    position_limit: int = 10,
) -> dict[str, Any]:
    left_qty, right_qty = _pair_unit_sizes(beta, position_limit)
    hedge_sign = -1 if beta >= 0 else 1
    train_residual = train_left - beta * train_right
    history = list(train_residual[-lookback:])

    left_pos = 0
    right_pos = 0
    cash = 0.0
    trade_events = 0
    side_switches = 0
    current_side = 0
    active_steps = 0
    equity_path: list[float] = []

    mid_left = test_left["mid"]
    mid_right = test_right["mid"]
    bid_left = test_left["bid"]
    ask_left = test_left["ask"]
    bid_right = test_right["bid"]
    ask_right = test_right["ask"]

    for i in range(len(mid_left) - 1):
        previous_side = current_side
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
        if current_side != previous_side:
            side_switches += 1
        if current_side != 0:
            active_steps += 1

        target_left = current_side * left_qty
        target_right = current_side * hedge_sign * right_qty
        delta_left = target_left - left_pos
        delta_right = target_right - right_pos
        j = i + 1
        if delta_left:
            cash += _cashflow_with_slippage(delta_left, bid_left[j], ask_left[j], slippage)
            left_pos += delta_left
            trade_events += 1
        if delta_right:
            cash += _cashflow_with_slippage(delta_right, bid_right[j], ask_right[j], slippage)
            right_pos += delta_right
            trade_events += 1

        equity_path.append(float(cash + left_pos * mid_left[j] + right_pos * mid_right[j]))
        history.append(float(mid_left[i] - beta * mid_right[i]))
        if len(history) > lookback:
            history = history[-lookback:]

    final_idx = len(mid_left) - 1
    if left_pos > 0:
        cash += left_pos * (bid_left[final_idx] - slippage)
    elif left_pos < 0:
        cash += left_pos * (ask_left[final_idx] + slippage)
    if right_pos > 0:
        cash += right_pos * (bid_right[final_idx] - slippage)
    elif right_pos < 0:
        cash += right_pos * (ask_right[final_idx] + slippage)

    quarters = _quarter_pnls(equity_path, float(cash))
    return {
        "final_pnl": float(cash),
        "max_drawdown": _max_drawdown(equity_path),
        "trade_events": float(trade_events),
        "side_switches": float(side_switches),
        "active_steps": float(active_steps),
        "active_share": float(active_steps / max(1, len(mid_left) - 1)),
        "positive_quarters": int(sum(value > 0 for value in quarters)),
        "quarter_pnls": quarters,
    }


def _pair_row_from_folds(
    spec: dict[str, Any],
    lookback: int,
    entry_z: float,
    exit_z: float,
    slippage: float,
    fold_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fold_labels = [label for label, _, _ in WALKFORWARD_FOLDS]
    pnls = [fold_results[label]["final_pnl"] for label in fold_labels]
    return {
        "family": spec["family"],
        "pair": f"{spec['left']} vs {spec['right']}",
        "branch": f"pair:{spec['left']}|{spec['right']}",
        "lookback": int(lookback),
        "entry_z": float(entry_z),
        "exit_z": float(exit_z),
        "slippage": float(slippage),
        "fit_day2_test_day3_pnl": float(pnls[0]),
        "fit_day3_test_day4_pnl": float(pnls[1]),
        "worst_fold_pnl": float(min(pnls)),
        "mean_fold_pnl": float(np.mean(pnls)),
        "positive_folds": int(sum(pnl > 0 for pnl in pnls)),
        "mean_trade_events": float(np.mean([fold_results[label]["trade_events"] for label in fold_labels])),
        "mean_max_drawdown": float(np.mean([fold_results[label]["max_drawdown"] for label in fold_labels])),
        "positive_quarters": int(sum(fold_results[label]["positive_quarters"] for label in fold_labels)),
        "mean_active_share": float(np.mean([fold_results[label]["active_share"] for label in fold_labels])),
    }


def research_pair_branches(data: dict[str, Any]) -> dict[str, Any]:
    price_data = data["price_data"]
    grid_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []

    for spec in PAIR_CANDIDATES:
        left = str(spec["left"])
        right = str(spec["right"])
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

        for lookback in PAIR_LOOKBACK_GRID:
            for entry_z in PAIR_ENTRY_GRID:
                for exit_z in PAIR_EXIT_GRID:
                    fold_results = {}
                    for fold_label, _, _ in WALKFORWARD_FOLDS:
                        cached = fold_cache[fold_label]
                        fold_results[fold_label] = _run_pair_fold(
                            cached["train_left"],
                            cached["train_right"],
                            cached["test_left"],
                            cached["test_right"],
                            cached["beta"],
                            lookback=lookback,
                            entry_z=entry_z,
                            exit_z=exit_z,
                            slippage=0.0,
                        )
                    grid_rows.append(_pair_row_from_folds(spec, lookback, entry_z, exit_z, 0.0, fold_results))

    summaries = []
    for spec in PAIR_CANDIDATES:
        pair_name = f"{spec['left']} vs {spec['right']}"
        candidate_rows = [row for row in grid_rows if row["pair"] == pair_name]
        best = max(
            candidate_rows,
            key=lambda row: (
                row["positive_folds"],
                row["worst_fold_pnl"],
                row["mean_fold_pnl"],
                -row["mean_max_drawdown"],
            ),
        )
        robust_rows = [row for row in candidate_rows if row["positive_folds"] == 2 and row["worst_fold_pnl"] > 0]
        neighbor_rows = [
            row
            for row in candidate_rows
            if abs(row["lookback"] - best["lookback"]) <= 60
            and abs(row["entry_z"] - best["entry_z"]) <= 0.5
            and row["exit_z"] == best["exit_z"]
        ]
        summaries.append(
            {
                **best,
                "robust_config_count": int(len(robust_rows)),
                "robust_config_share": float(len(robust_rows) / max(1, len(candidate_rows))),
                "neighbor_worst_median": float(np.median([row["worst_fold_pnl"] for row in neighbor_rows]))
                if neighbor_rows
                else float("nan"),
                "tested_configs": int(len(candidate_rows)),
            }
        )

        left = str(spec["left"])
        right = str(spec["right"])
        for slippage in PAIR_SLIPPAGE_GRID:
            fold_results = {}
            for fold_label, train_day, test_day in WALKFORWARD_FOLDS:
                beta = _fit_beta(price_data[left][train_day]["mid"], price_data[right][train_day]["mid"])
                fold_results[fold_label] = _run_pair_fold(
                    price_data[left][train_day]["mid"],
                    price_data[right][train_day]["mid"],
                    price_data[left][test_day],
                    price_data[right][test_day],
                    beta,
                    lookback=int(best["lookback"]),
                    entry_z=float(best["entry_z"]),
                    exit_z=float(best["exit_z"]),
                    slippage=slippage,
                )
            stress_rows.append(
                _pair_row_from_folds(
                    spec,
                    int(best["lookback"]),
                    float(best["entry_z"]),
                    float(best["exit_z"]),
                    slippage,
                    fold_results,
                )
            )

    return {
        "grid_rows": grid_rows,
        "summary": summaries,
        "stress_rows": stress_rows,
    }


def _product_family(product: str) -> str:
    for family, products in FAMILIES.items():
        if product in products:
            return family
    return "OTHER"


def analyze_570532_log() -> dict[str, Any]:
    log_path = ROOT / "LOGS" / "570532.json"
    with log_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    product_stats: dict[str, dict[str, float]] = {}
    days = set()
    reader = csv.DictReader(io.StringIO(payload["activitiesLog"]), delimiter=";")
    for row in reader:
        product = row["product"]
        pnl = float(row["profit_and_loss"])
        days.add(int(row["day"]))
        stats = product_stats.setdefault(
            product,
            {
                "final_pnl": pnl,
                "peak_pnl": pnl,
                "max_drawdown": 0.0,
                "updates": 0.0,
            },
        )
        stats["final_pnl"] = pnl
        stats["peak_pnl"] = max(stats["peak_pnl"], pnl)
        stats["max_drawdown"] = max(stats["max_drawdown"], stats["peak_pnl"] - pnl)
        stats["updates"] += 1.0

    graph = []
    for row in csv.DictReader(io.StringIO(payload["graphLog"]), delimiter=";"):
        graph.append(float(row["value"]))
    graph_drawdown = _max_drawdown(graph)

    requested_products = {product for products in FAMILIES.values() for product in products}
    family_rows = []
    for family, products in FAMILIES.items():
        values = [product_stats.get(product, {}).get("final_pnl", 0.0) for product in products]
        family_rows.append(
            {
                "family": family,
                "log_final_pnl": float(sum(values)),
                "negative_products": int(sum(value < 0 for value in values)),
                "flat_products": int(sum(abs(value) < 1e-9 for value in values)),
            }
        )

    product_rows = [
        {
            "family": _product_family(product),
            "product": product,
            "log_final_pnl": float(stats["final_pnl"]),
            "log_max_drawdown": float(stats["max_drawdown"]),
        }
        for product, stats in product_stats.items()
        if product in requested_products
    ]
    product_rows.sort(key=lambda row: row["log_final_pnl"])

    positions = [
        item
        for item in payload.get("positions", [])
        if item.get("symbol") in requested_products and int(item.get("quantity", 0)) != 0
    ]
    positions.sort(key=lambda item: abs(int(item["quantity"])), reverse=True)

    return {
        "profit": float(payload["profit"]),
        "status": payload["status"],
        "days": sorted(days),
        "graph_max_drawdown": float(graph_drawdown),
        "family_rows": family_rows,
        "product_rows": product_rows,
        "bottom_products": product_rows[:12],
        "top_products": sorted(product_rows, key=lambda row: row["log_final_pnl"], reverse=True)[:12],
        "open_positions": positions,
    }


def _branch_decision(row: dict[str, Any], branch_type: str) -> str:
    if row["positive_folds"] < 2 or row["worst_fold_pnl"] <= 0:
        return "reject"
    if branch_type == "passive":
        if row["positive_quarters"] >= 5 and row["robust_config_share"] >= 0.20 and row["total_fills"] >= 20:
            return "implement"
        return "small_size"
    if row["positive_quarters"] >= 5 and row["robust_config_share"] >= 0.30:
        return "implement"
    return "small_size"


def build_decision_rows(passive: dict[str, Any], pairs: dict[str, Any], log: dict[str, Any]) -> list[dict[str, Any]]:
    log_by_product = {row["product"]: row for row in log["product_rows"]}
    rows = []
    for row in passive["summary"]:
        stress = [item for item in passive["stress_rows"] if item["target"] == row["target"] and item["fill_through"] == 0.25]
        stress_worst = stress[0]["worst_fold_pnl"] if stress else float("nan")
        rows.append(
            {
                "branch": row["branch"],
                "family": row["family"],
                "decision": _branch_decision(row, "passive"),
                "best_config": f"{row['model']} e={row['edge']} s={row['skew']} cap={row['cap']}",
                "worst_fold_pnl": row["worst_fold_pnl"],
                "mean_fold_pnl": row["mean_fold_pnl"],
                "stress_worst": stress_worst,
                "positive_quarters": row["positive_quarters"],
                "robust_share": row["robust_config_share"],
                "log_pnl": log_by_product.get(row["target"], {}).get("log_final_pnl", float("nan")),
            }
        )
    for row in pairs["summary"]:
        rows.append(
            {
                "branch": row["branch"],
                "family": row["family"],
                "decision": _branch_decision(row, "pair"),
                "best_config": f"lb={row['lookback']} entry={row['entry_z']} exit={row['exit_z']}",
                "worst_fold_pnl": row["worst_fold_pnl"],
                "mean_fold_pnl": row["mean_fold_pnl"],
                "stress_worst": next(
                    (
                        item["worst_fold_pnl"]
                        for item in pairs["stress_rows"]
                        if item["pair"] == row["pair"] and item["slippage"] == 0.5
                    ),
                    float("nan"),
                ),
                "positive_quarters": row["positive_quarters"],
                "robust_share": row["robust_config_share"],
                "log_pnl": float("nan"),
            }
        )
    rows.sort(key=lambda row: (row["decision"] != "implement", -row["worst_fold_pnl"]))
    return rows


def build_combined_rows(decisions: list[dict[str, Any]], passive: dict[str, Any], pairs: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [row for row in decisions if row["decision"] in {"implement", "small_size"}]
    fold_labels = [label for label, _, _ in WALKFORWARD_FOLDS]
    out = []
    for fold_index, fold_label in enumerate(fold_labels):
        total = 0.0
        implement_total = 0.0
        for row in selected:
            if row["branch"].startswith("passive:"):
                target = row["branch"].split(":", 1)[1]
                source = next(item for item in passive["summary"] if item["target"] == target)
            else:
                source = next(item for item in pairs["summary"] if item["branch"] == row["branch"])
            pnl = source["fit_day2_test_day3_pnl"] if fold_index == 0 else source["fit_day3_test_day4_pnl"]
            if row["decision"] == "small_size":
                pnl *= 0.5
            total += pnl
            if row["decision"] == "implement":
                implement_total += pnl
        out.append(
            {
                "fold": fold_label,
                "candidate_portfolio_pnl": float(total),
                "full_size_only_pnl": float(implement_total),
                "branch_count": int(len(selected)),
            }
        )
    return out


def _top_rows(rows: list[dict[str, Any]], count: int = 8) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["positive_folds"], row["worst_fold_pnl"], row["mean_fold_pnl"]), reverse=True)[:count]


def format_report(results: dict[str, Any]) -> str:
    passive = results["passive"]
    pairs = results["pairs"]
    log = results["log_570532"]
    decisions = results["decisions"]
    combined = results["combined"]

    lines = [
        "# Round 5 optimal algorithm research",
        "",
        "This report is a targeted robustness pass after the broad `family_strategy_scan.py` scan. It tests the strongest candidate branches across walk-forward folds, parameter neighborhoods, fill stress, markouts, cap sensitivity, slippage, and current-bot log diagnostics from `LOGS/570532.json`.",
        "",
        "## Executive decision matrix",
        _markdown_table(
            decisions,
            [
                ("branch", "branch"),
                ("family", "family"),
                ("decision", "decision"),
                ("best_config", "best config"),
                ("worst_fold_pnl", "worst fold"),
                ("mean_fold_pnl", "mean fold"),
                ("stress_worst", "stress worst"),
                ("positive_quarters", "+ quarters"),
                ("robust_share", "robust share"),
                ("log_pnl", "570532 pnl"),
            ],
            digits=2,
        ),
        "",
        "Decision rule: implement only if both walk-forward folds are positive, the surrounding parameter neighborhood is not fragile, and intraday quarter stability is acceptable. `small_size` means the branch has positive evidence but should start as a capped overlay rather than a full-size engine.",
        "",
        "Passive `stress worst` uses a strict fill-through requirement. The Round 5 public tape mostly fills at the touch, so fill-through values above zero often produce no fills. Treat that column as queue/fill-availability sensitivity, not as a direct rejection test for passive quoting.",
        "",
        "## Candidate portfolio approximation",
        _markdown_table(
            combined,
            [
                ("fold", "fold"),
                ("candidate_portfolio_pnl", "portfolio pnl"),
                ("full_size_only_pnl", "full-size only"),
                ("branch_count", "branches"),
            ],
            digits=1,
        ),
        "",
        "This approximation sums non-overlapping branch simulations. It is not a replacement for a full trader backtest because live branches can still compete for shared product limits and alpha overlays.",
        "",
        "## Passive fair-value branches",
        _markdown_table(
            sorted(passive["summary"], key=lambda row: row["worst_fold_pnl"], reverse=True),
            [
                ("target", "target"),
                ("model", "model"),
                ("edge", "edge"),
                ("skew", "skew"),
                ("cap", "cap"),
                ("fit_day2_test_day3_pnl", "d2->d3"),
                ("fit_day3_test_day4_pnl", "d3->d4"),
                ("worst_fold_pnl", "worst"),
                ("total_fills", "fills"),
                ("positive_quarters", "+ quarters"),
                ("robust_config_count", "robust cfgs"),
                ("robust_config_share", "robust share"),
                ("neighbor_worst_median", "neighbor median"),
                ("markout_20", "m20"),
            ],
            digits=2,
        ),
        "",
        "### Passive fill stress",
        _markdown_table(
            sorted(passive["stress_rows"], key=lambda row: (row["target"], row["fill_through"])),
            [
                ("target", "target"),
                ("model", "model"),
                ("fill_through", "fill-through"),
                ("worst_fold_pnl", "worst"),
                ("mean_fold_pnl", "mean"),
                ("total_fills", "fills"),
                ("positive_quarters", "+ quarters"),
            ],
            digits=2,
        ),
        "",
        "### Passive cap sensitivity",
        _markdown_table(
            sorted(passive["cap_rows"], key=lambda row: (row["target"], row["cap"])),
            [
                ("target", "target"),
                ("cap", "cap"),
                ("worst_fold_pnl", "worst"),
                ("mean_fold_pnl", "mean"),
                ("total_fills", "fills"),
                ("mean_max_drawdown", "mean dd"),
            ],
            digits=2,
        ),
        "",
        "### Passive signed markouts",
        _markdown_table(
            sorted(passive["markout_rows"], key=lambda row: (row["target"], row["horizon"])),
            [
                ("target", "target"),
                ("model", "model"),
                ("horizon", "bars"),
                ("avg_signed_markout", "avg signed markout"),
            ],
            digits=3,
        ),
        "",
        "## Residual pair branches",
        _markdown_table(
            sorted(pairs["summary"], key=lambda row: row["worst_fold_pnl"], reverse=True),
            [
                ("pair", "pair"),
                ("lookback", "lookback"),
                ("entry_z", "entry"),
                ("exit_z", "exit"),
                ("fit_day2_test_day3_pnl", "d2->d3"),
                ("fit_day3_test_day4_pnl", "d3->d4"),
                ("worst_fold_pnl", "worst"),
                ("mean_trade_events", "trades"),
                ("mean_max_drawdown", "mean dd"),
                ("positive_quarters", "+ quarters"),
                ("robust_config_count", "robust cfgs"),
                ("robust_config_share", "robust share"),
                ("neighbor_worst_median", "neighbor median"),
            ],
            digits=2,
        ),
        "",
        "### Pair slippage stress",
        _markdown_table(
            sorted(pairs["stress_rows"], key=lambda row: (row["pair"], row["slippage"])),
            [
                ("pair", "pair"),
                ("slippage", "slippage"),
                ("worst_fold_pnl", "worst"),
                ("mean_fold_pnl", "mean"),
                ("mean_trade_events", "trades"),
                ("mean_max_drawdown", "mean dd"),
                ("positive_quarters", "+ quarters"),
            ],
            digits=2,
        ),
        "",
        "## Current bot log diagnostics: 570532",
        f"- Status: `{log['status']}`.",
        f"- Reported submission profit: `{log['profit']:.2f}`.",
        f"- Activities log days present: `{log['days']}`.",
        f"- Total graph max drawdown: `{log['graph_max_drawdown']:.2f}`.",
        f"- Requested-family open positions at the end: `{len(log['open_positions'])}`.",
        "",
        "### 570532 family PnL",
        _markdown_table(
            sorted(log["family_rows"], key=lambda row: row["log_final_pnl"], reverse=True),
            [
                ("family", "family"),
                ("log_final_pnl", "log final pnl"),
                ("negative_products", "negative products"),
                ("flat_products", "flat products"),
            ],
            digits=2,
        ),
        "",
        "### 570532 weakest requested products",
        _markdown_table(
            log["bottom_products"],
            [
                ("family", "family"),
                ("product", "product"),
                ("log_final_pnl", "log final pnl"),
                ("log_max_drawdown", "log max dd"),
            ],
            digits=2,
        ),
        "",
        "### 570532 strongest requested products",
        _markdown_table(
            log["top_products"],
            [
                ("family", "family"),
                ("product", "product"),
                ("log_final_pnl", "log final pnl"),
                ("log_max_drawdown", "log max dd"),
            ],
            digits=2,
        ),
        "",
        "## Implementation blueprint for the next trader",
        "",
        "1. Keep the profitable existing core, but separate hard-avoid products from branch-specific allowlists. `PANEL_2X2` should stay blocked from the generic engine while the dedicated panel-geometry passive branch is allowed to quote it.",
        "2. Add only the `implement` branches at normal size. Add `small_size` branches at half-size or with stricter live markout throttles.",
        "3. For passive branches, track product-specific live fill markouts and realized branch PnL. Use 20-bar markouts for UV/SNACK/GALAXY, but use shorter 5-10 bar markouts for `PANEL_2X2` because the panel branch earns the spread before the longer-horizon reversal in this scan.",
        "4. For pair branches, use the researched lookback/entry/exit values but cap overlay strength. Pair signals should alter reservation prices or target inventory, not force broad aggressive basket trades.",
        "5. Add branch-level PnL and markout accounting in `traderData` so the next log can tell us which branch made each trade.",
        "",
        "## Research caveats",
        "",
        "- Passive fills use public trade-through checks, which are conservative but not identical to exchange queue priority.",
        "- Pair branches cross the next bar in simulation; live implementation as alpha overlays will usually realize smaller but smoother edge.",
        "- The combined portfolio table is additive because the selected branches do not intentionally share product limits. Full validation still requires `prosperity4bt` once runtime is acceptable.",
    ]
    return "\n".join(lines)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_research() -> dict[str, Any]:
    data = load_round5_data()
    passive = research_passive_branches(data)
    pairs = research_pair_branches(data)
    log = analyze_570532_log()
    decisions = build_decision_rows(passive, pairs, log)
    combined = build_combined_rows(decisions, passive, pairs)
    results = {
        "data_dir": data["data_dir"],
        "passive": passive,
        "pairs": pairs,
        "log_570532": log,
        "decisions": decisions,
        "combined": combined,
        "top_passive_configs": _top_rows(passive["grid_rows"], 20),
        "top_pair_configs": _top_rows(pairs["grid_rows"], 20),
    }
    REPORT_PATH.write_text(format_report(results), encoding="utf-8")
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    return results


if __name__ == "__main__":
    research = run_research()
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {RESULTS_PATH}")
    print("Decision summary:")
    for row in research["decisions"]:
        print(
            f"- {row['decision']:10s} {row['branch']:55s} "
            f"worst={row['worst_fold_pnl']:.1f} mean={row['mean_fold_pnl']:.1f} "
            f"stress={row['stress_worst']:.1f}"
        )
