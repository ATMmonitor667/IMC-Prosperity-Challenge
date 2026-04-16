"""Quick data analysis of Round 1 price/trade CSVs.

Goal: figure out which kind of strategy (market-making, mean-reversion,
momentum, pair trade) best fits each product, based on statistical
properties of the mid-price / spread / returns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROUND1 = Path(__file__).parent
DAYS = [-2, -1, 0]


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(ROUND1 / f"prices_round_1_day_{d}.csv", sep=";")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["day", "timestamp", "product"], kind="stable").reset_index(drop=True)
    df = df[(df["mid_price"] > 0) & df["mid_price"].notna()].copy()
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    return df


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(ROUND1 / f"trades_round_1_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def describe_product(prices: pd.DataFrame, product: str) -> dict:
    df = prices[prices["product"] == product].copy()
    df = df.sort_values(["day", "timestamp"])
    mids = df["mid_price"].dropna().to_numpy()
    rets = np.diff(mids) / mids[:-1]
    rets = rets[np.isfinite(rets)]

    spreads = df["spread"].dropna().to_numpy()

    mean_mid = float(np.mean(mids))
    std_mid = float(np.std(mids))
    dev = mids - mean_mid

    zero_crossings = int(np.sum(np.diff(np.sign(dev)) != 0))
    autocorr1 = float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 2 else float("nan")

    deviations_above_1sigma = float(np.mean(np.abs(dev) > std_mid))

    return {
        "product": product,
        "n_rows": int(len(df)),
        "mean_mid": mean_mid,
        "std_mid": std_mid,
        "min_mid": float(np.min(mids)),
        "max_mid": float(np.max(mids)),
        "range": float(np.max(mids) - np.min(mids)),
        "mean_spread": float(np.mean(spreads)),
        "median_spread": float(np.median(spreads)),
        "pct_spread_eq_1": float(np.mean(spreads == 1)),
        "pct_spread_le_2": float(np.mean(spreads <= 2)),
        "ret_std": float(np.std(rets)),
        "ret_autocorr_lag1": autocorr1,
        "mean_crossings": zero_crossings,
        "pct_above_1sigma": deviations_above_1sigma,
    }


def correlation_between_products(prices: pd.DataFrame, a: str, b: str) -> dict:
    pa = prices[prices["product"] == a].set_index(["day", "timestamp"])["mid_price"]
    pb = prices[prices["product"] == b].set_index(["day", "timestamp"])["mid_price"]
    joined = pd.concat([pa, pb], axis=1, keys=[a, b]).dropna()
    corr_level = float(joined[a].corr(joined[b]))
    ret_a = joined[a].pct_change().dropna()
    ret_b = joined[b].pct_change().dropna()
    aligned = pd.concat([ret_a, ret_b], axis=1, keys=[a, b]).dropna()
    corr_ret = float(aligned[a].corr(aligned[b]))
    return {
        "n_joined": int(len(joined)),
        "corr_mid_level": corr_level,
        "corr_returns": corr_ret,
    }


def mean_reversion_backtest(prices: pd.DataFrame, product: str, z_entry: float = 1.0) -> dict:
    """Toy z-score mean reversion on the full window's mean.

    Not a real backtest (uses in-sample mean), just to gauge if prices revert.
    Simulates: if z > z_entry -> short 1 unit; if z < -z_entry -> long 1 unit; exit at z ~ 0.
    Returns total PnL in raw price units and number of trades.
    """
    df = prices[prices["product"] == product].sort_values(["day", "timestamp"]).copy()
    mids = df["mid_price"].dropna().to_numpy()
    mu = np.mean(mids)
    sigma = np.std(mids)
    if sigma == 0:
        return {"pnl": 0.0, "trades": 0}

    z = (mids - mu) / sigma
    position = 0
    entry_price = 0.0
    pnl = 0.0
    trades = 0
    for i, (zi, pi) in enumerate(zip(z, mids)):
        if position == 0:
            if zi > z_entry:
                position = -1
                entry_price = pi
            elif zi < -z_entry:
                position = 1
                entry_price = pi
        else:
            exit_now = (position == -1 and zi <= 0) or (position == 1 and zi >= 0)
            if exit_now:
                pnl += position * (pi - entry_price)
                position = 0
                trades += 1
    return {"pnl": float(pnl), "trades": int(trades)}


def momentum_backtest(prices: pd.DataFrame, product: str, lookback: int = 20) -> dict:
    """Toy momentum: hold +1 if last 'lookback' return is positive, else -1.

    Uses mid-to-mid next-step return for PnL.
    """
    df = prices[prices["product"] == product].sort_values(["day", "timestamp"]).copy()
    mids = df["mid_price"].dropna().to_numpy()
    if len(mids) <= lookback + 1:
        return {"pnl": 0.0, "trades": 0}
    signals = np.sign(mids[lookback:] - mids[:-lookback])
    next_moves = np.diff(mids)[lookback - 1 : -1]
    m = min(len(signals) - 1, len(next_moves))
    signals = signals[:m]
    next_moves = next_moves[:m]
    pnl = float(np.sum(signals * next_moves))
    trades = int(np.sum(np.abs(np.diff(signals))))
    return {"pnl": pnl, "trades": trades}


def market_make_backtest(prices: pd.DataFrame, product: str, fair_window: int = 50,
                         edge: int = 1, max_pos: int = 80) -> dict:
    """Toy market-making sim.

    - Fair price = rolling mean of mid over `fair_window` ticks.
    - Each tick we 'quote' fair-edge and fair+edge.
    - Fill model: if best_ask <= our_bid AND pos < max, we buy 1 at our_bid.
                  if best_bid >= our_ask AND pos > -max, we sell 1 at our_ask.
    - PnL is marked against final mid.
    """
    df = prices[prices["product"] == product].sort_values(["day", "timestamp"]).copy()
    df["fair"] = df["mid_price"].rolling(fair_window, min_periods=1).mean()
    pos = 0
    cash = 0.0
    fills = 0
    for bid, ask, fair in zip(df["bid_price_1"].to_numpy(),
                               df["ask_price_1"].to_numpy(),
                               df["fair"].to_numpy()):
        if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(fair)):
            continue
        my_bid = fair - edge
        my_ask = fair + edge
        if ask <= my_bid and pos < max_pos:
            cash -= ask
            pos += 1
            fills += 1
        if bid >= my_ask and pos > -max_pos:
            cash += bid
            pos -= 1
            fills += 1
    last_mid = df["mid_price"].iloc[-1]
    pnl = cash + pos * last_mid
    return {"pnl": float(pnl), "fills": int(fills), "final_pos": int(pos)}


def main() -> None:
    prices = load_prices()
    trades = load_trades()

    products = sorted(prices["product"].dropna().unique().tolist())
    print(f"Products detected: {products}\n")

    rows = [describe_product(prices, p) for p in products]
    stats = pd.DataFrame(rows).set_index("product")

    print("=== Per-product summary ===")
    with pd.option_context("display.float_format", lambda x: f"{x:,.4f}"):
        print(stats.T)
    print()

    if len(products) >= 2:
        a, b = products[0], products[1]
        corr = correlation_between_products(prices, a, b)
        print(f"=== Cross-product correlation: {a} vs {b} ===")
        print(corr)
        print()

    print("=== Toy mean-reversion (z>1 entry, exit at z=0) ===")
    for p in products:
        print(p, mean_reversion_backtest(prices, p, z_entry=1.0))
    print()

    print("=== Toy momentum (sign of 20-step return) ===")
    for p in products:
        print(p, momentum_backtest(prices, p, lookback=20))
    print()

    print("=== Toy market making (fair = 50-tick SMA, edge=1, max_pos=80) ===")
    for p in products:
        print(p, market_make_backtest(prices, p, fair_window=50, edge=1, max_pos=80))
    print()

    print("=== Trade tape summary ===")
    for p in products:
        t = trades[trades["symbol"] == p]
        print(f"{p}: {len(t)} public trades, avg qty {t['quantity'].abs().mean():.2f}, "
              f"avg price {t['price'].mean():.2f}")


if __name__ == "__main__":
    main()
