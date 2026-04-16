"""Quick targeted data analysis to confirm / extend insight docs."""
from __future__ import annotations
import pandas as pd
import numpy as np

frames = []
for d in (-2, -1, 0):
    df = pd.read_csv(f"prices_round_1_day_{d}.csv", sep=";")
    df["day_idx"] = d
    frames.append(df)
raw = pd.concat(frames, ignore_index=True)
raw = raw[raw["mid_price"] > 0].copy()

# Per-product stats
for prod in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
    sub = raw[raw["product"] == prod].copy().sort_values(["day_idx", "timestamp"]).reset_index(drop=True)
    sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
    sub["microprice"] = (sub["bid_price_1"] * sub["ask_volume_1"] + sub["ask_price_1"] * sub["bid_volume_1"]) / (sub["bid_volume_1"] + sub["ask_volume_1"])
    print(f"\n=== {prod} ===")
    print(f"n={len(sub)}  mid mean={sub['mid_price'].mean():.3f}  std={sub['mid_price'].std():.3f}")
    print(f"spread mean={sub['spread'].mean():.3f}  median={sub['spread'].median()}")
    print(f"bid_vol_1 mean={sub['bid_volume_1'].mean():.2f}  ask_vol_1 mean={sub['ask_volume_1'].mean():.2f}")

    # Drift regression per day
    for d in sorted(sub["day_idx"].unique()):
        s = sub[sub["day_idx"] == d]
        slope = np.polyfit(s["timestamp"].values, s["mid_price"].values, 1)[0]
        print(f"day {d}: open={s.iloc[0]['mid_price']:.1f} close={s.iloc[-1]['mid_price']:.1f}  slope/ts={slope:.6f}")

    # Autocorr
    r = sub["mid_price"].diff().dropna()
    print(f"lag-1 autocorr of dmid: {r.autocorr():.4f}")

# Pepper-specific: detrended residual
pep = raw[raw["product"] == "INTARIAN_PEPPER_ROOT"].copy().sort_values(["day_idx", "timestamp"]).reset_index(drop=True)
# Within-day detrend with slope 0.01/ts from opening
resid = []
for d, s in pep.groupby("day_idx"):
    slope = 0.01
    anchor = s["mid_price"].iloc[0]
    t0 = s["timestamp"].iloc[0]
    r = s["mid_price"] - (anchor + slope * (s["timestamp"] - t0))
    resid.extend(r.tolist())
resid = np.array(resid)
print(f"\n--- Pepper residual vs slope=0.01 ---")
print(f"mean={resid.mean():.3f}  std={resid.std():.3f}  min={resid.min():.2f}  max={resid.max():.2f}")
print(f"fraction within ±5: {(np.abs(resid) <= 5).mean()*100:.1f}%")
print(f"fraction within ±10: {(np.abs(resid) <= 10).mean()*100:.1f}%")

# Check Pepper spread distribution precisely
pep["spread"] = pep["ask_price_1"] - pep["bid_price_1"]
print("\nPepper spread distribution:")
print(pep["spread"].value_counts().sort_index().head(20))

# Osmium spread
osm = raw[raw["product"] == "ASH_COATED_OSMIUM"].copy()
osm["spread"] = osm["ask_price_1"] - osm["bid_price_1"]
print("\nOsmium spread distribution:")
print(osm["spread"].value_counts().sort_index().head(15))

# Pepper: how often is ask BELOW the trend-projected fair?
print("\n--- Pepper: ask < trend_fair fraction ---")
for d, s in pep.groupby("day_idx"):
    t0 = s["timestamp"].iloc[0]
    anchor = s["mid_price"].iloc[0]
    fair = anchor + 0.01 * (s["timestamp"] - t0)
    frac_ask = ((s["ask_price_1"] < fair)).mean()
    frac_bid = ((s["bid_price_1"] > fair)).mean()
    print(f"day {d}: ask<fair={frac_ask*100:.1f}% bid>fair={frac_bid*100:.1f}%")
