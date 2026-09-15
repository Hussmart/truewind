"""Validate the regime detector across multiple real, unrelated assets.

A single asset (BTC-USD) is not enough to claim the method behaves
consistently -- this runs it, the naive z-score baseline, and an
independent changepoint-detection cross-check (PELT via `ruptures`,
a widely-used library, not something this project invented) across a
diverse basket of real tickers, and reports whether the pattern seen on
BTC-USD (consensus flags fewer, more targeted windows; an independent
method confirms the flagged episode is a genuine structural break) holds
up elsewhere.

This is still not a trading backtest or a statistical significance test
across a large asset universe -- see the README for what this is and
isn't a claim about.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from truewind.data.price_data import fetch_price_history
from truewind.regime import (
    RegimeAnomalyDetector,
    changepoint_breakpoints,
    episode_confirmed_by_changepoint,
    naive_zscore_baseline,
)
from truewind.viz import plot_multi_asset_summary

DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "NVDA", "SPY", "GLD", "MSTR"]
OUTPUT_DIR = Path(__file__).parent / "output"


def validate_ticker(ticker: str, period: str, window_size: int) -> dict:
    prices = fetch_price_history(ticker, period=period)
    consensus = RegimeAnomalyDetector(window_size=window_size).fit(prices)
    baseline = naive_zscore_baseline(prices, window_size=window_size)
    bkps = changepoint_breakpoints(prices)
    confirmed = episode_confirmed_by_changepoint(consensus.anomaly_timestamps, bkps)

    flagged_std = consensus.window_stds[consensus.anomaly_mask].mean() if consensus.anomaly_mask.any() else float("nan")
    baseline_std = consensus.window_stds[~consensus.anomaly_mask].mean()

    return {
        "ticker": ticker,
        "n_windows": len(consensus.anomaly_mask),
        "consensus_flagged": int(consensus.anomaly_mask.sum()),
        "naive_flagged": int(baseline.anomaly_mask.sum()),
        "flagged_vol_ratio": round(flagged_std / baseline_std, 2) if consensus.anomaly_mask.any() else None,
        "ruptures_confirms_episode": confirmed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--save", action="store_true", help="save chart + CSV to examples/output/")
    args = parser.parse_args()

    rows = []
    for ticker in args.tickers:
        try:
            rows.append(validate_ticker(ticker, args.period, args.window_size))
            print(f"{ticker}: OK")
        except Exception as e:
            print(f"{ticker}: FAILED ({e})")

    df = pd.DataFrame(rows).set_index("ticker")
    print()
    print(df.to_string())

    n_confirmed = df["ruptures_confirms_episode"].sum()
    n_with_episode = (df["consensus_flagged"] > 0).sum()
    print(f"\n{n_confirmed}/{n_with_episode} assets with a flagged episode were independently confirmed by ruptures/PELT.")
    print(f"Consensus flagged fewer windows than the naive baseline on {(df['consensus_flagged'] <= df['naive_flagged']).sum()}/{len(df)} assets.")

    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / "multi_asset_validation.csv")
        plot_multi_asset_summary(df, save_path=str(OUTPUT_DIR / "multi_asset_validation.png"))
        print(f"\nSaved to {OUTPUT_DIR}/multi_asset_validation.{{csv,png}}")
    else:
        plot_multi_asset_summary(df)
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
