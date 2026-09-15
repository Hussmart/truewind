"""Demo: flag exchanges whose quoted price disagrees with the group consensus.

By default uses a synthetic multi-exchange price panel with one exchange
biased for a window of time, so the demo runs offline and reproducibly.

Pass --history to fetch real historical daily closes from several real
exchanges via ccxt (the recommended way to see this on real data), or
--live for a single real-time snapshot instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from truewind.data.exchange_data import (
    fetch_multi_exchange_ohlcv_history,
    fetch_multi_exchange_snapshot,
    generate_synthetic_exchange_snapshots,
)
from truewind.exchange_consensus import ExchangeConsensusDetector
from truewind.viz import plot_exchange_consensus

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="fetch real historical daily closes via ccxt")
    parser.add_argument("--live", action="store_true", help="fetch a single live snapshot via ccxt")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--days", type=int, default=180, help="days of history for --history")
    parser.add_argument("--save", action="store_true", help="save chart to examples/output/")
    args = parser.parse_args()

    if args.history:
        snapshots = fetch_multi_exchange_ohlcv_history(args.symbol, limit=args.days)
        print(f"Fetched {len(snapshots)} days of real history from: {list(snapshots.columns)}")
    elif args.live:
        snapshot = fetch_multi_exchange_snapshot(args.symbol)
        print("Live snapshot:")
        print(snapshot)
        snapshots = snapshot.to_frame().T
    else:
        snapshots = generate_synthetic_exchange_snapshots()
        manipulated = snapshots.attrs.get("manipulated_exchange")
        window = snapshots.attrs.get("manipulation_window")
        print(f"Using synthetic panel. Injected bias on '{manipulated}' during index range {window}")

    detector = ExchangeConsensusDetector()
    result = detector.fit(snapshots)

    print("\nOutlier rate per exchange (fraction of timestamps flagged):")
    print(result.outlier_rate.to_string())

    save_path = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path = str(OUTPUT_DIR / "exchange_consensus.png")

    title = "Cross-Exchange Price Consensus"
    if args.history:
        title += f" — {args.symbol}, real daily closes"
    plot_exchange_consensus(snapshots, result, title=title, save_path=save_path)
    if save_path:
        print(f"Chart saved to {save_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
