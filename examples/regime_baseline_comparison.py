"""Demo: consensus regime detection vs. a naive z-score baseline, side by side.

Answers "why does the consensus method flag so few windows?" concretely: it
requires a window's volatility to *also* be jointly anomalous, not just its
direction. A naive z-score-on-mean-return baseline flags every sufficiently
negative stretch -- including ordinary grinding declines -- while the
consensus method isolates only the episode with an anomalous volatility
signature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from truewind.data.price_data import fetch_price_history, generate_synthetic_price_series
from truewind.regime import RegimeAnomalyDetector, naive_zscore_baseline
from truewind.viz import plot_regime_comparison

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", type=str, default="BTC-USD")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--z-threshold", type=float, default=1.5)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data instead of a real ticker")
    parser.add_argument("--save", action="store_true", help="save chart to examples/output/")
    args = parser.parse_args()

    prices = generate_synthetic_price_series() if args.synthetic else fetch_price_history(args.ticker, period=args.period)

    consensus = RegimeAnomalyDetector(window_size=args.window_size).fit(prices)
    baseline = naive_zscore_baseline(prices, window_size=args.window_size, z_threshold=args.z_threshold)

    print(f"Consensus method flagged: {consensus.anomaly_mask.sum()} windows")
    print(f"Naive z-score baseline flagged: {baseline.anomaly_mask.sum()} windows")
    print(f"\nNaive-only episodes (flagged by baseline, missed by consensus):")
    consensus_ts = set(consensus.anomaly_timestamps)
    naive_only = sorted(t for t in baseline.anomaly_timestamps if t not in consensus_ts)
    for t in naive_only:
        print(f"  - {t.date()}")

    save_path = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path = str(OUTPUT_DIR / "regime_baseline_comparison.png")

    title = f"Consensus vs. Naive Z-Score — {args.ticker}" if not args.synthetic else "Consensus vs. Naive Z-Score — synthetic data"
    plot_regime_comparison(prices, consensus, baseline, title=title, save_path=save_path)
    if save_path:
        print(f"Chart saved to {save_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
