"""Demo: flag anomalous / regime-break windows in a price series.

By default uses a synthetic price series with injected shocks so the demo
runs offline and reproducibly. Pass --ticker to fetch a real symbol via
yfinance instead (requires network access).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from truewind.data.price_data import fetch_price_history, generate_synthetic_price_series
from truewind.regime import RegimeAnomalyDetector
from truewind.viz import plot_regime_anomalies

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", type=str, default=None, help="e.g. AAPL, BTC-USD (requires internet)")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--save", action="store_true", help="save chart to examples/output/")
    args = parser.parse_args()

    if args.ticker:
        prices = fetch_price_history(args.ticker, period=args.period)
        title = f"Regime-Break Detection — {args.ticker}, real data via yfinance"
    else:
        prices = generate_synthetic_price_series()
        injected = prices.attrs.get("injected_anomaly_positions", [])
        print(f"Using synthetic series. Injected shock indices: {injected}")
        title = "Regime-Break Detection — synthetic data"

    detector = RegimeAnomalyDetector(window_size=args.window_size)
    result = detector.fit(prices)

    print(f"Windows analyzed: {len(result.anomaly_mask)}")
    print(f"Flagged as anomaly / regime break: {result.anomaly_mask.sum()}")
    for ts in result.anomaly_timestamps:
        print(f"  - {ts.date()}")

    save_path = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path = str(OUTPUT_DIR / "regime_anomalies.png")

    plot_regime_anomalies(prices, result, title=title, save_path=save_path)
    if save_path:
        print(f"Chart saved to {save_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
