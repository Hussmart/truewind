"""Demo: robustly align news events to the price moves they actually caused.

Uses a synthetic price series plus a mix of genuine and coincidental news
events (ground truth known) so the demo is offline and reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from truewind.data.news_data import generate_synthetic_news_and_price
from truewind.news_price_alignment import NewsPriceAligner
from truewind.viz import plot_news_price_alignment

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="save chart to examples/output/")
    args = parser.parse_args()

    prices, news = generate_synthetic_news_and_price()
    n_genuine = int(news["genuine"].sum())
    n_noise = len(news) - n_genuine
    print(f"Generated {len(news)} news events ({n_genuine} genuine, {n_noise} noise)")

    aligner = NewsPriceAligner()
    result = aligner.fit(prices, news)

    print(f"Candidate (news, price-jump) pairs considered: {len(result.candidates)}")
    print(f"Confirmed as genuine by consensus: {len(result.confirmed)}")

    confirmed_news_idx = {c.news_index for c in result.confirmed}
    true_positives = sum(1 for i in confirmed_news_idx if news.loc[i, "genuine"])
    print(f"  of which correctly match a truly genuine news event: {true_positives}/{len(confirmed_news_idx)}")

    save_path = None
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        save_path = str(OUTPUT_DIR / "news_price_alignment.png")

    plot_news_price_alignment(prices, result, save_path=save_path)
    if save_path:
        print(f"Chart saved to {save_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
