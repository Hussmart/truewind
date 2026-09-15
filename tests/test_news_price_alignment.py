from truewind.data.news_data import generate_synthetic_news_and_price
from truewind.news_price_alignment import NewsPriceAligner


def test_confirms_mostly_genuine_events():
    prices, news = generate_synthetic_news_and_price(
        n_genuine_events=4, n_noise_news=12, reaction_lag_days=1, jump_magnitude=0.07, seed=2
    )
    aligner = NewsPriceAligner()
    result = aligner.fit(prices, news)

    assert len(result.confirmed) > 0

    confirmed_news_idx = {c.news_index for c in result.confirmed}
    genuine_confirmed = sum(1 for i in confirmed_news_idx if news.loc[i, "genuine"])

    assert genuine_confirmed / len(confirmed_news_idx) >= 0.5


def test_no_candidates_returns_empty_result():
    prices, _ = generate_synthetic_news_and_price(n_genuine_events=0, n_noise_news=0)
    import pandas as pd

    empty_news = pd.DataFrame(columns=["timestamp", "sentiment"])
    aligner = NewsPriceAligner()
    result = aligner.fit(prices, empty_news)

    assert len(result.candidates) == 0
    assert len(result.confirmed) == 0
