"""Synthetic news-event data for the news-to-price alignment demo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_news_and_price(
    n_days: int = 250,
    start_price: float = 100.0,
    volatility: float = 0.008,
    n_genuine_events: int = 4,
    n_noise_news: int = 10,
    reaction_lag_days: int = 1,
    jump_magnitude: float = 0.06,
    seed: int | None = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate a price series plus a mix of genuine and coincidental news events.

    ``n_genuine_events`` news items each cause a same-direction price jump
    ``reaction_lag_days`` later. ``n_noise_news`` additional items are
    timestamped and signed at random and have no causal relationship to any
    price move -- a realistic proxy for headline noise / irrelevant chatter.
    The ground truth (which news are genuine) is stashed in the returned
    DataFrame's ``genuine`` column for evaluation.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, volatility, n_days)
    index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    genuine_days = rng.choice(
        np.arange(10, n_days - reaction_lag_days - 5), size=n_genuine_events, replace=False
    )

    records = []
    for day in genuine_days:
        sentiment = int(rng.choice([-1, 1]))
        magnitude = jump_magnitude * rng.uniform(0.8, 1.2)
        returns[day + reaction_lag_days] += sentiment * magnitude
        records.append({"timestamp": index[day], "sentiment": sentiment, "genuine": True})

    for _ in range(n_noise_news):
        day = int(rng.integers(0, n_days))
        sentiment = int(rng.choice([-1, 1]))
        records.append({"timestamp": index[day], "sentiment": sentiment, "genuine": False})

    prices = start_price * np.cumprod(1 + returns)
    price_series = pd.Series(prices, index=index, name="SYNTH")

    news = pd.DataFrame(records, columns=["timestamp", "sentiment", "genuine"])
    news = news.sort_values("timestamp").reset_index(drop=True)
    return price_series, news
