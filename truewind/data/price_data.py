"""Price history data sources: live (yfinance) and synthetic (offline)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.Series:
    """Fetch close-price history for ``ticker`` using yfinance.

    Requires network access. Raises the underlying yfinance exception on
    failure so callers can fall back to :func:`generate_synthetic_price_series`.
    """
    import yfinance as yf

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close


def generate_synthetic_price_series(
    n_points: int = 500,
    start_price: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.01,
    n_anomalies: int = 3,
    anomaly_magnitude: float = 0.08,
    seed: int | None = 7,
) -> pd.Series:
    """Generate a synthetic price series with injected regime-break shocks.

    Useful for demos, tests, and CI where network access to a live price
    feed isn't available or reproducibility is required.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=volatility, size=n_points)

    anomaly_positions = rng.choice(
        np.arange(int(n_points * 0.15), int(n_points * 0.9)),
        size=n_anomalies,
        replace=False,
    )
    for pos in anomaly_positions:
        shock = anomaly_magnitude * rng.choice([-1, 1])
        returns[pos] += shock

    prices = start_price * np.cumprod(1 + returns)
    index = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_points, freq="D")
    series = pd.Series(prices, index=index, name="SYNTH")
    series.attrs["injected_anomaly_positions"] = sorted(int(p) for p in anomaly_positions)
    return series
