"""Multi-exchange price snapshots: live (ccxt) and synthetic (offline)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

DEFAULT_EXCHANGES = ["binance", "kraken", "coinbase", "kucoin", "bitstamp"]


def fetch_multi_exchange_snapshot(symbol: str = "BTC/USDT", exchanges: list[str] | None = None) -> pd.Series:
    """Fetch the current last-trade price for ``symbol`` from several exchanges via ccxt.

    Requires network access and that ``symbol`` is listed on each exchange
    (not all exchanges quote every pair, e.g. some only have BTC/USD).
    Exchanges that fail to respond are silently skipped.
    """
    import ccxt

    exchanges = exchanges or DEFAULT_EXCHANGES
    prices = {}
    for name in exchanges:
        try:
            exchange_cls = getattr(ccxt, name)
            exchange = exchange_cls({"enableRateLimit": True})
            ticker = exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price:
                prices[name] = float(price)
        except Exception:
            continue

    if len(prices) < 2:
        raise ValueError(
            f"Could not fetch '{symbol}' from at least 2 exchanges (got {list(prices)})."
        )
    return pd.Series(prices, name=symbol)


def generate_synthetic_exchange_snapshots(
    n_timestamps: int = 60,
    exchanges: list[str] | None = None,
    true_price: float = 60000.0,
    price_drift: float = 0.0,
    normal_noise: float = 0.0008,
    manipulated_exchange: str | None = "kucoin",
    manipulation_bias: float = 0.03,
    manipulation_window: tuple[int, int] = (25, 40),
    seed: int | None = 11,
) -> pd.DataFrame:
    """Generate synthetic per-exchange price snapshots over time.

    One exchange (``manipulated_exchange``) is biased away from the true
    consensus price for a window of timestamps, simulating a wash-trading
    / stale order-book scenario. Returns a DataFrame indexed by timestamp
    with one column per exchange.
    """
    exchanges = exchanges or DEFAULT_EXCHANGES
    rng = np.random.default_rng(seed)

    true_prices = true_price * np.cumprod(1 + rng.normal(price_drift, 0.001, n_timestamps))

    data = {}
    for exch in exchanges:
        noise = rng.normal(0, normal_noise, n_timestamps)
        prices = true_prices * (1 + noise)
        if exch == manipulated_exchange:
            lo, hi = manipulation_window
            prices[lo:hi] *= 1 + manipulation_bias
        data[exch] = prices

    index = pd.date_range(end=datetime.now(timezone.utc), periods=n_timestamps, freq="min")
    df = pd.DataFrame(data, index=index)
    df.attrs["manipulated_exchange"] = manipulated_exchange
    df.attrs["manipulation_window"] = manipulation_window
    return df
