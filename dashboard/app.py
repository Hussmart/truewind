"""Interactive Streamlit dashboard for the three TrueWind detectors.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from truewind.data.exchange_data import (
    fetch_multi_exchange_ohlcv_history,
    generate_synthetic_exchange_snapshots,
)
from truewind.data.news_data import generate_synthetic_news_and_price
from truewind.data.price_data import fetch_price_history, generate_synthetic_price_series
from truewind.exchange_consensus import ExchangeConsensusDetector
from truewind.news_price_alignment import NewsPriceAligner
from truewind.regime import RegimeAnomalyDetector
from truewind.viz import plot_exchange_consensus, plot_news_price_alignment, plot_regime_anomalies

st.set_page_config(page_title="TrueWind", layout="wide")
st.title("TrueWind")
st.caption(
    "Robust financial anomaly detection via consensus-maximization on a graph -- "
    "the same relaxation MIT-ACL's CLIPPER uses for robot point-cloud registration, "
    "reimplemented here for markets. See the README for details."
)

tab1, tab2, tab3 = st.tabs(
    ["Regime-break detection", "Cross-exchange consensus", "News ↔ price alignment"]
)

with tab1:
    st.subheader("Flag anomalous / regime-breaking windows in a price series")
    col1, col2 = st.columns([1, 3])
    with col1:
        use_live = st.checkbox("Use real data (yfinance)", value=True, key="regime_live")
        ticker = st.text_input("Ticker", "BTC-USD", key="regime_ticker") if use_live else None
        period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1, key="regime_period") if use_live else None
        window_size = st.slider("Window size (days)", 5, 40, 15)

    try:
        prices = fetch_price_history(ticker, period=period) if use_live else generate_synthetic_price_series()
    except Exception as e:
        st.error(f"Failed to fetch live data ({e}); falling back to synthetic series.")
        prices = generate_synthetic_price_series()

    detector = RegimeAnomalyDetector(window_size=window_size)
    result = detector.fit(prices)
    title = f"Regime-Break Detection — {ticker}, real data" if use_live else "Regime-Break Detection — synthetic data"
    st.pyplot(plot_regime_anomalies(prices, result, title=title))
    st.metric("Flagged windows", int(result.anomaly_mask.sum()), f"of {len(result.anomaly_mask)} total")

with tab2:
    st.subheader("Flag exchanges whose quoted price disagrees with the group consensus")
    use_real_exchanges = st.checkbox("Use real exchange data (ccxt)", value=True, key="exchange_live")
    symbol = st.text_input("Symbol", "BTC/USDT", key="exchange_symbol") if use_real_exchanges else None
    days = st.slider("Days of history", 30, 400, 180, key="exchange_days") if use_real_exchanges else None

    if use_real_exchanges:
        try:
            snapshots = fetch_multi_exchange_ohlcv_history(symbol, limit=days)
            st.caption(f"Real daily closes from: {', '.join(snapshots.columns)}")
        except Exception as e:
            st.error(f"Failed to fetch real exchange data ({e}); falling back to synthetic panel.")
            snapshots = generate_synthetic_exchange_snapshots()
    else:
        st.caption("Synthetic multi-exchange panel with one exchange biased for a window of time.")
        snapshots = generate_synthetic_exchange_snapshots()

    exchange_detector = ExchangeConsensusDetector()
    exchange_result = exchange_detector.fit(snapshots)
    exchange_title = f"Cross-Exchange Price Consensus — {symbol}" if use_real_exchanges else "Cross-Exchange Price Consensus — synthetic data"
    st.pyplot(plot_exchange_consensus(snapshots, exchange_result, title=exchange_title))
    st.dataframe(exchange_result.outlier_rate.rename("outlier rate"))

with tab3:
    st.subheader("Robustly align news events to the price moves they actually caused")
    st.caption("Synthetic mix of genuine (causal) and coincidental news events -- no free real-time news API is wired in yet, see the roadmap.")
    news_prices, news = generate_synthetic_news_and_price()
    aligner = NewsPriceAligner()
    news_result = aligner.fit(news_prices, news)
    st.pyplot(plot_news_price_alignment(news_prices, news_result))
    st.metric("Confirmed genuine news-driven moves", len(news_result.confirmed), f"of {len(news_result.candidates)} candidates")
