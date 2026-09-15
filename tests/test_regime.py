from truewind.data.price_data import generate_synthetic_price_series
from truewind.regime import RegimeAnomalyDetector


def test_flags_at_least_one_injected_anomaly():
    prices = generate_synthetic_price_series(n_points=400, n_anomalies=3, anomaly_magnitude=0.1, seed=3)
    injected = prices.attrs["injected_anomaly_positions"]

    detector = RegimeAnomalyDetector(window_size=10, step=1)
    result = detector.fit(prices)

    assert result.anomaly_mask.sum() > 0

    anomaly_positions = set(prices.index.get_indexer(result.anomaly_timestamps))
    close_to_injected = any(
        any(abs(pos - inj) <= 10 for inj in injected) for pos in anomaly_positions
    )
    assert close_to_injected


def test_raises_on_too_short_series():
    prices = generate_synthetic_price_series(n_points=5)
    detector = RegimeAnomalyDetector(window_size=15)
    try:
        detector.fit(prices)
        assert False, "expected ValueError"
    except ValueError:
        pass
