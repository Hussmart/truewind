from truewind.data.price_data import generate_synthetic_price_series
from truewind.regime import RegimeAnomalyDetector, naive_zscore_baseline


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


def test_naive_baseline_disagrees_with_consensus():
    # The two methods score different things (direction-only vs. joint
    # mean+volatility), so neither should consistently flag a superset of
    # the other -- just check both produce sensible, non-identical results.
    prices = generate_synthetic_price_series(n_points=400, n_anomalies=3, anomaly_magnitude=0.1, seed=3)

    consensus = RegimeAnomalyDetector(window_size=10, step=1).fit(prices)
    baseline = naive_zscore_baseline(prices, window_size=10, step=1)

    assert consensus.anomaly_mask.sum() > 0
    assert baseline.anomaly_mask.sum() > 0
    assert not (consensus.anomaly_mask == baseline.anomaly_mask).all()
