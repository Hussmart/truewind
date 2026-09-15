from truewind.data.exchange_data import generate_synthetic_exchange_snapshots
from truewind.exchange_consensus import ExchangeConsensusDetector


def test_flags_the_manipulated_exchange_most_often():
    snapshots = generate_synthetic_exchange_snapshots(
        manipulated_exchange="kucoin",
        manipulation_window=(20, 40),
        manipulation_bias=0.05,
    )
    detector = ExchangeConsensusDetector()
    result = detector.fit(snapshots)

    assert result.outlier_rate.idxmax() == "kucoin"
    assert result.outlier_rate["kucoin"] > 0


def test_consensus_price_close_to_unbiased_exchanges():
    snapshots = generate_synthetic_exchange_snapshots(manipulated_exchange=None)
    detector = ExchangeConsensusDetector()
    result = detector.fit(snapshots)

    mean_abs_dev = (snapshots.mean(axis=1) - result.consensus_price).abs().mean()
    assert mean_abs_dev < snapshots.mean(axis=1).mean() * 0.01
