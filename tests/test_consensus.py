import numpy as np

from truewind.consensus import ConsensusSolver


def _block_consistency_matrix(n_inliers: int, n_outliers: int, seed: int = 0) -> np.ndarray:
    """A clique of n_inliers mutually-consistent nodes plus loosely-connected outliers."""
    rng = np.random.default_rng(seed)
    n = n_inliers + n_outliers
    M = np.zeros((n, n))

    M[:n_inliers, :n_inliers] = rng.uniform(0.85, 1.0, size=(n_inliers, n_inliers))
    M[:n_inliers, :n_inliers] = (M[:n_inliers, :n_inliers] + M[:n_inliers, :n_inliers].T) / 2

    M[n_inliers:, :] = rng.uniform(0.0, 0.1, size=(n_outliers, n))
    M[:, n_inliers:] = M[n_inliers:, :].T

    np.fill_diagonal(M, 1.0)
    return M


def test_recovers_clean_clique():
    M = _block_consistency_matrix(n_inliers=8, n_outliers=4)
    solver = ConsensusSolver()
    result = solver.solve(M)

    assert set(result.inlier_indices) == set(range(8))


def test_all_consistent_graph_keeps_everyone():
    M = np.ones((6, 6)) * 0.9
    np.fill_diagonal(M, 1.0)
    solver = ConsensusSolver()
    result = solver.solve(M)

    assert result.n_inliers == 6


def test_rejects_non_square_matrix():
    solver = ConsensusSolver()
    try:
        solver.solve(np.ones((3, 4)))
        assert False, "expected ValueError"
    except ValueError:
        pass
