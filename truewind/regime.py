"""Regime-break / anomaly detection via consensus-maximization.

Splits a price series into rolling windows, scores pairwise statistical
consistency between windows, and uses :class:`ConsensusSolver` to find the
largest mutually-consistent set of windows ("normal regime"). Windows left
out of that consensus set are flagged as anomalies / regime breaks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .consensus import ConsensusSolver


@dataclass
class RegimeResult:
    window_index: pd.DatetimeIndex
    window_means: np.ndarray
    window_stds: np.ndarray
    u: np.ndarray
    anomaly_mask: np.ndarray

    @property
    def anomaly_timestamps(self) -> pd.DatetimeIndex:
        return self.window_index[self.anomaly_mask]


class RegimeAnomalyDetector:
    """Detects anomalous / regime-breaking windows in a price series."""

    def __init__(
        self,
        window_size: int = 15,
        step: int = 1,
        bandwidth: float = 1.5,
        solver: ConsensusSolver | None = None,
    ) -> None:
        self.window_size = window_size
        self.step = step
        self.bandwidth = bandwidth
        self.solver = solver or ConsensusSolver(threshold_ratio=0.2, prune_rounds=3)

    def _windows(self, returns: np.ndarray):
        n = len(returns)
        starts = range(0, n - self.window_size + 1, self.step)
        for s in starts:
            yield returns[s : s + self.window_size]

    def fit(self, prices: pd.Series) -> RegimeResult:
        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        values = returns.to_numpy()
        index = returns.index

        window_list = list(self._windows(values))
        if len(window_list) < 3:
            raise ValueError("Series too short for the given window_size/step")

        means = np.array([w.mean() for w in window_list])
        stds = np.array([w.std() for w in window_list])

        window_end_idx = [
            min(s + self.window_size - 1, len(index) - 1)
            for s in range(0, len(values) - self.window_size + 1, self.step)
        ]
        window_index = index[window_end_idx]

        mean_scale = means.std() + 1e-9
        std_scale = stds.std() + 1e-9
        feat = np.stack([means / mean_scale, stds / std_scale], axis=1)

        diff = feat[:, None, :] - feat[None, :, :]
        dist2 = np.sum(diff**2, axis=-1)
        M = np.exp(-dist2 / (2 * self.bandwidth**2))
        np.fill_diagonal(M, 1.0)

        result = self.solver.solve(M)

        anomaly_mask = np.ones(len(window_list), dtype=bool)
        anomaly_mask[result.inlier_indices] = False

        return RegimeResult(
            window_index=window_index,
            window_means=means,
            window_stds=stds,
            u=result.u,
            anomaly_mask=anomaly_mask,
        )


def naive_zscore_baseline(prices: pd.Series, window_size: int = 15, step: int = 1, z_threshold: float = 1.5) -> RegimeResult:
    """A standard "z-score on rolling return" anomaly flagger, for comparison.

    Flags a window whenever its mean return is more than ``z_threshold``
    standard deviations below the average window mean across the series --
    the common baseline approach. Unlike :class:`RegimeAnomalyDetector`, it
    only looks at the *direction* of the window (mean return), not whether
    its volatility is *also* jointly unusual -- so it tends to flag every
    sufficiently negative stretch, not just the ones with an anomalous
    volatility signature.
    """
    prices = prices.dropna()
    returns = prices.pct_change().dropna()
    values = returns.to_numpy()
    index = returns.index

    n = len(values)
    starts = list(range(0, n - window_size + 1, step))
    if len(starts) < 3:
        raise ValueError("Series too short for the given window_size/step")

    means = np.array([values[s : s + window_size].mean() for s in starts])
    stds = np.array([values[s : s + window_size].std() for s in starts])
    window_index = index[[min(s + window_size - 1, n - 1) for s in starts]]

    z = (means - means.mean()) / (means.std() + 1e-9)
    anomaly_mask = z < -z_threshold

    return RegimeResult(
        window_index=window_index,
        window_means=means,
        window_stds=stds,
        u=np.where(anomaly_mask, 0.0, 1.0),
        anomaly_mask=anomaly_mask,
    )
