"""Cross-exchange price-consensus / manipulation flagging.

At each timestamp, treats each exchange's quoted price as a node in a
consistency graph (nodes agree if their prices are close relative to the
group). :class:`ConsensusSolver` finds the largest mutually-consistent
subset -- the "true price" consensus -- and any exchange left out is
flagged as a pricing outlier for that timestamp (stale book, thin
liquidity, or possible wash trading / manipulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .consensus import ConsensusSolver


@dataclass
class ExchangeConsensusResult:
    outlier_flags: pd.DataFrame
    consensus_price: pd.Series
    outlier_rate: pd.Series


class ExchangeConsensusDetector:
    """Flags exchanges whose quoted price disagrees with the group consensus."""

    def __init__(self, rel_tolerance: float = 0.01, solver: ConsensusSolver | None = None) -> None:
        self.rel_tolerance = rel_tolerance
        self.solver = solver or ConsensusSolver(threshold_ratio=0.2, prune_rounds=3)

    def _consistency_matrix(self, prices: np.ndarray) -> np.ndarray:
        diff = np.abs(prices[:, None] - prices[None, :])
        med = np.median(prices)
        rel_diff = diff / med
        M = np.clip(1 - rel_diff / self.rel_tolerance, 0, 1)
        np.fill_diagonal(M, 1.0)
        return M

    def fit(self, snapshots: pd.DataFrame) -> ExchangeConsensusResult:
        exchanges = snapshots.columns
        outlier_flags = pd.DataFrame(False, index=snapshots.index, columns=exchanges)
        consensus_price = pd.Series(index=snapshots.index, dtype=float)

        for ts, row in snapshots.iterrows():
            prices = row.to_numpy(dtype=float)
            valid = ~np.isnan(prices)
            if valid.sum() < 2:
                continue

            M = self._consistency_matrix(prices[valid])
            result = self.solver.solve(M)

            valid_idx = np.where(valid)[0]
            inlier_global_idx = valid_idx[result.inlier_indices]

            row_flags = np.ones(len(exchanges), dtype=bool)
            row_flags[inlier_global_idx] = False
            row_flags[~valid] = False
            outlier_flags.loc[ts] = row_flags

            consensus_price.loc[ts] = prices[inlier_global_idx].mean() if len(inlier_global_idx) else np.nan

        outlier_rate = outlier_flags.mean(axis=0).sort_values(ascending=False)

        return ExchangeConsensusResult(
            outlier_flags=outlier_flags,
            consensus_price=consensus_price,
            outlier_rate=outlier_rate,
        )
