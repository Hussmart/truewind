"""Robust news-to-price-move alignment via consensus-maximization.

This is the financial-domain analogue of CLIPPER's *original* point-cloud
registration use case. There, a candidate correspondence between two
points is consistent with another correspondence if the pairwise distance
between the two points is preserved under a common rigid transform.

Here, a candidate correspondence is a (news event, price jump) pair with a
same-direction sentiment/return and a time lag between them. Two candidate
pairs are consistent if they imply a *similar reaction lag* -- the
financial analogue of "distance preserved under a common transform": a
real news-driven market reacts with a roughly consistent delay, while
coincidental (news, jump) pairings have inconsistent, effectively random
lags. Candidate pairs that would double-assign the same news item or the
same price jump are also marked inconsistent, mirroring the one-to-one
matching constraint in point-cloud registration.

The consensus solver then finds the largest mutually-consistent set of
(news, jump) pairs -- the news events we can be confident actually drove
a market reaction -- and rejects the rest as noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .consensus import ConsensusSolver


@dataclass
class Candidate:
    news_index: int
    jump_index: int
    news_timestamp: pd.Timestamp
    jump_timestamp: pd.Timestamp
    delay_days: float
    sentiment: int
    price_return: float


@dataclass
class NewsPriceAlignmentResult:
    candidates: list[Candidate]
    confirmed_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    u: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def confirmed(self) -> list[Candidate]:
        return [self.candidates[i] for i in self.confirmed_indices]

    @property
    def rejected(self) -> list[Candidate]:
        confirmed_set = set(self.confirmed_indices.tolist())
        return [c for i, c in enumerate(self.candidates) if i not in confirmed_set]


class NewsPriceAligner:
    """Finds the largest mutually-consistent set of (news, price-jump) pairs.

    Parameters
    ----------
    jump_zscore:
        A return is treated as a "jump" candidate if ``|return| > jump_zscore *
        std(returns)``.
    max_lag_days:
        Only (news, jump) pairs within this many days of each other, with the
        jump *after* the news, are considered candidates at all.
    lag_tolerance_days:
        Bandwidth of the Gaussian kernel scoring how similar two candidates'
        reaction lags must be to be called mutually consistent. Smaller values
        demand a tighter, more consistent reaction lag across confirmed events.
    """

    def __init__(
        self,
        jump_zscore: float = 1.5,
        max_lag_days: float = 3.0,
        lag_tolerance_days: float = 1.0,
        solver: ConsensusSolver | None = None,
    ) -> None:
        self.jump_zscore = jump_zscore
        self.max_lag_days = max_lag_days
        self.lag_tolerance_days = lag_tolerance_days
        self.solver = solver or ConsensusSolver(threshold_ratio=0.2, prune_rounds=3)

    def detect_price_jumps(self, prices: pd.Series) -> pd.DataFrame:
        returns = prices.pct_change().dropna()
        thresh = returns.std() * self.jump_zscore
        jumps = returns[returns.abs() > thresh]
        return pd.DataFrame({"timestamp": jumps.index, "return": jumps.values})

    def _build_candidates(self, news: pd.DataFrame, jumps: pd.DataFrame) -> list[Candidate]:
        candidates = []
        for ni, nrow in news.iterrows():
            for ji, jrow in jumps.iterrows():
                delay = (jrow["timestamp"] - nrow["timestamp"]).total_seconds() / 86400.0
                if 0 <= delay <= self.max_lag_days and np.sign(nrow["sentiment"]) == np.sign(jrow["return"]):
                    candidates.append(
                        Candidate(
                            news_index=int(ni),
                            jump_index=int(ji),
                            news_timestamp=nrow["timestamp"],
                            jump_timestamp=jrow["timestamp"],
                            delay_days=delay,
                            sentiment=int(nrow["sentiment"]),
                            price_return=float(jrow["return"]),
                        )
                    )
        return candidates

    def fit(self, prices: pd.Series, news: pd.DataFrame) -> NewsPriceAlignmentResult:
        """``news`` must have ``timestamp`` and ``sentiment`` (+1/-1) columns."""
        jumps = self.detect_price_jumps(prices)
        candidates = self._build_candidates(news, jumps)

        if len(candidates) < 2:
            return NewsPriceAlignmentResult(candidates=candidates)

        n = len(candidates)
        delays = np.array([c.delay_days for c in candidates])
        news_ids = np.array([c.news_index for c in candidates])
        jump_ids = np.array([c.jump_index for c in candidates])

        diff = np.abs(delays[:, None] - delays[None, :])
        M = np.exp(-(diff**2) / (2 * self.lag_tolerance_days**2))

        conflict = (news_ids[:, None] == news_ids[None, :]) | (jump_ids[:, None] == jump_ids[None, :])
        np.fill_diagonal(conflict, False)
        M[conflict] = 0.0
        np.fill_diagonal(M, 1.0)

        result = self.solver.solve(M)
        return NewsPriceAlignmentResult(candidates=candidates, confirmed_indices=result.inlier_indices, u=result.u)
