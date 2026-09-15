"""Pure-Python reimplementation of CLIPPER's consensus-maximization core.

CLIPPER (Lusk, Fathian & How, MIT ACL, https://github.com/mit-acl/clipper)
frames robust data association as: given a pairwise "consistency" score
between candidate matches, find the largest mutually-consistent subset
while rejecting outliers. It relaxes the underlying maximum-clique problem
to a continuous quadratic program over the nonnegative simplex and solves
it with projected gradient ascent (a spectral/replicator-dynamics style
relaxation of the Motzkin-Straus max-clique formulation).

This module reimplements that same relaxation from scratch in numpy. It is
*inspired by* CLIPPER's formulation, not a port of its C++ code -- the
original library adds additional refinements (e.g. certified bounds,
denser pruning heuristics) that this version does not attempt to match
exactly. See the CLIPPER paper for the full derivation:
Lusk, P. C., Fathian, K., & How, J. P. (2021). CLIPPER: A Graph-Theoretic
Framework for Robust Data Association. ICRA 2021.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConsensusResult:
    """Result of running :class:`ConsensusSolver` on a consistency matrix."""

    u: np.ndarray
    inlier_indices: np.ndarray

    @property
    def n_inliers(self) -> int:
        return len(self.inlier_indices)


class ConsensusSolver:
    """Finds the largest mutually-consistent subset of a weighted graph.

    Parameters
    ----------
    tol:
        Convergence tolerance on the solution vector between iterations.
    max_iter:
        Maximum projected-gradient-ascent iterations per round.
    threshold_ratio:
        A node is kept in the consensus set if its score is at least
        ``threshold_ratio * max(u)``. Lower values keep more (looser)
        consensus; higher values are stricter.
    prune_rounds:
        Number of prune-and-resolve rounds: after each solve, nodes
        below the threshold are dropped and the (smaller) problem is
        re-solved, which sharpens the result on noisy graphs.
    """

    def __init__(
        self,
        tol: float = 1e-6,
        max_iter: int = 1000,
        threshold_ratio: float = 0.15,
        prune_rounds: int = 2,
    ) -> None:
        self.tol = tol
        self.max_iter = max_iter
        self.threshold_ratio = threshold_ratio
        self.prune_rounds = prune_rounds

    def _power_iterate(self, M: np.ndarray) -> np.ndarray:
        n = M.shape[0]
        u = np.full(n, 1.0 / np.sqrt(n))
        for _ in range(self.max_iter):
            grad = M @ u
            u_new = np.maximum(grad, 0.0)
            norm = np.linalg.norm(u_new)
            if norm < 1e-12:
                return u_new
            u_new /= norm
            if np.linalg.norm(u_new - u) < self.tol:
                return u_new
            u = u_new
        return u

    def solve(self, M: np.ndarray) -> ConsensusResult:
        """Solve for the maximum-consensus subset of ``M``.

        ``M`` must be a square, symmetric matrix with entries in
        ``[0, 1]`` where ``M[i, j]`` is the pairwise consistency between
        candidates ``i`` and ``j`` (higher = more consistent).
        """
        M = np.asarray(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("M must be a square consistency matrix")

        n = M.shape[0]
        active_idx = np.arange(n)
        Mc = M
        u_active = np.ones(n) / max(n, 1)

        for _ in range(self.prune_rounds + 1):
            if len(active_idx) < 2:
                break
            u_active = self._power_iterate(Mc)
            if u_active.max() <= 0:
                break
            keep = u_active >= self.threshold_ratio * u_active.max()
            if keep.sum() == len(active_idx) or keep.sum() < 2:
                break
            active_idx = active_idx[keep]
            Mc = Mc[np.ix_(keep, keep)]

        full_u = np.zeros(n)
        full_u[active_idx] = u_active[: len(active_idx)]

        if full_u.max() > 0:
            inlier_mask = full_u >= self.threshold_ratio * full_u.max()
            inlier_idx = np.where(inlier_mask)[0]
        else:
            inlier_idx = active_idx

        return ConsensusResult(u=full_u, inlier_indices=np.sort(inlier_idx))
