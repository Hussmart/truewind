"""Matplotlib plotting helpers for the two demo modules."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .exchange_consensus import ExchangeConsensusResult
from .news_price_alignment import NewsPriceAlignmentResult
from .regime import RegimeResult


def _contiguous_spans(timestamps: pd.DatetimeIndex, gap_tolerance: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(timestamps) == 0:
        return []
    ts = timestamps.sort_values()
    spans = []
    start = prev = ts[0]
    for t in ts[1:]:
        if t - prev > gap_tolerance:
            spans.append((start, prev))
            start = t
        prev = t
    spans.append((start, prev))
    return spans


def plot_regime_anomalies(prices: pd.Series, result: RegimeResult, title: str = "Regime-Break Detection", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prices.index, prices.values, color="#2b6cb0", linewidth=1.2, label="Price", zorder=2)

    anomaly_ts = result.anomaly_timestamps
    if len(anomaly_ts):
        median_gap = pd.Series(prices.index).diff().median()
        spans = _contiguous_spans(anomaly_ts, gap_tolerance=median_gap * 3)
        for i, (lo, hi) in enumerate(spans):
            ax.axvspan(lo, hi, color="#e53e3e", alpha=0.15, zorder=1, label="Flagged regime break" if i == 0 else None)
        ax.scatter(
            anomaly_ts,
            prices.reindex(anomaly_ts),
            color="#e53e3e",
            s=28,
            zorder=5,
        )
        ax.text(
            0.01,
            0.02,
            f"{len(anomaly_ts)} flagged windows across {len(spans)} episode(s)",
            transform=ax.transAxes,
            fontsize=9,
            color="#742a2a",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_exchange_consensus(
    snapshots: pd.DataFrame,
    result: ExchangeConsensusResult,
    title: str = "Cross-Exchange Price Consensus",
    save_path: str | None = None,
    zoom_window: pd.Timedelta | None = None,
):
    """Plot the full price history plus, if any exchange was flagged, a zoomed-in
    panel around the first flagged event where the divergence is actually visible
    (on real data the exchanges normally overlap almost exactly, so a full-history
    view alone hides a one-day divergence)."""
    any_flagged = result.outlier_flags.to_numpy().any()
    n_rows = 2 if any_flagged else 1
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(11, 5 if n_rows == 1 else 8), gridspec_kw={"height_ratios": [2, 1.3][:n_rows]}
    )
    ax1 = axes[0] if n_rows > 1 else axes

    for col in snapshots.columns:
        ax1.plot(snapshots.index, snapshots[col], linewidth=1, alpha=0.6, label=col)
    ax1.plot(
        result.consensus_price.index,
        result.consensus_price.values,
        color="black",
        linewidth=1.3,
        linestyle="--",
        alpha=0.8,
        label="Consensus price",
    )

    flagged_mask = result.outlier_flags.any(axis=1)
    if flagged_mask.any():
        flagged_ts = result.outlier_flags.index[flagged_mask]
        ax1.scatter(
            flagged_ts,
            snapshots.loc[flagged_ts].max(axis=1),
            color="#e53e3e",
            s=50,
            zorder=6,
            marker="v",
            label="Flagged divergence",
        )

    ax1.set_ylabel("Price")
    ax1.set_title(title)
    ax1.legend(fontsize=8, ncol=3)

    if any_flagged:
        ax2 = axes[1]
        first_flag_ts = result.outlier_flags.index[flagged_mask][0]
        window = zoom_window or max((snapshots.index[-1] - snapshots.index[0]) * 0.01, pd.Timedelta(days=4))
        lo, hi = first_flag_ts - window, first_flag_ts + window
        zoomed = snapshots.loc[(snapshots.index >= lo) & (snapshots.index <= hi)]

        outlier_cols = result.outlier_flags.loc[first_flag_ts]
        outlier_cols = outlier_cols[outlier_cols].index.tolist()

        for col in zoomed.columns:
            is_outlier = col in outlier_cols
            ax2.plot(
                zoomed.index,
                zoomed[col],
                linewidth=2.4 if is_outlier else 1.2,
                color="#e53e3e" if is_outlier else "#2b6cb0",
                alpha=1.0 if is_outlier else 0.55,
                marker="o",
                markersize=5,
                label=f"{col} (flagged)" if is_outlier else col,
            )

        if outlier_cols:
            outlier_price = snapshots.loc[first_flag_ts, outlier_cols[0]]
            consensus_at_ts = result.consensus_price.loc[first_flag_ts]
            pct_gap = (outlier_price - consensus_at_ts) / consensus_at_ts * 100
            ax2.annotate(
                f"{outlier_cols[0]}: {pct_gap:+.2f}% vs. consensus",
                xy=(first_flag_ts, outlier_price),
                xytext=(12, 18),
                textcoords="offset points",
                fontsize=9,
                color="#e53e3e",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e53e3e"),
            )

        ax2.axvline(first_flag_ts, color="#a0aec0", linestyle=":", linewidth=1, zorder=0)
        ax2.set_title(f"Zoomed on flagged event: {first_flag_ts.date()}", fontsize=10)
        ax2.set_ylabel("Price")
        ax2.set_xlabel("Time")
        ax2.legend(fontsize=8, ncol=3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_news_price_alignment(prices: pd.Series, result: NewsPriceAlignmentResult, title: str = "News ↔ Price-Move Alignment", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prices.index, prices.values, color="#2b6cb0", linewidth=1.2, label="Price", zorder=1)

    for c in result.confirmed:
        ax.axvline(c.news_timestamp, color="#38a169", alpha=0.5, linewidth=1)
        ax.scatter([c.jump_timestamp], [prices.reindex([c.jump_timestamp]).iloc[0]], color="#38a169", s=70, zorder=5)

    for c in result.rejected:
        ax.axvline(c.news_timestamp, color="#a0aec0", alpha=0.3, linewidth=1, linestyle=":")

    ax.plot([], [], color="#38a169", marker="o", linestyle="-", linewidth=1, label="Confirmed news → price reaction")
    ax.plot([], [], color="#a0aec0", linestyle=":", label="Rejected candidate (noise)")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_regime_comparison(
    prices: pd.Series,
    consensus_result: RegimeResult,
    baseline_result: RegimeResult,
    title: str = "Consensus vs. Naive Z-Score Baseline",
    save_path: str | None = None,
):
    """Show both detectors' flags on the same price series to make the
    difference concrete: the naive baseline flags every sufficiently
    negative-return window, while the consensus method only flags windows
    whose volatility is *also* jointly anomalous."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(prices.index, prices.values, color="#2b6cb0", linewidth=1.2, label="Price", zorder=2)

    base_ts = baseline_result.anomaly_timestamps
    if len(base_ts):
        for i, (lo, hi) in enumerate(_contiguous_spans(base_ts, gap_tolerance=pd.Timedelta(days=3))):
            ax.axvspan(lo, hi, color="#ed8936", alpha=0.18, zorder=0, label="Naive z-score flag" if i == 0 else None)

    cons_ts = consensus_result.anomaly_timestamps
    if len(cons_ts):
        for i, (lo, hi) in enumerate(_contiguous_spans(cons_ts, gap_tolerance=pd.Timedelta(days=3))):
            ax.axvspan(lo, hi, color="#e53e3e", alpha=0.28, zorder=1, label="Consensus flag" if i == 0 else None)

    ax.text(
        0.01,
        0.02,
        f"Naive: {len(base_ts)} windows flagged  |  Consensus: {len(cons_ts)} windows flagged",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig
