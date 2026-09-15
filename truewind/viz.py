"""Matplotlib plotting helpers for the two demo modules."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .exchange_consensus import ExchangeConsensusResult
from .news_price_alignment import NewsPriceAlignmentResult
from .regime import RegimeResult


def plot_regime_anomalies(prices: pd.Series, result: RegimeResult, title: str = "Regime-Break Detection", save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(prices.index, prices.values, color="#2b6cb0", linewidth=1.2, label="Price")

    anomaly_ts = result.anomaly_timestamps
    if len(anomaly_ts):
        ax.scatter(
            anomaly_ts,
            prices.reindex(anomaly_ts),
            color="#e53e3e",
            s=60,
            zorder=5,
            label="Flagged anomaly / regime break",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=140)
    return fig


def plot_exchange_consensus(snapshots: pd.DataFrame, result: ExchangeConsensusResult, title: str = "Cross-Exchange Price Consensus", save_path: str | None = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    for col in snapshots.columns:
        ax1.plot(snapshots.index, snapshots[col], linewidth=1, alpha=0.6, label=col)
    ax1.plot(
        result.consensus_price.index,
        result.consensus_price.values,
        color="black",
        linewidth=2,
        linestyle="--",
        label="Consensus price",
    )
    ax1.set_ylabel("Price")
    ax1.set_title(title)
    ax1.legend(fontsize=8, ncol=3)

    for i, col in enumerate(snapshots.columns):
        flagged = result.outlier_flags[col]
        ax2.scatter(
            snapshots.index[flagged],
            [i] * flagged.sum(),
            color="#e53e3e",
            s=20,
        )
    ax2.set_yticks(range(len(snapshots.columns)))
    ax2.set_yticklabels(snapshots.columns)
    ax2.set_ylabel("Flagged outlier")
    ax2.set_xlabel("Time")

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
