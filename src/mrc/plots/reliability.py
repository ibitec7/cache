"""Reliability diagram for the hazard probabilities.

A hazard model estimates `h_b(x) = P(R in bin b | R >= b, x)`; the MRC inherits its
accuracy from how faithful those probabilities are, so the diagram is reported next
to the curve. Predictions are grouped into equal-width bins on [0, 1]; each occupied
bin contributes its mean prediction against the observed hit frequency, and the
diagonal marks perfect calibration.

The bin table is public because the partition is an invariant, not a drawing detail:
the bin counts partition the predictions, so the fractions sum to one.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mrc.plots.style import PALETTE, apply_style, fig_single

DIAGONAL_COLOUR = "0.6"


def _bin_index(pred: Sequence[float] | np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin index on [0, 1]; 1.0 belongs to the last bin."""
    p = np.asarray(pred, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("pred is empty")
    if not np.all(np.isfinite(p)):
        raise ValueError("pred must be finite")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("pred must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)


def bin_fractions(pred: Sequence[float] | np.ndarray, n_bins: int = 12) -> np.ndarray:
    """Fraction of the predictions that falls in each of the `n_bins` bins."""
    idx = _bin_index(pred, n_bins)
    return np.bincount(idx, minlength=n_bins) / idx.size


def reliability_table(
    pred: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_bins: int = 12,
) -> dict[str, np.ndarray]:
    """Per-bin calibration table, one row per bin, empty bins kept as NaN.

    pred_mean  mean predicted probability in the bin; NaN when the bin is empty
    observed   observed frequency of y in the bin; NaN when the bin is empty
    count      predictions in the bin
    fraction   count / number of predictions; the fractions sum to one
    """
    p = np.asarray(pred, dtype=float).ravel()
    yv = np.asarray(y, dtype=float).ravel()
    if yv.shape != p.shape:
        raise ValueError(f"pred has length {p.size}, y has length {yv.size}")
    if not np.all(np.isfinite(yv)) or yv.min() < 0.0 or yv.max() > 1.0:
        raise ValueError("y must be a finite frequency in [0, 1]")

    idx = _bin_index(p, n_bins)
    count = np.bincount(idx, minlength=n_bins)
    occupied = count > 0
    pred_mean = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    np.divide(np.bincount(idx, weights=p, minlength=n_bins), count, out=pred_mean, where=occupied)
    np.divide(np.bincount(idx, weights=yv, minlength=n_bins), count, out=observed, where=occupied)
    return {
        "pred_mean": pred_mean,
        "observed": observed,
        "count": count,
        "fraction": count / idx.size,
    }


def plot_reliability(
    pred: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    path: str | Path,
    n_bins: int = 12,
) -> Path:
    """Predicted against observed hit frequency, with the diagonal, as a PDF."""
    import matplotlib.pyplot as plt

    table = reliability_table(pred, y, n_bins=n_bins)
    occupied = table["count"] > 0

    apply_style()
    fig, ax = fig_single(2.1)
    ax.plot([0.0, 1.0], [0.0, 1.0], color=DIAGONAL_COLOUR, linewidth=0.8,
            linestyle="--", label="ideal")
    ax.plot(table["pred_mean"][occupied], table["observed"][occupied],
            color=PALETTE[0], marker="o", markersize=3, label="hazard")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted hit probability")
    ax.set_ylabel("Observed hit frequency")
    ax.legend(loc="upper left")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out
