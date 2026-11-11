"""Risk-set expansion for the discrete-time hazard over reuse distance.

Input contract
--------------
``X``       float32 ``(n, n_features)``, row i is the design matrix of request i in trace order
``rtilde``  int64 ``(n,)``, capped reuse distance, ``-1`` for ineligible (cold) requests
``delta``   bool ``(n,)``, True when the re-access is observed inside the label window
``edges``   int64 ``(M + 1,)``, increasing; bins ``0..M-1``, the last one is the overflow bin

Both expanders read ``rtilde``, never the raw reuse distance ``R``: a censored
request contributes the bins it survived, never the distance it never reached.

Grouped variant (canonical): one row per (request, bin at risk). The label is 1
in the bin where the observed event lands, 0 in every earlier bin. Every row is
one Bernoulli trial of the within-bin hazard, so the fitting weight is 1. The
master plan proposed weighting each row by the bin width, the grouped-likelihood
approximation to the per-step product; that is only correct under a complementary
log-log link with an offset of log(width). Under the binary log-loss used here it
over-weights wide (late) bins by up to six orders of magnitude, and measured on
the 1M sample it raised test MAE from 0.0050 to 0.0113. Cost is O(M) per request instead of O(R); measured 28.4x fewer rows on
cluster045 (745,784 vs 21,174,922 for 200,000 requests).

Per-step variant (ablation only): one row per (request, step at risk), the exact
likelihood. It refuses above ``PER_STEP_MAX_REQUESTS`` and above
``MAX_GROUPED_ROWS``: one request at the ceiling contributes one row per unit of
distance, so the request guard alone does not bound the row count.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mrc.config import MAX_GROUPED_ROWS, PER_STEP_MAX_REQUESTS
from mrc.models.hazard import assign_bins


def _bin_widths(edges: np.ndarray) -> np.ndarray:
    """Width of each bin in distance units, clipped at 1."""
    w = np.diff(edges).astype(np.float32)
    w[w < 1] = 1.0
    return w


def _within_request_index(counts: np.ndarray, dtype: Any) -> np.ndarray:
    """``0..count_i - 1`` for each request i, concatenated. O(total) and vectorised."""
    total = int(counts.sum())
    starts = (np.cumsum(counts) - counts).astype(np.int32)
    local = np.arange(total, dtype=np.int32) - np.repeat(starts, counts)
    return local.astype(dtype, copy=False)


def grouped_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray,
                 edges: np.ndarray, n_features: int) -> dict:
    """Expand eligible requests into one row per (request, bin at risk).

    ``request_index`` indexes the kept eligible rows, ``eligible_index`` holds the
    original request indices, so ``eligible_index[request_index]`` recovers the
    trace position of every row.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError(f"X must have shape (n, n_features={n_features}), got {X.shape}")

    bins = assign_bins(np.asarray(rtilde, dtype=np.int64), edges)
    keep = bins >= 0
    b = bins[keep].astype(np.int64)
    d = np.asarray(delta, dtype=bool)[keep]

    counts = b + 1
    total = int(counts.sum())
    if total > MAX_GROUPED_ROWS:
        raise MemoryError(f"{total} grouped rows exceeds MAX_GROUPED_ROWS={MAX_GROUPED_ROWS}")

    request_index = np.repeat(np.arange(b.shape[0], dtype=np.int64), counts)
    bin_index = _within_request_index(counts, np.int16)
    y = (bin_index == np.repeat(b, counts)).astype(np.float32) * np.repeat(d, counts)
    return {
        "X": X[keep][request_index],
        "bin": bin_index,
        "y": y,
        "w": np.ones(total, dtype=np.float32),
        "bin_width": _bin_widths(edges)[bin_index],
        "request_index": request_index,
        "eligible_index": np.flatnonzero(keep),
        "n_rows": total,
    }


def per_step_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray,
                  edges: np.ndarray | None = None) -> dict:
    """Expand eligible requests into one row per (request, step at risk).

    ``edges`` is accepted so the ablation runs through the same adapter as
    ``grouped_rows``; the per-step expansion carries steps, not bins, so it is
    otherwise unused.
    """
    X = np.asarray(X, dtype=np.float32)
    rt_all = np.asarray(rtilde, dtype=np.int64)
    eligible = rt_all >= 0
    rt = rt_all[eligible]
    d = np.asarray(delta, dtype=bool)[eligible]

    if rt.shape[0] > PER_STEP_MAX_REQUESTS:
        raise MemoryError(
            f"{rt.shape[0]} requests exceeds PER_STEP_MAX_REQUESTS={PER_STEP_MAX_REQUESTS}")
    counts = rt + 1
    total = int(counts.sum())
    if total > MAX_GROUPED_ROWS:
        raise MemoryError(f"{total} per-step rows exceeds MAX_GROUPED_ROWS={MAX_GROUPED_ROWS}")

    request_index = np.repeat(np.arange(rt.shape[0], dtype=np.int64), counts)
    step = _within_request_index(counts, np.int32)
    y = (step == np.repeat(rt, counts)).astype(np.float32) * np.repeat(d, counts)
    return {
        "X": X[eligible][request_index],
        "step": step,
        "y": y,
        "request_index": request_index,
        "eligible_index": np.flatnonzero(eligible),
        "n_rows": total,
    }
