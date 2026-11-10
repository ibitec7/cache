"""Miss-ratio curves from a reuse-distance sample.

Under LRU a request hits iff the objects requested since its last access, plus
the requested object itself, still fit in the cache:

    unit-size capacities:   hit  <=>  R_i < C
    byte capacities:        hit  <=>  B_i + s_i < C

R_i is the number of distinct keys requested since the last access, B_i is the
total size of those distinct keys, s_i is the size of the requested object.
Both give

    MRC(C) = P(no hit at capacity C),

an empirical average over requests: a distribution, not a point estimate.
Compulsory accesses (R_i = B_i = -1) always miss.
"""
from __future__ import annotations

import numpy as np

_SENTINEL = np.iinfo(np.int64).max


def mrc_curve(distance, capacities, own_size=None):
    """Empirical MRC at each capacity.

    distance  R (objects) or B (bytes), -1 for compulsory accesses
    own_size  per-request object size, required for byte capacities
    """
    d = np.asarray(distance, dtype=np.int64).copy()
    compulsory = d < 0
    if own_size is not None:
        d = d + np.asarray(own_size, dtype=np.int64)
    d[compulsory] = _SENTINEL      # compulsory accesses always miss
    d_sorted = np.sort(d)
    n = d_sorted.size
    idx = np.searchsorted(d_sorted, np.asarray(capacities), side="left")
    return 1.0 - idx / n


def capacity_grid(max_capacity, points=24):
    """Log-spaced capacities, integers, strictly increasing."""
    grid = np.unique(np.round(np.geomspace(1, max_capacity, points)).astype(np.int64))
    return grid[grid >= 1]


def mrc_error(pred, truth):
    """Pointwise and aggregate error between two curves on the same grid."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    err = pred - truth
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "mre": float(np.mean(np.abs(err) / np.maximum(truth, 1e-12))),
    }
