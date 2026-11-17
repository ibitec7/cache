"""Contract tests for the pipeline stage that produces every headline number.

The implementation is already correct, so these tests are mutation-first: each
one names a measured failure mode of `mrc.pipeline` and goes red when that
specific line changes.

1. The capacity grid must be a subset of the bin upper edges. A grid log-spaced
   independently of the edges charges the model for the width of the bin it lands
   in: measured inside a bin, truth 0.473 against prediction 0.327, max error
   0.147, versus 0.021 on the edges.
2. The fitted target is the backward distance `rtilde`, not the forward event
   `delta`. Fitting on `delta` gave test MAE 0.170 with +0.170 bias on cluster45
   against 0.00027 for the backward target.
3. The bin grid is taken from the raw backward distance `R`, never from the
   capped `rtilde`, whose upper quantiles are inflated by the censoring ceiling.

Inputs are synthetic arrays, so the file runs in well under a second and needs no
trace data.
"""
from __future__ import annotations

import numpy as np

from mrc.config import CAP_QUANTILE, N_CAPACITIES
from mrc.eval.mrc import mrc_curve
from mrc.gt.stackdist import capacity_grid_from_edges
from mrc.models.hazard import assign_bins
from mrc.pipeline import cluster_edges, prepare

NAMES = ["f0", "f1", "f2"]


def _X(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, len(NAMES))).astype(np.float32)


def _labels(r: np.ndarray, eligible: np.ndarray | None = None) -> dict:
    """Minimal label dict with the two fields `prepare` reads, plus `n_keys`."""
    r = np.asarray(r, dtype=np.int64)
    if eligible is None:
        eligible = r >= 0
    return {"R": r, "eligible": np.asarray(eligible, dtype=bool),
            "n_keys": np.array([int(r.max()) + 1], dtype=np.int64)}


def _split(labels: dict) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Edges, capacities and the (capacities, truth curve) pair, as `run_one` builds them."""
    edges = cluster_edges(labels)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    return edges, caps, (caps, mrc_curve(labels["R"], caps))


def test_the_capacity_grid_is_bin_upper_edges():
    n = 4_000
    rng = np.random.default_rng(11)
    r = rng.integers(1, 20_000, n)
    r[rng.random(n) < 0.05] = -1                       # compulsory accesses
    labels = _labels(r)
    edges, _caps, truth = _split(labels)
    prep = prepare("synthetic", labels, _X(n), NAMES, truth)

    inner = edges[1:-2]                                # bin upper edges, overflow excluded
    assert np.all(np.isin(prep.capacities, inner))
    assert np.all(np.diff(prep.capacities) > 0)
    assert int(prep.capacities.max()) < int(edges[-2])
    assert prep.capacities.size >= 8


def test_the_fitted_target_is_the_backward_distance():
    n = 400
    pos = 100
    rng = np.random.default_rng(5)
    r = rng.integers(1, 60, n)
    # The premise: four requests observed beyond the cap. Their backward distance
    # is observable (rtilde = CAP) but the forward event delta is False.
    beyond = np.arange(100, 104)
    r[beyond] = CAP_QUANTILE + rng.integers(1, 5_000, beyond.size)
    labels = _labels(r)
    _edges, _caps, truth = _split(labels)
    prep = prepare("synthetic", labels, _X(n), NAMES, truth)

    assert not prep.delta[pos], "premise: forward event is False for an R beyond the cap"
    assert int(prep.rtilde[pos]) == CAP_QUANTILE

    tr = prep.masks[0] & prep.eligible
    tr_idx = np.flatnonzero(tr)
    assert prep.n_fit_requests == int(tr.sum())        # default budget keeps every request

    rows = prep.rows
    assert rows is not None
    # The risk set is built from the backward distance, so every eligible training
    # request is in it and a request's row count is its own bin plus one.
    assert np.unique(rows["eligible_index"]).size == prep.n_fit_requests
    rt_capped = np.minimum(prep.rtilde[tr_idx], int(prep.edges[-1]))
    assert int(rows["n_rows"]) == int((assign_bins(rt_capped, prep.edges) + 1).sum())
    # The four beyond-cap requests are censored at the ceiling, not dropped: no
    # event, but they still contribute the bins they survived.
    assert np.count_nonzero(rows["y"] == 1) == int(prep.delta[tr].sum())

    fit_pos = int(np.flatnonzero(tr_idx == pos)[0])
    sel = rows["request_index"] == fit_pos
    assert sel.sum() == int(assign_bins(np.array([rt_capped[fit_pos]]), prep.edges)[0]) + 1
    assert rows["y"][sel].sum() == 0
    assert int(rows["bin"][sel][-1]) == int(assign_bins(np.array([rt_capped[fit_pos]]),
                                                        prep.edges)[0])


def test_the_bin_grid_comes_from_the_backward_distance_not_the_cap():
    n = 2_000
    rng = np.random.default_rng(17)
    r = rng.integers(1, 1_000, n)
    censored = rng.choice(np.arange(50, 150), 40, replace=False)
    r[censored] = CAP_QUANTILE + rng.integers(1, 10_000, censored.size)
    labels = _labels(r)
    assert int(np.percentile(r, 99.9)) > CAP_QUANTILE   # premise: the tail is beyond the cap
    _edges, _caps, truth = _split(labels)
    prep = prepare("synthetic", labels, _X(n), NAMES, truth)

    assert int(prep.edges[-1]) > CAP_QUANTILE


def test_prepare_is_deterministic():
    n = 800
    rng = np.random.default_rng(3)
    r = rng.integers(1, 5_000, n)
    labels = _labels(r)
    _edges, _caps, truth = _split(labels)
    budget = 400                                       # force the subsample path
    a = prepare("synthetic", labels, _X(n), NAMES, truth, row_budget=budget)
    b = prepare("synthetic", labels, _X(n), NAMES, truth, row_budget=budget)

    eligible_in_train = int((a.masks[0] & labels["eligible"]).sum())
    assert 0 < a.n_fit_requests < eligible_in_train
    assert a.n_fit_requests == b.n_fit_requests
    rows_a, rows_b = a.rows, b.rows
    assert rows_a is not None and rows_b is not None
    for key in ("X", "bin", "y", "request_index", "eligible_index"):
        assert np.array_equal(rows_a[key], rows_b[key]), key
    for m_a, m_b in zip(a.masks, b.masks):
        assert np.array_equal(m_a, m_b)
