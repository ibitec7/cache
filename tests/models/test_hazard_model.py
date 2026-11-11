"""HazardModel: grouped fit, block split, survival shape and the identification guard."""
import numpy as np
import pytest

from mrc.models.hazard import HazardModel, _split_rows, block_masks
from mrc.models.risk_set import grouped_rows

PARAMS = {"objective": "binary", "verbosity": -1, "num_threads": 4, "seed": 7,
          "deterministic": True, "force_row_wise": True}
NAMES = [f"f{i}" for i in range(4)]


def _synthetic(n=20_000, seed=5, requested_bins=24):
    rng = np.random.default_rng(seed)
    edges = np.unique(np.round(np.geomspace(2, 4096, requested_bins)).astype(np.int64))
    edges = np.concatenate(([0, 1], edges))
    rtilde = rng.integers(0, 4500, n).astype(np.int64)
    delta = rng.random(n) < 0.35
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, rtilde, delta, edges


def test_fit_grouped_reports_validation_logloss_and_monotone_survival():
    X, rtilde, delta, edges = _synthetic()
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    m = HazardModel(params=dict(PARAMS), nbins=len(edges) - 1)
    m.fit_grouped(rows, edges, NAMES, num_boost_round=60)
    S = m.predict_survival(X[:100], np.arange(len(edges) - 1))
    assert S.shape == (100, len(edges) - 1)
    assert np.all(np.diff(S, axis=1) <= 1e-9)      # non-increasing in step
    assert np.all((S > 0) & (S <= 1))
    assert m.validation_logloss is not None and np.isfinite(m.validation_logloss)
    assert m.edges is edges and m.feature_names == NAMES


def test_split_boundaries_use_original_request_indices():
    X, rtilde, delta, edges = _synthetic(6_000)
    rtilde[:2_000] = -1                            # eligible fraction varies over the trace
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    m = HazardModel(params=dict(PARAMS), nbins=len(edges) - 1)
    m.fit_grouped(rows, edges, NAMES, num_boost_round=10, n_requests=6_000)

    tr, va, te = block_masks(6_000)
    assert int(tr.sum()) == 4_200 and int(va.sum()) == 900 and int(te.sum()) == 900
    assert m.split_n == 6_000

    # the row masks are those of the ORIGINAL request index, not the eligible subset
    orig = rows["eligible_index"][rows["request_index"]]
    tr_rows, va_rows, te_rows = _split_rows(rows, 6_000)
    assert int(tr_rows.sum()) == int((orig < 4_200).sum())
    assert orig[tr_rows].min() >= 2_000 and orig[tr_rows].max() < 4_200
    assert orig[va_rows].min() >= 4_200 and orig[va_rows].max() < 5_100
    assert int(tr_rows.sum() + va_rows.sum()) == int((orig < 5_100).sum())
    assert int(tr_rows.sum() + va_rows.sum() + te_rows.sum()) == orig.size
    assert not (tr_rows & va_rows).any()

    with pytest.raises(ValueError, match="outside n_requests"):
        _split_rows(rows, 900)


def test_two_fits_with_the_same_seed_are_identical():
    X, rtilde, delta, edges = _synthetic(5_000)
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    caps = np.array([1, 10, 100, 1_000])
    curves = []
    for _ in range(2):
        m = HazardModel(params=dict(PARAMS), nbins=len(edges) - 1)
        m.fit_grouped(rows, edges, NAMES, num_boost_round=30, n_requests=5_000)
        curves.append(m.predict_mrc(X[:200], caps, edges, p_compulsory=0.1))
    assert np.array_equal(curves[0], curves[1])
