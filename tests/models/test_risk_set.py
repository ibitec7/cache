"""Risk-set expansion: counts, labels, weights, index mapping and the frozen guards."""
import numpy as np
import pytest

from mrc.models.hazard import assign_bins
from mrc.models.risk_set import grouped_rows


def test_grouped_rows_counts_and_labels():
    edges = np.array([0, 1, 2, 4, 8, 16])          # 5 bins
    rtilde = np.array([0, 2, 4, -1], dtype=np.int64)
    delta = np.array([True, True, False, False])
    eligible = np.array([True, True, True, False])
    X = np.arange(8, dtype=np.float32).reshape(4, 2)
    b = assign_bins(rtilde, edges)
    rows = grouped_rows(X, rtilde, delta, edges, n_features=2)
    assert rows["n_rows"] == int((b[eligible] + 1).sum())
    assert rows["X"].shape == (rows["n_rows"], 2)
    assert rows["bin"].max() == b[eligible].max()
    # exactly one positive label in the first bin, one in the bin of rtilde=2 (bin 1)
    assert int(rows["y"].sum()) == 2
    # weights are bin widths, clipped at 1
    assert rows["w"].min() >= 1.0


def test_overflow_bin_is_the_last_edge_minus_one():
    edges = np.array([0, 1, 2, 4, 8])
    assert assign_bins(np.array([7]), edges)[0] == 3
    assert assign_bins(np.array([8]), edges)[0] == 3   # capped, never index 4
    assert assign_bins(np.array([10_000]), edges)[0] == 3


def test_grouped_rows_columns_and_index_mapping():
    edges = np.array([0, 1, 2, 4, 8])
    rtilde = np.array([-1, 3, 0, 7, -1], dtype=np.int64)   # eligible requests 1, 2, 3
    delta = np.array([False, True, False, True, False])
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    rows = grouped_rows(X, rtilde, delta, edges, n_features=2)

    # contract dtypes
    assert rows["X"].dtype == np.float32
    assert rows["bin"].dtype == np.int16
    assert rows["y"].dtype == np.float32
    assert rows["w"].dtype == np.float32
    assert rows["request_index"].dtype == np.int64
    assert rows["eligible_index"].dtype == np.int64

    # eligible_index indexes all requests, request_index indexes the eligible subset
    assert rows["eligible_index"].tolist() == [1, 2, 3]
    assert rows["request_index"].tolist() == [0, 0, 0, 1, 2, 2, 2, 2]
    assert rows["bin"].tolist() == [0, 1, 2, 0, 0, 1, 2, 3]
    assert rows["n_rows"] == 8
    # weights are the clipped bin widths of edges = [0, 1, 2, 4, 8]
    assert rows["w"].tolist() == [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 4.0]
    # one event in the bin where the observed reuse lands, none for the censored request
    assert rows["y"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    # rows of a request carry that request's features, in bin order
    assert rows["X"][0].tolist() == [2.0, 3.0]
    assert rows["X"][-1].tolist() == [6.0, 7.0]
    assert np.all(np.bincount(rows["request_index"], weights=rows["y"]) <= 1.0)


def test_grouped_rows_refuses_above_the_row_budget(monkeypatch):
    import mrc.models.risk_set as rs

    monkeypatch.setattr(rs, "MAX_GROUPED_ROWS", 5)
    edges = np.array([0, 1, 2, 4, 8])
    rtilde = np.array([3, 3], dtype=np.int64)      # 4 + 4 = 8 rows
    delta = np.array([True, True])
    X = np.zeros((2, 1), dtype=np.float32)
    with pytest.raises(MemoryError, match="MAX_GROUPED_ROWS"):
        rs.grouped_rows(X, rtilde, delta, edges, n_features=1)


def test_grouped_rows_rejects_a_mismatched_feature_count():
    edges = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="n_features"):
        grouped_rows(np.zeros((2, 3), dtype=np.float32), np.array([1, 0]), np.ones(2, bool),
                     edges, n_features=2)


def test_grouped_and_per_step_agree_on_survival():
    from mrc.models.risk_set import per_step_rows

    rng = np.random.default_rng(4)
    n = 5_000
    edges = np.unique(np.round(np.geomspace(2, 512, 16)).astype(np.int64))
    edges = np.concatenate(([0, 1], edges))
    rtilde = rng.integers(0, 600, n).astype(np.int64)
    delta = rng.random(n) < 0.4
    rtilde[:500] = -1             # ineligible: both expanders must drop it
    delta[500:2_000] = False      # censored: present at risk, no positive label
    X = rng.normal(size=(n, 3)).astype(np.float32)

    g = grouped_rows(X, rtilde, delta, edges, 3)
    p = per_step_rows(X, rtilde, delta)
    # different granularity: bins at risk vs unit steps at risk
    assert abs(g["n_rows"] - p["n_rows"]) > 0
    # same requests, same events, same order
    assert set(g["eligible_index"].tolist()) == set(p["eligible_index"].tolist())
    assert g["eligible_index"].tolist() == p["eligible_index"].tolist()
    assert g["y"].sum() == p["y"].sum() == int(delta[500:].sum())
    assert np.all(np.bincount(g["request_index"], weights=g["y"]) <= 1.0)
    assert np.all(np.bincount(p["request_index"], weights=p["y"]) <= 1.0)
