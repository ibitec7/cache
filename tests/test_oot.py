"""Exact-first gates for the out-of-time reduction.

The claim these tests defend: reducing windows to histograms and pooled rows does
not change a single number. Tolerance is 0.0 for the curves, the grid and the fit
rows, because every one of them is a function of counts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from mrc.data.trace import Trace

from mrc.config import CAP_QUANTILE, N_BINS, N_CAPACITIES
from mrc.data.hist import DistanceHistogram
from mrc.eval.mrc import capacity_grid, mrc_curve
from mrc.features import build_features
from mrc.gt.stackdist import capacity_grid_from_edges
from mrc.models.hazard import assign_bins, bin_edges
from mrc.models.risk_set import grouped_rows
from mrc.pipeline import km_curve, km_from_hist, prepare, prepare_windows


def synthetic_trace(n: int, n_keys: int, seed: int) -> Trace:
    """A synthetic `Trace` with a skewed key distribution, in cache-request order."""
    from mrc.data.trace import Trace

    rng = np.random.default_rng(seed)
    vocab = np.array([f"key{i:06d}" for i in range(n_keys)], dtype=object)
    key = rng.zipf(1.2, n).astype(np.int64) % n_keys
    return Trace(
        ts=np.arange(n, dtype=np.int64),
        key=key.astype(np.int32),
        obj_size=rng.integers(64, 4096, n).astype(np.int64),
        client=rng.integers(0, 5, n).astype(np.int32),
        op=rng.integers(0, 4, n).astype(np.int8),
        ttl=np.zeros(n, dtype=np.int64),
        key_vocab=vocab,
    )


def labels_for(n: int, n_keys: int, seed: int) -> tuple[dict, np.ndarray, list[str]]:
    from mrc.data.labels import labels_from_trace

    labels = labels_from_trace(synthetic_trace(n, n_keys, seed), cap=CAP_QUANTILE)
    x, names, _cat = build_features(labels)
    return labels, x, names


@pytest.fixture(scope="module")
def parts() -> list[tuple[str, dict, np.ndarray]]:
    blocks = [("train", 40_000, 11), ("train", 40_000, 12), ("valid", 20_000, 13),
              ("test", 20_000, 14)]
    return [(block, *labels_for(n, 900, seed)[:2]) for block, n, seed in blocks]


def test_histogram_percentile_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    r = np.minimum(rng.zipf(1.25, 300_000), 500_000)
    r[rng.random(r.size) < 0.3] = -1
    hist = DistanceHistogram().add(r)
    for q in (50.0, 90.0, 99.0, 99.9):
        assert int(hist.percentile(q)) == int(np.percentile(r[r >= 0], q))


def test_histogram_mrc_is_bit_identical_to_mrc_curve() -> None:
    rng = np.random.default_rng(4)
    r = np.minimum(rng.zipf(1.3, 200_000), 300_000)
    r[rng.random(r.size) < 0.25] = -1
    hist = DistanceHistogram().add(r)
    caps = capacity_grid(50_000, N_CAPACITIES)
    assert np.array_equal(hist.mrc(caps), mrc_curve(r, caps))


def test_histogram_bins_match_assign_bins() -> None:
    rng = np.random.default_rng(5)
    r = np.minimum(rng.zipf(1.35, 200_000), 300_000)
    r[rng.random(r.size) < 0.2] = -1
    edges = bin_edges(int(np.percentile(r[r >= 0], 99.9)), N_BINS)
    hist = DistanceHistogram().add(r)
    bins = assign_bins(r, edges)
    expected = np.bincount(bins[bins >= 0], minlength=len(edges) - 1)
    assert np.array_equal(hist.bin_counts(edges), expected[:len(hist.bin_counts(edges))])


def test_km_from_hist_matches_km_curve(parts) -> None:
    labels, x, names = labels_for(60_000, 900, 21)
    edges = bin_edges(int(np.percentile(labels["R"][labels["eligible"]], 99.9)), N_BINS)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    prep = prepare("unit", labels, x, names, (caps, mrc_curve(labels["R"], caps)),
                   edges=edges, row_budget=1_000_000)
    hist = DistanceHistogram().add(labels["R"][prep.masks[0]])
    assert np.array_equal(km_from_hist(hist, edges, caps, prep.p_compulsory["test"]),
                          km_curve(prep, "test"))


def test_prepare_windows_reproduces_the_array_path(parts) -> None:
    """Truth curves, compulsory mass and bin grid equal the concatenated arrays."""
    train = [(b, l, x) for b, l, x in parts if b == "train"]
    r_all = np.concatenate([l["R"] for _b, l, _x in train])
    el_all = np.concatenate([l["eligible"] for _b, l, _x in train])
    names = labels_for(1000, 900, 99)[2]
    prep = prepare_windows("unit", parts, names, row_budget=10_000_000,
                           n_train_windows=len(train), pool_factor=10**6)
    edges = bin_edges(int(np.percentile(r_all[el_all], 99.9)), N_BINS)
    assert np.array_equal(prep.edges, edges)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    assert np.array_equal(prep.capacities, caps)
    assert np.array_equal(prep.truth_blocks["train"], mrc_curve(r_all, caps))
    assert prep.p_compulsory["train"] == float(1.0 - el_all.mean())


def test_prepare_windows_rows_equal_grouped_rows_on_the_pool(parts) -> None:
    """With an unbounded pool and budget, the fit rows are grouped_rows of the union."""
    train = [(b, l, x) for b, l, x in parts if b == "train"]
    names = labels_for(1000, 900, 99)[2]
    prep = prepare_windows("unit", parts, names, row_budget=10**9,
                           n_train_windows=len(train), pool_factor=10**6)
    r = np.concatenate([l["R"] for _b, l, _x in train])
    el = np.concatenate([l["eligible"] for _b, l, _x in train])
    x = np.concatenate([x_ for _b, _l, x_ in train])
    rt = np.minimum(r[el], CAP_QUANTILE)
    d = r[el] <= CAP_QUANTILE
    expected = grouped_rows(x[el], rt, d, prep.edges, len(names))
    assert prep.n_fit_requests == int(el.sum())
    assert np.array_equal(prep.rows["bin"], expected["bin"])
    assert np.array_equal(prep.rows["y"], expected["y"])
    assert np.array_equal(prep.rows["X"], expected["X"])


def test_split_plumbing_defaults_to_frozen_behaviour(parts) -> None:
    labels, x, names = labels_for(50_000, 800, 31)
    edges = bin_edges(int(np.percentile(labels["R"][labels["eligible"]], 99.9)), N_BINS)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    truth = (caps, mrc_curve(labels["R"], caps))
    legacy = prepare("unit", labels, x, names, truth, edges=edges, row_budget=500_000)
    assert legacy.stop_split == (0.85, 1.0)
    assert legacy.meta["split"] == [0.70, 0.85]
    oot = prepare("unit", labels, x, names, truth, edges=edges, row_budget=500_000,
                  split=(0.60, 0.75), stop_on_validation=True)
    assert oot.stop_split == (0.60, 0.75)
    n = x.shape[0]
    assert oot.masks[0].sum() == int(n * 0.60)
    assert oot.masks[1].sum() == int(n * 0.75) - int(n * 0.60)
    # the grid and the truth curves do not depend on the split, only the masks do
    assert np.array_equal(legacy.edges, oot.edges)
