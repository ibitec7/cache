"""The prediction path must be a rewrite, not an approximation.

`predict_survival` builds a grid of (request, step) pairs, predicts the hazard on
it, and scans it into a survival curve. The optimised version allocates that grid
once in float32 and does the clip, complement and prefix scan in place, which is
only legitimate if every produced number is the same one: the grid must be the
same bytes (nothing is rounded that was not rounded before) and the scan must run
in the same order. These gates are `array_equal` with no tolerance, over several
chunk sizes so that the chunk boundary itself is exercised.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pytest

from mrc.models.hazard import HazardModel, assign_bins


def reference_survival(booster: lgb.Booster, X: np.ndarray, steps: np.ndarray,
                       max_grid_rows: int = 2_000_000) -> np.ndarray:
    """The pre-optimisation body, kept as the reference the rewrite must match."""
    X = np.asarray(X, dtype=np.float32)
    n, K = X.shape[0], len(steps)
    step = np.asarray(steps, dtype=np.int64)
    out = np.empty((n, K), dtype=np.float64)
    per = max(1, int(max_grid_rows) // max(K, 1))
    for start in range(0, n, per):
        block = X[start:start + per]
        grid = np.column_stack([np.repeat(block, K, axis=0),
                                np.tile(step, block.shape[0])])
        pred = np.asarray(booster.predict(grid.astype(np.float32)), dtype=np.float64)
        h = pred.reshape(-1, K)
        out[start:start + block.shape[0]] = np.cumprod(
            1.0 - np.clip(h, 1e-9, 1 - 1e-9), axis=1)
    return out


@pytest.fixture(scope="module")
def fitted() -> tuple[lgb.Booster, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, d = 4000, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    step = rng.integers(0, 5, n)
    Xr = np.column_stack([X, step]).astype(np.float32)
    y = (rng.random(len(Xr)) < 0.3).astype(np.float32)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 7,
                         "min_data_in_leaf": 20, "deterministic": True, "seed": 7},
                        lgb.Dataset(Xr, label=y), num_boost_round=10)
    return booster, X, np.array([1, 2, 3, 7, 15, 31, 63], dtype=np.int64)


def test_grid_is_the_same_bytes(fitted) -> None:
    _booster, X, steps = fitted
    block = X[:37]
    K = len(steps)
    old = np.column_stack([np.repeat(block, K, axis=0),
                           np.tile(np.asarray(steps, np.int64), block.shape[0])])
    old = old.astype(np.float32)
    new = np.empty((block.shape[0] * K, X.shape[1] + 1), dtype=np.float32)
    new[:, :X.shape[1]].reshape(block.shape[0], K, X.shape[1])[:] = block[:, None, :]
    new[:, X.shape[1]].reshape(block.shape[0], K)[:] = np.asarray(steps, np.float32)
    assert np.array_equal(new, old)


@pytest.mark.parametrize("max_grid_rows", [8, 64, 2_000_000])
def test_survival_matches_reference(fitted, max_grid_rows: int) -> None:
    booster, X, steps = fitted
    model = HazardModel(params={}, edges=None, booster=booster)
    assert np.array_equal(model.predict_survival(X, steps, max_grid_rows=max_grid_rows),
                          reference_survival(booster, X, steps, max_grid_rows))


def test_mrc_matches_reference(fitted) -> None:
    booster, X, _steps = fitted
    edges = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 1 << 21])
    caps = np.array([1, 3, 7, 15, 40, 100], dtype=np.int64)
    model = HazardModel(params={}, edges=edges, booster=booster)
    p_comp = 0.12
    bins_c = np.clip(assign_bins(caps - 1, edges), 0, len(edges) - 2)
    kmax = int(min(len(edges) - 2, assign_bins(np.array([int(caps.max())]), edges)[0])) + 1
    ref = reference_survival(booster, X, np.arange(kmax + 1))
    want = p_comp + (1.0 - p_comp) * ref[:, bins_c].mean(axis=0)
    assert np.array_equal(model.predict_mrc(X, caps, edges, p_compulsory=p_comp), want)
