"""Chunked survival prediction must equal one-shot prediction exactly."""
import numpy as np

from mrc.models.hazard import HazardModel


def _toy_model() -> HazardModel:
    """Two-round booster on random rows: enough to exercise the predict path."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 13)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_threads": 1},
                        lgb.Dataset(X, label=y), num_boost_round=2)
    model = HazardModel(params={}, nbins=8)
    model.booster = booster
    return model


def test_chunked_prediction_is_bit_identical_to_one_shot():
    model = _toy_model()
    X = np.random.default_rng(1).normal(size=(137, 12)).astype(np.float32)
    steps = np.arange(7)

    one_shot = model.predict_survival(X, steps, max_grid_rows=10**12)
    chunked = model.predict_survival(X, steps, max_grid_rows=13 * 7 // 3)  # 3 chunks
    per_row = model.predict_survival(X, steps, max_grid_rows=1)           # one row at a time

    assert np.array_equal(one_shot, chunked)
    assert np.array_equal(one_shot, per_row)


def test_survival_is_monotone_and_bounded():
    model = _toy_model()
    X = np.random.default_rng(2).normal(size=(50, 12)).astype(np.float32)
    S = model.predict_survival(X, np.arange(6), max_grid_rows=64)

    assert S.shape == (50, 6)
    assert np.all(S <= 1.0) and np.all(S >= 0.0)
    assert np.all(np.diff(S, axis=1) <= 1e-12)
