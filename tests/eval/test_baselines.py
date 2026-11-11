"""Contract tests for the three evaluation baselines (Phase E3).

Each baseline stands in for the hazard model at a different level of sophistication: a
covariate-free Kaplan-Meier curve, a one-row-per-request ablation, and a regression that
turns requested frequency into a reuse-distance proxy. The tests pin the curve contract
(shape, range, monotonicity), a hand-computed KM answer that fixes the censoring
convention, and the rule that the fitting block and the scoring block can differ (audit
M-22).
"""
import numpy as np
import pytest

from mrc.eval.baselines import (
    fit_predict_hitcount,
    fit_predict_km,
    fit_predict_single_row,
)
from mrc.eval.mrc import mrc_curve
from mrc.models.hazard import bin_edges

CAPACITIES = np.array([1, 8, 64, 256, 512, 900])
EDGES = bin_edges(2048, 16)


def _toy(n: int = 4000, seed: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    rtilde = rng.integers(0, 2000, n).astype(np.int64)
    delta = rng.random(n) < 0.4
    return X, rtilde, delta


def _assert_curve(curve: np.ndarray, capacities: np.ndarray, p_compulsory: float = 0.0) -> None:
    assert curve.shape == capacities.shape
    assert np.all(np.isfinite(curve))
    assert np.all((curve >= p_compulsory - 1e-12) & (curve <= 1.0 + 1e-12))
    assert np.all(np.diff(curve) <= 1e-12)


def _mae(curve: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(curve - truth)))


def _frequency_trace(
    n: int = 8000, seed: int = 41, eval_signal: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A trace where frequency predicts distance in the train block only.

    ``freq`` is generated from the latent covariate ``s``; distance falls with ``s``.
    With ``eval_signal=False`` the eval block draws distances independently of ``s``,
    so any mapping learned on train carries no information there.
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    X = np.column_stack([s, rng.normal(size=n)]).astype(np.float32)
    freq = np.maximum(1, np.round(np.exp(1.0 * s + 0.2 * rng.normal(size=n)))).astype(np.int64)
    train = np.arange(n) < int(0.7 * n)
    distance = 600.0 * np.exp(-1.0 * s) + 30.0 * rng.random(n)
    if not eval_signal:
        distance[~train] = rng.uniform(0.0, 4000.0, int((~train).sum()))
    rtilde = np.round(distance).astype(np.int64)
    rtilde[rng.random(n) < 0.2] = -1
    return X, freq, rtilde, train


def _with_deep_censored(early: list[int], events: list[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Early requests plus deep censored ones, so the identification limit is high."""
    rtilde = np.concatenate([np.asarray(early, dtype=np.int64), np.full(994, 100_000)])
    delta = np.concatenate([np.asarray(events, dtype=bool), np.zeros(994, dtype=bool)])
    return rtilde, delta


def test_km_curve_reproduces_a_hand_computed_survival():
    rtilde, delta = _with_deep_censored([0, 0, 1, 1, 2, 2], [True] * 6)
    # censoring sits beyond every capacity, so S(k) = 1 - cumulative events / 1000
    curve = fit_predict_km(rtilde, delta, np.array([1, 2, 3]), p_compulsory=0.1)
    assert np.allclose(curve, 0.1 + 0.9 * np.array([0.998, 0.996, 0.994]))


def test_km_curve_counts_censored_requests_in_the_risk_set_only():
    rtilde = np.concatenate([np.array([1, 1], dtype=np.int64), np.full(100, 100_000)])
    delta = np.concatenate([np.array([True, False]), np.zeros(100, dtype=bool)])
    # bin 0 is empty (h=0); bin 1 has 102 at risk and one event (h=1/102)
    curve = fit_predict_km(rtilde, delta, np.array([1, 2]))
    assert np.allclose(curve, [1.0, 1.0 - 1.0 / 102.0])


def test_km_curve_defaults_to_the_compulsory_fraction_measured_on_the_arrays():
    rtilde = np.concatenate([np.array([-1, -1, 0, 1, 2]), np.full(995, 100_000)])
    delta = np.concatenate([np.ones(5, dtype=bool), np.zeros(995, dtype=bool)])
    # p_c = 2/1000 and S(2) = 995/998, so the curve is 1 - 3/1000
    curve = fit_predict_km(rtilde, delta, np.array([3]))
    assert np.allclose(curve, [0.997])


def test_km_curve_is_monotone_and_in_range():
    _, rtilde, delta = _toy()
    curve = fit_predict_km(rtilde, delta, CAPACITIES, p_compulsory=0.1)
    _assert_curve(curve, CAPACITIES, p_compulsory=0.1)


def test_km_curve_refuses_capacities_at_the_identification_limit():
    rtilde = np.arange(100, dtype=np.int64)
    delta = np.ones(100, dtype=bool)
    with pytest.raises(ValueError, match="identification limit"):
        fit_predict_km(rtilde, delta, np.array([10_000]))


def test_single_row_curve_is_monotone_bounded_and_finite():
    X, rtilde, delta = _toy()
    curve = fit_predict_single_row(X, rtilde, delta, EDGES, CAPACITIES, p_compulsory=0.1)
    _assert_curve(curve, CAPACITIES, p_compulsory=0.1)


def test_single_row_does_not_fit_on_the_block_it_scores():
    rng = np.random.default_rng(71)
    n = 6000
    s = rng.normal(size=n)
    X = np.column_stack([s, rng.normal(size=n)]).astype(np.float32)
    rtilde = np.round(700.0 * np.exp(-0.9 * s) + 20.0 * rng.random(n)).astype(np.int64)
    train = np.arange(n) < int(0.7 * n)
    test = ~train
    delta = np.zeros(n, dtype=bool)
    delta[train] = rng.random(int(train.sum())) < 0.5      # no signal in train
    delta[test] = s[test] > 0.0                            # signal only in test
    fair = fit_predict_single_row(
        X[train], rtilde[train], delta[train], EDGES, CAPACITIES, p_compulsory=0.1,
        eval_X=X[test], eval_rtilde=rtilde[test], eval_delta=delta[test],
    )
    leaky = fit_predict_single_row(
        X[test], rtilde[test], delta[test], EDGES, CAPACITIES, p_compulsory=0.1
    )
    assert not np.array_equal(fair, leaky)
    truth = mrc_curve(rtilde[test], CAPACITIES)
    assert _mae(fair, truth) > _mae(leaky, truth)


def test_single_row_refuses_a_partial_eval_block():
    X, rtilde, delta = _toy()
    with pytest.raises(ValueError, match="eval"):
        fit_predict_single_row(X, rtilde, delta, EDGES, CAPACITIES, eval_X=X)


def test_hitcount_curve_is_monotone_bounded_and_finite():
    X, freq, rtilde, train = _frequency_trace()
    curve = fit_predict_hitcount(
        X[train], freq[train], CAPACITIES, rtilde=rtilde[train], p_compulsory=0.2
    )
    _assert_curve(curve, CAPACITIES, p_compulsory=0.2)


def test_hitcount_beats_a_constant_on_a_frequency_signal():
    X, freq, rtilde, train = _frequency_trace()
    test = ~train
    curve = fit_predict_hitcount(
        X[train], freq[train], CAPACITIES, rtilde=rtilde[train],
        eval_X=X[test], eval_freq=freq[test], p_compulsory=0.2,
    )
    truth = mrc_curve(rtilde[test], CAPACITIES)
    flat = np.full(CAPACITIES.shape, truth.mean())
    assert _mae(curve, truth) < _mae(flat, truth)


def test_hitcount_is_worse_on_eval_when_the_signal_exists_only_in_train():
    X, freq, rtilde, train = _frequency_trace(eval_signal=False)
    test = ~train
    on_train = fit_predict_hitcount(
        X[train], freq[train], CAPACITIES, rtilde=rtilde[train], p_compulsory=0.2
    )
    on_eval = fit_predict_hitcount(
        X[train], freq[train], CAPACITIES, rtilde=rtilde[train],
        eval_X=X[test], eval_freq=freq[test], p_compulsory=0.2,
    )
    truth_train = mrc_curve(rtilde[train], CAPACITIES)
    truth_eval = mrc_curve(rtilde[test], CAPACITIES)
    flat_eval = np.full(CAPACITIES.shape, truth_eval.mean())
    assert _mae(on_train, truth_train) < _mae(on_eval, truth_eval)
    assert _mae(on_eval, truth_eval) > _mae(flat_eval, truth_eval)


def test_hitcount_needs_training_distances_to_calibrate_the_mapping():
    X, rtilde, _ = _toy()
    freq = np.ones(rtilde.shape[0], dtype=np.int64)
    with pytest.raises(ValueError, match="rtilde"):
        fit_predict_hitcount(X, freq, CAPACITIES)
