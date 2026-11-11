"""Baselines the hazard model has to beat, each returning one MRC vector.

The comparison is what the risk set and the covariates buy, at three levels of
sophistication.

``fit_predict_km``
    Kaplan-Meier hazard on the binned reuse distance, no covariates. This is the
    floor: the censoring correction enters only through the risk set. A censored
    request is at risk up to the bin where observation stops and contributes no
    event, which is the same convention ``mrc.models.risk_set`` uses.
``fit_predict_single_row``
    One row per request, label ``delta`` at the observed distance, and no risk-set
    expansion. Survival is read off directly: a request whose own bin is past the
    capacity bin cannot hit, so its survival is 1 there, and everywhere else the
    survival is the model's ``1 - P(event)``. The approximation is cheap and
    biased, because a request contributes one outcome instead of one row per step
    it survived.
``fit_predict_hitcount``
    Regress ``log1p(freq)`` on the design matrix, then map the predicted frequency
    to a reuse-distance proxy through the training deciles of the prediction: for
    decile ``k`` the proxy is the median training ``rtilde`` of the requests whose
    prediction fell in ``k``. The curve is ``mean(rtilde_hat >= C)``. This is the
    practitioner's shortcut the project argues against: frequency is not a
    distance, so it has to be converted through a marginal, not a per-request,
    statement. Empty deciles, possible only when every prediction is identical,
    take the global training median.

Every function returns float64 on ``capacities``, finite, in ``[p_compulsory, 1]``,
and non-increasing in capacity. The grids are refused at or above the
identification limit ``edges[-2]``, exactly like ``HazardModel.predict_mrc``:
above that edge the overflow bin collects every capped distance and the curve is
not identified.

``p_compulsory=None`` estimates the compulsory fraction as ``mean(rtilde < 0)`` on
the block the curve is averaged over, which is the evaluation block when one is
given. ``fit_predict_hitcount`` can only see the training distances, so a caller
scoring a block with a different cold rate passes ``p_compulsory`` explicitly.

Deviation from the task brief: ``fit_predict_hitcount`` takes ``rtilde`` as a
keyword, because calibrating frequency to distance needs the training reuse
distances and the given signature had no way to receive them. ``fit_predict_km``
does not reuse ``mrc.models.hazard.km_hazard`` because that helper treats every
observed distance as an event and ignores ``delta``; under censoring this
overstates the hazard. It is reimplemented here with the event indicator.
"""
from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import ArrayLike

from mrc.config import N_BINS, SEED
from mrc.models.hazard import assign_bins, bin_edges

PARAMS_CLF: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "num_threads": 4,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
}

PARAMS_REG: dict[str, Any] = {
    **PARAMS_CLF,
    "objective": "regression",
    "metric": "rmse",
}

NUM_BOOST_ROUND = 150
FREQ_DECILES = 10


def _check_capacities(capacities: ArrayLike) -> np.ndarray:
    """Capacities as a strictly increasing vector of positive integers."""
    caps = np.asarray(capacities, dtype=np.int64).ravel()
    if caps.size == 0:
        raise ValueError("capacities is empty")
    if int(caps.min()) < 1:
        raise ValueError(f"capacities must be at least 1, got {int(caps.min())}")
    if caps.size > 1 and not np.all(np.diff(caps) > 0):
        raise ValueError("capacities must be strictly increasing")
    return caps


def _check_X(X: ArrayLike, name: str) -> np.ndarray:
    """A finite 2-D float32 design matrix."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} has no rows")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    return arr


def _check_edges(edges: ArrayLike) -> np.ndarray:
    """Bin boundaries as an increasing integer vector with at least one real bin."""
    arr = np.asarray(edges, dtype=np.int64).ravel()
    if arr.size < 2:
        raise ValueError(f"edges needs at least two boundaries, got {arr.size}")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("edges must be strictly increasing")
    return arr


def _compulsory(p_compulsory: float | None, rtilde: np.ndarray) -> float:
    """Compulsory fraction, measured on the scored block when not supplied."""
    if p_compulsory is None:
        return float(np.mean(np.asarray(rtilde) < 0))
    p = float(p_compulsory)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p_compulsory must be in [0, 1], got {p}")
    return p


def _eval_block(
    eval_arrays: tuple[ArrayLike | None, ...], fit_arrays: tuple[ArrayLike, ...]
) -> tuple[Any, ...]:
    """Use the evaluation block only if all of it was given.

    A partial block would silently average part of the curve over the fitting
    rows and part over the scored rows, which is the leakage this interface
    exists to prevent.
    """
    provided = [array is not None for array in eval_arrays]
    if any(provided) and not all(provided):
        raise ValueError(
            "pass every eval array or none of them: a partial eval block mixes "
            "fitting and evaluation rows"
        )
    return tuple(fit_arrays) if not any(provided) else tuple(eval_arrays)


def _fit_booster(
    X: np.ndarray, y: np.ndarray, params: dict[str, Any] | None, base: dict[str, Any]
) -> lgb.Booster:
    """Train a LightGBM booster; ``params`` overrides the module defaults."""
    config = {**base, **(params or {})}
    rounds = int(config.pop("num_boost_round", NUM_BOOST_ROUND))
    dataset = lgb.Dataset(np.asarray(X, dtype=np.float32), label=y)
    return lgb.train(config, dataset, num_boost_round=rounds)


def fit_predict_km(
    rtilde: ArrayLike,
    delta: ArrayLike,
    capacities: ArrayLike,
    *,
    p_compulsory: float | None = None,
) -> np.ndarray:
    """Kaplan-Meier hazard over the binned reuse distance, then the MRC curve.

    ``h_k = events_k / at_risk_k`` with ``at_risk_k = #{rtilde bin >= k}`` and
    ``events_k = #{delta and rtilde bin == k}``; ``MRC(C) = p_c + (1 - p_c) *
    prod_{b <= bin(C-1)} (1 - h_b)``.
    """
    rt = np.asarray(rtilde, dtype=np.int64).ravel()
    event = np.asarray(delta, dtype=bool).ravel()
    if rt.shape != event.shape:
        raise ValueError(f"rtilde and delta differ in length: {rt.size} vs {event.size}")
    if rt.size == 0:
        raise ValueError("rtilde is empty")
    caps = _check_capacities(capacities)

    eligible = rt >= 0
    if not eligible.any():
        raise ValueError("no eligible request: every rtilde is compulsory")
    max_distance = int(np.percentile(rt[eligible], 99.9))
    edges = bin_edges(max_distance, N_BINS)
    limit = int(edges[-2])
    if int(caps.max()) >= limit:
        raise ValueError(
            f"capacity {int(caps.max())} is at or above the identification limit "
            f"{limit}: the overflow bin makes every capacity from {limit} upward "
            "unidentified. Truncate the grid below the limit."
        )

    bins = assign_bins(rt, edges)
    keep = bins >= 0
    n_bins = len(edges) - 1
    hist = np.bincount(bins[keep], minlength=n_bins).astype(np.float64)
    events = np.bincount(
        bins[keep], weights=event[keep].astype(np.float64), minlength=n_bins
    )
    at_risk = np.cumsum(hist[::-1])[::-1]
    hazard = np.divide(events, at_risk, out=np.zeros(n_bins), where=at_risk > 0)
    survival = np.cumprod(1.0 - hazard)

    bins_c = np.clip(assign_bins(caps - 1, edges), 0, n_bins - 1)
    p = _compulsory(p_compulsory, rt)
    return p + (1.0 - p) * survival[bins_c]


def fit_predict_single_row(
    X: ArrayLike,
    rtilde: ArrayLike,
    delta: ArrayLike,
    edges: ArrayLike,
    capacities: ArrayLike,
    *,
    eval_X: ArrayLike | None = None,
    eval_rtilde: ArrayLike | None = None,
    eval_delta: ArrayLike | None = None,
    p_compulsory: float | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """One row per eligible request, label only ``delta``: biased but cheap.

    The model is ``P(event | x, own bin)``. For a capacity ``C`` with bin
    ``k = bin(C - 1)``, a request whose own bin is past ``k`` never hits, so its
    survival is 1; otherwise survival is ``1 - P(event)``. That is

        MRC(C) = p_c + (1 - p_c) * mean(1 - P(event_i) * 1[bin_i <= bin(C - 1)]).

    The task brief writes the indicator as ``bin_i < bin(C - 1)``; the frozen
    definition hits when ``bin_i <= bin(C - 1)``, so the comparison is ``<=``
    here, which keeps the off-by-one out of the curve.

    ``eval_delta`` is accepted so the evaluation block is passed whole; the curve
    reads the predicted event probability, not the realised labels.
    """
    X_fit = _check_X(X, "X")
    rt_fit = np.asarray(rtilde, dtype=np.int64).ravel()
    d_fit = np.asarray(delta, dtype=bool).ravel()
    if not (X_fit.shape[0] == rt_fit.shape[0] == d_fit.shape[0]):
        raise ValueError(
            f"X, rtilde and delta differ in length: {X_fit.shape[0]}, "
            f"{rt_fit.shape[0]}, {d_fit.shape[0]}"
        )
    edges_arr = _check_edges(edges)
    caps = _check_capacities(capacities)
    limit = int(edges_arr[-2])
    if int(caps.max()) >= limit:
        raise ValueError(
            f"capacity {int(caps.max())} is at or above the identification limit "
            f"{limit}: the overflow bin leaves every capacity from {limit} upward "
            "unidentified"
        )

    X_eval, rt_eval, d_eval = _eval_block(
        (eval_X, eval_rtilde, eval_delta), (X_fit, rt_fit, d_fit)
    )
    X_eval = _check_X(X_eval, "eval_X")
    rt_eval = np.asarray(rt_eval, dtype=np.int64).ravel()
    d_eval = np.asarray(d_eval, dtype=bool).ravel()
    if not (X_eval.shape[0] == rt_eval.shape[0] == d_eval.shape[0]):
        raise ValueError(
            f"eval_X, eval_rtilde and eval_delta differ in length: {X_eval.shape[0]}, "
            f"{rt_eval.shape[0]}, {d_eval.shape[0]}"
        )

    bins_fit = assign_bins(rt_fit, edges_arr)
    keep = bins_fit >= 0
    if not keep.any():
        raise ValueError("no eligible request to fit the single-row baseline on")
    design = np.column_stack([X_fit[keep], bins_fit[keep]]).astype(np.float32)
    booster = _fit_booster(
        design, d_fit[keep].astype(np.float32), params, PARAMS_CLF
    )

    bins_eval = assign_bins(rt_eval, edges_arr)
    eligible = bins_eval >= 0
    if not eligible.any():
        raise ValueError("no eligible request in the evaluation block")
    predict_X = np.column_stack([X_eval[eligible], bins_eval[eligible]]).astype(np.float32)
    p_event = np.asarray(booster.predict(predict_X), dtype=np.float64)

    n_bins = len(edges_arr) - 1
    bins_c = np.clip(assign_bins(caps - 1, edges_arr), 0, n_bins - 1)
    own_bin = bins_eval[eligible]
    survival = np.array(
        [np.where(own_bin <= k, 1.0 - p_event, 1.0).mean() for k in bins_c]
    )
    p = _compulsory(p_compulsory, rt_eval)
    return p + (1.0 - p) * survival


def fit_predict_hitcount(
    X: ArrayLike,
    freq: ArrayLike,
    capacities: ArrayLike,
    *,
    rtilde: ArrayLike | None = None,
    eval_X: ArrayLike | None = None,
    eval_freq: ArrayLike | None = None,
    p_compulsory: float | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Regress ``log1p(freq)`` on ``X`` and map the prediction to a distance proxy.

    ``rtilde`` is the training reuse distance used only to calibrate the mapping:
    predicted frequency falls in a decile, and the proxy is the median training
    ``rtilde`` of the requests in that decile. The curve is
    ``p_c + (1 - p_c) * mean(rtilde_hat >= C)``.
    """
    X_fit = _check_X(X, "X")
    f_fit = np.asarray(freq, dtype=np.float64).ravel()
    if X_fit.shape[0] != f_fit.shape[0]:
        raise ValueError(f"X and freq differ in length: {X_fit.shape[0]} vs {f_fit.size}")
    if np.any(f_fit < 0.0):
        raise ValueError("freq must be non-negative")
    caps = _check_capacities(capacities)
    if rtilde is None:
        raise ValueError(
            "fit_predict_hitcount needs the training rtilde to calibrate the "
            "frequency-to-distance mapping"
        )
    rt_fit = np.asarray(rtilde, dtype=np.int64).ravel()
    if rt_fit.shape[0] != X_fit.shape[0]:
        raise ValueError(
            f"X and rtilde differ in length: {X_fit.shape[0]} vs {rt_fit.size}"
        )
    eligible = rt_fit >= 0
    if not eligible.any():
        raise ValueError("no eligible request to calibrate the mapping on")

    X_eval, f_eval = _eval_block((eval_X, eval_freq), (X_fit, f_fit))
    X_eval = _check_X(X_eval, "eval_X")
    f_eval = np.asarray(f_eval, dtype=np.float64).ravel()
    if X_eval.shape[0] != f_eval.shape[0]:
        raise ValueError(
            f"eval_X and eval_freq differ in length: {X_eval.shape[0]} vs {f_eval.size}"
        )
    if np.any(f_eval < 0.0):
        raise ValueError("eval_freq must be non-negative")

    booster = _fit_booster(
        X_fit, np.log1p(f_fit).astype(np.float32), params, PARAMS_REG
    )
    fit_freq = np.expm1(booster.predict(X_fit)).astype(np.float64)
    cuts = np.quantile(fit_freq, np.linspace(0.0, 1.0, FREQ_DECILES + 1))[1:-1]
    fit_decile = np.clip(
        np.searchsorted(cuts, fit_freq, side="right"), 0, FREQ_DECILES - 1
    )
    global_median = float(np.median(rt_fit[eligible]))
    proxy = np.array(
        [
            float(np.median(rt_fit[eligible & (fit_decile == k)]))
            if np.any(eligible & (fit_decile == k))
            else global_median
            for k in range(FREQ_DECILES)
        ]
    )

    eval_freq = np.expm1(booster.predict(X_eval)).astype(np.float64)
    eval_decile = np.clip(
        np.searchsorted(cuts, eval_freq, side="right"), 0, FREQ_DECILES - 1
    )
    rtilde_hat = proxy[eval_decile]

    survival = np.array([np.mean(rtilde_hat >= capacity) for capacity in caps])
    p = _compulsory(p_compulsory, rt_fit)
    return p + (1.0 - p) * survival
