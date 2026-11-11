"""Error metrics for miss-ratio curves.

Every metric is defined on one capacity grid, takes `(pred, truth, capacities)` in
that order, and returns a plain Python float or int, so a metric dict goes into the
metrics json and into MLflow without a numpy scalar leaking through.

Relative error is reported only where the truth clears `mre_floor`, and the number
of points that cleared it is reported beside it: dividing by a vanishing miss ratio
measures nothing, so a reader has to see how thin the average is.

ABC integrates `|pred - truth|` in log capacity and divides by the log range, so it
is an average displacement over the grid, not a number that grows with the count of
capacities. Two curves on different grids are therefore not comparable, which is why
the grid is frozen (Phase B) and every ABC is reported with its `n`.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _curves(pred: ArrayLike, truth: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The two curves as 1-D float arrays, refused when they are not the same length."""
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    if p.shape != t.shape:
        raise ValueError(f"pred and truth differ in length: {p.size} vs {t.size}")
    return p, t


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule."""
    return float(np.trapezoid(y, x))


def abc(pred: ArrayLike, truth: ArrayLike, capacities: ArrayLike) -> float:
    """Area between the curves, integrated in log C and normalised by the log range."""
    p, t = _curves(pred, truth)
    caps = np.asarray(capacities, dtype=float).ravel()
    if caps.size != p.size:
        raise ValueError(f"{caps.size} capacities for {p.size} curve points")
    if caps.size < 2:
        raise ValueError("abc needs at least two capacities")
    lg = np.log(caps)
    if not np.all(np.diff(lg) > 0.0):
        raise ValueError("capacities must be strictly increasing")
    return _trapezoid(np.abs(p - t), lg) / float(lg[-1] - lg[0])


def curve_metrics(
    pred: ArrayLike, truth: ArrayLike, capacities: ArrayLike, mre_floor: float = 0.01
) -> dict[str, float | int]:
    """MAE, MAX, bias, MRE and log-normalised ABC on one capacity grid.

    `mre` is `mean(|err| / truth)` over the points with `truth >= mre_floor`, and is
    `nan` when no point qualifies; `mre_n` says how many points the average rests on.
    """
    p, t = _curves(pred, truth)
    err = p - t
    keep = t >= mre_floor
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "mre": float(np.mean(np.abs(err[keep]) / t[keep])) if keep.any() else float("nan"),
        "mre_n": int(keep.sum()),
        "abc": abc(p, t, capacities),
        "n": int(p.size),
    }
