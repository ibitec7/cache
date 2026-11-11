"""Feature attribution for the fitted hazard model.

Three views of the same question, in increasing order of intervention:

    permutation  shuffle one column, keep its marginal distribution, score the drop
    SHAP         mean |tree contribution| per column, additive within the booster
    ablation     zero a whole feature family, refit, score the MRC curve

Contracts
---------
Attribution is measured on the validation block only; the test block is never
read. `drop` is `base_score - score(permuted)`, so a positive value means the
column helped. SHAP contributions drop LightGBM's final bias column, so the
feature axis matches `names`; a shape mismatch is a `ValueError`, never a silent
misalignment.

`ablation_matrix` zeroes the dropped columns instead of deleting them. The matrix
keeps its shape and column order, so the refit callable sees the same feature
axis for every group and a categorical code stays a code. Zero is a real value
for every column of the design matrix, so a zeroed column is an intervention, not
an absence: a tree can still split on it. That is the declared limitation of this
view, and the reason it is reported beside SHAP rather than instead of it.

The ablation truth curve is the empirical survival of the capped distance
`rtilde`, which equals the exact stack-distance MRC at every capacity below the
cap. The frozen capacity grid is truncated below the overflow edge, so it is
inside that range; a capacity at or above the cap would silently read as a hit.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np

from mrc.eval.mrc import mrc_curve, mrc_error
from mrc.models.hazard import block_masks


def permutation_importance(
    score: Callable[[np.ndarray], float],
    X: np.ndarray,
    names: Sequence[str],
    n_repeats: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Rows `{feature, drop, std}` sorted by mean drop, descending.

    `score` maps a design matrix to a scalar where larger is better (negative MAE,
    negated log-loss, accuracy). The base score is computed once; each repeat
    copies `X`, permutes one column, and scores again, so the shuffled column
    keeps its marginal distribution and loses only its association with the
    target. `std` is the population standard deviation over the repeats.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[1] != len(names):
        raise ValueError(f"X has {X.shape[1]} columns but {len(names)} names were given")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be at least 1, got {n_repeats}")

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    base = float(score(X))
    rows: list[dict] = []
    for j, name in enumerate(names):
        drops = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(n), j]
            drops[r] = base - float(score(Xp))
        rows.append({"feature": name, "drop": float(drops.mean()), "std": float(drops.std())})
    return sorted(rows, key=lambda row: row["drop"], reverse=True)


def shap_summary(
    booster: lgb.Booster,
    X: np.ndarray,
    names: Sequence[str],
    top: int = 12,
    path: str | Path | None = None,
) -> Path | list[dict]:
    """Mean |SHAP contribution| per feature, optionally written as json.

    `booster` is a fitted LightGBM booster; `X` is the matrix it was fitted on, in
    column order, including any extra column the caller appended (the hazard
    step), with `names` naming every one of them. With `path` the json is
    `{"top": [{"feature", "mean_abs_contribution"}], "n_rows"}`, `len(top)` is
    `min(top, len(names))`, and the path is returned. Without a path the rows are
    returned and nothing is written.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if top < 1:
        raise ValueError(f"top must be at least 1, got {top}")

    contrib = np.asarray(booster.predict(X, pred_contrib=True), dtype=np.float64)
    if contrib.ndim != 2 or contrib.shape[1] != len(names) + 1:
        raise ValueError(
            f"pred_contrib returned shape {contrib.shape}; expected "
            f"({X.shape[0]}, {len(names) + 1}) with the bias column last. "
            "Pass every column the booster was fitted on, in order."
        )
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    order = np.argsort(-mean_abs, kind="stable")[:top]
    rows = [{"feature": names[j], "mean_abs_contribution": float(mean_abs[j])} for j in order]
    if path is None:
        return rows
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"top": rows, "n_rows": int(X.shape[0])}, indent=2) + "\n")
    return out


# Frozen ablation families, fixed before any tuning run. There is no family for
# `op` and `is_write`: they are the request type, not a covariate family.
GROUPS: dict[str, tuple[str, ...]] = {
    "all": (),
    "no_namespace": ("ns_code",),
    "no_sizes": ("log_obj_size", "log_key_size", "log_val_size"),
    "no_timing": ("log_ttl", "has_ttl", "log_gap", "log_age"),
    "no_popularity": ("log_freq",),
    "no_client": ("client_bucket",),
}


class FitPredict(Protocol):
    """Refit on a design matrix and return the predicted MRC on `capacities`.

    An implementation applies the frozen `block_masks(n_requests)` split itself
    and uses the hyperparameters frozen by the caller, so every group sees the
    same rows, the same target and the same parameters. `names` names the columns
    of `X` in order; the group list is deliberately not passed, so an
    implementation cannot change the feature axis it is handed.
    """

    def __call__(
        self,
        X: np.ndarray,
        names: list[str],
        rtilde: np.ndarray,
        delta: np.ndarray,
        edges: np.ndarray,
        capacities: np.ndarray,
        n_requests: int,
    ) -> np.ndarray: ...


def ablation_matrix(
    model_fit_predict: FitPredict,
    X: np.ndarray,
    names: Sequence[str],
    rtilde: np.ndarray,
    delta: np.ndarray,
    edges: np.ndarray,
    capacities: np.ndarray,
    n_requests: int,
    *,
    groups: Mapping[str, Sequence[str]] = GROUPS,
) -> list[dict]:
    """Refit once per feature family with that family zeroed; rows sorted by MAE.

    Truth is the empirical survival of `rtilde` on the validation block, exact for
    every capacity below the cap (module docstring). Each row is
    `{group, mae, max_abs, bias, dropped_features}`; `dropped_features` is the
    group's declared list in its declared order. An unknown feature name is a
    `KeyError` raised while the groups are built, before any fit, so a typo cannot
    quietly shrink the matrix. The full model is the `all` group, whose drop list
    is empty.
    """
    X = np.asarray(X, dtype=np.float32)
    names = list(names)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError(f"X must be ({X.shape[0]}, {len(names)}), got {X.shape}")

    rt = np.asarray(rtilde, dtype=np.int64)
    if rt.shape[0] != X.shape[0]:
        raise ValueError(f"rtilde has {rt.shape[0]} rows but X has {X.shape[0]}")

    index = {name: j for j, name in enumerate(names)}
    columns = {group: [index[feature] for feature in drop] for group, drop in groups.items()}

    caps = np.asarray(capacities, dtype=np.int64)
    _train, valid, _test = block_masks(int(n_requests))
    truth = mrc_curve(rt[valid], caps)

    rows: list[dict] = []
    for group, drop in groups.items():
        Xg = X.copy()
        Xg[:, columns[group]] = 0.0
        pred = np.asarray(
            model_fit_predict(
                Xg,
                names,
                rt,
                np.asarray(delta, dtype=bool),
                np.asarray(edges, dtype=np.int64),
                caps,
                int(n_requests),
            ),
            dtype=np.float64,
        )
        if pred.shape != truth.shape:
            raise ValueError(
                f"group {group!r}: prediction has shape {pred.shape}, expected {truth.shape}"
            )
        error = mrc_error(pred, truth)
        rows.append(
            {
                "group": group,
                "mae": error["mae"],
                "max_abs": error["max_abs"],
                "bias": error["bias"],
                "dropped_features": list(drop),
            }
        )
    return sorted(rows, key=lambda row: row["mae"])
