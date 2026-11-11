"""Interpretability driver: permutation, SHAP and feature-family ablation for one cluster.

All three run on the *same* time-block split as the main fit. The ablation refits on a
fixed random subsample of requests (600k), because six refits at the full row budget
would cost an hour; the subsample keeps the time order, so the split stays contiguous.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np

from mrc.config import ARTIFACTS, N_CAPACITIES, SEED
from mrc.data.labels import load_labels
from mrc.gt.stackdist import capacity_grid_from_edges, mrc_from_labels
from mrc.interp.importance import GROUPS, ablation_matrix, permutation_importance, shap_summary
from mrc.models.risk_set import grouped_rows
from mrc.pipeline import (
    DEFAULT_PARAMS,
    cluster_edges,
    ensure_features,
    fit,
    labels_path,
    predict_block,
    prepare,
)

TABLE_DIR = ARTIFACTS / "tables"


def _subsample(labels: dict, n: int, seed: int = SEED) -> tuple[dict, np.ndarray | None]:
    """A uniform random subset of requests, kept in trace order. None index = all rows."""
    total = int(np.asarray(labels["R"]).shape[0])
    if n >= total:
        return labels, None
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    sub = {
        key: np.asarray(value)[idx]
        for key, value in labels.items()
        if np.ndim(value) > 0 and len(value) == total
    }
    sub["n_keys"] = np.asarray(labels["n_keys"])
    return sub, idx


def run_interpret(cluster: str, params: dict | None = None, n_requests: int = 600_000,
                  rounds: int = 200, n_repeats: int = 5) -> dict:
    """Fit on a subsample, then write the importance, SHAP and ablation artefacts."""
    labels = load_labels(labels_path(cluster))
    X, names, _ = ensure_features(cluster, labels)
    edges = cluster_edges(labels)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    truth = mrc_from_labels(labels, caps)
    lab, idx = _subsample(labels, n_requests)
    Xs = X if idx is None else X[idx]
    prep = prepare(cluster, lab, Xs, names, (caps, truth), row_budget=6_000_000,
                   edges=edges)
    model = fit(prep, params or DEFAULT_PARAMS, num_boost_round=rounds)

    valid = prep.masks[1] & prep.eligible
    Xv = prep.X[valid]
    truth_v = prep.truth_blocks["valid"]
    pc = prep.p_compulsory["valid"]

    def score(Xp: np.ndarray) -> float:
        pred = model.predict_mrc(Xp, caps, edges, p_compulsory=pc)
        return -float(np.mean(np.abs(pred - truth_v)))

    importance = permutation_importance(score, Xv, names, n_repeats=n_repeats, seed=SEED)

    rows = grouped_rows(Xv, np.minimum(prep.rtilde[valid], int(edges[-1])),
                        prep.delta[valid], edges, len(names))
    take = np.arange(min(200_000, rows["n_rows"]))
    shap_path = TABLE_DIR / f"{cluster}_shap.json"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    X_shap = np.column_stack([rows["X"][take], rows["bin"][take]]).astype(np.float32)
    if model.booster is None:
        raise RuntimeError('fit produced no booster')
    shap_summary(model.booster, X_shap, names + ["hazard_step"],
                 top=len(names), path=shap_path)

    def refit_predict(Xg, gnames, rt, delta, gedges, gcaps, n_req):
        lab_g = {k: np.asarray(v) for k, v in lab.items() if k not in ("n_keys",)}
        prep_g = prepare(cluster, lab_g, Xg, list(gnames), (gcaps, truth),
                         row_budget=6_000_000, edges=np.asarray(gedges))
        model_g = fit(prep_g, params or DEFAULT_PARAMS, num_boost_round=rounds)
        return predict_block(model_g, prep_g, "valid")

    ablation = ablation_matrix(refit_predict, prep.X, names, prep.rtilde, prep.delta,
                               edges, caps, int(prep.X.shape[0]))

    payload = {
        "cluster": cluster, "block": "valid", "n_requests": int(prep.X.shape[0]),
        "n_repeats": n_repeats, "base_score": score(Xv),
        "features": importance, "shap_top": json.loads(shap_path.read_text())["top"],
        "ablation": ablation, "groups": {k: list(v) for k, v in GROUPS.items()},
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = TABLE_DIR / f"{cluster}_importance.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    return payload
