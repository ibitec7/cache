"""End-to-end orchestration for one cluster: labels -> truth -> features -> fit -> metrics.

Design
------
Every stage is a pure function of the artefacts it reads, and every artefact is
cached on disk with a manifest. A rerun is a cache hit, not a recompute.

The hazard model is fitted on a random subsample of the training block's eligible
requests, sized to a fixed grouped-row budget. The likelihood is a sum over
requests, so a uniform subsample is unbiased with more variance; without it a
5M-request window needs 50-85M rows because every censored request contributes one
row per bin at risk. Prediction and scoring use every request in the validation and
test blocks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mrc.config import (
    CAP_QUANTILE,
    LABEL_REQUESTS,
    N_BINS,
    N_CAPACITIES,
    NUM_THREADS,
    PROCESSED,
    REPO,
    SEED,
)
from mrc.data.labels import build_labels, labels_from_trace, load_labels, save_labels
from mrc.data.store import StoreShard
from mrc.eval.metrics import curve_metrics
from mrc.eval.mrc import mrc_curve
from mrc.features import FEATURE_VERSION, build_features
from mrc.gt.stackdist import capacity_grid_from_edges, mrc_from_labels
from mrc.gt.write import written_truth
from mrc.models.hazard import HazardModel, assign_bins, bin_edges, block_masks
from mrc.models.risk_set import grouped_rows

ARTIFACTS = REPO / "artifacts"
ROW_BUDGET = 20_000_000          # grouped rows held in memory for one fit
TRUTH_DIR = ARTIFACTS / "truth"
TABLE_DIR = ARTIFACTS / "tables"

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": NUM_THREADS,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
}


# ---------------------------------------------------------------- artefacts

def labels_path(cluster: str) -> Path:
    return PROCESSED / cluster / "labels.npz"


def features_path(cluster: str) -> Path:
    return PROCESSED / cluster / "features.npz"


def truth_path(cluster: str) -> Path:
    return TRUTH_DIR / f"{cluster}.npz"


def labels_are_cached(path: Path, source: str, limit: int | None, cap: int) -> bool:
    """True when the sidecar manifest names the same source, window and cap."""
    man = path.with_name(path.stem + ".manifest.json")
    if not (path.exists() and man.exists()):
        return False
    try:
        meta = json.loads(man.read_text())
    except json.JSONDecodeError:
        return False
    return (meta.get("source") == str(source) and meta.get("limit") == limit
            and meta.get("cap") == cap)


Source = str | Path | StoreShard


def source_key(source: Source) -> str:
    """Stable cache key for a CSV path or a store shard."""
    return source.key() if isinstance(source, StoreShard) else str(source)


def ensure_labels(cluster: str, source: Source, limit: int | None = LABEL_REQUESTS,
                  cap: int = CAP_QUANTILE) -> dict:
    """Load the label cache, or build and save it. Returns the label dict.

    A `StoreShard` is read as a request window and labelled on its own, so the
    window is the unit of work: peak memory is set by the window, not by the shard
    the window came from.
    """
    out = labels_path(cluster)
    key = source_key(source)
    if labels_are_cached(out, key, limit, cap):
        return load_labels(out)
    labels = (labels_from_trace(source.read(), cap=cap) if isinstance(source, StoreShard)
              else build_labels(source, limit=limit, cap=cap))
    save_labels(out, labels, source=key, limit=limit, cap=cap)
    return labels


def ensure_features(cluster: str, labels: dict) -> tuple[np.ndarray, list[str], list[int]]:
    """Load the design matrix cache, or build it. The labels are the input contract."""
    out = features_path(cluster)
    man = out.with_name(out.stem + ".manifest.json")
    if out.exists():
        with np.load(out, allow_pickle=False) as z:
            return z["X"], list(z["names"]), z["cat_idx"].tolist()
    X, names, cat_idx = build_features(labels)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, names=np.array(names), cat_idx=np.array(cat_idx))
    man.write_text(json.dumps({
        "n_rows": int(X.shape[0]), "n_cols": int(X.shape[1]),
        "feature_version": FEATURE_VERSION, "names": names,
        "source": str(labels_path(cluster)),
    }, indent=2) + "\n")
    return X, names, cat_idx


def ensure_truth(cluster: str, labels: dict, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth curve and its capacity grid, cached under artifacts/truth."""
    out = truth_path(cluster)
    man = out.with_name(out.stem + ".manifest.json")
    if out.exists() and man.exists():
        with np.load(out, allow_pickle=False) as z:
            return z["capacities"], z["mrc"]
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    written_truth(out, labels, n_keys=int(labels["n_keys"][0]),
                  overflow_edge=int(edges[-2]), edges=edges, points=N_CAPACITIES,
                  source_labels=labels_path(cluster))
    return caps, mrc_from_labels(labels, caps)


def cluster_edges(labels: dict, nbins: int = N_BINS) -> np.ndarray:
    """Bin edges from the observed backward distance over eligible requests.

    `rtilde` mixes backward distances with forward censoring bounds, and its upper
    quantiles are inflated by the bounds. The bin grid must resolve the quantity
    being predicted, so it comes from `R` over eligible requests.
    """
    eligible = np.asarray(labels["eligible"], dtype=bool)
    r = np.asarray(labels["R"])[eligible]
    top = int(np.percentile(r, 99.9)) if r.size else 3
    return bin_edges(max(top, 3), nbins)


# ---------------------------------------------------------------- preparation

@dataclass
class Prepared:
    """Everything one fit needs, built once and reused by every Optuna trial."""

    cluster: str
    X: np.ndarray
    names: list[str]
    edges: np.ndarray
    capacities: np.ndarray
    truth: np.ndarray
    masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    eligible: np.ndarray
    rtilde: np.ndarray
    delta: np.ndarray
    p_compulsory: dict[str, float] = field(default_factory=dict)
    truth_blocks: dict[str, np.ndarray] = field(default_factory=dict)
    rows: dict | None = None
    n_fit_requests: int = 0
    meta: dict = field(default_factory=dict)


def prepare(cluster: str, labels: dict, X: np.ndarray, names: list[str],
            truth: tuple[np.ndarray, np.ndarray], nbins: int = N_BINS,
            row_budget: int = ROW_BUDGET, seed: int = SEED,
            edges: np.ndarray | None = None) -> Prepared:
    """Build the bin grid, the split, and the training rows for one cluster.

    Pass `edges` when the labels are a subsample: the bin grid must come from the
    full trace, or the capacity grid and the model's identification limit disagree.
    """
    edges = cluster_edges(labels, nbins) if edges is None else np.asarray(edges)
    caps, truth_curve = truth
    n = int(X.shape[0])
    masks = block_masks(n)
    tr, va, te = masks
    # Target: the BACKWARD reuse distance, which is what MRC(C) = P(R >= C) is a
    # survival function of. Every eligible request has it observed; the only
    # censoring is at the observation ceiling. The forward event (`delta` from
    # compute_reuse) answers a different question and is biased here: measured on
    # cluster45 it gave test MAE 0.170 against 0.005 for the backward target.
    eligible = np.asarray(labels["eligible"], dtype=bool)
    raw_r = np.asarray(labels["R"], dtype=np.int64)
    rtilde = np.where(eligible, np.minimum(raw_r, CAP_QUANTILE), -1).astype(np.int64)
    delta = eligible & (raw_r <= CAP_QUANTILE)

    p_comp = {name: float((~eligible[m]).mean())
              for name, m in (("train", tr), ("valid", va), ("test", te))}

    # Training rows: eligible requests in the training block, capped at the top
    # edge (a request censored beyond the grid contributes no information above it).
    idx = np.flatnonzero(tr & eligible)
    rt_capped = np.minimum(rtilde, int(edges[-1]))
    per_request = float(np.mean(assign_bins(rt_capped[idx], edges) + 1)) if idx.size else 0.0
    max_requests = int(row_budget / max(per_request, 1.0))
    if idx.size > max_requests > 0:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_requests, replace=False))

    truth_blocks = {
        name: mrc_curve(np.asarray(labels["R"])[m], caps)
        for name, m in (("train", tr), ("valid", va), ("test", te))
    }

    rows = grouped_rows(X[idx], rt_capped[idx], delta[idx], edges, len(names))
    return Prepared(
        cluster=cluster, X=X, names=names, edges=edges, capacities=caps,
        truth=truth_curve, masks=masks, eligible=eligible, rtilde=rtilde,
        delta=delta, p_compulsory=p_comp, truth_blocks=truth_blocks,
        rows=rows, n_fit_requests=int(idx.size),
        meta={"n_requests": n, "n_eligible": int(eligible.sum()),
              "grouped_rows": int(rows["n_rows"]), "row_budget": int(row_budget),
              "edges_last": int(edges[-1]), "b_M_minus_1": int(edges[-2]),
              "fit_requests": int(idx.size), "per_request_rows": per_request,
              "target": "backward_reuse_distance"},
    )


# ---------------------------------------------------------------- fitting

def fit(prep: Prepared, params: dict | None = None, num_boost_round: int = 400,
        early_stopping: int = 50) -> HazardModel:
    """Fit one hazard model on the prepared rows, early-stopping on the last 15 %."""
    if prep.rows is None:
        raise ValueError('prepared rows are missing; call prepare() first')
    model = HazardModel(params=dict(params or DEFAULT_PARAMS), nbins=len(prep.edges) - 1)
    model.fit_grouped(prep.rows, prep.edges, prep.names,
                      num_boost_round=num_boost_round, n_requests=prep.n_fit_requests,
                      train=0.85, valid=1.0)
    return model


def predict_block(model: HazardModel, prep: Prepared, block: str) -> np.ndarray:
    """Predicted MRC on one block, averaging over every eligible request in it."""
    mask = dict(zip(("train", "valid", "test"), prep.masks))[block]
    sel = mask & prep.eligible
    Xe = prep.X[sel]
    return model.predict_mrc(Xe, prep.capacities, prep.edges,
                             p_compulsory=prep.p_compulsory[block])


def truth_block(prep: Prepared, block: str) -> np.ndarray:
    """Ground-truth MRC on one block, from every request's backward distance."""
    return prep.truth_blocks[block]


def km_curve(prep: Prepared, block: str) -> np.ndarray:
    """Kaplan-Meier hazard fitted on the training block, scored on `block`."""
    tr = prep.masks[0]
    rt = np.minimum(prep.rtilde[tr], int(prep.edges[-1]))
    keep = prep.eligible[tr]
    bins = assign_bins(rt[keep], prep.edges)
    nb = len(prep.edges) - 1
    events = np.bincount(bins, minlength=nb, weights=None).astype(float)
    hist = np.bincount(bins, minlength=nb).astype(float)
    at_risk = np.cumsum(hist[::-1])[::-1]
    h = np.where(at_risk > 0, events / np.maximum(at_risk, 1e-12), 0.0)
    S = np.cumprod(1.0 - np.clip(h, 0.0, 1.0))
    bin_c = np.clip(assign_bins(prep.capacities - 1, prep.edges), 0, nb - 1)
    pc = prep.p_compulsory[block]
    return pc + (1 - pc) * S[bin_c]


def score(pred: np.ndarray, truth: np.ndarray, caps: np.ndarray) -> dict:
    return curve_metrics(pred, truth, caps)


# ---------------------------------------------------------------- end to end

def capacity_for_target(caps: np.ndarray, curve: np.ndarray, target_hit: float) -> float:
    """Smallest capacity whose predicted hit ratio reaches `target_hit`.

    Returns `nan` when the curve never reaches it on the grid. Interpolates in log
    capacity between the bracketing points, because the grid is log-spaced.
    """
    miss = 1.0 - float(target_hit)
    ok = np.flatnonzero(curve <= miss)
    if ok.size == 0:
        return float("nan")
    i = int(ok[0])
    if i == 0:
        return float(caps[0])
    c0, c1 = float(caps[i - 1]), float(caps[i])
    m0, m1 = float(curve[i - 1]), float(curve[i])
    if m0 == m1:
        return c1
    frac = (m0 - miss) / (m0 - m1)
    return float(np.exp(np.log(c0) + frac * (np.log(c1) - np.log(c0))))


PROVISION_FRACTIONS = (0.60, 0.80, 0.95)


def provisioning(caps, truth, pred, p_compulsory: float,
                 fractions=PROVISION_FRACTIONS) -> list[dict]:
    """Capacity needed for a fraction of the achievable hit-ratio ceiling.

    Compulsory misses bound the hit ratio at `1 - p_compulsory`, so an absolute
    90 % target can be unreachable on a cold-heavy trace. Targets are therefore
    expressed as fractions of that ceiling and reported with it.
    """
    ceiling = 1.0 - float(p_compulsory)
    rows = []
    for fraction in fractions:
        target = fraction * ceiling
        c_true = capacity_for_target(caps, truth, target)
        c_pred = capacity_for_target(caps, pred, target)
        ratio = c_pred / c_true if np.isfinite(c_true) and c_true > 0 else float("nan")
        rows.append({"fraction": float(fraction), "target_hit": float(target),
                     "ceiling": ceiling, "capacity_true": c_true,
                     "capacity_pred": c_pred, "ratio": float(ratio)})
    return rows


def _log_mlflow(cluster: str, params: dict, metrics: dict, artifacts: list[Path],
                prep: Prepared) -> None:
    """One MLflow run per cluster. Silent failure is not allowed: it raises."""
    import mlflow

    from mrc.config import MLFLOW_URI

    Path(MLFLOW_URI.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mrc")
    with mlflow.start_run(run_name=cluster):
        mlflow.log_params({k: v for k, v in params.items() if k != "objective"})
        mlflow.log_params({f"prep_{k}": v for k, v in prep.meta.items()})
        mlflow.log_param("cluster", cluster)
        flat = {}
        for block in ("valid", "test"):
            for name, m in metrics[block].items():
                for key, value in m.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        flat[f"{block}_{name}_{key}"] = float(value)
        mlflow.log_metrics(flat)
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))


def run_one(cluster: str, source: Source, limit: int | None = LABEL_REQUESTS,
            params: dict | None = None, num_boost_round: int = 400,
            nbins: int = N_BINS, row_budget: int = ROW_BUDGET,
            with_mlflow: bool = True, keep_cache: bool = True) -> dict:
    """Run every stage for one cluster and write the metrics artefact."""
    labels = ensure_labels(cluster, source, limit=limit)
    X, names, cat_idx = ensure_features(cluster, labels)
    edges = cluster_edges(labels, nbins)
    truth = ensure_truth(cluster, labels, edges)
    caps, _truth_curve = truth
    # Labels carry the request columns and the censoring intermediates as well as the four
    # arrays the fit reads. Dropping the rest before the risk set is built frees about
    # 375 MB at 5M requests (R, eligible, freq and n_keys are 85 MB of a 460 MB dict).
    labels = {name: labels[name] for name in ("R", "eligible", "freq", "n_keys")}
    prep = prepare(cluster, labels, X, names, truth, nbins=nbins, row_budget=row_budget)
    model = fit(prep, params, num_boost_round=num_boost_round)

    from mrc.eval.baselines import fit_predict_hitcount, fit_predict_single_row

    freq = np.asarray(labels["freq"])
    tr_sel = prep.masks[0] & prep.eligible
    fit_X, fit_rt, fit_delta = prep.X[tr_sel], prep.rtilde[tr_sel], prep.delta[tr_sel]
    fit_freq = freq[tr_sel]

    metrics: dict[str, dict] = {"valid": {}, "test": {}}
    curves: dict[str, np.ndarray] = {"capacities": caps}
    # Only the reported blocks. A train-block row would fit the baselines on the
    # block they are scored on, and no table reads it: it cost 70 % of this stage.
    for block in ("valid", "test"):
        truth_b = truth_block(prep, block)
        pred = predict_block(model, prep, block)
        km = km_curve(prep, block)
        metrics.setdefault(block, {})
        metrics[block]["gbdt"] = score(pred, truth_b, caps)
        metrics[block]["km"] = score(km, truth_b, caps)
        curves[f"truth_{block}"] = truth_b
        curves[f"gbdt_{block}"] = pred
        curves[f"km_{block}"] = km

        sel = dict(zip(("train", "valid", "test"), prep.masks))[block] & prep.eligible
        pc = prep.p_compulsory[block]
        single = fit_predict_single_row(
            fit_X, fit_rt, fit_delta, prep.edges, caps,
            eval_X=prep.X[sel], eval_rtilde=prep.rtilde[sel],
            eval_delta=prep.delta[sel], p_compulsory=pc,
        )
        hit = fit_predict_hitcount(
            fit_X, fit_freq, caps, rtilde=fit_rt,
            eval_X=prep.X[sel], eval_freq=freq[sel], p_compulsory=pc,
        )
        metrics[block]["single_row"] = score(single, truth_b, caps)
        metrics[block]["hitcount"] = score(hit, truth_b, caps)
        curves[f"single_row_{block}"] = single
        curves[f"hitcount_{block}"] = hit

    metrics["provisioning"] = {
        "gbdt": provisioning(caps, curves["truth_test"], curves["gbdt_test"],
                             prep.p_compulsory["test"]),
        "km": provisioning(caps, curves["truth_test"], curves["km_test"],
                           prep.p_compulsory["test"]),
    }
    meta = {"cluster": cluster, "source": source_key(source), "limit": limit,
            "params": dict(params or DEFAULT_PARAMS), "num_boost_round": num_boost_round,
            "features": names, "cat_idx": cat_idx, **prep.meta}
    payload = {"meta": meta, "capacities": caps.tolist(), "metrics": metrics,
               "curves": {k: v.tolist() for k, v in curves.items()}}

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / f"{cluster}_metrics.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    curve_path = TABLE_DIR / f"{cluster}_curves.npz"
    np.savez_compressed(curve_path, **curves)  # type: ignore[arg-type]

    model_dir = ARTIFACTS / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{cluster}.txt"
    if model.booster is None:
        raise RuntimeError('fit produced no booster')
    model.booster.save_model(str(model_path))
    payload["meta"]["model_path"] = str(model_path)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    if with_mlflow:
        _log_mlflow(cluster, dict(params or DEFAULT_PARAMS), metrics, [out, curve_path], prep)
    if not keep_cache:
        # A window sweep keeps the metrics and drops the per-window arrays: labels and
        # features are 165 MB per 5M-request window, which is 443 GB over the full shard.
        for path in (labels_path(cluster), features_path(cluster), truth_path(cluster)):
            path.unlink(missing_ok=True)
            path.with_name(path.stem + ".manifest.json").unlink(missing_ok=True)
    return payload
