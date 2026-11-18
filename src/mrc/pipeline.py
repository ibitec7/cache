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
import time
from collections.abc import Iterable
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
    SPLIT_TRAIN,
    SPLIT_VALID,
)
from mrc.data.hist import DistanceHistogram
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
                  cap: int = CAP_QUANTILE, save: bool = True) -> dict:
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
    if save:
        save_labels(out, labels, source=key, limit=limit, cap=cap)
    return labels


def _label_sha256(cluster: str) -> str:
    """The sha256 recorded beside the label cache, empty when it is unreadable."""
    sidecar = labels_path(cluster).with_name("labels.manifest.json")
    try:
        return str(json.loads(sidecar.read_text()).get("sha256", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def _git_sha() -> str:
    """Short HEAD of the repo, empty when git cannot answer."""
    import subprocess

    try:
        done = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def ensure_features(cluster: str, labels: dict,
                    save: bool = True) -> tuple[np.ndarray, list[str], list[int]]:
    """Load the design matrix cache, or build it. The labels are the input contract."""
    out = features_path(cluster)
    man = out.with_name(out.stem + ".manifest.json")
    if out.exists() and man.exists():
        try:
            meta = json.loads(man.read_text())
        except json.JSONDecodeError:
            meta = {}
        if (meta.get("source") == str(labels_path(cluster))
                and meta.get("feature_version") == FEATURE_VERSION):
            with np.load(out, allow_pickle=False) as z:
                return z["X"], list(z["names"]), z["cat_idx"].tolist()
    X, names, cat_idx = build_features(labels)
    if save:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=X, names=np.array(names), cat_idx=np.array(cat_idx))
        man.write_text(json.dumps({
            "n_rows": int(X.shape[0]), "n_cols": int(X.shape[1]),
            "feature_version": FEATURE_VERSION, "names": names,
            "source": str(labels_path(cluster)),
        }, indent=2) + "\n")
    return X, names, cat_idx


def ensure_truth(cluster: str, labels: dict, edges: np.ndarray,
                 save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth curve and its capacity grid, cached under artifacts/truth."""
    out = truth_path(cluster)
    man = out.with_name(out.stem + ".manifest.json")
    if out.exists() and man.exists():
        try:
            meta = json.loads(man.read_text())
        except json.JSONDecodeError:
            meta = {}
        if (meta.get("labels_sha256") == _label_sha256(cluster)
                and meta.get("overflow_edge") == int(edges[-2])):
            with np.load(out, allow_pickle=False) as z:
                return z["capacities"], z["mrc"]
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    if save:
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
    # Request-index interval the early-stopping set comes from. The legacy default
    # stops on the last 15 % of the training block, which is what every committed
    # artefact was fitted with; an out-of-time run sets this to the validation block
    # so the stopping set is the period the design says is validation.
    stop_split: tuple[float, float] = (0.85, 1.0)


def prepare(cluster: str, labels: dict, X: np.ndarray, names: list[str],
            truth: tuple[np.ndarray, np.ndarray], nbins: int = N_BINS,
            row_budget: int = ROW_BUDGET, seed: int = SEED,
            edges: np.ndarray | None = None,
            split: tuple[float, float] | None = None,
            stop_on_validation: bool = False) -> Prepared:
    """Build the bin grid, the split, and the training rows for one cluster.

    Pass `edges` when the labels are a subsample: the bin grid must come from the
    full trace, or the capacity grid and the model's identification limit disagree.

    `split` is the (train_end, valid_end) fraction of the request index, and
    `stop_on_validation` moves the early-stopping set onto the validation block.
    Both default to the frozen behaviour, so every existing artefact reproduces.
    """
    edges = cluster_edges(labels, nbins) if edges is None else np.asarray(edges)
    caps, truth_curve = truth
    n = int(X.shape[0])
    split = (SPLIT_TRAIN, SPLIT_VALID) if split is None else (float(split[0]), float(split[1]))
    masks = block_masks(n, split[0], split[1])
    stop_split = split if stop_on_validation else (0.85, 1.0)
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
        rows=rows, n_fit_requests=int(idx.size), stop_split=stop_split,
        meta={"n_requests": n, "n_eligible": int(eligible.sum()),
              "grouped_rows": int(rows["n_rows"]), "row_budget": int(row_budget),
              "edges_last": int(edges[-1]), "b_M_minus_1": int(edges[-2]),
              "fit_requests": int(idx.size), "per_request_rows": per_request,
              "split": list(split), "stop_split": list(stop_split),
              "target": "backward_reuse_distance"},
    )


# ---------------------------------------------------------------- out of time

WindowPart = tuple[str, dict, np.ndarray]


@dataclass
class ScoredWindow:
    """One non-train window kept so a study can score it without re-reading it.

    `histogram` holds the full window's distances, so `mrc(capacities)` is the exact
    truth curve; `x` is a uniform sample of the eligible design rows and only ever
    feeds the *prediction* side, which makes a sampled score an unbiased estimate of
    the full-window score rather than a different quantity.
    """

    block: str
    p_comp: float
    histogram: DistanceHistogram
    x: np.ndarray
    n_requests: int
    n_eligible: int


@dataclass
class WindowPrepared:
    """Fit-ready reduction of many windows; no request-index split survives.

    `Prepared` describes one contiguous array, so its masks index requests inside
    that array. An out-of-time run has no such array: the train, validation and
    test blocks are different windows, read and discarded one at a time. What
    survives is the bin grid, the sampled fit rows, and one reduced curve and
    compulsory fraction per block.

    `n_fit_requests` counts pooled requests, so `stop_split` splits the pool: the
    pool is a uniform sample of each train window in time order, which makes the
    last 15 % a sample of the latest train windows rather than a random draw.
    """

    cluster: str
    names: list[str]
    edges: np.ndarray
    capacities: np.ndarray
    rows: dict
    n_fit_requests: int
    stop_split: tuple[float, float]
    truth_blocks: dict[str, np.ndarray]
    p_compulsory: dict[str, float]
    histograms: dict[str, DistanceHistogram]
    meta: dict = field(default_factory=dict)
    scored: list[ScoredWindow] = field(default_factory=list)


def prepare_windows(cluster: str, parts: Iterable[WindowPart], names: list[str],
                    nbins: int = N_BINS, row_budget: int = ROW_BUDGET,
                    seed: int = SEED, pool_factor: int = 6,
                    n_train_windows: int | None = None,
                    keep_scored: str | None = None,
                    scored_per_window: int = 200_000,
                    edges: np.ndarray | None = None) -> WindowPrepared:
    """Fit-ready reduction of a time-ordered sequence of windows.

    Each part is `(block, labels, X)` with `block` in train / valid / test, and the
    caller streams it: peak memory is one window plus the pooled fit rows, never
    the whole trace. Every quantity the fit and the report need from `R` is
    accumulated into a histogram (see `mrc.data.hist`), so nothing is
    approximated: the bin grid and the block curves are the ones the arrays give.

    Two choices are deliberate. The bin grid comes from the TRAIN block alone,
    because at deployment time the later periods do not exist yet. The fit rows
    are a uniform sample of the train period, sized by `row_budget`, with a pool
    `pool_factor` times larger so the final draw is uniform in requests rather
    than in bins.

    `edges` overrides that grid. Pass the grid of the full pass when screening with
    a window stride: striding shrinks the train period, which shrinks the p99.9
    distance, which coarsens the grid, so a strided run otherwise tunes against a
    different hazard discretisation than the model it is choosing parameters for.

    `keep_scored` names a block (in practice `valid`) whose windows are retained as
    `ScoredWindow`s: a per-window histogram plus a uniform sample of up to
    `scored_per_window` eligible rows. A hyper-parameter study needs to score the
    validation windows repeatedly, and re-reading them per trial would cost more
    than every fit in the study. `scored_per_window` is a real floor on precision,
    not a free knob: on cluster10, scoring 50,000 of 2.5M eligible requests moved a
    window's MAE by 8 %, 200,000 moved it by 0.4 %.
    """
    hists: dict[str, DistanceHistogram] = {}
    counted: dict[str, list[int]] = {}
    scored: list[ScoredWindow] = []
    pool_x: list[np.ndarray] = []
    pool_rt: list[np.ndarray] = []
    pool_delta: list[np.ndarray] = []
    per_window = max(1, (pool_factor * row_budget) // (nbins + 1))
    if n_train_windows:
        per_window = max(1, per_window // int(n_train_windows))

    for i, (block, labels, X) in enumerate(parts):
        r = np.asarray(labels["R"], dtype=np.int64)
        eligible = np.asarray(labels["eligible"], dtype=bool)
        hists.setdefault(block, DistanceHistogram()).add(r)
        slot = counted.setdefault(block, [0, 0])
        slot[0] += int(r.size)
        slot[1] += int(eligible.sum())
        if keep_scored is not None and block == keep_scored:
            idx = np.flatnonzero(eligible)
            if scored_per_window and idx.size > scored_per_window:
                pick = np.random.default_rng(seed + 7919 * i).choice(
                    idx, scored_per_window, replace=False)
                idx = np.sort(pick)
            scored.append(ScoredWindow(
                block=block, p_comp=1.0 - float(eligible.mean()),
                histogram=DistanceHistogram().add(r),
                x=np.asarray(X[idx], dtype=np.float32),
                n_requests=int(r.size), n_eligible=int(eligible.sum())))
        if block != "train":
            continue
        idx = np.flatnonzero(eligible)
        if idx.size > per_window:
            pick = np.random.default_rng(seed + i).choice(idx, per_window, replace=False)
            idx = np.sort(pick)
        pool_x.append(np.asarray(X[idx], dtype=np.float32))
        pool_rt.append(np.minimum(r[idx], CAP_QUANTILE))
        pool_delta.append(r[idx] <= CAP_QUANTILE)

    if "train" not in hists:
        raise ValueError("no train window was supplied; the bin grid comes from the train period")

    if edges is None:
        top = int(max(hists["train"].percentile(99.9), 3))
        edges = bin_edges(top, nbins)
        edges_from = "train_period"
    else:
        edges = np.asarray(edges, dtype=np.int64)
        edges_from = "given"
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    truth_blocks = {block: h.mrc(caps) for block, h in hists.items()}
    p_comp = {block: 1.0 - counted[block][1] / counted[block][0] for block in hists}

    x_all = np.concatenate(pool_x) if pool_x else np.empty((0, len(names)), np.float32)
    rt_all = np.concatenate(pool_rt) if pool_rt else np.empty(0, np.int64)
    d_all = np.concatenate(pool_delta) if pool_delta else np.empty(0, bool)
    del pool_x, pool_rt, pool_delta
    rt_capped = np.minimum(rt_all, int(edges[-1]))
    per_request = float(np.mean(assign_bins(rt_capped, edges) + 1)) if rt_capped.size else 0.0
    max_requests = int(row_budget / max(per_request, 1.0))
    if rt_capped.size > max_requests > 0:
        pick = np.random.default_rng(seed).choice(rt_capped.size, max_requests, replace=False)
        pick.sort()
        x_all, rt_capped, d_all = x_all[pick], rt_capped[pick], d_all[pick]

    rows = grouped_rows(x_all, rt_capped, d_all, edges, len(names))
    return WindowPrepared(
        cluster=cluster, names=list(names), edges=edges, capacities=caps,
        rows=rows, n_fit_requests=int(rt_capped.size), stop_split=(0.85, 1.0),
        truth_blocks=truth_blocks, p_compulsory=p_comp, histograms=hists,
        scored=scored,
        meta={"n_requests": {b: counted[b][0] for b in counted},
              "n_eligible": {b: counted[b][1] for b in counted},
              "grouped_rows": int(rows["n_rows"]), "row_budget": int(row_budget),
              "pool_requests": int(rt_all.size), "fit_requests": int(rt_capped.size),
              "per_request_rows": per_request, "edges_last": int(edges[-1]),
              "b_M_minus_1": int(edges[-2]), "edges_from": edges_from,
              "stop_split": [0.85, 1.0], "target": "backward_reuse_distance"},
    )


def km_from_hist(hist: DistanceHistogram, edges: np.ndarray, caps: np.ndarray,
                 p_compulsory: float) -> np.ndarray:
    """Kaplan-Meier miss curve from a bin histogram, identical to `km_curve`.

    `km_curve` bins the training block's distances, applies the same binning to the
    capacity grid, and returns `p_comp + (1 - p_comp) * S[bin(C - 1)]`. The bin
    counts are a property of the histogram, so the same curve is available after
    the window has been discarded.
    """
    hist_counts = hist.bin_counts(edges).astype(np.float64)
    nb = len(hist_counts)
    at_risk = np.cumsum(hist_counts[::-1])[::-1]
    h = np.where(at_risk > 0, hist_counts / np.maximum(at_risk, 1e-12), 0.0)
    survival = np.cumprod(1.0 - np.clip(h, 0.0, 1.0))
    bin_c = np.clip(assign_bins(np.asarray(caps, dtype=np.int64) - 1, edges), 0, nb - 1)
    return p_compulsory + (1.0 - p_compulsory) * survival[bin_c]


# ---------------------------------------------------------------- fitting

def fit(prep: Prepared | WindowPrepared, params: dict | None = None,
        num_boost_round: int = 400, early_stopping: int = 50) -> HazardModel:
    """Fit one hazard model on the prepared rows.

    The early-stopping set comes from `prep.stop_split`: the last 15 % of the
    training block by default, the validation block when the prepared object came
    from an out-of-time split.
    """
    if prep.rows is None:
        raise ValueError('prepared rows are missing; call prepare() first')
    model = HazardModel(params=dict(params or DEFAULT_PARAMS), nbins=len(prep.edges) - 1)
    model.fit_grouped(prep.rows, prep.edges, prep.names,
                      num_boost_round=num_boost_round, n_requests=prep.n_fit_requests,
                      train=prep.stop_split[0], valid=prep.stop_split[1])
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


def log_mlflow(cluster: str, params: dict, metrics: dict, artifacts: list[Path],
                meta: dict, *, source: str, params_source: str,
                timings: dict[str, float], tag: str | None = None) -> None:
    """One MLflow run per window (per cluster when the source is a whole trace). Raises on failure."""
    import mlflow

    from mrc.config import MLFLOW_URI

    Path(MLFLOW_URI.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("mrc")
    with mlflow.start_run(run_name=tag or cluster):
        mlflow.set_tags({"cluster": cluster, "params_source": params_source,
                         "git_sha": _git_sha()})
        mlflow.log_params({k: v for k, v in params.items() if k != "objective"})
        mlflow.log_params({f"prep_{k}": v for k, v in meta.items()
                           if isinstance(v, (int, float, str, bool))})
        mlflow.log_params({"cluster": cluster, "source": source})
        flat = {}
        for block in ("valid", "test"):
            for name, m in metrics[block].items():
                for key, value in m.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        flat[f"{block}_{name}_{key}"] = float(value)
        flat.update({f"stage_{name}_seconds": float(value) for name, value in timings.items()})
        mlflow.log_metrics(flat)
        for path in artifacts:
            if Path(path).exists():
                mlflow.log_artifact(str(path))


def run_one(cluster: str, source: Source, limit: int | None = LABEL_REQUESTS,
            params: dict | None = None, num_boost_round: int = 400,
            nbins: int = N_BINS, row_budget: int = ROW_BUDGET,
            with_mlflow: bool = True, keep_cache: bool = True) -> dict:
    """Run every stage for one cluster and write the metrics artefact."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    labels = ensure_labels(cluster, source, limit=limit, save=keep_cache)
    timings["labels"] = time.perf_counter() - start
    start = time.perf_counter()
    X, names, cat_idx = ensure_features(cluster, labels, save=keep_cache)
    timings["features"] = time.perf_counter() - start
    edges = cluster_edges(labels, nbins)
    start = time.perf_counter()
    truth = ensure_truth(cluster, labels, edges, save=keep_cache)
    timings["truth"] = time.perf_counter() - start
    caps, _truth_curve = truth
    # Labels carry the request columns and the censoring intermediates as well as the four
    # arrays the fit reads. Dropping the rest before the risk set is built frees about
    # 375 MB at 5M requests (R, eligible, freq and n_keys are 85 MB of a 460 MB dict).
    labels = {name: labels[name] for name in ("R", "eligible", "freq", "n_keys")}
    start = time.perf_counter()
    prep = prepare(cluster, labels, X, names, truth, nbins=nbins, row_budget=row_budget)
    timings["prepare"] = time.perf_counter() - start
    start = time.perf_counter()
    model = fit(prep, params, num_boost_round=num_boost_round)
    timings["fit"] = time.perf_counter() - start

    from mrc.eval.baselines import fit_predict_hitcount, fit_predict_single_row

    freq = np.asarray(labels["freq"])
    tr_sel = prep.masks[0] & prep.eligible
    fit_X, fit_rt, fit_delta = prep.X[tr_sel], prep.rtilde[tr_sel], prep.delta[tr_sel]
    fit_freq = freq[tr_sel]

    start = time.perf_counter()
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

    timings["metrics"] = time.perf_counter() - start
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
               "edges": [int(e) for e in edges],
               "curves": {k: v.tolist() for k, v in curves.items()}}

    start = time.perf_counter()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    # A windowed source needs its own artefact names: every worker of a sweep writes
    # into the same directories, and keying on the cluster alone made concurrent
    # windows of one cluster overwrite each other's metrics, curves and model.
    # `analysis._WINDOW_TAG` recognises this suffix and keeps windows out of the
    # reported per-cluster tables.
    tag = cluster if not isinstance(source, StoreShard) else f"{cluster}_w{source.start}_{source.rows}"
    out = TABLE_DIR / f"{tag}_metrics.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    curve_path = TABLE_DIR / f"{tag}_curves.npz"
    np.savez_compressed(curve_path, **curves)  # type: ignore[arg-type]

    model_dir = ARTIFACTS / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{tag}.txt"
    if model.booster is None:
        raise RuntimeError('fit produced no booster')
    model.booster.save_model(str(model_path))
    payload["meta"]["model_path"] = str(model_path)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    timings["save"] = time.perf_counter() - start

    if with_mlflow:
        artifacts = [out, curve_path, model_path]
        for path in (labels_path(cluster).with_name("labels.manifest.json"),
                     features_path(cluster).with_name("features.manifest.json"),
                     truth_path(cluster).with_name(f"{cluster}.manifest.json")):
            if path.exists():
                artifacts.append(path)
        log_mlflow(cluster, dict(params or DEFAULT_PARAMS), metrics, artifacts, prep.meta,
                    source=source_key(source),
                    params_source="tuned" if params is not None else "default",
                    timings=timings, tag=tag)
    return payload
