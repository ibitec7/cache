"""Fit and predict benchmark on one real store window.

Two subcommands so a fixture is built once and every configuration is measured on
exactly the same matrix:

    mrc-bench build  --store data/publish/cluster45 --window 5000000 --out /tmp/fx
    mrc-bench run    --fixture /tmp/fx --config cpu_rowwise --config cpu_colwise

`build` writes `rows.npz` (the grouped risk-set rows `fit_grouped` consumes),
`eval.npz` (the valid and test request matrices, the capacity grid, the truth
curve) and `meta.json`. `run` fits each configuration on those rows, scores the
test block, and appends one JSON record per configuration, so a result is only
ever read back from the file it was written to.

Input contract: `--store` points at a publish directory with `manifest.json`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from mrc.config import CAP_QUANTILE, N_BINS, N_CAPACITIES, NUM_THREADS, SEED
from mrc.data.labels import labels_from_trace
from mrc.data.store import StoreShard
from mrc.eval.metrics import curve_metrics
from mrc.features import build_features
from mrc.gt.stackdist import capacity_grid_from_edges, mrc_from_labels
from mrc.models.hazard import assign_bins
from mrc.pipeline import DEFAULT_PARAMS, cluster_edges, prepare


def cmd_build(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = StoreShard(args.store, args.start, args.window)
    started = time.perf_counter()
    trace = store.read()
    read_s = time.perf_counter() - started

    started = time.perf_counter()
    labels = labels_from_trace(trace, cap=CAP_QUANTILE)
    labels_s = time.perf_counter() - started
    del trace

    started = time.perf_counter()
    X, names, _cat = build_features(labels)
    features_s = time.perf_counter() - started

    edges = cluster_edges(labels, N_BINS)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    truth = (caps, mrc_from_labels(labels, caps))
    keep = ("R", "eligible", "freq", "n_keys")
    slim = {k: labels[k] for k in keep}

    started = time.perf_counter()
    prep = prepare(args.cluster, slim, X, names, truth, row_budget=args.row_budget)
    prepare_s = time.perf_counter() - started

    rows = prep.rows
    assert rows is not None
    np.savez(out / "rows.npz", X=rows["X"], bin=rows["bin"], y=rows["y"], w=rows["w"],
             request_index=rows["request_index"], eligible_index=rows["eligible_index"])

    te = prep.masks[2] & prep.eligible
    np.savez(out / "eval.npz", X_eval=prep.X[te], capacities=caps, edges=edges,
             truth_test=prep.truth_blocks["test"], truth_valid=prep.truth_blocks["valid"])
    meta = {
        "cluster": args.cluster, "store": str(args.store), "start": args.start,
        "window": args.window, "row_budget": args.row_budget,
        "features": names, "n_rows": int(rows["n_rows"]),
        "n_eval": int(te.sum()), "n_requests": int(prep.meta["n_requests"]),
        "p_compulsory_test": float(prep.p_compulsory["test"]),
        "edges_last": int(edges[-1]), "edges_bins": int(len(edges) - 1),
        "seconds": {"read": read_s, "labels": labels_s, "features": features_s,
                    "prepare": prepare_s},
        "default_params": {k: v for k, v in DEFAULT_PARAMS.items() if k != "objective"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "features"}, indent=1))


CONFIGS: dict[str, dict] = {
    "cpu_rowwise": {},
    "cpu_colwise": {"force_row_wise": False, "force_col_wise": True},
}


def _fit(rows: dict, names: list[str], params: dict, rounds: int, threads: int,
         train: float, valid: float, n_requests: int) -> tuple[lgb.Booster, float, float]:
    """Same construction as `HazardModel.fit_grouped`, timed end to end."""
    started = time.perf_counter()
    full = names + ["hazard_step"]
    X = np.empty((rows["X"].shape[0], len(full)), dtype=np.float32)
    X[:, :-1] = rows["X"]
    X[:, -1] = rows["bin"]
    y = np.asarray(rows["y"], dtype=np.float32)
    w = np.asarray(rows["w"], dtype=np.float32)
    orig = rows["eligible_index"][rows["request_index"]]
    tr = orig < int(n_requests * train)
    va = (orig >= int(n_requests * train)) & (orig < int(n_requests * valid))
    dtrain = lgb.Dataset(X[tr], label=y[tr], weight=w[tr], feature_name=full)
    dvalid = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtrain)
    del X, y, w
    p = dict(params)
    p.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
              "seed": SEED, "deterministic": True, "num_threads": threads})
    booster = lgb.train(p, dtrain, num_boost_round=rounds,
                        valid_sets=[dvalid], callbacks=[lgb.early_stopping(50, verbose=False)])
    elapsed = time.perf_counter() - started
    return booster, elapsed, float(booster.best_score["valid_0"]["binary_logloss"])


def _predict(booster: lgb.Booster, X_eval: np.ndarray, caps: np.ndarray,
             edges: np.ndarray, p_comp: float, threads: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    limit = int(edges[-2])
    if int(caps.max()) >= limit:
        raise ValueError(f"capacity {int(caps.max())} at or above limit {limit}")
    kmax = int(min(len(edges) - 2, assign_bins(np.array([int(caps.max())]), edges)[0])) + 1
    steps = np.arange(kmax + 1)
    X = np.asarray(X_eval, dtype=np.float32)
    n, K = X.shape[0], steps.size
    S = np.empty((n, K), dtype=np.float64)
    per = max(1, 2_000_000 // K)
    for start in range(0, n, per):
        block = X[start:start + per]
        grid = np.empty((block.shape[0] * K, X.shape[1] + 1), dtype=np.float32)
        grid[:, :-1] = np.repeat(block, K, axis=0)
        grid[:, -1] = np.tile(steps, block.shape[0])
        h = np.asarray(booster.predict(grid, num_threads=threads)).reshape(-1, K)
        S[start:start + block.shape[0]] = np.cumprod(1.0 - np.clip(h, 1e-9, 1 - 1e-9), axis=1)
    bins_c = np.clip(assign_bins(np.asarray(caps, dtype=np.int64) - 1, edges), 0, kmax)
    return p_comp + (1.0 - p_comp) * S[:, bins_c].mean(axis=0), time.perf_counter() - started


def cmd_run(args: argparse.Namespace) -> None:
    fx = Path(args.fixture)
    meta = json.loads((fx / "meta.json").read_text())
    with np.load(fx / "rows.npz") as z:
        rows = {k: z[k] for k in z.files}
    with np.load(fx / "eval.npz") as z:
        ev = {k: z[k] for k in z.files}
    results = Path(args.results)
    for name in args.config:
        if name not in CONFIGS:
            raise SystemExit(f"unknown config {name}; have {sorted(CONFIGS)}")
        params = dict(DEFAULT_PARAMS)
        params.update(CONFIGS[name])
        params.pop("objective", None)
        booster, fit_s, logloss = _fit(rows, meta["features"], params, args.rounds,
                                       args.threads, 0.85, 1.0, meta["n_requests"])
        pred, pred_s = _predict(booster, ev["X_eval"], ev["capacities"], ev["edges"],
                               meta["p_compulsory_test"], args.threads)
        mae = curve_metrics(pred, ev["truth_test"], ev["capacities"])["mae"]
        rec = {"config": name, "params": {k: params[k] for k in sorted(params)},
               "fit_seconds": round(fit_s, 2), "predict_seconds": round(pred_s, 2),
               "best_iteration": int(booster.best_iteration or 0),
               "valid_logloss": logloss, "test_mae": mae,
               "n_rows": meta["n_rows"], "threads": args.threads,
               "lightgbm": lgb.__version__}
        with results.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--store", required=True)
    b.add_argument("--cluster", required=True)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--window", type=int, default=5_000_000)
    b.add_argument("--row-budget", type=int, default=10_000_000)
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--config", action="append", default=[])
    r.add_argument("--rounds", type=int, default=400)
    r.add_argument("--threads", type=int, default=NUM_THREADS)
    r.add_argument("--results", default="bench_results.jsonl")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
