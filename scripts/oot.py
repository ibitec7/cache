"""Out-of-time generalisation run: fit on the first 60 % of a trace, test on the last 25 %.

Why this exists
---------------
Every committed result fits and scores inside one 5M-request window, so a 70/15/15
request split answers "does the model interpolate inside a window". The deployment
question is different: a model fitted on this week's trace is applied to next
week's cache, and the key population, the working set and the operation mix all
move. This script measures that, with time as the split axis.

Design
------
* Blocks are windows of the trace in order: train `[0, 0.60 N)`, validation
  `[0.60 N, 0.75 N)`, test `[0.75 N, N)`. A window never straddles a boundary.
* The bin grid, the fit rows and the training-period curve come from the train
  block only, because at deployment time the later periods do not exist.
* Each window is read, reduced to a histogram, and discarded. Peak memory is one
  window plus the pooled fit rows, not the trace (`prepare_windows`).
* The unit of replication is the window, not the request: the report is per-window
  MAE, its median and IQR, and a moving-block bootstrap CI, because neighbouring
  windows are not independent.

Usage
-----
    uv run python scripts/oot.py --store data/publish/cluster45 --cluster cluster45
    uv run python scripts/oot.py --store DIR --cluster C --limit-windows 3   # smoke

The output is one JSON file with `meta`, `per_window` and `summary`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mrc.config import CAP_QUANTILE, N_BINS, SEED, SPLIT_OOT
from mrc.data.labels import labels_from_trace
from mrc.data.store import Store, StoreShard
from mrc.eval.metrics import curve_metrics, curve_metrics_ci
from mrc.eval.mrc import mrc_curve
from mrc.features import build_features
from mrc.models.hazard import assign_bins
from mrc.pipeline import DEFAULT_PARAMS, ROW_BUDGET, fit, km_from_hist, prepare_windows
from mrc.tuning.search import run_windows_study


def windows_for(n_rows: int, window: int, split: tuple[float, float]) -> list[tuple[str, int, int]]:
    """Time-ordered windows `(block, start, stop)`, none straddling a block boundary."""
    edges = [(0, int(n_rows * split[0]), "train"),
             (int(n_rows * split[0]), int(n_rows * split[1]), "valid"),
             (int(n_rows * split[1]), int(n_rows), "test")]
    out: list[tuple[str, int, int]] = []
    for lo, hi, block in edges:
        start = lo
        while start < hi:
            stop = min(start + window, hi)
            out.append((block, start, stop))
            start = stop
    return out


def load_window(root: str | Path, start: int, stop: int) -> tuple[dict, np.ndarray, list[str]]:
    """Labels and design matrix for one window; nothing is cached, windows are disposable."""
    labels = labels_from_trace(StoreShard(root, start, stop - start).read(), cap=CAP_QUANTILE)
    x, names, _cat = build_features(labels)
    return labels, x, names


def thin_windows(windows: list[tuple[str, int, int]], stride: int = 1,
                 max_windows: int = 0) -> list[tuple[str, int, int]]:
    """Keep every `stride`-th window of each block, never crossing a block boundary.

    Striding is computed inside each block, so no selection ever draws from a later
    period: the train / valid / test split stays exactly the split `windows_for`
    produced. Windows stay in time order within a block, which the fit relies on
    (the last 15 % of the pool is the latest train windows).
    """
    out: list[tuple[str, int, int]] = []
    for block in ("train", "valid", "test"):
        sel = [w for w in windows if w[0] == block][::max(1, stride)]
        if max_windows:
            sel = sel[:max_windows]
        out.extend(sel)
    return out


def block_bootstrap(values: np.ndarray, length: int, reps: int = 2000,
                    seed: int = SEED) -> tuple[float, float]:
    """Percentile CI for the mean of an autocorrelated series of window scores.

    Moving-block bootstrap with a fixed block length: resampling single windows
    would treat neighbouring windows as independent and report a CI that is too
    narrow, which is the failure mode this whole run exists to avoid.
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 3:
        return (float("nan"), float("nan"))
    length = int(max(1, min(length, n - 1)))
    starts = np.arange(0, n - length + 1)
    need = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for r in range(reps):
        pick = rng.choice(starts, size=need, replace=True)
        sample = np.concatenate([v[s:s + length] for s in pick])[:n]
        means[r] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(records: list[dict], block: str) -> dict:
    mae = np.array([r["gbdt"]["mae"] for r in records], dtype=float)
    km = np.array([r["km"]["mae"] for r in records], dtype=float)
    lo, hi = block_bootstrap(mae, length=max(1, round(mae.size ** (1 / 3))))
    return {
        "windows": int(mae.size),
        "mae_mean": float(mae.mean()), "mae_median": float(np.median(mae)),
        "mae_iqr": [float(np.percentile(mae, 25)), float(np.percentile(mae, 75))],
        "mae_ci95_block_bootstrap": [lo, hi],
        "km_mae_mean": float(km.mean()),
        "gbdt_beats_km_windows": int((mae < km).sum()),
        "mae_ratio_to_km": float(mae.mean() / km.mean()) if km.mean() else float("nan"),
        "block": block,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--window", type=int, default=5_000_000)
    ap.add_argument("--split", type=float, nargs=2, default=list(SPLIT_OOT))
    ap.add_argument("--row-budget", type=int, default=ROW_BUDGET)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--params", help="JSON file with LightGBM params")
    ap.add_argument("--limit-windows", type=int, default=0,
                    help="use at most N windows per block (smoke test)")
    ap.add_argument("--window-stride", type=int, default=1,
                    help="keep every K-th window of each block; the split never moves")
    ap.add_argument("--max-windows", type=int, default=0,
                    help="keep at most N windows per block after striding")
    ap.add_argument("--score-sample", type=int, default=0,
                    help="score at most N eligible requests per window (0 = all)")
    ap.add_argument("--trials", type=int, default=0,
                    help="run an Optuna study on the validation windows before fitting")
    ap.add_argument("--study-storage", default=None,
                    help="Optuna storage URL; default is the configured sqlite file")
    ap.add_argument("--edges-from", default=None,
                    help="reuse the bin grid recorded in this run's json instead of "
                         "deriving it from the strided train period")
    ap.add_argument("--save-booster", default=None,
                    help="write the fitted booster here (probes and reuse)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    params = dict(DEFAULT_PARAMS)
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))

    store = Store(args.store)
    started = time.perf_counter()
    windows = windows_for(store.n_rows, args.window, tuple(args.split))
    windows = thin_windows(windows, args.window_stride,
                           args.max_windows or args.limit_windows)
    if not windows:
        raise SystemExit("no window left after striding")
    n_train = sum(1 for w in windows if w[0] == "train")
    print(f"{len(windows)} windows over {store.n_rows:,} requests, {n_train} for training, "
          f"stride {args.window_stride}, score sample {args.score_sample or 'all'}")

    names: list[str] = []
    plan = [w for w in windows if w[0] in ("train", "valid")]
    train_plan = [w for w in plan if w[0] == "train"]
    if not train_plan:
        raise SystemExit("no training window in the plan; shorten --window or check --split")
    # The feature names come from the first window, so it is read once and then
    # replayed through the generator instead of being read twice.
    head_block, head_start, head_stop = train_plan[0]
    head_labels, head_x, names = load_window(args.store, head_start, head_stop)
    print(f"  train window [{head_start:,}, {head_stop:,}) (features known)", flush=True)

    def fit_parts():
        """Train windows (pooled into the fit rows) and validation windows (curve only).

        The test block is deliberately absent: nothing about it may reach the fit or
        the early-stopping set, so it is only read after the model is frozen.
        """
        yield head_block, head_labels, head_x
        for block, start, stop in plan[1:]:
            labels, x, _ = load_window(args.store, start, stop)
            print(f"  {block} window [{start:,}, {stop:,})", flush=True)
            yield block, labels, x

    grid = None
    if args.edges_from:
        grid = np.asarray(json.loads(Path(args.edges_from).read_text())["meta"]["edges"],
                          dtype=np.int64)
    prep = prepare_windows(args.cluster, fit_parts(), names, nbins=N_BINS,
                           row_budget=args.row_budget, n_train_windows=len(train_plan),
                           keep_scored="valid" if args.trials else None,
                           scored_per_window=args.score_sample or 200_000,
                           edges=grid)
    print(f"fit rows {prep.rows['n_rows']:,} from {prep.n_fit_requests:,} pooled requests, "
          f"edges top {int(prep.edges[-1]):,}, {len(prep.scored)} scored valid windows")

    if args.trials:
        study = run_windows_study(prep, args.trials, cluster=args.cluster,
                                  num_boost_round=args.rounds,
                                  score_sample=args.score_sample,
                                  storage=args.study_storage)
        params.update(study.best_params)
        print(f"study mrc-oot-{args.cluster}: {len(study.trials)} trials, "
              f"best valid mae {study.best_value:.6f}, params "
              f"{json.dumps(study.best_params, sort_keys=True)}", flush=True)

    model = fit(prep, params, num_boost_round=args.rounds)
    if args.save_booster and model.booster is not None:
        model.booster.save_model(args.save_booster)
    fit_seconds = time.perf_counter() - started
    train_km = km_from_hist(prep.histograms["train"], prep.edges, prep.capacities,
                            prep.p_compulsory["train"])

    per_window: list[dict] = []
    for block, start, stop in windows:
        if block == "train":
            continue
        t0 = time.perf_counter()
        labels, x, _ = load_window(args.store, start, stop)
        r = np.asarray(labels["R"], dtype=np.int64)
        eligible = np.asarray(labels["eligible"], dtype=bool)
        # Both sides stay honest under sampling: `p_comp` and the truth curve come
        # from the whole window, only the predicted curve is averaged over a sample.
        p_comp = 1.0 - float(eligible.mean())
        truth = mrc_curve(r, prep.capacities)
        idx = np.flatnonzero(eligible)
        if args.score_sample and idx.size > args.score_sample:
            pick = np.random.default_rng(SEED + int(start) % 1_000_003).choice(
                idx, args.score_sample, replace=False)
            idx = np.sort(pick)
        pred_rows = model.predict_mrc_rows(x[idx], prep.capacities, prep.edges,
                                           p_compulsory=p_comp)
        metrics: dict[str, object] = dict(curve_metrics(pred_rows.mean(axis=0), truth,
                                                        prep.capacities))
        if idx.size < int(eligible.sum()):
            metrics["mae_ci95_block_bootstrap"] = list(curve_metrics_ci(
                pred_rows, truth, prep.capacities, seed=SEED))
            metrics["n_scored"] = int(idx.size)
            metrics["n_eligible"] = int(eligible.sum())
        per_window.append({
            "block": block, "start": int(start), "stop": int(stop),
            "requests": int(r.size), "p_compulsory": p_comp,
            "gbdt": metrics,
            "km": curve_metrics(train_km, truth, prep.capacities),
            "seconds": round(time.perf_counter() - t0, 1),
            "bins_used": int(assign_bins(np.asarray(prep.capacities) - 1, prep.edges).max()),
        })
        print(f"  {block} window [{start:,}, {stop:,}) mae {per_window[-1]['gbdt']['mae']:.6f} "
              f"km {per_window[-1]['km']['mae']:.6f}", flush=True)

    summary = {block: summarise([w for w in per_window if w["block"] == block], block)
               for block in ("valid", "test")}
    payload = {
        "meta": {"cluster": args.cluster, "store": str(args.store),
                 "window": args.window, "split": list(args.split),
                 "row_budget": args.row_budget, "rounds": args.rounds,
                 "window_stride": args.window_stride,
                 "max_windows_per_block": args.max_windows or args.limit_windows,
                 "score_sample": args.score_sample,
                 "trials": args.trials,
                 "params": params, "prep": prep.meta,
                 "train_windows": n_train,
                 "train_km": train_km.tolist(),
                 "capacities": prep.capacities.tolist(),
                 "edges": prep.edges.tolist(),
                 "fit_seconds": round(fit_seconds, 1),
                 "total_seconds": round(time.perf_counter() - started, 1)},
        "per_window": per_window,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
