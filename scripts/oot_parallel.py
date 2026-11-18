"""Fit once, then score every held-out window in parallel; the farm driver for OOT and LOCO.

Why a second driver
-------------------
`scripts/oot.py` is sequential by design: one model, windows read one at a time, so
peak memory is one window. On a 128-vCPU box that leaves 127 cores idle while a
673-window test block scores at ten seconds per window. The work that parallelises
is the *prediction* pass, not the fit: the model is frozen before any test window is
opened, so each window is independent and writes its own JSON.

Three phases, each resumable and each able to run without the others:

  fit     stream the train block, optionally tune on the validation block, freeze
          the booster, write `model.json` (grid, capacities, logit-side constants).
  score   fan out `--workers` subprocesses over the windows of one block; existing
          per-window files are skipped, so a killed run resumes where it stopped.
  reduce  concatenate the per-window files into the same payload `oot.py` writes,
          so every downstream table and plot reads one schema.

Draw one window at a time with `window`, which is what the fan-out calls; it exists
as a subcommand so a worker can be reproduced by hand for a single window.

Usage
-----
    uv run python scripts/oot_parallel.py fit --cluster cluster45 \\
        --store data/publish/cluster45 --out artifacts/oot/cluster45
    uv run python scripts/oot_parallel.py score --model artifacts/oot/cluster45/model.json \\
        --block test --workers 48
    uv run python scripts/oot_parallel.py reduce --model artifacts/oot/cluster45/model.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from mrc.config import N_BINS, SPLIT_OOT
from mrc.models.hazard import HazardModel
from mrc.pipeline import DEFAULT_PARAMS, ROW_BUDGET

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oot import load_window, score_window, summarise, thin_windows, windows_for


def split_specs(specs: list[str]) -> list[tuple[str, str]]:
    """`cluster:/path` pairs, in the order given."""
    out = []
    for s in specs:
        name, _, path = s.partition(":")
        if not path:
            raise SystemExit(f"--store wants cluster:path, got {s!r}")
        out.append((name, path))
    return out


def cmd_fit(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params = dict(DEFAULT_PARAMS)
    if args.params:
        params.update(json.loads(Path(args.params).read_text()))
    stores = split_specs(args.store)

    model, prep, train_km, params, names = fit_pooled(stores, params, args)
    booster = out / "booster.txt"
    if model.booster is not None:
        model.booster.save_model(str(booster))
    spec = {
        "cluster": args.cluster, "stores": [{"name": n, "path": p} for n, p in stores],
        "window": args.window, "split": list(args.split), "row_budget": args.row_budget,
        "rounds": args.rounds, "trials": args.trials, "score_sample": args.score_sample,
        "window_stride": args.window_stride, "max_windows": args.max_windows,
        "params": params, "prep": prep.meta, "nbins": int(len(prep.edges) - 1),
        "edges": prep.edges.tolist(), "capacities": prep.capacities.tolist(),
        "train_km": train_km.tolist(), "feature_names": list(names),
        "booster": booster.name, "fit_seconds": prep.meta.get("seconds"),
    }
    (out / "model.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(f"wrote {out / 'model.json'}")


def fit_pooled(sources: list[tuple[str, str]], params: dict, args):
    """One model over several stores: their train windows pool into the fit rows.

    The grid, the pooled rows and `p_compulsory` therefore come from every
    training cluster and never from the held-out one, which is the LOCO rule.
    """
    from mrc.data.store import Store
    from mrc.pipeline import fit, km_from_hist, prepare_windows

    plans: list[tuple[str, str, list[tuple[str, int, int]]]] = []
    for name, path in sources:
        st = Store(path)
        w = thin_windows(windows_for(st.n_rows, args.window, tuple(args.split)),
                         args.window_stride, args.max_windows)
        plans.append((name, path, w))
        print(f"{name}: {len(w)} windows over {st.n_rows:,} requests", flush=True)

    n_train = sum(1 for _n, _p, w in plans for b, _s, _e in w if b == "train")
    if not n_train:
        raise SystemExit("no training window in the plan")
    names = load_window(plans[0][1], next(w for w in plans[0][2] if w[0] == "train")[1],
                        next(w for w in plans[0][2] if w[0] == "train")[2])[2]

    def parts():
        # Store by store, block by block, in time order inside each store: the fit
        # relies on the pool's last 15 % being the latest train windows.
        for name, path, w in plans:
            for block, start, stop in w:
                if block == "test":
                    continue
                labels, x, _ = load_window(path, start, stop)
                print(f"  {name} {block} [{start:,}, {stop:,})", flush=True)
                yield block, labels, x

    prep = prepare_windows(args.cluster, parts(), names, nbins=N_BINS,
                           row_budget=args.row_budget, n_train_windows=n_train,
                           keep_scored="valid" if args.trials else None,
                           scored_per_window=args.score_sample or 200_000)
    print(f"fit rows {prep.rows['n_rows']:,} from {prep.n_fit_requests:,} pooled requests, "
          f"edges top {int(prep.edges[-1]):,}", flush=True)
    if args.trials:
        from mrc.tuning.search import run_windows_study
        study = run_windows_study(prep, args.trials, cluster=args.cluster,
                                  num_boost_round=args.rounds,
                                  score_sample=args.score_sample,
                                  storage=args.study_storage)
        params = {**params, **study.best_params}
        print(f"study {args.cluster}: {len(study.trials)} trials, best "
              f"{study.best_value:.6f}", flush=True)
    t0 = time.perf_counter()
    model = fit(prep, params, num_boost_round=args.rounds)
    prep.meta["seconds"] = round(time.perf_counter() - t0, 1)
    train_km = km_from_hist(prep.histograms["train"], prep.edges, prep.capacities,
                            prep.p_compulsory["train"])
    return model, prep, train_km, params, names


def cmd_window(args: argparse.Namespace) -> None:
    """Score one window with a frozen booster (what each worker runs)."""
    import lightgbm as lgb

    spec = json.loads(Path(args.model).read_text())
    edges = np.asarray(spec["edges"], dtype=np.int64)
    model = HazardModel(params=spec["params"], nbins=int(spec["nbins"]),
                        booster=lgb.Booster(model_file=str(Path(args.model).parent
                                                          / spec["booster"])),
                        edges=edges, feature_names=spec.get("feature_names"))

    class _Spec:  # the three attributes `score_window` reads off a WindowPrepared
        pass

    prep = _Spec()
    prep.capacities = np.asarray(spec["capacities"])
    prep.edges = edges
    rec = score_window(model, prep, np.asarray(spec["train_km"]), args.store,
                       args.block, args.start, args.stop, spec["score_sample"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec) + "\n")


def windows_for_block(spec: dict, block: str, store_path: str) -> list[tuple[int, int]]:
    """Every window of `block` in the store being scored.

    `max_windows` caps the fit only: a capped test block would turn the per-window
    distribution into a sample of it, and that distribution is the reported result.
    """
    from mrc.data.store import Store

    st = Store(store_path)
    w = windows_for(st.n_rows, spec["window"], tuple(spec["split"]))
    return [(s, e) for b, s, e in w if b == block]


def cmd_score(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.model).read_text())
    store = args.store or spec["stores"][0]["path"]  # LOCO scores the held-out store
    windows = windows_for_block(spec, args.block, store)
    if args.limit:
        windows = windows[: args.limit]
    outdir = Path(args.out or (Path(args.model).parent / args.block))
    outdir.mkdir(parents=True, exist_ok=True)
    todo = []
    for start, stop in windows:
        target = outdir / f"{start}.json"
        if target.exists() and not args.force:
            continue
        todo.append((start, stop, target))
    print(f"{args.block}: {len(windows)} windows, {len(todo)} to score, "
          f"{args.workers} workers", flush=True)
    if not todo:
        return
    started = time.perf_counter()
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", str(args.threads))

    def run(item):
        start, stop, target = item
        cmd = [sys.executable, __file__, "window", "--model", args.model,
               "--store", store, "--block", args.block, "--start", str(start),
               "--stop", str(stop), "--out", str(target)]
        for attempt in (1, 2):
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
            if p.returncode == 0:
                return start, None
            last = p.stderr[-800:]
        return start, last

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start, err in pool.map(run, todo):
            done += 1
            if err:
                print(f"FAILED [{start}] {err}", file=sys.stderr, flush=True)
            elif done % 25 == 0 or done == len(todo):
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"  {done}/{len(todo)} windows, {rate:.2f}/s, "
                      f"eta {(len(todo) - done) / max(rate, 1e-9):.0f}s", flush=True)


def cmd_reduce(args: argparse.Namespace) -> None:
    spec = json.loads(Path(args.model).read_text())
    root = Path(args.model).parent
    per_window: list[dict] = []
    for block in ("valid", "test"):
        for f in sorted((root / block).glob("*.json"), key=lambda p: int(p.stem)):
            per_window.append(json.loads(f.read_text()))
    if not per_window:
        raise SystemExit(f"no per-window files under {root}")
    per_window.sort(key=lambda r: (r["block"], r["start"]))
    summary = {b: summarise([w for w in per_window if w["block"] == b], b)
               for b in ("valid", "test")
               if any(w["block"] == b for w in per_window)}
    payload = {"meta": {**spec, "n_scored_windows": len(per_window)},
               "per_window": per_window, "summary": summary}
    out = Path(args.out or (root / "result.json"))
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--cluster", required=True)
    f.add_argument("--store", action="append", required=True,
                   help="cluster:path, repeatable; several stores pool the fit rows")
    f.add_argument("--window", type=int, default=5_000_000)
    f.add_argument("--split", type=float, nargs=2, default=list(SPLIT_OOT))
    f.add_argument("--row-budget", type=int, default=ROW_BUDGET)
    f.add_argument("--rounds", type=int, default=200)
    f.add_argument("--trials", type=int, default=0)
    f.add_argument("--score-sample", type=int, default=200_000)
    f.add_argument("--window-stride", type=int, default=1)
    f.add_argument("--max-windows", type=int, default=0)
    f.add_argument("--params", default=None)
    f.add_argument("--study-storage", default=None)
    f.add_argument("--out", required=True)

    s = sub.add_parser("score")
    s.add_argument("--model", required=True)
    s.add_argument("--store", default=None)
    s.add_argument("--block", choices=["valid", "test"], default="test")
    s.add_argument("--workers", type=int, default=48)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--out", default=None)

    r = sub.add_parser("reduce")
    r.add_argument("--model", required=True)
    r.add_argument("--out", default=None)

    w = sub.add_parser("window")
    w.add_argument("--model", required=True)
    w.add_argument("--store", required=True)
    w.add_argument("--block", required=True)
    w.add_argument("--start", type=int, required=True)
    w.add_argument("--stop", type=int, required=True)
    w.add_argument("--out", required=True)

    args = ap.parse_args()
    {"fit": cmd_fit, "score": cmd_score, "reduce": cmd_reduce,
     "window": cmd_window}[args.cmd](args)


if __name__ == "__main__":
    main()
