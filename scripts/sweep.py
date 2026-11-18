"""Run one training window per process over a store, resuming from what already exists.

Why subprocesses and not a pool
-------------------------------
`NUM_THREADS` is read at import time by `mrc.config`, and LightGBM plus numba each
take every core they are given. A worker therefore has to pick its thread count
before the interpreter starts, which a subprocess does cleanly and an in-process
pool cannot. Each worker writes its own artefacts (the tag in `run_one` keys them
on the window), so nothing is shared and nothing needs locking.

Resume is the point: a 2,685-window trace is hours of work, so a re-run must skip
windows whose metrics file already exists. `--force` recomputes them.

    uv run python scripts/sweep.py --store data/publish/cluster52 --cluster cluster52 \
        --workers 8 --window 5000000 --row-budget 10000000 --out artifacts/tables/sweep52
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

from mrc.config import NUM_THREADS, REPO
from mrc.data.store import Store


def plan_windows(n_rows: int, window: int, limit: int | None) -> list[tuple[int, int]]:
    out = []
    start = 0
    while start < n_rows:
        stop = min(start + window, n_rows)
        out.append((start, stop - start))
        start = stop
    return out[:limit] if limit else out


def tag_for(cluster: str, start: int, rows: int) -> str:
    return f"{cluster}_w{start}_{rows}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--window", type=int, default=5_000_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0,
                    help="LightGBM threads per worker; 0 = cores // workers")
    ap.add_argument("--row-budget", type=int, default=10_000_000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", required=True, help="prefix for the results file and log dir")
    args = ap.parse_args()

    n_rows = Store(args.store).n_rows
    windows = plan_windows(n_rows, args.window, args.limit or None)
    threads = args.threads or max(1, (os.cpu_count() or 8) // args.workers)
    tables = REPO / "artifacts" / "tables"
    out = Path(args.out)
    log_dir = out.with_suffix(".logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    results = out.with_suffix(".jsonl")

    todo = []
    for start, rows in windows:
        metrics = tables / f"{tag_for(args.cluster, start, rows)}_metrics.json"
        if metrics.exists() and not args.force:
            continue
        todo.append((start, rows))
    print(f"{len(windows):,} windows, {len(windows) - len(todo):,} cached, {len(todo):,} to run, "
          f"{args.workers} workers x {threads} threads", flush=True)

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["MRC_THREADS"] = str(threads)
    started = time.perf_counter()
    failed: list[tuple[int, str]] = []

    def one(item: tuple[int, int]) -> None:
        start, rows = item
        tag = tag_for(args.cluster, start, rows)
        log = log_dir / f"{tag}.log"
        cmd = [sys.executable, "-m", "mrc.cli", "train", args.cluster,
               "--store", str(args.store), "--start", str(start), "--window", str(rows),
               "--rows", str(args.row_budget), "--no-keep-cache", "--no-mlflow"]
        with log.open("w") as fh:
            done = subprocess.run(cmd, env=env, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                  check=False)
        if done.returncode != 0:
            failed.append((start, str(log)))
            print(f"FAILED {tag} -> {log}", flush=True)
            return
        print(f"done {tag} ({len(failed)} failures so far)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))

    payload = {"cluster": args.cluster, "store": str(args.store), "window": args.window,
               "n_windows": len(windows), "ran": len(todo) - len(failed),
               "failed": failed, "workers": args.workers, "threads_per_worker": threads,
               "seconds": round(time.perf_counter() - started, 1)}
    with results.open("a") as fh:
        fh.write(json.dumps(payload) + "\n")
    print(json.dumps(payload, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} windows failed; see {log_dir}")


if __name__ == "__main__":
    main()
