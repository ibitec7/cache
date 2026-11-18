# Handoff: cluster52 parquet, the CPU farm, and generalization experiments

Status: 2026-11-18. Read this before running anything.

The pipeline works and is tested. The open work is: finish and upload the cluster52
parquet shards, optimize the parquet reader, then run three generalization
experiments. Do the phases in order.

## 0. What this repo is

Predicts the miss-ratio curve (MRC) of a production cache from a request trace.
Each request's reuse distance is treated as a survival time; LightGBM estimates a
discrete hazard on 12 request features; the curve is read off the survival
function. Truth is exact LRU stack distance.

Four traces are used:

| Trace | Reuse shape | Role |
|---|---|---|
| cluster45 | long tail, R p99 about 1.01M | development, Optuna study |
| cluster26 | dense, R p99 about 17.7k | development |
| cluster10 | tiny working set, R p99 about 8 | development, sanity check |
| cluster52 | long tail, R p99 about 489k, 13.4B requests | held-out from tuning, scale test |

Each reported model is fit on its own trace's first 5M requests, 70% train block.
There is no cross-cluster trained model yet. That is what the handoff experiments
add.

## 1. Current data state

| Item | State |
|---|---|
| `ibitec/cache-trace-cluster10-parquet` | uploaded, 139,150,615 rows, 959 MB |
| `data/parquet/cluster26.parquet` | ready, 156,594,627 rows, 1.27 GB, not uploaded |
| `data/parquet/cluster45.parquet` | ready, 227,773,739 rows, 1.75 GB, not uploaded |
| `data/parquet/cluster52/` | building at the time of writing: 76 shards, 75 GB, target about 107 GB |
| `data/publish/cluster52/` | created at completion; contains `manifest.json`, `README.md`, and hard links to every shard |

At the time of writing the cluster52 conversion is still running:

```bash
du -sh data/parquet/cluster52
ls data/parquet/cluster52 | wc -l
pgrep -af convert_shards
```

Do not upload a partial directory. Wait for the process to exit, then build the
manifest:

```bash
cd /home/ibrahim/uni/cache
env -u PYTHONPATH uv run python scripts/build_manifest.py \
  cluster52 data/parquet/cluster52 data/raw/traces/cluster52.sort.zst data/publish/cluster52
```

Verify every shard before upload:

```bash
env -u PYTHONPATH uv run python - <<'PY'
import os, re, pyarrow.parquet as pq
d = "data/parquet/cluster52"
files = sorted((f for f in os.listdir(d) if f.endswith(".parquet")),
               key=lambda f: int(re.search(r"\d+", f).group(0)))
rows, prev = 0, -1
for f in files:
    pf = pq.ParquetFile(os.path.join(d, f))
    if pf.metadata.num_row_groups == 0:
        raise SystemExit(f"empty shard {f}")
    t0 = pf.read_row_group(0, columns=["timestamp"]).column("timestamp")[0].as_py()
    if t0 < prev:
        raise SystemExit(f"out of order at {f}")
    rows += pf.metadata.num_rows
    prev = t0
print("parts", len(files), "rows", rows)
PY
```

If `data_0.parquet` has no footer, regenerate the first shard with DuckDB and the
same `FILE_SIZE_BYTES`, or rerun the conversion. The manifest builder fails loudly
if a part is unreadable.

Upload with 8 workers. Measured throughput is about 18 MB/s, so 107 GB is about
1.7 hours.

```bash
hf repos create ibitec/cache-trace-cluster52-parquet --type dataset --public
hf upload-large-folder ibitec/cache-trace-cluster52-parquet data/publish/cluster52 \
  --type dataset --num-workers 8
```

Use the same pattern for cluster26 and cluster45 after building a per-cluster
parts directory (one file per cluster):

```bash
mkdir -p data/parquet/cluster26_parts
ln data/parquet/cluster26.parquet data/parquet/cluster26_parts/
env -u PYTHONPATH uv run python scripts/build_manifest.py \
  cluster26 data/parquet/cluster26_parts data/raw/traces/cluster26.sort.zst data/publish/cluster26
```

Do not point `build_manifest.py` at `data/parquet/` directly for the single-file
clusters. It would mix cluster10, cluster26 and cluster45 into one manifest.

## 2. CPU farm

Recommended instance:

| Type | Spec | $/h | Use |
|---|---|---|---|
| `n2d-highcpu-128` | 128 vCPU, 128 GB | 3.16 | primary, 52 workers at 2 threads |
| `n2d-standard-96` | 96 vCPU, 384 GB | 3.21 | safer RAM headroom, 48 workers |

At a 5M-row budget a window is 117 s at 2 threads and 2.09 GB. Fifty-two workers
use about 109 GB. All 2,685 windows of cluster52 take 1.68 h.

Setup:

```bash
brev create cache52 --type n2d-highcpu-128 --disk 300
# on the instance
git clone <this repo> ~/cache && cd ~/cache
uv sync --all-groups
hf download ibitec/cache-trace-cluster52-parquet --repo-type dataset \
  --local-dir data/parquet/cluster52_hf
```

The downloaded directory contains `manifest.json` and the shards. Point
`--store` at it.

## 3. Phase 0: optimize the parquet read path first

Do this before any experiment. The current reader is correct but not optimized for
a 52-worker sweep.

Benchmark:

1. `Store.read_trace(0, 5_000_000)` against `read_trace(csv, limit=5_000_000)`.
2. Repeat for 1M and 10M windows, with `/usr/bin/time -v` for peak RSS.
3. Run one `mrc train --store ... --no-keep-cache` window and record wall and RSS.

Known current numbers: parquet 0.91 s per 5M window, CSV 4.35 s; labels from both
are bit-identical.

Candidate changes, in order:

1. Cache one `pyarrow.parquet.ParquetFile` per part inside a worker, instead of
   reopening it for every window.
2. Read only the row groups that overlap the window. Keep the part offset index in
   the manifest and use `read_row_group` directly.
3. Pass `memory_map=True` and only the needed columns.
4. Batch several windows per worker process so the numba cache and imports are
   paid once.
5. If key strings still dominate, keep the parquet dictionary indices and build
   the window vocabulary from the dictionary pages instead of materializing
   strings.

Acceptance gates:

- labels computed from the parquet window are bit-identical to the CSV window;
- read wall time is at or below the CSV reader;
- peak RSS is at or below the CSV reader;
- `pytest`, `ruff` and `mypy` pass;
- record the benchmark table in `logs/WORKLOG.md`.

Do not change the label definition, the bin grid, or the split while optimizing.

## 4. Experiments, in this order

### 4.1 Out-of-time (OOT) within each cluster: primary

This is the strongest deployment claim. Same distribution, same capacity grid, no
cross-cluster confound.

1. Sort the full trace by request index.
2. Use the first 60% of requests for training, the next 15% for validation, the
   last 25% for test. Split on request index, never randomly.
3. Fit one model on a sample of the train period (20 to 30 windows of 5M requests
   is enough to bound compute). Use the validation period for early stopping and
   Optuna.
4. Freeze the model and the capacity grid, then predict every test window.
5. Report the distribution of per-window MAE, not one pooled number.

Why: this isolates temporal drift from distribution shift. It answers "will the
model still work later in the week on the same cache?"

Leakage rule: no test window may contribute rows, labels or grid edges to the fit.

### 4.2 Leave-one-cluster-out (LOCO): secondary

This is the direct "new cluster" test.

1. Hold out one cluster. Train on the other three clusters' train periods.
2. Run Optuna on the training clusters' validation periods only. Objective: mean
   validation MAE across the three.
3. Freeze the best params, fit on the training clusters, test on the held-out
   cluster's test period.
4. Four folds, one per cluster.

Use a shared capacity grid derived from the training clusters, or report relative
error. Do not derive the grid from the held-out cluster.

Expect degradation. The four clusters span R p99 about 8 to 1M, so zero-shot
transfer will be worse than a per-cluster fit. That result is the point: it tells
you whether the method transfers and whether a new cluster needs local data.

### 4.3 Minimal retraining curve: bridging

For each held-out cluster:

1. Fit on 0.5M, 1M, 2M and 5M early requests of that cluster.
2. Test on a later window.
3. Plot MAE against the amount of local data.

This answers "how much data from a new cluster before OOT works?" It is the most
useful number for an operator.

## 5. Optuna and telemetry

- OOT: study name `mrc-oot-<cluster>`, tune on train, validate on the validation
  period, test untouched.
- LOCO: study name `mrc-loco-<heldout>`, tune on the three training clusters,
  test untouched.
- 15 to 30 trials per study is enough. Log every study to MLflow under
  `mrc-optuna`.
- Log each run with `mrc train` or `mrc log`; stage timings and data manifests are
  recorded.
- Never tune on cluster52's test period, and never tune on a held-out cluster.

## 6. Leakage and correctness gates

The code enforces:

1. Contiguous request blocks; no interleaving.
2. `FEATURE_NAMES` contains no `R`, `delta`, `gamma` or `rtilde`.
3. Target is the backward reuse distance. The forward-event target was measured
   biased and removed.
4. Truth is exact LRU stack distance.
5. Optuna sees validation only; test is untouched.
6. Cluster52 is refused by the tuning path.
7. Feature and truth caches validate against the label manifest.

Not yet covered:

- no LOCO or OOT test in the repo;
- no automated assertion that a future feature cannot enter the matrix;
- no cross-cluster benchmark.

Keep the `single_row` baseline out of headline tables. It reads the scored
request's own distance and is an oracle, not a model.

## 7. Deliverables

1. Parquet benchmark table and the optimized reader.
2. `scripts/oot.py`, `scripts/loco.py`, `scripts/retrain_curve.py` (or one script
   with subcommands) using `prepare`, `fit`, `predict_block`, `score`.
3. OOT table: per cluster, hazard vs Kaplan-Meier vs hit-count, MAE/MAX/bias/MRE,
   provisioning ratios.
4. LOCO table: same metrics per held-out cluster.
5. Minimal-retrain figure: MAE against local data.
6. MLflow runs and Optuna studies for every fold.
7. `logs/WORKLOG.md` entry, updated report tables, and a short results section.

## 8. Known gaps

- Only four of the 54 clusters are used.
- No trained-on-3-tested-on-1 result exists yet.
- The full cluster52 sweep is within-trace robustness, not cross-cluster.
- Optuna studies exist only for cluster45 and cluster10.
- The first cluster52 shard needs a footer check before upload.
