# Reproduction

Everything needed to rebuild the environment, the data tiers, every artifact and
this site. Commands assume the repository root, `/home/ibrahim/uni/cache`.

## 1. Environment

```bash
cd /home/ibrahim/uni/cache
uv sync --all-groups
```

`env -u PYTHONPATH` is required for every module invocation in this repository:
the shell exports a `PYTHONPATH` that shadows the project's `src/` layout, and
`uv run` inherits it.

```bash
env -u PYTHONPATH uv run python -c "import mrc, lightgbm, libcachesim; print('ok')"
```

| Group | Contents | Needed for |
|---|---|---|
| main | lightgbm, numpy, pandas, polars, duckdb, libcachesim, optuna, mlflow, matplotlib, numba | all modelling |
| `viz` | mkdocs-material, pymdown-extensions | this site only |
| `dev` | pytest, ruff, mypy, pre-commit | tests, lint |

## 2. Data tiers

| Tier | Command | Content | Disk |
|---|---|---|---|
| T0 | `make data-samples` | 54 sample files, one per cluster | ~1 GB |
| T1 | `make data-t1` | clusters 45, 26, 10 | ~4 GB |
| T2 | `make data-t2` | cluster 52, full shard | 87 GB |

```bash
make data-samples        # always start here
make data-t1             # three smallest clusters, enough for the model
make data-t2             # optional, one full shard
```

Acquisition properties, from `scripts/fetch_traces.sh`:

| Property | Value |
|---|---|
| Mirror | `https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source` |
| Resume | `curl -C -`, 5 attempts with 10 s backoff |
| Integrity | `zstd -t` per file, sha256 appended to `data/raw/traces/CHECKSUMS.sha256` |
| Budget guard | `BUDGET_GB` (default 150) aborts before an oversized transfer |
| Free-space guard | aborts under 5 GB free |

## 3. Pipeline

```bash
export MLFLOW_TRACKING_URI=sqlite:///$PWD/artifacts/mlflow/mlflow.db

make smoke                                   # fast end-to-end check
make inspect F=data/raw/cache-trace/samples/2020Mar/cluster045
make reuse   F=data/raw/traces/cluster45.sort.zst OUT=data/interim/cluster45.npz
make mrc     R=data/interim/cluster45.npz
make train   C=cluster45
make plots
```

| Stage | Reads | Writes |
|---|---|---|
| `inspect` | one trace file | stdout summary only |
| `reuse` | trace | `data/interim/*.npz` with $d_i$ per request |
| `mrc` | interim npz | curve table, printed |
| `train` | trace | labels, features, truth, metrics, MLflow run, model |
| `plots` | metrics artifacts | `report/figures/`, `report/tables/`, `artifacts/tables/` |

## 4. Dashboards

```bash
make ui      # MLflow on :5000, Optuna dashboard on :8080
```

MLflow backend store is `sqlite:///artifacts/mlflow/mlflow.db`; Optuna storage is
`sqlite:///artifacts/optuna/study.db`. Both are gitignored.

## 5. Tests and lint

```bash
make test    # full pytest suite
make lint    # ruff check src tests && mypy src
```

## 6. This site

```bash
cd wiki
env -u PYTHONPATH uv run --group viz mkdocs build --strict
```

or from the root, `make wiki`. Output lands in `wiki/site/` (gitignored).
To preview with reload:

```bash
cd wiki && env -u PYTHONPATH uv run --group viz mkdocs serve -a 127.0.0.1:8000
```

`--strict` turns any broken internal link, missing nav page or unrecognised
markdown extension into a build failure. That is intentional: the wiki is part of
the deliverable, not a scratchpad.

## 7. Expected artifacts

| Path | Produced by | Content |
|---|---|---|
| `data/raw/cache-trace/` | `make data-samples` | cloned corpus metadata, samples, `stat/` |
| `data/raw/traces/*.sort.zst` | `make data-t1`, `make data-t2` | verified compressed shards |
| `data/interim/*.npz` | `make reuse` | per-request reuse distance |
| `data/processed/<cluster>/{labels,features}.npz` | `make train` | label cache and design matrix |
| `artifacts/mlflow/mlflow.db` | `make train` | runs, params, metrics, artifacts |
| `artifacts/optuna/study.db` | `make tune` | trial history |
| `artifacts/tables/mrc_accuracy.csv` | `make plots` | the table on [Results](results.md) |
| `artifacts/figures/*.pdf` | `make plots` | curves and residuals |
| `wiki/site/index.html` | `make wiki` | this site |

## 8. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mrc` | stale `PYTHONPATH` | prefix with `env -u PYTHONPATH` |
| `zstd: unsupported format` | truncated download | delete the file, rerun the tier target |
| MLflow writes to `./mlruns` | `MLFLOW_TRACKING_URI` unset | export it as in section 3 |
| mkdocs `strict` fails on a link | renamed page or heading anchor | fix the link, do not drop `--strict` |
| Optuna trial fails instantly | bucket count $K$ larger than shard length | lower `K` in `src/mrc/tuning/search.py` |

## Limits

The corpus is 2.8 TB compressed and 14 TB uncompressed; this project never
downloads it whole. Every reported number is produced from T0 plus T1, with T2
used for a single-cluster spot check. Any claim about "all 54 clusters" must be
read as "the 54 sample files", which are prefixes, not full traces.
