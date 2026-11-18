# Cache hit-ratio curves from reuse distance

Predicting the miss-ratio curve (MRC) of a production cache instead of a single
hit-rate number, using a discrete-time hazard model on request metadata, compared
against the classical stack-distance curve computed from the trace itself.

Course project (Machine Learning), CityU HK. Data: [twitter/cache-trace](https://github.com/twitter/cache-trace),
54 production clusters, one-week traces, March 2020.

## Why

A hit rate is not a property of a workload, it is a property of a workload **and a
capacity**. System designers still size memory with a regression on aggregate hit
counts, which cannot answer "what happens at 1.5x the cache". The curve can, and
it is a distribution rather than a point estimate.

## What is built

| Layer | Implementation | Contract |
|---|---|---|
| Trace ingest | pyarrow CSV/zstd reader | `timestamp,key,key_size,value_size,client,op,ttl` |
| Trace store | zstd parquet shards, 5M requests per part, 8.4 bytes per request | request window by row group, manifest with part ranges |
| Reuse distance | Fenwick tree over last-occurrence positions, numba kernel, `O(n log n)` | distinct-key distance `R`, distinct-byte distance `B` |
| Ground truth | exact LRU replay plus naive reference for tests | `MRC(C) = P(R >= C)` unit sizes, `P(B + s >= C)` bytes |
| Model | LightGBM discrete-time hazard over the risk set | `S_R(k|x) = prod (1 - h_b(x))`, `MRC(C) = p_comp + (1-p_comp) S(bin(C-1))` |
| Tuning | Optuna (TPE), one SQLite study | objective: MRC error against ground truth |
| Telemetry | MLflow (params, metrics, figures), Optuna dashboard | all local, no cloud |
| Output | matplotlib with a frozen style, LaTeX report, Beamer deck, mkdocs wiki | see `AGENTS.md` |

## Quickstart

```bash
uv sync --all-groups
make data-samples                 # T0: 54 sample traces
env -u PYTHONPATH uv run python -m mrc.cli inspect \
  data/raw/cache-trace/samples/2020Mar/cluster045
make reuse F=data/raw/cache-trace/samples/2020Mar/cluster045 OUT=data/interim/c45.npz
make mrc R=data/interim/c45.npz
make test
```

A full shard is processed as fixed request windows, each with the same memory ceiling:

```bash
make store F=data/raw/traces/cluster52.sort.zst D=data/processed/store52
make shard C=s52_00000 S=data/processed/store52 I=0 N=5000000
```

Telemetry: `make ui` starts MLflow on :5000 and the Optuna dashboard on :8080.

## Status

Test block, four production traces, first 5,000,000 requests each, 24-point capacity grid
(12 on cluster10). MAE against the exact Mattson curve, read from
`artifacts/tables/*_metrics.json`. cluster52 was held out: the model, grid, features and
hyperparameters were frozen before its labels were read.

| Estimator | cluster45 | cluster26 | cluster52 | cluster10 |
|---|---|---|---|---|
| Hazard GBDT | 0.00027 | 0.00548 | 0.00027 | 0.00772 |
| Kaplan-Meier | 0.02000 | 0.01178 | 0.02481 | 0.04387 |
| Hit-count regression | 0.10968 | 0.07927 | 0.11342 | 0.07930 |

The hazard GBDT is the finished model, and the capacity it implies is within 7.5% of truth
on all four traces: the model-over-truth capacity ratio spans 0.925 to 1.009
(`report/tables/provisioning.tex`). On cluster45 and cluster52 the curve error is inside the
sampling noise of the truth block (`scripts/noise_floor.py`, `report/tables/noise_floor.tex`).

## Documentation

| File | Content |
|---|---|
| `docs/PROJECT_PLAN.md` | problem, architecture, parallel build strategy |
| `docs/STATISTICAL_SKELETON.md` | full statistical specification, every symbol defined |
| `docs/DATA.md` | acquisition, parsing, sampling, scraping, known blockers |
| `docs/LIGHTGBM_CUDA.md` | CUDA build recipe, GPU measurements, reproducibility limits |
| `docs/MILESTONES.md` | M0 to M4 with verifiable exit gates |
| `docs/RISKS.md` | risk register |
| `docs/LITERATURE.md` | verified sources with method of verification |
| `wiki/` | searchable site, `make wiki` |
| `AGENTS.md` | the rules every agent follows |

## Layout

```
src/mrc/{data,models,eval,plots}   library
tests/                             exact-first gates
scripts/fetch_traces.sh            bounded, resumable acquisition
report/ slides/ wiki/ docs/        outputs
data/ artifacts/                   gitignored payloads
```

## Citing the dataset

Yang, Yue, Vinayak. *A large scale analysis of hundreds of in-memory cache clusters at
Twitter.* OSDI 2020. Traces are CC-BY 4.0.
