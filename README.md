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
| Reuse distance | Fenwick tree over last-occurrence positions, numba kernel, `O(n log n)` | distinct-key distance `R`, distinct-byte distance `B` |
| Ground truth | exact LRU replay plus naive reference for tests | `MRC(C) = P(R >= C)` unit sizes, `P(B + s >= C)` bytes |
| Model | LightGBM discrete-time hazard over the risk set | `S_R(k|x) = prod (1 - h_b(x))`, `MRC(C) = p_comp + (1-p_comp) S(bin(C-1))` |
| Tuning | Optuna (TPE), one SQLite study | objective: MRC error against ground truth |
| Telemetry | MLflow (params, metrics, figures), Optuna dashboard, DuckDB over parquet | all local, no cloud |
| Output | matplotlib with a frozen style, LaTeX report, Beamer deck, mkdocs wiki | see `AGENTS.md` |

## Quickstart

```bash
uv sync --all-groups
make data-samples                 # T0: 54 sample traces
env -u PYTHONPATH uv run python -m mrc.cli inspect \
  data/raw/cache-trace/samples/2020Mar/cluster045
make reuse F=data/raw/cache-trace/samples/2020Mar/cluster045 OUT=data/interim/c45.npz
make mrc R=data/interim/c45.npz
make fit R=data/interim/c45.npz
make test
```

Telemetry: `make ui` starts MLflow on :5000 and the Optuna dashboard on :8080.

## Status

Measured on the 200,000-request `cluster045` sample, not the final T2 numbers. Recorded
here so the pipeline is auditable end to end.

| Curve | MAE vs empirical MRC (16 capacities) | Max abs error | Bias |
|---|---|---|---|
| Kaplan-Meier hazard, no features | 0.0018 | 0.0073 | -0.0018 |
| GBDT hazard, 5 features, default params | 0.0057 | 0.0099 | +0.0057 |

The GBDT overestimates the miss ratio here because it is fitted on five coarse features
and has not been tuned; the KM baseline is unbiased by construction. Closing that gap is
the modelling work.

## Documentation

| File | Content |
|---|---|
| `docs/PROJECT_PLAN.md` | problem, architecture, parallel build strategy |
| `docs/STATISTICAL_SKELETON.md` | full statistical specification, every symbol defined |
| `docs/DATA.md` | acquisition, parsing, sampling, scraping, known blockers |
| `docs/MILESTONES.md` | M0 to M4 with verifiable exit gates |
| `docs/RISKS.md` | risk register |
| `docs/LITERATURE.md` | verified sources with method of verification |
| `wiki/` | searchable site, `make wiki` |
| `AGENTS.md` | the rules every agent follows |

## Layout

```
src/mrc/{data,models,eval,plots}   library
tests/                             exact-first gates
configs/                           data, model, search configs
scripts/fetch_traces.sh            bounded, resumable acquisition
report/ slides/ wiki/ docs/        outputs
data/ artifacts/                   gitignored payloads
```

## Citing the dataset

Yang, Yue, Vinayak. *A large scale analysis of hundreds of in-memory cache clusters at
Twitter.* OSDI 2020. Traces are CC-BY 4.0.
