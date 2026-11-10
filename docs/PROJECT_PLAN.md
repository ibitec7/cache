# Project Plan: Miss-Ratio Curves from Reuse-Distance Hazard Models

Status: draft for M0 review. Every numeric result in this plan is TBD until the
corresponding milestone gate passes.

## 1. Problem statement

Cache sizing is a budget decision. The operator picks a DRAM or SSD budget `B`,
and that budget buys a capacity `C(B)`; latency and backend load are then
determined by the miss ratio at that capacity. The quantity that matters is
therefore a function, `MRC(C)`, not a scalar.

Current practice estimates the function indirectly. Designers regress total hit
counts on capacity across a handful of deployments or on synthetic sweeps, then
read a single operating point off the fit. Two failures follow.

1. **No curve.** A hit-count regression at one capacity identifies one point.
   Hit ratio is concave in capacity over most of the useful range and flattens
   near the working set size, so a local slope extrapolated from one point
   over-predicts the benefit of the next gigabyte in the flat region and
   under-predicts it near the knee. Over-allocation burns budget, under-allocation
   burns tail latency and backend QPS.
2. **No censoring model.** Production traces are logs with a fixed end. An
   object whose next reference falls after the log ends has an unobserved reuse
   distance. Dropping those objects biases the empirical distribution toward
   short distances and under-estimates the capacity needed for a target hit
   ratio. Treating them as "never reused" biases the other way.

This project builds the curve directly and treats truncation explicitly. Each
request is modelled as a discrete-time hazard over reuse distance, conditional
on request features. The fitted model supplies a survival function
`S_R(r | x)`, and the miss-ratio curve is its expectation over the request
population. The target is validated against Mattson's stack-distance MRC, which
is exact for a fully observed trace and therefore serves as ground truth inside
the sample.

| Approach | Output | Handles truncation | Reusable on new traffic |
|---|---|---|---|
| Hit-count regression | one point per fit | no | no |
| Mattson stack distance | full curve | no (needs full replay) | no |
| Hazard model (this project) | full curve | yes (censored likelihood) | yes (conditional on features) |

Scope: one week of one Twitter cache cluster per trace file (Mar 2020), plus
the four smallest clusters for robustness. Generalisation beyond that vintage is
a stated caveat, not a claim.

## 2. Statistical skeleton

### 2.1 Objects and distance

Fix one trace. Requests arrive in time order, each naming an object `o` via its
anonymised key. For the `j`-th reference to `o`, the reuse distance `R` is the
number of distinct other objects referenced since the previous reference to `o`.
Under LRU with capacity `C` objects, that reference hits if and only if
`R < C`, because exactly the `R` distinct objects above it in the stack must
still fit. Hence

```
MRC(C) = P(R >= C) = 1 - F_R(C - 1)
```

which is Mattson's identity. The whole modelling problem is estimating the
distribution of `R` at the granularity of one reference.

### 2.2 Discrete-time hazard

Distances are binned into `B` bins with edges `0 = e_0 < e_1 < ... < e_B`, bin
`r` covering `[e_r, e_{r+1})`; binning is log-spaced after the first few bins
because the mass spans several decades. For a request with feature vector `x`,

```
h_r(x) = P(R in bin r | R >= e_r, x),     r = 0..B-1
S_R(r | x) = prod_{k=0..r} (1 - h_k(x)) = P(R > e_r | x)
F_R(r | x) = 1 - S_R(r | x)
MRC(C) = 1 - E_x[F_R(b(C) - 1 | x)] = E_x[S_R(b(C) - 1 | x)]
```

with `b(C)` the bin containing capacity `C`. The second equality in the MRC line
is the one plotted; the first is the definition from section 2.1. Taking the
expectation over requests (not objects) matches the reference-weighted
stack-distance MRC.

Estimation is a single LightGBM binary classifier on person-period rows. Each
observed reuse contributes one row per bin at risk, labelled `1` in the bin
where the reuse lands and `0` in earlier bins. Censored reuses contribute `0`
rows up to the last bin fully observed before the log ends. The pooled
likelihood, with `d_r` events and `n_r` at-risk rows in bin `r`, is

```
L = prod_r h_r^{d_r} (1 - h_r)^{n_r - d_r}
```

which is why one classifier with bin index as a feature recovers all per-bin
hazards; the discrete-time hazard model and the pooled binomial likelihood are
the same object. `S_R` is then a cumulative product, enforced non-increasing by
an isotonic projection if the raw classifier is locally non-monotone.

### 2.3 Notation

| Symbol | Meaning | Type / domain |
|---|---|---|
| `o` | object identity (anonymised key) | string |
| `j` | index of a reference within one object's history | positive integer |
| `R` | reuse distance, distinct objects between consecutive references | non-negative integer, `0..N` |
| `r` | bin index over `R` | `0..B-1` |
| `B` | number of distance bins | integer, TBD (provisional 48) |
| `e_r` | left edge of bin `r` in distinct-object units | integer |
| `b(C)` | bin index containing capacity `C` | integer |
| `x` | request feature vector (namespace, op, sizes, TTL, client, gap stats) | `R^d` |
| `h_r(x)` | discrete hazard, probability the reuse lands in bin `r` given survival to `e_r` | `[0,1]` |
| `S_R(r\|x)` | survival function, `P(R > e_r \| x)` | `[0,1]`, non-increasing in `r` |
| `F_R(r\|x)` | CDF, `1 - S_R(r\|x)` | `[0,1]` |
| `C` | cache capacity in objects (bytes via size columns for the byte MRC) | positive integer |
| `MRC(C)` | miss ratio at capacity `C`, `1 - E_x[F_R(b(C)-1\|x)]` | `[0,1]` |
| `delta` | censoring indicator, `1` reuse observed inside the log, `0` truncated | `{0,1}` |
| `n_r` | at-risk count in bin `r` | non-negative integer |
| `d_r` | event count in bin `r` | non-negative integer, `d_r <= n_r` |
| `g_ij` | gap in distinct objects between reference `j-1` and `j` of `o` | integer |
| `N` | number of requests in the trace slice | integer |

### 2.4 Estimator and evaluation

The fitted curve is compared to the Mattson curve on held-out time slices of the
same trace, never on the fitting slice. Metrics: MAE of miss ratio over a fixed
capacity grid, maximum absolute error, error at the knee (capacity where the
slope drops below a fixed threshold), and error at the 90% hit-ratio capacity.
The byte-indexed MRC uses `key size + value size` per object and is reported
separately because it is the number an operator budgets against.

Known bias source kept in scope: TTL expiry and explicit deletes remove objects
before `R` would have evicted them, so `R < C` is necessary but not sufficient
for a hit. TTL is a feature and expiry is modelled as a competing removal, with
the residual measured by the LRU-simulator cross-check rather than assumed away.

## 3. System architecture

Four packages under `src/mrc`, each with one contract. No package imports
sideways except through the interfaces named below.

| Module | Contract (one line) |
|---|---|
| `data/fetch.py` | Given a shard name and mirror URL, download, verify sha256 against the published manifest, and append one manifest record; never re-download a verified shard. |
| `data/decode.py` | Stream a `.sort.zst` file and write typed parquet in bounded batches with a fixed row count per file; memory stays under a configured cap. |
| `data/schema.py` | Own the canonical request schema, expose `validate(table) -> Report`, and refuse any table whose dtypes, nullability or units deviate. |
| `data/features.py` | Turn validated requests into feature columns with no look-ahead: every feature uses only information available at or before the current request. |
| `data/labels.py` | Compute reuse distance and censoring, then emit person-period rows (or sampled at-risk rows with weights) for training. |
| `data/stackdist.py` | Compute exact LRU stack distance and the Mattson MRC for a shard using numba; deterministic, resumable, checked against a small brute-force oracle. |
| `models/hazard.py` | Fit the pooled LightGBM hazard classifier, expose `predict_hazard(frame) -> ndarray[B]` per request and `predict_survival()`. |
| `models/survival.py` | Turn per-request hazards into `S_R`, then into object- and byte-indexed MRC on a capacity grid. |
| `models/tuning.py` | Optuna objective over a declared search space, pruning on validation MRC error, returning the best params and trial table. |
| `models/track.py` | Wrap MLflow: one run per (trace, config), log params, metrics, model artifact and figure paths; fail loudly if the tracking URI is unreachable. |
| `eval/metrics.py` | Pure functions from two curves plus a capacity grid to the metric dict; no I/O. |
| `eval/baseline.py` | Reference curves: Mattson, exact LRU simulation, and the constant hit-rate regression baseline the project argues against. |
| `eval/libcachesim_adapter.py` | Thin, quarantined wrapper around libcachesim 0.3.5 used only as a third opinion, with both known defects detected and the result marked untrusted. |
| `plots/style.py` | Frozen rcParams (Nimbus Roman, sizes, palette, grid); imported by every plotting module and by the paper. |
| `plots/mrc_plots.py` | One function per figure, writing PDF next to PNG, no global state. |

Why the split: labels and features are the expensive, error-prone part and are
tested independently of modelling; the model package never touches parquet; the
eval package never touches the model object, only curves. The libcachesim
adapter is isolated because its pybind layer is known-bad (see RISKS.md) and a
defect there must not contaminate the pipeline.

## 4. Data flow

```
clusterN.sort.zst
  -> fetch.py        data/raw/         + manifest.json (sha256, bytes, row count)
  -> decode.py       data/interim/     typed parquet, one file per 5M rows
  -> schema.py       validate          reject on drift, log schema hash
  -> features.py     feature table     prior-gap stats, sizes, TTL, namespace, op
  -> labels.py       person-period     (x, bin, label, weight) rows
  -> hazard.py       LightGBM          p(h_r | x) for every bin
  -> survival.py     S_R(r|x)          cumulative product, isotonic fix
  -> eval/metrics.py curve vs Mattson  MAE, max error, knee error
  -> plots/          figures           PDF + PNG, style frozen
  -> report/         paper             8-10 pages
```

| Artifact | Path | Owner | Frozen at |
|---|---|---|---|
| raw shard + manifest | `data/raw/`, `data/manifest.json` | mrc-data | M1 |
| request parquet | `data/interim/*.parquet` | mrc-data | M1, schema freeze |
| feature/label table | `data/processed/` | mrc-data | M2 |
| stack-distance MRC | `data/processed/mrc_truth/` | mrc-data | M2 |
| model runs | `mlruns/` (MLflow) | mrc-run | M3 |
| figures | `report/figures/` | mrc-write, produced by mrc-run | M4 |
| paper | `report/main.tex` | mrc-write | M4 |

Streaming is mandatory: no stage may hold a full shard in memory. decode.py
writes parquet in bounded batches; labels.py consumes parquet in row-group
order so one shard can be processed on a 16 GB machine.

## 5. Parallel build strategy

Four workstreams run concurrently and integrate at fixed file contracts, not at
meetings. Each maps to one Hermes profile, one Kanban lane, and one owner of
record.

| Workstream | Profile | Owns | Produces | Consumes |
|---|---|---|---|---|
| Planning and statistics | `mrc-plan` | `docs/`, `Makefile` | plan, milestones, risks, schema contract, gate targets | nothing |
| Data | `mrc-data` | `src/mrc/data/`, `data/` | parquet, labels, ground-truth MRC | schema contract |
| Experiments | `mrc-run` | `src/mrc/models/`, `src/mrc/eval/`, `mlruns/` | hazard model, tuning table, metric table | parquet + labels |
| Writing | `mrc-write` | `report/`, `src/mrc/plots/`, `docs/paper/` | figures, paper | metric table + style freeze |

Contract points, each a versioned artifact with a hash recorded in the Kanban
card:

1. **Parquet schema freeze** (`docs/contracts/requests_schema.json`). Column
   names, dtypes, units, null semantics. Any change is a new version and a new
   card; consumers pin a version. Gate: `make schema`.
2. **Style freeze** (`src/mrc/plots/style.py`). Fonts, sizes, palette, figure
   widths for the two-column layout. After freeze only additive changes, approved
   by mrc-write. Gate: `make style-check`.
3. **Refs freeze** (`docs/paper/refs.bib`). Citation keys for every claim.
   Writing may not introduce an uncited number. Gate: `make refs-check`.

Why this shape: the three hard dependencies (schema, style, refs) are exactly
the ones that cause silent rework, so they are frozen early and validated by a
command. Everything else is soft-coupled through files on disk, so a stalled
workstream blocks only its own lane.

Kanban board is the coordination mechanism:

| Column | Meaning | Rule |
|---|---|---|
| Backlog | known work, no owner commitment | one card per deliverable |
| Ready | entry criteria met, owner named | max 2 cards per profile |
| In progress | actively worked | WIP limit 1 per profile |
| Review | exit gate command run, artifact attached | reviewer is a different profile |
| Done | gate output pasted on the card | immutable, no re-open, new card instead |

Integration gates are cards of their own (`G1 schema`, `G2 style`, `G3 refs`),
each with a due position before the dependent milestone and a verifiable
command. A milestone card cannot move to Done while a gate card it depends on
is open.

## 6. Repository layout

```
cache/
├── pyproject.toml
├── Makefile                     # all gates; owner mrc-plan
├── README.md
├── AGENTS.md
├── .pre-commit-config.yaml
├── configs/
│   ├── t1.yaml                  # four small clusters
│   └── t2.yaml                  # cluster52, final curves
├── data/
│   ├── raw/                     # *.sort.zst, gitignored
│   ├── interim/                 # decoded parquet
│   ├── processed/               # features, labels, mrc_truth
│   └── manifest.json
├── src/mrc/
│   ├── __init__.py
│   ├── cli.py                   # click entry point, one subcommand per stage
│   ├── data/{fetch,decode,schema,features,labels,stackdist}.py
│   ├── models/{hazard,survival,tuning,track}.py
│   ├── eval/{metrics,baseline,libcachesim_adapter}.py
│   └── plots/{style,mrc_plots}.py
├── tests/
│   ├── data/                    # schema, decode round-trip, stackdist oracle
│   ├── models/                  # hazard monotonicity, survival consistency
│   └── eval/                    # metric identities, MRC endpoints
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── MILESTONES.md
│   ├── RISKS.md
│   ├── contracts/               # frozen schema, style note, refs note
│   └── paper/{main.tex,refs.bib}
├── report/
│   ├── main.pdf
│   └── figures/
└── wiki/                        # mkdocs-material notes
```

`data/raw`, `data/interim`, `mlruns/` and `report/figures/` are generated and
gitignored; only manifests, hashes and summary tables are committed.

## 7. Definition of done per milestone

Full entry criteria and gates live in MILESTONES.md. Summary:

| Milestone | Done when |
|---|---|
| M0 Plan | this file plus MILESTONES.md, RISKS.md, schema contract and Makefile targets exist and `make check` passes on the empty skeleton |
| M1 Data | one tiny cluster decoded end to end, schema validated, manifest reproducible by hash, request count matches an independent DuckDB count |
| M2 Ground truth | exact stack-distance MRC for T1 matches a brute-force oracle on 1e5 requests and the exact LRU simulator on all T1 clusters within 1e-6 |
| M3 Model | hazard model on T1 beats the constant-hit-rate baseline on held-out slices and its MRC is within a stated tolerance of Mattson at the 90% hit ratio capacity; tuning and MLflow runs reproducible from a config hash |
| M4 Write | all figures regenerated by one command from frozen artifacts, paper within the page limit, every number traceable to an MLflow run or a committed table |

Numbers marked TBD are filled at the gate that produces them. No claim in the
paper may precede the gate that measures it.
