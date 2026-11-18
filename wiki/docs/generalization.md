# Generalization and handoff

This page records what the four traces prove, what they do not prove, and the
experiment plan for a new cluster. The operational runbook is `HANDOFF.md` in the
repository root.

## What is trained on what

Four experiment families, all on 5,000,000-request contiguous windows. A window is
the unit of training and of scoring, so every number below is a distribution over
windows, not one block read once.

| Family | Train | Tune | Test |
|---|---|---|---|
| Single block (`mrc train`) | first 5M requests, 70% train block | Optuna on the validation block | last 15% of the 5M |
| Out-of-time (OOT) | up to 20 windows from the first 60% of the trace | Optuna, 8 to 34 trials | every window of the last 25% |
| Leave-one-cluster-out (LOCO) | 12 windows of 5M from each of the other three traces | Optuna, 6 trials | every window of the held-out trace |
| Retrain curve | prefix of the trace, `f = budget_rows / total_rows` | none (`--trials 0`) | every window of the last 25% |

The split is `[0.60, 0.75]` in request-index space: train is the first 60%, validation
the next 15%, test the last 25%. Blocks are contiguous, never shuffled.

*Out-of-time* keeps the trace, the capacity grid and the feature distribution fixed and
moves only the time. *Leave-one-cluster-out* moves the distribution and keeps the code.
The two together separate temporal drift from workload transfer, which a single
transfer number cannot do.

Training volume was measured, not assumed: OOT models saw 83.5M (cluster10), 94.0M
(cluster26), 100M (cluster45) and 120M (cluster52) training requests; every LOCO model
saw 180M. So the LOCO models were trained on **more** data than the OOT models, which
makes any LOCO loss attributable to distribution, not to starvation.

## What the four traces show

Test-block MAE against the exact LRU curve:

| Trace | Hazard | Kaplan-Meier | Hit-count |
|---|---|---|---|
| cluster45 | 0.00027 | 0.02000 | 0.10968 |
| cluster26 | 0.00548 | 0.01178 | 0.07927 |
| cluster52 | 0.00027 | 0.02481 | 0.11342 |
| cluster10 | 0.00772 | 0.04387 | 0.07930 |

The model wins on all four, and the capacity it implies is within 7.5% of truth.
That is evidence that the **method** transfers across workloads. It is not evidence
that one set of weights transfers.

On cluster52 the pre-registered ordering and provisioning predictions held, but
the predicted margin did not. Removing the timing features multiplies error by
234x, so the covariates, not the marginal, carry the result.

## Results of the three pre-registered tests

The tests were pre-registered on this page in the order below. All three have now run
on all four traces, and the numbers are recomputed from `artifacts/oot/`,
`artifacts/loco/` and `artifacts/retrain/`; the per-window mean recomputed from the raw
window files equals the reported mean exactly.

### 1. Out-of-time, per trace (primary)

Fit on the first 60%, score every window of the last 25%. `n` is the number of scored
windows. The ratio is the model's MAE divided by the Kaplan-Meier MAE on the same
windows.

| Trace | n | MAE | median | KM MAE | ratio | MRE | KM MRE | windows won |
|---|---|---|---|---|---|---|---|---|
| cluster10 | 7 | 0.003637 | 0.003273 | 0.027656 | 0.132 | 0.61% | 4.03% | 7/7 |
| cluster26 | 8 | 0.001518 | 0.001794 | 0.005109 | 0.297 | 0.38% | 1.09% | 8/8 |
| cluster45 | 12 | 0.001313 | 0.000977 | 0.024877 | 0.053 | 0.21% | 4.07% | 12/12 |
| cluster52 | 672 | 0.003834 | 0.003184 | 0.030559 | 0.125 | 0.70% | 4.72% | 672/672 |

cluster52 is the only trace whose test block is a distribution rather than a handful of
windows: 672 of 672 windows, 94 shards, no subsampling. Its block-bootstrap 95%
interval on the mean MAE is **[0.003416, 0.004318]**, and its per-window IQR is
[0.002122, 0.005872].

The model beats Kaplan-Meier on all 699 scored test windows across the four traces,
with no exception.

### 2. Leave-one-cluster-out (secondary)

| Held out | n | MAE | KM MAE | ratio | MRE | KM MRE | LOCO / OOT |
|---|---|---|---|---|---|---|---|
| cluster10 | 7 | 0.002756 | 0.160934 | 0.017 | 0.47% | 31.2% | **0.76x (better)** |
| cluster26 | 8 | 0.006770 | 0.156697 | 0.043 | 3.78% | 139.8% | 4.46x |
| cluster45 | 12 | 0.005445 | 0.097702 | 0.056 | 0.97% | 17.8% | 4.15x |
| cluster52 | 667 | 0.016266 | 0.144415 | 0.113 | 3.24% | 24.7% | 4.24x |

Reading:

1. **The functional form transfers, the absolute level does not.** Every fold beats the
   covariate-free floor by 9x to 60x, so nothing collapses on an unseen trace. Three of
   four folds lose about 4x in absolute MAE, and the held-out cluster52 error is 1.63
   percentage points of miss ratio. That is enough to mis-size a cache by a factor.
2. **The loss is distributional.** LOCO models had 180M training requests against 83.5M
   to 120M for the OOT models, so data volume cannot explain the gap.
3. **cluster10 inverts, and the retrain curve explains why.** Its LOCO model
   (0.002756) is better than its own-data model (0.003637). cluster10's own-data error
   is still falling steeply at 60M rows (0.004259), so at 83M its binding constraint is
   volume, and 180M requests of other traces is a better prior than its own small
   sample. Two independent experiments, one mechanism.
4. **cluster45 transfers without bias.** Its LOCO bias is -0.00056 against a MAE of
   0.005445, so |bias|/MAE is 0.10: the level moved, the shape did not. Every other
   fold has |bias|/MAE between 0.84 and 0.96.
5. **Out of domain the error stops being a clean offset.** The worst grid point is 2.5
   to 3.5 times the MAE in-domain but 9 to 10 times out of domain for cluster10 and
   cluster26, so the in-domain residual is a level shift and the out-of-domain residual
   is not.

### 3. Retrain curve (bridging)

Fixed hyper-parameters (`--trials 0`), only the data budget moves, same test windows at
every budget.

| Trace | 5M requests | 25M | 60M | n |
|---|---|---|---|---|
| cluster10 | 0.017567 | 0.006046 | 0.004259 | 7 |
| cluster26 | 0.003057 | 0.001861 | 0.001706 | 8 |
| cluster45 | 0.001398 | 0.001313 | 0.001333 | 12 |

cluster10 buys 4.1x from 12x more data and has not plateaued. cluster26 buys 1.8x and
flattens. cluster45 is already saturated at 5M requests, and its three points sit
inside its own per-window spread (0.0007 to 0.0035), so this curve says nothing about
cluster45's data demand. That null is reported as a null.

### The residual in domain is a level offset

| Trace | MAE | bias | \|bias\|/MAE | mean worst point | worst / MAE |
|---|---|---|---|---|---|
| cluster10 | 0.003637 | -0.00336 | 0.93 | 0.0102 | 2.8x |
| cluster26 | 0.001518 | -0.00134 | 0.88 | 0.0044 | 2.9x |
| cluster45 | 0.001313 | -0.00098 | 0.74 | 0.0046 | 3.5x |
| cluster52 | 0.003834 | -0.00373 | 0.97 | 0.0098 | 2.5x |

The in-domain error is a near-constant under-prediction of the miss ratio, uniform
across capacities, of 0.1 to 0.4 percentage points. That is the failure mode that
under-provisions a cache, and it is the cheapest thing to fix: a per-trace scalar
recalibration. It is a hypothesis, not a measured result; it has not been run.

### What this proves and what it does not

Proved: the model predicts the MRC on windows it never trained on, on every trace, and
beats the covariate-free floor 100% of the time. The method transfers across workloads.

Not proved: that one set of weights transfers at in-domain accuracy. A model trained on
other traces is accurate to about 1% relative on a new trace, not 0.2%, so a new cluster
needs local data before its curves are trusted. The retrain curve bounds that requirement
for three traces at 25M to 60M requests.

## Pipeline in one screen

1. Ingest the trace (CSV.zst or zstd parquet shards).
2. Compute reuse distance `R`, byte distance `B`, censoring bound `gamma`, and
   eligibility with a numba Fenwick pass.
3. Compute the exact LRU truth curve `MRC(C) = P(R >= C)`.
4. Build 12 causal features from request metadata only.
5. Bin the backward reuse distance into 33 log-spaced bins.
6. Expand each eligible request into one row per bin at risk.
7. Fit LightGBM on the binary hazard; survival is `prod(1 - h)`.
8. Score MAE, MAX, bias, MRE and ABC against truth on contiguous valid and test
   blocks, plus capacity provisioning.

Measured cost per 5M-request window: 2.09 s reuse distance, 32 s fit at 4 threads,
117 s end to end at 2 threads. The full cluster52 sweep is 2,685 windows, 1.68 h
on 52 workers.

## Data and parquet

The source data ships as CSV.zst. PyArrow streams it and stops at the 5M prefix
(4.35 s per window). For the full 13.4B-request shard, `mrc.data.store` converts
the trace to zstd parquet shards and reads a window by row group (0.91 s per 5M
window). The full shard is 92 shards, 98.3 GB, 13,426,570,607 rows.

Labels and features stay npz. They are dense numeric arrays of 98 MB and 67 MB at
5M rows and load in under a second. Parquet would add row-group overhead with no
gain at this size. The model consumes numpy, not Arrow.

## Leakage guarantees

Enforced in code:

1. contiguous 70/15/15 request blocks, no interleaving;
2. the feature list contains no `R`, `delta`, `gamma` or `rtilde`;
3. target is the backward reuse distance; the forward target was removed after
   measuring a large bias;
4. truth is exact LRU stack distance;
5. Optuna sees validation only; test is untouched;
6. cluster52 is refused by the tuning path;
7. feature and truth caches validate against the label manifest.

Now covered: the leave-one-cluster-out test and the out-of-time test, both on all four
traces (see above). Still not covered: an automated assertion that no future feature
enters the matrix.

The `single_row` baseline takes `eval_rtilde` for the block it scores, and its own bin
is the feature it reads, so it reconstructs the truth by construction. It is an oracle
by design and is excluded from headline tables. Its MAE of 0.000000 in
`artifacts/tables/cluster*_metrics.json` is that oracle, not a competitor.

## Handoff

The runbook for the next agent, the CPU farm commands, the parquet optimization
task and the deliverables are in `HANDOFF.md` at the repository root.
