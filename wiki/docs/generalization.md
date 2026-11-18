# Generalization and handoff

This page records what the four traces prove, what they do not prove, and the
experiment plan for a new cluster. The operational runbook is `HANDOFF.md` in the
repository root.

## What is trained on what

Each reported model is fit on its own trace:

| Trace | Fit data | Tuned on it | Role |
|---|---|---|---|
| cluster45 | first 5M requests, 70% train block | yes, 15 Optuna trials | development |
| cluster26 | first 5M requests, 70% train block | no study | development |
| cluster10 | first 5M requests, 70% train block | 2-trial study | development |
| cluster52 | first 5M requests, 70% train block | no, held out from tuning | scale and falsification |

There is **no cross-cluster trained model** yet. Cluster52 is held out from tuning,
but its model still learns from cluster52's own train block. "Held out" means the
features, grid and hyperparameters were frozen before its labels were read.

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

## What to test next, in order

### 1. Out-of-time within each cluster, primary

Train on the first 60% of the week, validate on the next 15%, test on the last
25%. Split by request index, never randomly. Fit one model on the train period,
freeze it, then predict every test window and report the distribution of MAE.

This is the strongest deployment claim. It keeps the distribution and the grid,
and it isolates temporal drift.

### 2. Leave-one-cluster-out, secondary

Hold out one cluster, train on the other three, run Optuna on the training
clusters' validation periods, then test on the held-out cluster. Four folds.

This is the direct new-cluster test, but it confounds model transfer with
distribution shift. The four traces span R p99 from about 8 to about 1M, so expect
degradation. That degradation is the result.

### 3. Minimal retraining curve, bridging

Fit on 0.5M, 1M, 2M and 5M early requests of a held-out cluster, test on a later
window, and plot MAE against the amount of local data. This answers how much data
a new cluster needs before out-of-time evaluation works.

Do not tune on the test period or on the held-out cluster.

## Why not train on three and test on the fourth as the only claim

It answers a different question. A single transfer number mixes three effects:

1. the model's ability to use request features;
2. the shift in reuse scale between clusters;
3. the difference in capacity grids.

Out-of-time keeps 2 and 3 fixed. Leave-one-cluster-out changes all three. Run both:
out-of-time for the deployment claim, leave-one-cluster-out for the transfer
claim, and the retraining curve to connect them.

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

Not yet covered: a leave-one-cluster-out test, an out-of-time test, and an
automated assertion that no future feature enters the matrix. The `single_row`
baseline is an oracle by design and is excluded from headline tables.

## Handoff

The runbook for the next agent, the CPU farm commands, the parquet optimization
task and the deliverables are in `HANDOFF.md` at the repository root.
