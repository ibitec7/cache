# Results

## What it is

The reporting surface for the experiment matrix on [Experiments](experiments.md).
Numbers are written here only after they exist in `artifacts/tables/` and are
traceable to an MLflow run id. Nothing on this page is estimated, rounded from
memory, or illustrative.

!!! warning "Status: pending"
    No corpus run has been executed yet. The tables below are the schema the
    reporting script fills, with `pending` in every cell rather than placeholder
    numbers. The only numbers on this page that are real are the toy-trace
    algebra checks at the bottom, and they are labelled as such.

## Why it is needed here

An MRC claim can be true on average and false where it matters: the interesting
region is small $C$, where the curve is steep and a small capacity error is a
large ratio error. So results are reported three ways, and a claim is only made
when all three agree:

| View | Question |
|---|---|
| Per-capacity residual | where on the curve does the model fail |
| Per-cluster MAPE | is the failure clustered in particular workloads |
| Cost | memory and wall-clock against the exact algorithm |

## Formal statement

For each run $r$, capacity $C_j$ and cluster $c$, the reporting table stores

$$
e_{r,c,j} = \hat m_{r,c}(C_j) - m^\ast_{c}(C_j),
$$

with summary statistics MAPE, MaxErr, MAE as defined on
[Ground truth MRC](ground-truth.md). A run is accepted only if all of:

1. MAPE $\le$ the SHARDS baseline at matched memory;
2. MaxErr $\le 0.05$ absolute on every capacity in the grid;
3. the predicted curve is monotone non-increasing after isotonic correction,
   with the raw curve's violation count reported alongside.

| Symbol | Meaning | Domain |
|---|---|---|
| $r$ | run id (MLflow) | string |
| $c$ | cluster index | $1..54$ |
| $C_j$ | capacity grid point | $[0,U]$ |
| $e_{r,c,j}$ | signed residual | $[-1,1]$ |
| $\hat m_{r,c}$ | predicted miss ratio | $[0,1]$ |
| $m^\ast_c$ | simulated miss ratio | $[0,1]$ |

## Worked micro-example

The two capacities the toy trace supports, computed end to end, as the template
for what a real row looks like:

| Quantity | $C=2$ | $C=3$ |
|---|---|---|
| Exact stack distance, $\hat m$ | 0.8333 | 0.6667 |
| Kaplan-Meier, $\hat m$ | 0.8333 | 0.6667 |
| Simulator ground truth, $m^\ast$ | 0.8333 | 0.6667 |
| Residual $e$ | 0.0000 | 0.0000 |

Both estimators reproduce the simulator exactly on six requests, which is what
the algebra on [Hazard model](hazard-model.md) promises and nothing more. With
$N=6$ the standard error of any miss-ratio estimate is on the order of
$\sqrt{0.8333 \times 0.1667 / 6} = 0.152$, so a six-request trace cannot
distinguish a good model from a bad one. Real evaluation uses $10^8$ requests per
cluster, where the same bound is about $4 \times 10^{-5}$ and model error
dominates sampling error.

## Tables to be filled

### Curve accuracy by cluster (test split)

| Cluster | MAPE exact | MAPE KM | MAPE GBDT default | MAPE GBDT Optuna | MaxErr best | Run id |
|---|---|---|---|---|---|---|
| 45 | pending | pending | pending | pending | pending | pending |
| 26 | pending | pending | pending | pending | pending | pending |
| 10 | pending | pending | pending | pending | pending | pending |
| 52 | pending | pending | pending | pending | pending | pending |

### Cost

| Method | Memory | Passes over trace | Wall clock |
|---|---|---|---|
| Exact stack distance | pending | pending | pending |
| Counter stacks | pending | pending | pending |
| SHARDS $q=1/32$ | pending | pending | pending |
| Hazard GBDT | pending | pending | pending |

### Cross-cluster transfer (leave-one-out, T1)

| Held out | Train on | MAPE | MaxErr | Run id |
|---|---|---|---|---|
| 45 | 26, 10 | pending | pending | pending |
| 26 | 45, 10 | pending | pending | pending |
| 10 | 45, 26 | pending | pending | pending |

## Where the numbers come from

| Artifact | Content |
|---|---|
| `artifacts/tables/mrc_accuracy.csv` | the per-cluster table above |
| `artifacts/tables/policy_compare.csv` | LRU, FIFO, S3-FIFO reference curves |
| `artifacts/figures/mrc_<cluster>.pdf` | curve plus residuals |
| `artifacts/mlflow/mlflow.db` | runs, params, metrics, tags |
| `artifacts/optuna/study.db` | trial history |

Regenerate everything with `make plots`; the tables are written by
`src/mrc/report.py` from the MLflow store, not by hand.

## Where this lives in the code

`src/mrc/report.py` (table assembly from MLflow), `src/mrc/plot.py` (figures),
`src/mrc/metrics.py` (error definitions).
