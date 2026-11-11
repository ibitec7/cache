# Results

## What it is

The reporting surface for the experiment matrix on [Experiments](experiments.md).
Every number here is read from `artifacts/tables/<cluster>_metrics.json`, which the
pipeline writes after a run. Nothing is estimated, rounded from memory, or
illustrative.

!!! note "Status: three production traces measured"
    cluster45, cluster26 and cluster10, first 5,000,000 requests each, object
    capacities, test block scored once. The exact Mattson curve is the ground
    truth on the same block.

## What is reported, and why

A curve claim can be true on average and false where it matters: the interesting
region is small capacity, where a small capacity error is a large ratio error. A
result is reported four ways:

| View | Question | Artefact |
|---|---|---|
| Per-capacity residual | where on the curve does the model fail | `fig_error.pdf` |
| Curve error | how far is the whole curve | `<cluster>_metrics.json` |
| Provisioning | what capacity does the error cost | `fig_capacity.pdf`, `metrics.json` |
| Attribution | which features carry the curve | `fig_importance.pdf`, `<cluster>_importance.json` |

## Formal statement

For capacity $C_j$, cluster $c$, predicted curve $\hat m$ and exact curve $m^\ast$,

$$e_{c,j} = \hat m_c(C_j) - m^\ast_c(C_j).$$

Reported statistics are MAE, MAX, signed bias, MRE over points with truth at least
0.01, and ABC (area between curves, integrated in log capacity and normalised by
the log range). Capacities are the upper edges of the hazard bins, so the step
function and $P(R \ge C)$ refer to the same event.

| Symbol | Meaning | Domain |
|---|---|---|
| $c$ | cluster | {45, 26, 10} |
| $C_j$ | capacity grid point, j-th bin upper edge | objects, $[1, b_{M-1})$ |
| $\hat m_c$ | predicted miss ratio | $[0,1]$ |
| $m^\ast_c$ | exact stack-distance miss ratio | $[0,1]$ |
| $e_{c,j}$ | signed residual | $[-1,1]$ |

## Test-block accuracy

| Cluster | Model | MAE | MAX | bias | MRE |
|---|---|---|---|---|---|
| cluster45 | hazard GBDT | 0.00027 | 0.00107 | -0.00017 | 0.0005 |
| cluster45 | Kaplan-Meier | 0.02000 | 0.03429 | -0.01645 | 0.0356 |
| cluster45 | hit-count regression | 0.10968 | 0.19838 | -0.07341 | 0.1838 |
| cluster26 | hazard GBDT | 0.00548 | 0.02057 | -0.00539 | 0.0106 |
| cluster26 | Kaplan-Meier | 0.01178 | 0.01537 | -0.01178 | 0.0201 |
| cluster26 | hit-count regression | 0.07927 | 0.27053 | 0.04805 | 0.1477 |
| cluster10 | hazard GBDT | 0.00772 | 0.01902 | -0.00772 | 0.0127 |
| cluster10 | Kaplan-Meier | 0.04387 | 0.12042 | -0.04387 | 0.0623 |
| cluster10 | hit-count regression | 0.07930 | 0.33206 | -0.06852 | 0.1112 |

## Provisioning

Capacity ratio, model over truth, for a target hit ratio expressed as a fraction of
the achievable ceiling $1-p_c$. Below one under-provisions.

| Cluster | Model | 60% | 80% | 95% |
|---|---|---|---|---|
| cluster45 | hazard GBDT | 1.003 | 0.976 | 1.004 |
| cluster45 | Kaplan-Meier | 0.922 | 0.239 | 0.524 |
| cluster26 | hazard GBDT | 0.977 | 0.981 | 0.999 |
| cluster26 | Kaplan-Meier | 0.971 | 0.985 | 0.996 |
| cluster10 | hazard GBDT | 0.947 | 0.935 | 0.925 |
| cluster10 | Kaplan-Meier | 0.635 | 0.681 | 0.744 |

## Attribution

Mean absolute SHAP contribution, top three per cluster:

| Cluster | First | Second | Third |
|---|---|---|---|
| cluster45 | log gap (2.66) | hazard step (1.42) | log age (0.12) |
| cluster26 | log gap (1.32) | hazard step (1.03) | log age (0.10) |
| cluster10 | log gap (3.29) | hazard step (3.29) | log object size (0.08) |

Ablation, validation MAE with one feature family zeroed: removing the timing
family (ttl, gap, age) raises MAE 30x on cluster45 (0.00051 to 0.01557) and 4.6x on
cluster10; the other families move it by less than 10%.

## Worked micro-example

The two capacities the toy trace supports, computed end to end, as the template
for what a real row looks like:

| Quantity | $C=2$ | $C=3$ |
|---|---|---|
| Exact stack distance, $m^\ast$ | 0.8333 | 0.6667 |
| Kaplan-Meier, $\hat m$ | 0.8333 | 0.6667 |
| Residual $e$ | 0.0000 | 0.0000 |

With $N=6$ the standard error of any miss-ratio estimate is about 0.152, so a
six-request trace cannot distinguish a good model from a bad one. The production
runs use $5 \times 10^6$ requests, where the same bound is about $2 \times 10^{-4}$.

## Where the numbers come from

| Artifact | Content |
|---|---|
| `artifacts/tables/<cluster>_metrics.json` | curves, per-model metrics, provisioning |
| `artifacts/tables/<cluster>_importance.json` | permutation, SHAP, ablation |
| `artifacts/tables/metrics.md` | generated markdown tables |
| `artifacts/figures/fig_mrc.pdf` | curves, three panels |
| `artifacts/figures/fig_error.pdf` | absolute error against capacity |
| `artifacts/figures/fig_capacity.pdf` | capacity ratio at three targets |
| `artifacts/figures/fig_importance.pdf` | SHAP top six |
| `artifacts/mlflow/mlflow.db` | one run per cluster, params and metrics |
| `artifacts/optuna/study.db` | 15-trial study, best validation MAE 0.00036 |

Regenerate the figures and tables with `mrc plots`; the metrics come from
`mrc train <cluster>`.
