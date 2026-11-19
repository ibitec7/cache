# Results

## What it is

The reporting surface for the experiment matrix on [Experiments](experiments.md).
Every number here is read from `artifacts/tables/`, which the pipeline writes after a
run (`<cluster>_metrics.json` for accuracy and provisioning, `method_ablation.json` for
the feature-family ablation). Nothing is estimated, rounded from memory, or
illustrative.

!!! note "Status: four production traces measured, two protocols"
    **Single block**: cluster45, cluster26, cluster52 and cluster10, first 5,000,000
    requests each, 70/15/15, one test block scored once. This is the table below.

    **Windowed**: the same four traces under out-of-time (OOT) and
    leave-one-cluster-out (LOCO) evaluation, 5,000,000 requests per window, every test
    window scored. See [Windowed evaluation](#windowed-evaluation). The exact Mattson
    curve is the ground truth in both. cluster52 is the held-out trace: every design
    choice was frozen before its labels were read.

## What is reported, and why

A curve claim can be true on average and false where it matters: the interesting
region is small capacity, where a small capacity error is a large ratio error. A
result is reported four ways:

| View | Question | Artefact |
|---|---|---|
| Per-capacity residual | where on the curve does the model fail | `report/figures/cluster-error.pdf` |
| Curve error | how far is the whole curve | `<cluster>_metrics.json` |
| Provisioning | what capacity does the error cost | `report/figures/capacity-ratio.pdf`, `metrics.json` |
| Attribution | which features carry the curve | `report/figures/importance.pdf`, `<cluster>_importance.json` |

## Formal statement

For capacity $C_j$, cluster $c$, predicted curve $\hat m$ and exact curve $m^\ast$,

$$e_{c,j} = \hat m_c(C_j) - m^\ast_c(C_j).$$

Reported statistics are MAE, MAX, signed bias, MRE over points with truth at least
0.01, and ABC (area between curves, integrated in log capacity and normalised by
the log range). Capacities are the upper edges of the hazard bins, so the step
function and $P(R \ge C)$ refer to the same event.

| Symbol | Meaning | Domain |
|---|---|---|
| $c$ | cluster | {45, 26, 52, 10} |
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
| cluster52 | hazard GBDT | 0.00027 | 0.00079 | +0.00020 | 0.0007 |
| cluster52 | Kaplan-Meier | 0.02481 | 0.04035 | -0.02481 | 0.0614 |
| cluster52 | hit-count regression | 0.11342 | 0.25439 | -0.00889 | 0.2352 |
| cluster10 | hazard GBDT | 0.00772 | 0.01902 | -0.00772 | 0.0127 |
| cluster10 | Kaplan-Meier | 0.04387 | 0.12042 | -0.04387 | 0.0623 |
| cluster10 | hit-count regression | 0.07930 | 0.33206 | -0.06852 | 0.1112 |

## Windowed evaluation

The single-block table above reads one block once. The windowed families read every
window of the last 25% of each trace, so the unit of evidence is a window and the
reported number is a mean over windows. Design and interpretation are on
[Generalization](generalization.md); the numbers are here.

| Trace | Protocol | n | MAE | median | KM MAE | ratio | MRE | windows won |
|---|---|---|---|---|---|---|---|---|
| cluster10 | OOT | 7 | 0.003637 | 0.003273 | 0.027656 | 0.132 | 0.61% | 7/7 |
| cluster26 | OOT | 8 | 0.001518 | 0.001794 | 0.005109 | 0.297 | 0.38% | 8/8 |
| cluster45 | OOT | 12 | 0.001313 | 0.000977 | 0.024877 | 0.053 | 0.21% | 12/12 |
| cluster52 | OOT | 672 | 0.003834 | 0.003184 | 0.030559 | 0.125 | 0.70% | 672/672 |
| cluster10 | LOCO | 7 | 0.002756 | 0.002682 | 0.160934 | 0.017 | 0.47% | 7/7 |
| cluster26 | LOCO | 8 | 0.006770 | 0.006903 | 0.156697 | 0.043 | 3.78% | 8/8 |
| cluster45 | LOCO | 12 | 0.005445 | 0.005631 | 0.097702 | 0.056 | 0.97% | 12/12 |
| cluster52 | LOCO | 667 | 0.016266 | 0.015778 | 0.144415 | 0.113 | 3.24% | 667/667 |

cluster52 OOT is the only row whose test block is a full distribution: 672 of 672
windows, 94 shards, no subsampling, block-bootstrap 95% interval on the mean MAE
[0.003416, 0.004318]. LOCO reports 667 because three partial windows at the store
boundary merge.

Data budget, fixed hyper-parameters, same test windows at every budget:

| Trace | 5M requests | 25M | 60M | n |
|---|---|---|---|---|
| cluster10 | 0.017567 | 0.006046 | 0.004259 | 7 |
| cluster26 | 0.003057 | 0.001861 | 0.001706 | 8 |
| cluster45 | 0.001398 | 0.001313 | 0.001333 | 12 |

Two properties of this table are load bearing. The model wins every scored window
against Kaplan-Meier (699 of 699 across OOT, 694 of 694 across LOCO), and the
in-domain residual is almost entirely a signed level offset (|bias|/MAE 0.74 to 0.97,
always negative) rather than shape error.

## Hyper-parameter search: the first 12 trials were not a search

The Optuna studies are the one place where a naive read of the artefacts would be
wrong, so the finding is recorded here rather than in a log.

At 12 trials, cluster10, cluster26 and cluster45 returned **identical** chosen
hyper-parameters and a Spearman rank correlation of exactly 1.000 between the
trial-value rankings of every pair of traces. Seeded TPE spends its first 10 trials on
random draws, so the ranking was dominated by capacity, not by the trace.

Continuing the same studies past trial 20 diverged per trace (num_leaves 24, 20 and 25;
learning rate 0.110, 0.104, 0.103) and improved every test MAE: cluster10 0.003704 to
0.003637, cluster26 0.001762 to 0.001518, cluster45 0.001517 to 0.001313.

**Retracted**: the 12-trial numbers. **Reported**: the extended studies, which hold 35, 29,
30 and 20 complete trials for cluster10, cluster26, cluster45 and cluster52 (counted from
`artifacts/study_<cluster>.db`, `trials` joined on `trial_values` where `state = COMPLETE`).
A study that is reported must have run past trial 20, or it is random search wearing a
sampler's name. cluster52 is the shortest at 20 because its windows are the expensive ones.

## Baselines: what each one establishes

| Baseline | Status | Test-block MAE (single block) | Reading |
|---|---|---|---|
| Exact stack distance (Mattson) | ground truth | 0 by definition | the target, not a competitor |
| Kaplan-Meier, no covariates | run, all four traces | 0.0440 / 0.0118 / 0.0200 / 0.0248 (c10/c26/c45/c52) | the floor the covariates must beat |
| Hit-count regression | run, all four traces | 0.0793 / 0.0793 / 0.1097 / 0.1134 | 3x to 9x worse than the floor; frequency is not a distance |
| `single_row` | run, all four traces | 0.000000 | **oracle, not a baseline**: it reads `eval_rtilde` for the block it scores. Excluded from headline tables |

Hit-count is the practitioner's shortcut this project argues against, and it loses to
the covariate-free floor on every trace, which is the sharper result: the conversion
from predicted frequency to a reuse distance through marginal deciles destroys more
than the covariates add.

`single_row` is labelled `oracle$^\dagger$` in `artifacts/tables/method_ablation.json`
and the caption there says so. The raw `<cluster>_metrics.json` files carry the same
0.000000 without that footnote, so read the ablation table, not the metrics file.

**Cost of running them.** Measured on this machine, cluster10, 1,000,000 requests:
setup (labels, features, grid, prepare) 1.2 s, hit-count regression 0.6 s,
`single_row` 0.1 s, 1.9 s in total. Both baselines fit one booster on the training
design matrix and then evaluate per window, so the cost is dominated by streaming the
windows, not by the baselines. Four traces at the out-of-time design is a local job on
12 cores, not a cluster job. The scripted path is
`scripts/method_ablation.py` (nine variants, 1M requests per trace).

## Provisioning

Capacity ratio, model over truth, for a target hit ratio expressed as a fraction of
the achievable ceiling $1-p_c$. Below one under-provisions.

| Cluster | Model | 60% | 80% | 95% |
|---|---|---|---|---|
| cluster45 | hazard GBDT | 1.003 | 0.976 | 1.004 |
| cluster45 | Kaplan-Meier | 0.922 | 0.239 | 0.524 |
| cluster26 | hazard GBDT | 0.977 | 0.981 | 0.999 |
| cluster26 | Kaplan-Meier | 0.971 | 0.985 | 0.996 |
| cluster52 | hazard GBDT | 1.005 | 1.004 | 1.009 |
| cluster52 | Kaplan-Meier | 0.657 | 0.563 | 0.522 |
| cluster10 | hazard GBDT | 0.947 | 0.935 | 0.925 |
| cluster10 | Kaplan-Meier | 0.635 | 0.681 | 0.744 |

## Attribution

Mean absolute SHAP contribution, top three per cluster:

| Cluster | First | Second | Third |
|---|---|---|---|
| cluster45 | log gap (2.66) | hazard step (1.42) | log age (0.12) |
| cluster26 | log gap (1.32) | hazard step (1.03) | log age (0.10) |
| cluster10 | log gap (3.29) | hazard step (3.29) | log object size (0.08) |

Ablation, test-block MAE on a 1M-request subsample with one feature family zeroed:
removing the timing family (ttl, gap, age) raises MAE 51x on cluster45 (0.00042 to
0.02131) and 5.4x on cluster10 (0.00738 to 0.04015). On cluster45 and cluster10 every
other family stays within 17 % of the full model. On cluster26 the timing family costs
1.05x (0.00567 to 0.00598), while the client and popularity families lower MAE to 0.67x
of the full model.

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
| `artifacts/tables/method_ablation.json` | feature-family ablation, nine estimators, 1M subsample |
| `artifacts/tables/metrics.md` | generated markdown tables |
| `artifacts/oot/<trace>/result.json` | out-of-time summary, per-window metrics, chosen params |
| `artifacts/oot/<trace>/test/*.json` | one file per scored window, the evidence under every mean |
| `artifacts/loco/<trace>/result.json` | leave-one-cluster-out summary and per-window metrics |
| `artifacts/retrain/<trace>-<budget>/result.json` | retrain curve point, fixed hyper-parameters |
| `artifacts/study_<trace>.db` | Optuna study, one sqlite per trace, all trials |
| `artifacts/study_loco_<trace>.db` | Optuna study for the LOCO fold |
| `report/figures/mrc-curves.pdf` | curves, three panels |
| `report/figures/cluster-error.pdf` | absolute error against capacity |
| `report/figures/capacity-ratio.pdf` | capacity ratio at three targets |
| `report/figures/importance.pdf` | SHAP top six |
| `artifacts/mlflow/mlflow.db` | one run per cluster, params and metrics |
| `artifacts/optuna/study.db` | 15-trial study, best validation MAE 0.00036 |

Regenerate the figures and tables with `mrc plots`; the metrics come from
`mrc train <cluster>`.
