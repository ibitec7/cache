# Experiments

## What it is

The experimental protocol: data splits, baselines, model and search configuration,
metrics, and tracking. One run of the full protocol should be reproducible from
the commands on [Reproduction](reproduction.md).

## Why it is needed here

The claim under test is narrow and falsifiable: *a discrete-time hazard model over
request metadata predicts the MRC about as well as exact stack distance, at a
fraction of the memory, and degrades gracefully on clusters it never saw*. Each
clause needs its own comparison, so the protocol is built as a matrix rather than
one number.

## Formal statement

**Splits.** Time-ordered, per cluster, never shuffled (shuffling leaks future
reuse into training):

| Split | Fraction of requests | Use |
|---|---|---|
| train | first 0.60 | fit hazard model, fit Optuna trials |
| validation | next 0.20 | Optuna objective, early stopping |
| test | last 0.20 | all reported numbers |

Cross-cluster generalisation is a separate protocol: leave-one-cluster-out over the
T1 clusters, fit on the other two, report on the held-out one.

**Baselines.**

| Baseline | What it establishes |
|---|---|
| Exact stack distance (Mattson) | upper bound on accuracy under LRU |
| Counter stacks | cost of sublinear-space approximation |
| SHARDS at $q = 1/32$ | published approximation at matched memory |
| libCacheSim LRU | confirms the stack implementation |
| Hazard, Kaplan-Meier only (no covariates) | value added by features |
| Hazard, LightGBM, default params | value added by Optuna |

**Model and search.** LightGBM binary classifier over (episode, bucket) rows,
objective `binary`, early stopping on validation logloss. Optuna TPE, 3 seeds per
trial, objective = validation MAPE at fixed trial count:

| Hyperparameter | Range | Scale |
|---|---|---|
| `num_leaves` | 15 to 255 | int, log |
| `learning_rate` | 0.01 to 0.3 | float, log |
| `min_data_in_leaf` | 20 to 500 | int |
| `feature_fraction` | 0.4 to 1.0 | float |
| `bagging_fraction` | 0.4 to 1.0 | float |
| `lambda_l2` | $10^{-3}$ to $10$ | float, log |
| `max_depth` | 3 to 12 | int |

**Metrics.** MAPE over the capacity grid (primary), MaxErr, MAE, and $R^2$ on
$m(C)$. Reported per cluster and aggregated by median, never by mean alone, since
a few clusters with near-zero miss ratio dominate ratio errors.

## Worked micro-example

Take the toy trace A B A C B D and split it 4/1/1 in time order: train on requests
1 to 4, validation on 5, test on 6. The protocol then asks: fit on the first four
requests, score on the last one.

Training episodes from requests 1..4: e1 (A at 1, next at 3, $d=1$, event),
e2 (B at 2, censored at window end), e3 (A at 3, censored), e4 (C at 4, censored).
Risk sets at $K=4$: $n_1 = 4$, $d_1 = 1$, so $h_1 = 0.25$ and $S(1) = 0.75$; the
three censored episodes leave at bucket 4, so $S(t) = 0.75$ for $t \ge 1$.

Predicted curve from the four-request fit: $m(1) = 1.000$, $m(2) = 0.750$.
Test request 6 is D, a first access, so it misses at every capacity: observed
$m_{\text{test}}(1) = m_{\text{test}}(2) = 1.000$. MAPE for this fit is
$\frac{1}{2}(0 + \frac{0.25}{1.0}) = 0.125$.

The lesson is the one the protocol encodes: with a test set of one request, the
only possible score is 0 or 1 and the curve cannot be validated at all. This is
why every reported number uses the corpus, and why the toy trace is used for
algebra checks only. Fitting on 4 of 6 requests also shows the censoring mass
moving the curve by 0.0833 between the full-trace estimate $m(2)=0.8333$ and the
prefix estimate $m(2)=0.750$, purely from window length.

## Where this lives in the code

`src/mrc/splits.py` (time-ordered and leave-one-cluster-out splits),
`src/mrc/train.py` (LightGBM fit, early stopping),
`src/mrc/search.py` (Optuna study, pruning), `configs/` (YAML for data, model, search).
