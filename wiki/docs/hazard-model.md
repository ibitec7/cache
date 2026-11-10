# Hazard model

## What it is

A survival model for the reuse distance. Each request opens an episode: the time
until that key is requested again, measured in the reuse distance of the next
access. Episodes that never close inside the trace are right-censored. The
distribution of that lifetime gives the MRC by $m(C) = S(C-1)$.

Two estimators of the same quantity:

| Estimator | Form | Role |
|---|---|---|
| Kaplan-Meier | nonparametric product-limit, no covariates | consistency check, baseline |
| Discrete-time hazard GBDT | LightGBM binary classifier per distance bucket over metadata | the model this project contributes |

## Why it is needed here

The stack algorithm needs every key of every request and $O(U)$ memory, so a
54-cluster corpus costs a full pass per cluster. A hazard model replaces the pass
with a prediction from cheap per-request metadata. It also buys two things the
stack algorithm cannot do:

1. **Extrapolation in capacity** without ever materialising a stack at that size.
2. **Transfer across clusters**: fit on clusters with traces, predict the curve for
   a cluster with none.

The cost is that it is a model, so it must be checked against
[ground truth](ground-truth.md) at every capacity, not just on average.

## Formal statement

Discretise the lifetime into buckets $t = 1, \dots, K$ (bucket $t$ holds reuse
distances $t$; the last bucket also holds $\infty$). With $n_t$ episodes at risk
at the start of bucket $t$, $d_t$ events and $c_t$ censorings in it:

$$
h_t = \frac{d_t}{n_t}, \qquad S(t) = \prod_{s=1}^{t} (1 - h_s), \qquad n_{t+1} = n_t - d_t - c_t .
$$

Because a request hits at capacity $C$ exactly when its reuse distance is $< C$,

$$
m(C) = \Pr(d \ge C) = S(C-1), \qquad S(0) = 1 .
$$

Covariates enter by making the hazard a function of the feature vector $x$:

$$
h_t(x) = \Pr(d = t \mid d \ge t, x), \qquad S(t \mid x) = \prod_{s \le t} \bigl(1 - h_s(x)\bigr),
$$

and LightGBM is trained as a binary classifier on the expanded table of
(episode, bucket) rows, label $y = 1$ if the event falls in bucket $t$. The
censoring correction lives in the row construction: a censored episode contributes
rows with $y=0$ only up to its censoring bucket.

| Symbol | Meaning | Domain |
|---|---|---|
| $d$ | reuse distance of the next access (the lifetime) | $\mathbb{N}\cup\{\infty\}$ |
| $t$ | bucket index | $1..K$ |
| $K$ | number of buckets | $\mathbb{N}$ |
| $n_t$ | episodes at risk at start of bucket $t$ | $\mathbb{N}$ |
| $d_t$ | events (observed reuse) in bucket $t$ | $\mathbb{N}$ |
| $c_t$ | censorings in bucket $t$ | $\mathbb{N}$ |
| $h_t$ | discrete hazard, $d_t/n_t$ | $[0,1]$ |
| $S(t)$ | survival, $\Pr(d > t)$ in the discrete convention | $[0,1]$ |
| $m(C)$ | miss ratio at capacity $C$ | $[0,1]$ |
| $x$ | metadata feature vector of an episode | $\mathbb{R}^p$ |
| $\delta$ | event indicator, 1 observed, 0 censored | $\{0,1\}$ |

## Worked micro-example

Trace A B A C B D. Six episodes, one per request. Only two close inside the
window:

| Episode | Request | Next same key | $d$ | $\delta$ |
|---|---|---|---|---|
| e1 | A at 1 | A at 3 | 1 | 1 |
| e2 | B at 2 | B at 5 | 2 | 1 |
| e3 | A at 3 | none | - | 0 |
| e4 | C at 4 | none | - | 0 |
| e5 | B at 5 | none | - | 0 |
| e6 | D at 6 | none | - | 0 |

The four censored episodes are censored at the window end, bucket $K=4$, because
the trace cannot tell us whether they would have been reused at distance 5, 50 or
never. Dropping them instead would be the mistake described in
[Right-censoring](papers/right-censoring.md).

Kaplan-Meier table, $K=4$:

| $t$ | $n_t$ | $d_t$ | $c_t$ | $h_t$ | $S(t)$ |
|---|---|---|---|---|---|
| 1 | 6 | 1 | 0 | 0.1667 | 0.8333 |
| 2 | 5 | 1 | 0 | 0.2000 | 0.6667 |
| 3 | 4 | 0 | 0 | 0.0000 | 0.6667 |
| 4 | 4 | 0 | 4 | 0.0000 | 0.6667 |

Read the arithmetic: $S(1) = 1 - 1/6 = 0.8333$; $S(2) = 0.8333 \times (1 - 1/5) = 0.6667$;
buckets 3 and 4 carry no events, so the curve is flat and ends at 0.6667. The
survival function never reaches zero, and that residual 0.6667 is not a bug: it is
the mass of keys that were never reused inside the window, which is exactly the
compulsory floor $\kappa = 4/6$ from [Problem](problem.md).

Now the two required capacities:

$$
m(1) = S(0) = 1.000, \qquad m(2) = S(1) = 0.8333 .
$$

Both match the direct simulation in [Reuse distance](reuse-distance.md)
($m(1)=1.000$, $m(2)=0.833$), and $m(3)=S(2)=0.6667$ matches too. That agreement
on a six-request trace is a sanity check of the algebra, not evidence of accuracy;
accuracy is an empirical question answered on [Results](results.md).

For the GBDT form, the same episodes become classifier rows. Suppose the fitted
model returns $\hat h_1(x)=0.17$, $\hat h_2(x)=0.19$, $\hat h_3(x)=0.02$ for one
feature profile, computed as

$$
\hat S(1) = 1 - 0.17 = 0.8300, \quad
\hat S(2) = 0.8300 \times 0.81 = 0.6723, \quad
\hat S(3) = 0.6723 \times 0.98 = 0.6589,
$$

so that profile is predicted at $m(1)=1.000$, $m(2)=0.8300$, $m(3)=0.6723$.
The numbers here are illustrative hazards, chosen to show the product, not fitted
values from the corpus; fitted values appear only in `artifacts/tables/`.

## Where this lives in the code

`src/mrc/hazard.py` (episode construction, risk sets, Kaplan-Meier),
`src/mrc/features.py` (metadata to design matrix, bucket expansion),
`src/mrc/train.py` (LightGBM hazard classifier, calibration),
`src/mrc/search.py` (Optuna study).
