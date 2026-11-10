# Discrete-time survival analysis

## What it is

A survival model in which the lifetime is measured on a discrete grid (bucket 1,
2, 3, ...) and the model is a sequence of conditional probabilities: given survival
to bucket $t$, what is the chance the event happens in bucket $t$. It is the
discrete analogue of the Cox hazard, and its nonparametric estimator is the
Kaplan-Meier product.

## Why it is needed here

Reuse distance is an integer by construction: the number of distinct keys between
two accesses. There is no continuous time to model, so a continuous hazard would
have to invent a scale. Discrete time also makes the estimation problem ordinary
classification: each (episode, bucket) pair is a binary row, so any classifier,
including LightGBM, can produce the hazard. The classic treatment of this
reformulation is Allison (1982), *Discrete-Time Methods for the Analysis of Event
Histories*, Sociological Methodology 13, 61, DOI 10.2307/270718 (verified via
Crossref on 2026-11-10).

## Formal statement

Partition the lifetime support into buckets $1..K$, with $K$ absorbing every
lifetime longer than $K$ (including $\infty$). For an episode with features $x$:

$$
h_t(x) = \Pr(d = t \mid d \ge t, x), \qquad S(t \mid x) = \prod_{s=1}^{t}\bigl(1 - h_s(x)\bigr),
$$

$$
\Pr(d = t \mid x) = h_t(x)\, S(t-1 \mid x), \qquad m(C\mid x) = S(C-1 \mid x).
$$

Estimation by maximum likelihood on the expanded table is equivalent to fitting
$K$ independent binary classifiers only if each bucket's rows are weighted by the
risk set; in practice one classifier is fitted on all rows with the bucket index as
a feature and the cumulative product is taken afterwards, which is the
"single-model" variant used here.

| Symbol | Meaning | Domain |
|---|---|---|
| $t$ | bucket index | $1..K$ |
| $K$ | number of buckets, absorbing the tail | $\mathbb{N}$ |
| $d$ | lifetime, the reuse distance of the next access | $\mathbb{N}\cup\{\infty\}$ |
| $h_t(x)$ | discrete hazard in bucket $t$ | $[0,1]$ |
| $S(t\mid x)$ | survival function | $[0,1]$ |
| $m(C\mid x)$ | miss ratio at capacity $C$ | $[0,1]$ |
| $x$ | feature vector | $\mathbb{R}^p$ |
| $\delta$ | event indicator | $\{0,1\}$ |

## Worked micro-example

Trace A B A C B D, six episodes, two events, four censored at $K=4$.

Expanded table with the risk set made explicit ($y=1$ only where the event lands):

| Episode | Buckets with $y=0$ | Bucket with $y=1$ | Censoring bucket |
|---|---|---|---|
| e1 ($d=1$) | - | 1 | - |
| e2 ($d=2$) | 1 | 2 | - |
| e3 | 1, 2, 3 | - | 4 |
| e4 | 1, 2, 3 | - | 4 |
| e5 | 1, 2, 3 | - | 4 |
| e6 | 1, 2, 3 | - | 4 |

Column sums give $n_1=6$ rows, $d_1=1$; $n_2=5$, $d_2=1$; $n_3=4$, $d_3=0$;
$n_4=4$, $d_4=0$. Hence

$$
S(1) = 1 - \tfrac{1}{6} = 0.8333, \quad
S(2) = 0.8333 \times \tfrac{4}{5} = 0.6667, \quad
S(3) = S(4) = 0.6667,
$$

and the curve is $m(1)=1.0000$, $m(2)=0.8333$, $m(3)=0.6667$, $m(4)=0.6667$. This
is the same table as [Hazard model](../hazard-model.md), derived here from the
row-expanded view, which is how the code builds it.

The absorbing bucket matters: if the four censored episodes had been given
$y=1$ at bucket 4 instead of being censored, $S(4)$ would be $0$ and the model
would predict $m(5)=0$, asserting that every key is reused within distance 4. On a
week-long production trace that assertion is false and expensive.

## Where this lives in the code

`src/mrc/hazard.py` (`expand_episodes`, risk-set bookkeeping, cumulative product),
`src/mrc/features.py` (bucket index as a feature).
