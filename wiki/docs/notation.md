# Notation

Every symbol used on this site, with its meaning, domain and first page. Where a
symbol has two common meanings in the literature, the column *Not this* records
the meaning it does **not** have here.

## Traces and data

| Symbol | Meaning | Domain | Not this |
|---|---|---|---|
| $i, j$ | request index | $1..N$ | - |
| $N$ | number of requests in a trace (or segment) | $\mathbb{N}$ | not the number of keys |
| $U$ | number of distinct keys | $\mathbb{N}$ | not the number of objects cached |
| $k_i$ | key of request $i$ | categorical | - |
| $t_i$ | arrival timestamp of request $i$ | seconds | not the survival bucket $t$ |
| $s_i$ | object size, key size + value size | bytes | - |
| $\tau_i$ | TTL of a write request, 0 if not a write | seconds | not the lifetime $d$ |
| $g_i$ | inter-arrival gap, $t_i - t_{i-1}$ for the same key | seconds | - |
| $\pi(i)$ | index of previous request for key $k_i$, or $\bot$ | index | - |
| $c_i$ | client id of request $i$ | categorical | not censorings $c_t$ |
| $o_i$ | operation (get, set, ...) | categorical | - |
| $T_c$ | trace window length for cluster $c$ | seconds | - |

## Reuse distance and MRC

| Symbol | Meaning | Domain | Not this |
|---|---|---|---|
| $d_i$ | reuse distance, distinct keys between two accesses | $\mathbb{N}\cup\{\infty\}$ | not elapsed requests |
| $C$ | cache capacity in objects | $[0,U]$ | not bytes |
| $B$ | cache capacity in bytes | $[0,\infty)$ | - |
| $m(C)$ | miss ratio at capacity $C$ | $[0,1]$ | - |
| $H(C)$ | hit ratio, $1 - m(C)$ | $[0,1]$ | not the hazard |
| $m^\ast(C)$ | simulator miss ratio, ground truth | $[0,1]$ | - |
| $\hat m(C)$ | estimator output | $[0,1]$ | - |
| $U_{\text{unique}}$ | keys seen once only | $\mathbb{N}$ | - |
| $\kappa$ | compulsory miss ratio, floors the curve | $[0,1]$ | not the cold-start rate |
| $q$ | spatial sampling rate (SHARDS) | $(0,1]$ | - |
| $M$ | number of capacities in the evaluation grid | $\mathbb{N}$ | - |

## Survival analysis

| Symbol | Meaning | Domain | Not this |
|---|---|---|---|
| $d$ | lifetime of an episode, the reuse distance of the next access | $\mathbb{N}\cup\{\infty\}$ | not the request-level $d_i$ unless stated |
| $t$ | discrete bucket index | $1..K$ | not time in seconds |
| $K$ | number of buckets | $\mathbb{N}$ | - |
| $n_t$ | episodes at risk at the start of bucket $t$ | $\mathbb{N}$ | - |
| $d_t$ | events (observed reuse) in bucket $t$ | $\mathbb{N}$ | not $d_i$ |
| $c_t$ | censorings in bucket $t$ | $\mathbb{N}$ | not client id |
| $h_t$ | discrete hazard, $d_t/n_t$ | $[0,1]$ | - |
| $\lambda(t)$ | continuous-time hazard | $\ge 0$ | - |
| $S(t)$ | survival function, $\Pr(d > t)$ | $[0,1]$ | - |
| $F(t)$ | CDF, $1 - S(t)$ | $[0,1]$ | - |
| $\delta_i$ | event indicator, 1 observed, 0 censored | $\{0,1\}$ | - |
| $\tilde\tau$ | observed (possibly censored) lifetime | $\mathbb{N}$ | - |
| $\Lambda(t)$ | cumulative hazard, $-\log S(t)$ | $\ge 0$ | - |

## Model and search

| Symbol | Meaning | Domain | Not this |
|---|---|---|---|
| $x$ | metadata feature vector of an episode | $\mathbb{R}^p$ | - |
| $p$ | number of features | $\mathbb{N}$ | - |
| $\hat h_t(x)$ | predicted hazard for bucket $t$ given $x$ | $[0,1]$ | - |
| $\hat S(t\mid x)$ | predicted survival | $[0,1]$ | - |
| $\theta$ | LightGBM hyperparameters | - | - |
| $\mathcal{L}$ | Optuna objective (validation MAPE) | $\ge 0$ | - |
| $\eta$ | learning rate | $(0,1]$ | - |
| $L$ | number of leaves per tree | $\mathbb{N}$ | - |
| $R$ | minimum data in leaf | $\mathbb{N}$ | - |

## Evaluation

| Symbol | Meaning | Domain |
|---|---|---|
| MAPE | mean absolute percentage error over the capacity grid | $\ge 0$ |
| MaxErr | worst absolute error over the grid | $\ge 0$ |
| $R^2$ | coefficient of determination on $m(C)$ | $(-\infty,1]$ |
| MAE | mean absolute error | $\ge 0$ |

## Conventions

1. $C$ is always **objects**. Byte capacities are $B$ and are converted with the
   mean object size of the segment.
2. $m$ is non-increasing in $C$; any predicted curve is isotonic-corrected before
   scoring, and the raw version is reported separately so the correction is visible.
3. A hit at capacity $C$ means $d < C$, so $m(C) = S(C-1)$ on the discrete grid.
4. $\infty$ is treated as bucket $K$ everywhere in the hazard code.
