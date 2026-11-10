# Problem

## What it is

A cache operator must choose a capacity $C$ (bytes or objects) for each of 54
production cache clusters. The decision needs the *whole curve* $m(C)$, not one
operating point, because capacity is negotiated per service and the working set
moves with traffic. The curve is the miss ratio function

$$
m(C) = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\{\text{request } i \text{ misses with capacity } C\}.
$$

## Why it is needed here

Measuring $m(C)$ directly means replaying the trace once per candidate $C$. On the
Twitter corpus a single cluster shard is 87 GB compressed ([Data](data.md)), so a
20-point curve costs 20 passes, and every new cluster or month costs the same again.
The project asks two questions:

| Question | Approach |
|---|---|
| Can one pass give the full curve? | exact stack distance (Mattson) and its approximations |
| Can the curve be predicted from metadata, without the keys? | survival model over reuse distance, hazard classifier on metadata |

The second question is the novel one: if the reuse-distance distribution is
predictable from (namespace, operation, key size, value size, TTL, client id,
inter-arrival gaps), then a curve can be produced for a cluster whose traces were
never collected, and a new point can be added without simulation.

## Formal statement

Let the trace be a sequence of requests $r_1,\dots,r_N$ over keys. Fix a replacement
policy (LRU for the main experiments) and a capacity $C$ in objects.

- **Target.** The function $m: [0,U] \to [0,1]$ where $U$ is the number of distinct keys.
- **Estimator A.** Stack distance: $m(C) = \frac{1}{N}\sum_i \mathbf{1}\{d_i \ge C\}$,
  exact under the stack algorithm, see [Reuse distance](reuse-distance.md).
- **Estimator B.** Survival: $m(C) = S(C-1)$ where $S$ is the survival function of
  the reuse distance, estimated by Kaplan-Meier with right-censoring, or by a
  discrete-time hazard classifier, see [Hazard model](hazard-model.md).
- **Reference.** $\hat m_{\text{sim}}(C)$ from libCacheSim on held-out requests,
  see [Ground truth MRC](ground-truth.md).

Symbols are defined on [Notation](notation.md).

| Symbol | Meaning | Domain |
|---|---|---|
| $N$ | number of requests in the trace | $\mathbb{N}$ |
| $U$ | number of distinct keys | $\mathbb{N}$ |
| $C$ | capacity in objects (bytes divided by mean object size) | $[0,U]$ |
| $d_i$ | reuse distance of request $i$ | $\mathbb{N} \cup \{\infty\}$ |
| $m(C)$ | miss ratio at capacity $C$ | $[0,1]$ |
| $H(C)$ | hit ratio, $1-m(C)$ | $[0,1]$ |
| $S(t)$ | survival function of reuse distance | $[0,1]$ |
| $\kappa$ | compulsory miss ratio, $U_{\text{unique}}/N$ | $[0,1]$ |

## Worked micro-example

Trace A B A C B D, $N=6$, $U=4$, so $\kappa = 4/6 = 0.667$. That floor is the
statement that no cache, however large, can avoid the first touch of each key.

| $C$ | misses | $m(C)$ | $H(C)$ |
|---|---|---|---|
| 1 | 6 | 1.000 | 0.000 |
| 2 | 5 | 0.833 | 0.167 |
| 3 | 4 | 0.667 | 0.333 |
| 4 | 4 | 0.667 | 0.333 |
| 8 | 4 | 0.667 | 0.333 |

Two facts worth carrying forward:

1. The curve is a step function. Between integers it does not exist for object
   capacities, which matters when a model predicts continuous values.
2. The curve saturates at $\kappa$. A model that cannot reproduce the compulsory
   floor will look good at small $C$ and wrong at large $C$.

The second fact is why the hazard formulation on [Hazard model](hazard-model.md)
keeps censored (never-reused) objects in the risk set instead of dropping them:
they are exactly the mass at $\kappa$.

## Where this lives in the code

`src/mrc/cli.py` (subcommands `inspect`, `reuse`, `mrc`, `fit`, `plots`) and
`src/mrc/mrc.py` (curve assembly and the $\kappa$ floor).
