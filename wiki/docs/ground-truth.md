# Ground truth MRC

## What it is

The reference curve: replay the held-out request stream through a real cache
simulator (libCacheSim) at a grid of capacities, and record the miss ratio. This
is the only definition of "correct" the project uses.

## Why it is needed here

Both estimators make approximations: the stack algorithm is exact only for the
replacement policy it assumes, and the hazard model is a statistical fit. Without
a simulator reference there is no way to tell a modelling error from a policy
mismatch. The simulator also supplies the reference for policies the stack
algorithm cannot express in closed form (S3-FIFO, ARC, SLRU), and it is where
byte capacities become object capacities.

Three distinct roles, worth keeping separate:

| Role | What is replayed | Used for |
|---|---|---|
| Oracle curve | full held-out trace | the target $m^\ast(C)$ |
| Labelled sample | a subsample of held-out requests | labels for the hazard model |
| Policy reference | same trace, FIFO/S3-FIFO | shows how policy shifts the curve |

## Formal statement

For a capacity grid $C_1 < \dots < C_M$ and a trace segment $\mathcal{T}$,

$$
m^\ast(C_j) = \frac{\#\{\text{misses at } C_j \text{ on } \mathcal{T}\}}{|\mathcal{T}|} .
$$

Error of an estimator $\hat m$ is reported per capacity rather than pooled, because
the curve is used pointwise:

$$
\text{MAPE} = \frac{1}{M}\sum_{j=1}^{M} \frac{|\hat m(C_j) - m^\ast(C_j)|}{m^\ast(C_j)},
\qquad
\text{MaxErr} = \max_j |\hat m(C_j) - m^\ast(C_j)| .
$$

The stack-distance estimator is exact under LRU up to one caveat: object sizes.
With uniform objects it holds exactly; with variable sizes the object-capacity
curve must be converted through the byte-capacity curve, and that conversion is
approximate.

| Symbol | Meaning | Domain |
|---|---|---|
| $C_j$ | $j$-th capacity on the grid | $[0,U]$ |
| $M$ | number of grid points | $\mathbb{N}$ |
| $\mathcal{T}$ | held-out trace segment | sequence of requests |
| $m^\ast(C)$ | simulator miss ratio, ground truth | $[0,1]$ |
| $\hat m(C)$ | estimator output | $[0,1]$ |
| MAPE | mean absolute percentage error over the grid | $\ge 0$ |
| MaxErr | worst absolute error over the grid | $\ge 0$ |

## Worked micro-example

Replay A B A C B D through an LRU cache of 2 objects and of 3 objects, tracking
the resident set explicitly.

$C=2$:

| $i$ | key | resident before | hit? | resident after |
|---|---|---|---|---|
| 1 | A | - | no | A |
| 2 | B | A | no | A B |
| 3 | A | A B | yes | B A |
| 4 | C | B A | no | A C |
| 5 | B | A C | no | C B |
| 6 | D | C B | no | B D |

5 misses of 6, $m^\ast(2) = 0.8333$.

$C=3$:

| $i$ | key | resident before | hit? | resident after |
|---|---|---|---|---|
| 1 | A | - | no | A |
| 2 | B | A | no | A B |
| 3 | A | A B | yes | B A |
| 4 | C | B A | no | B A C |
| 5 | B | B A C | yes | A C B |
| 6 | D | A C B | no | C B D |

4 misses of 6, $m^\ast(3) = 0.6667$.

So the reference curve for this segment is $m^\ast = (1.0000, 0.8333, 0.6667)$ at
$C = (1,2,3)$, and MAPE for a perfect estimator is 0. A hazard model predicting
$\hat m = (1.0000, 0.8300, 0.6723)$, as in the illustrative product on
[Hazard model](hazard-model.md), scores

$$
\text{MAPE} = \tfrac{1}{3}\left(0 + \tfrac{0.0033}{0.8333} + \tfrac{0.0056}{0.6667}\right)
= \tfrac{1}{3}(0 + 0.0040 + 0.0084) = 0.0041 .
$$

That is 0.41 percent average relative error on three points, which is a plausible
magnitude for a good fit and must not be read as a result: it is arithmetic on one
toy segment, no corpus was involved.

## Where this lives in the code

`src/mrc/simulate.py` (libCacheSim driver, capacity grid, policy selection),
`src/mrc/metrics.py` (MAPE, MaxErr, per-capacity residuals).
