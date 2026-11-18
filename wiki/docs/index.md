# MRC Curve Wiki

Working documentation for the project *Cache hit-ratio curves from reuse distance
on Twitter production cache traces*.

The deliverable is a miss-ratio curve (MRC) estimator that predicts the miss ratio
$m(C)$ of a cache of capacity $C$ without replaying the trace at every capacity.
Two estimators are built and compared:

1. **Exact stack distance** (Mattson): one pass, exact $m(C)$ for every $C$, expensive in memory.
2. **Discrete-time hazard model** (Kaplan-Meier plus a LightGBM hazard classifier):
   predicts the survival function of the reuse distance from request metadata, so
   the curve can be extrapolated to capacities not simulated and to clusters not traced.

Ground truth for both is a full simulation with libCacheSim.

## What is on each page

| Page | Question it answers |
|---|---|
| [Problem](problem.md) | Why MRC, why these traces, what is measured |
| [Data](data.md) | What the 54 Twitter cluster traces contain |
| [Reuse distance](reuse-distance.md) | How $d_i$ is defined and turned into $m(C)$ |
| [Hazard model](hazard-model.md) | How survival analysis estimates the same curve |
| [Ground truth MRC](ground-truth.md) | How simulation defines the target |
| [Experiments](experiments.md) | Splits, baselines, metrics, tracking |
| [Results](results.md) | Where numbers land, per block and per window, and which numbers are retracted |
| [Generalization](generalization.md) | Out-of-time, leave-one-cluster-out and retrain results, and what they do not prove |
| [Notation](notation.md) | Every symbol used on this site |
| [Papers](papers/index.md) | Verified reading list, one page per source |
| [Glossary](glossary.md) | One-line definitions |
| [Reproduction](reproduction.md) | Exact commands and artifact paths |

## The toy trace used on every page

Every worked example on this site uses the same six-request trace, so the pages can
be read in any order and the numbers reconcile.

| Request | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Key | A | B | A | C | B | D |
| Reuse distance $d_i$ | $\infty$ | $\infty$ | 1 | $\infty$ | 2 | $\infty$ |
| Hit at $C=1$ | no | no | no | no | no | no |
| Hit at $C=2$ | no | no | yes | no | no | no |
| Hit at $C=3$ | no | no | yes | no | yes | no |

This gives $m(1)=1.000$, $m(2)=0.833$, $m(3)=0.667$, and $m(C)=0.667$ for all
$C \ge 3$, because only 4 of the 6 requests are compulsory misses and no capacity
can remove those. All four numbers appear again, derived independently, on
[Reuse distance](reuse-distance.md) and [Hazard model](hazard-model.md).

## Reading order

For the statistical argument: [Problem](problem.md) then [Hazard model](hazard-model.md).
For the systems argument: [Reuse distance](reuse-distance.md) then [Ground truth MRC](ground-truth.md).
For running anything: [Reproduction](reproduction.md).

## Headline results

Out-of-time, every window of the last 25% of each trace, against the exact stack-distance
curve on the same windows. Full tables and caveats on [Results](results.md).

| Trace | Windows | Model MAE | Kaplan-Meier MAE | Ratio |
|---|---|---|---|---|
| cluster10 | 7 | 0.003637 | 0.027656 | 0.132 |
| cluster26 | 8 | 0.001518 | 0.005109 | 0.297 |
| cluster45 | 12 | 0.001313 | 0.024877 | 0.053 |
| cluster52 | 672 | 0.003834 | 0.030559 | 0.125 |

The model beats the covariate-free floor on 699 of 699 scored test windows. Trained on
the other three traces and applied to a fourth it still beats the floor on every window
but loses about 4x in absolute accuracy, which is the boundary of the current claim:
the method transfers, one set of weights does not.

!!! note "Scope of this wiki"
    The `src/` tree is written by a separate worker. Code paths named in the
    "Where this lives in the code" sections are the agreed module layout, not
    necessarily files that already exist on disk.
