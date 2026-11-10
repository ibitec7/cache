# Mattson et al. 1970 — Evaluation Techniques for Storage Hierarchies

| Field | Value |
|---|---|
| Authors | R. L. Mattson, J. Gecsei, D. R. Slutz, I. L. Traiger |
| Title | Evaluation techniques for storage hierarchies |
| Venue | IBM Systems Journal, vol. 9, no. 2, pp. 78–117 |
| Year | 1970 |
| DOI | 10.1147/sj.92.0078 |
| URL fetched | https://api.crossref.org/works/10.1147/sj.92.0078 |
| Verified | 2026-11-10 |
| What I saw | Title, four authors in the order given above, `issued` 1970, container `IBM Systems Journal`, volume 9, issue 2, pages 78–117 |
| BibTeX key | `mattson1970evaluation` |

## What it did

Introduced the stack algorithm for storage hierarchies. The observation is that if
the cache is modelled as a stack ordered by recency, the position of a referenced
item in that stack (its stack distance) is independent of capacity: a request at
depth $d$ misses in every cache of size $\le d$ and hits in every larger one. One
pass over the reference string therefore yields the miss ratio at *all* capacities
simultaneously, instead of one pass per capacity. This is the origin of the reuse
distance and of the MRC as an object of study.

## The one idea this project borrows

Exactly one: $m(C) = \frac{1}{N}\sum_i \mathbf{1}\{d_i \ge C\}$, the tail-count
identity. It is Estimator A on [Reuse distance](../reuse-distance.md), it defines
the evaluation target for the model, and it is the reason the hazard model's
lifetime is measured in *distinct keys* rather than elapsed requests.

## What this project does differently

Mattson assumes the reference string is available and the stack fits in memory.
This project asks what happens when neither holds: the trace is 87 GB per shard,
the stack must be spilled or sampled, and the goal is to predict the tail of the
distribution from request metadata rather than to count it. The stack algorithm
appears here as the accuracy upper bound, not as the method.

## Where it lands in this code

`src/mrc/reuse.py` implements the stack algorithm over a counter stack;
`src/mrc/mrc.py` performs the tail count.
