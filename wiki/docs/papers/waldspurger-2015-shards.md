# Waldspurger et al. 2015 — Efficient MRC Construction with SHARDS

| Field | Value |
|---|---|
| Authors | Carl A. Waldspurger, Nohhyun Park, Alexander Garthwaite, Irfan Ahmad (CloudPhysics, Inc.) |
| Title | Efficient MRC Construction with SHARDS |
| Venue | 13th USENIX Conference on File and Storage Technologies (FAST '15) |
| Year | 2015 |
| URL fetched | https://www.usenix.org/conference/fast15/technical-sessions/presentation/waldspurger |
| Verified | 2026-11-10 |
| What I saw | Title, the four authors with CloudPhysics affiliation, FAST '15 programme entry and abstract |
| BibTeX key | `waldspurger2015efficient` |

## What it did

Proposed SHARDS (SHarded Approximate Reuse Distance Sampling), which approximates
reuse distances by hashing each key to a sample of $1/q$ and tracking accesses
only to sampled keys, then scaling the sampled distances back up by $q$. The
result is an MRC within a bounded error that costs memory proportional to the
sample, not to the working set. The paper also formalises the bias and variance
of sampled reuse distances and gives the correction that makes the estimate
unbiased, which is the part most reimplementations get wrong.

## The one idea this project borrows

Sample-based reuse distance as the *cost baseline*. SHARDS at $q = 1/32$ is the
comparison any learned model has to beat at matched memory, because if a GBDT
costs more memory than a hash sample and predicts worse, it has no reason to
exist. It also supplies the variance argument: sampled reuse distance is noisy,
and the noise floor bounds how accurate any MRC estimator can be on a shard.

## What this project does differently

SHARDS is an estimator of the same quantity the stack algorithm computes, and it
still needs the keys. This project's model never sees a key: it predicts the
reuse-distance distribution from namespace, sizes, operation, TTL, client and
arrival gaps. SHARDS is exact in expectation as the sample grows; the hazard model
is biased by construction and must show its bias is small on held-out clusters.

## Where it lands in this code

`src/mrc/reuse.py` (sampled reuse distance at configurable $q$, used as the
baseline), `src/mrc/metrics.py` (matched-memory comparison).
