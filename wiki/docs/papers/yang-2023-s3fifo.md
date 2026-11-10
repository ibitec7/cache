# Yang et al. 2023 — FIFO Queues Are All You Need for Cache Eviction

| Field | Value |
|---|---|
| Authors | Juncheng Yang, Yazhuo Zhang, Ziyue Qiu, Yao Yue, Rashmi Vinayak |
| Title | FIFO queues are all you need for cache eviction |
| Venue | Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23), pp. 130–149 |
| Year | 2023 |
| DOI | 10.1145/3600006.3613147 |
| URL fetched | https://api.crossref.org/works/10.1145/3600006.3613147 |
| Verified | 2026-11-10 |
| What I saw | Title, the five authors in the order above, container "Proceedings of the 29th Symposium on Operating Systems Principles", pages 130–149, issued 2023-10-23 |
| BibTeX key | `yang2023fifo` |
| Note | Reading list asked to verify the title; the registered title is the lowercase form above. |

## What it did

Proposed S3-FIFO, a policy made of three FIFO queues (small, main, ghost) with a
frequency sketch instead of a strict LRU order. Objects enter a small queue, are
promoted to a main queue on reuse, and evicted objects leave a ghost fingerprint
so that repeated failures to promote are detected. On the Twitter traces and on
several public storage traces it matches or beats LRU and state-of-the-art
policies while using less metadata and fewer per-request updates.

## The one idea this project borrows

Two things, both about honesty of the evaluation. First, S3-FIFO is the policy that
makes LRU curves look optimistic on skewed production traffic, so the reference
curves include a FIFO-family policy to show how much of the miss ratio is policy
versus capacity. Second, the paper's method of comparing curves at matched
*metadata* cost rather than matched capacity is applied here when the hazard model
is compared against SHARDS.

## What this project does differently

S3-FIFO changes the policy and measures the resulting curve. This project keeps the
policy fixed (LRU for the stack identity, FIFO-family as a reference) and changes
the *estimator*. The two are complementary: a better policy shifts the curve, a
better estimator computes it faster.

## Where it lands in this code

`src/mrc/simulate.py` (S3-FIFO avatars via libCacheSim, policy reference curves),
`artifacts/tables/policy_compare.csv`.
