# libCacheSim — Cache Simulation Toolkit

| Field | Value |
|---|---|
| Project | libCacheSim: a high performance library for building cache simulators |
| Maintainer | Juncheng Yang (1a1a11a), with contributors |
| Distribution | Python package `libcachesim` (>= 0.3.5) and C library |
| URLs fetched | https://libcachesim.readthedocs.io/en/latest/ and https://github.com/1a1a11a/libCacheSim |
| Verified | 2026-11-10 |
| What I saw | Read the Docs landing page "Welcome to libCacheSim's documentation!"; GitHub repository page titled "libCacheSim: a high performance library for building cache simulators"; `libcachesim.com` redirects to the GitHub repository |
| BibTeX key | `libcachesim` |

## What it did

Provides a fast, modular simulator for caches over request traces, with a large
set of implemented policies (LRU, FIFO, S3-FIFO, SIEVE, ARC, SLRU, LFU variants,
Belady), reader plugins for the common trace formats including the Twitter
`cache-trace` layout, and analysis tooling for MRCs. It is the reference
implementation used by the cache research community for policy comparison.

## The one idea this project borrows

The ground-truth engine, and the discipline of using an independent implementation
for it. Any MRC produced by this project's own stack code is compared against
libCacheSim replaying the same segment at the same capacities, so a bug in the
stack algorithm cannot silently become "ground truth" for the model.

## What this project does differently

libCacheSim answers "what is the miss ratio of policy P at capacity C". This
project answers "what is the miss ratio at capacity C without replaying", and uses
libCacheSim only to define the target and to draw reference curves for other
policies. The Python binding is used here; no C code is modified.

## Where it lands in this code

`src/mrc/simulate.py` (`Cache` objects over the Twitter reader, capacity grid,
policy selection), `tests/test_smoke.py` (agreement between the local stack
implementation and libCacheSim on a small shard).
