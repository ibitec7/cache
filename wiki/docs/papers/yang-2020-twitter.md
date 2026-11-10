# Yang, Yue, Rashmi 2020 — A Large Scale Analysis of Hundreds of In-Memory Cache Clusters at Twitter

| Field | Value |
|---|---|
| Authors | Juncheng Yang (CMU), Yao Yue (Twitter), K. V. Rashmi (CMU) |
| Title | A large scale analysis of hundreds of in-memory cache clusters at Twitter |
| Venue | 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20) |
| Year | 2020 |
| URL fetched | https://www.usenix.org/conference/osdi20/presentation/yang |
| Verified | 2026-11-10 |
| What I saw | Title, the three authors with the affiliations above, OSDI 20 programme page with abstract and PDF link |
| BibTeX key | `yang2020large` |

## What it did

Measured the production cache fleet behind Twitter: hundreds of clusters, tens of
thousands of machines, and over a year of telemetry, then released request traces
from 54 clusters collected in March 2020. The paper characterises workload
properties that had only been studied on key-value benchmarks: skewed popularity,
one-hit wonders, TTL-driven invalidation, write-heavy clusters, and the gap between
production miss ratios and what standard simulators predict. It is the source of the
dataset this project uses ([Data](../data.md)).

## The one idea this project borrows

The workload taxonomy, and specifically the finding that TTL expiry and write
traffic are first-class effects, not noise. That is why every trace row carries
`ttl`, why writes are kept rather than filtered out, and why censoring in this
project has two causes (end of window, expiry) instead of one.

## What this project does differently

The paper is descriptive: it reports distributions and argues about policy design.
This project is predictive: it fits a hazard model to the reuse-distance
distribution and scores the resulting curve against simulation. It also works at a
different granularity, one request's metadata rather than a cluster's aggregate
statistics, because the model must generalise across clusters rather than describe
one.

## Where it lands in this code

`src/mrc/data.py` (schema and TTL handling), `scripts/fetch_traces.sh` (acquisition
from the CMU PDL mirror named in the dataset README).
