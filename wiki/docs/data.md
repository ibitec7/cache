# Data

## What it is

`twitter/cache-trace`: anonymised request traces from Twitter's in-memory cache
clusters (Twemcache / Pelikan), released with the OSDI 2020 measurement paper
([Yang et al. 2020](papers/yang-2020-twitter.md)). Traces were collected from
**54 clusters in March 2020**, each **one week** long. The full corpus is 2.8 TB
compressed and 14 TB uncompressed; one shard file holds $10^9$ requests.

## Why it is needed here

It is the only public corpus with production cache traffic at this scale, with
TTL and value size attached to each request, and with write traffic included.
Those fields are exactly the feature set of the hazard classifier, and the TTL
field is what makes censoring non-trivial: an object may leave the cache by
expiry before it is ever reused.

## Formal statement

Each row is one request. Schema, verified against the repository README:

| Column | Type | Meaning | Used as |
|---|---|---|---|
| `timestamp` | integer seconds | request arrival time | inter-arrival gaps $g_i$ |
| `key` | string | anonymised key, namespaces preserved | grouping, reuse distance |
| `key_size` | integer bytes | size of the key | feature |
| `value_size` | integer bytes | size of the value | feature, object size |
| `client_id` | string | anonymised frontend service | feature, grouping |
| `operation` | enum | get, gets, set, add, replace, cas, append, prepend, delete, incr, decr | feature |
| `ttl` | integer seconds | TTL set by the client, **0 when not a write** | feature, censoring logic |

Keys are anonymised but namespaces are preserved, for example
`nz:u:eeW511W3dcH3de3d15ec` keeps `nz` and `u`. Delimiters differ per workload,
so namespace extraction is heuristic, not a fixed split on `:`.

| Symbol | Meaning | Domain |
|---|---|---|
| $T_c$ | cluster index | $\{1,\dots,54\}$ |
| $N_c$ | number of requests in cluster $c$ | $\mathbb{N}$ |
| $s_i$ | object size of request $i$, $s_i = \text{key\_size} + \text{value\_size}$ | bytes |
| $\tau_i$ | TTL attached to a write request | seconds, 0 if not a write |
| $g_i$ | gap $t_i - t_{i-1}$ for the same key | seconds |
| $c_i$ | client id | categorical |
| $o_i$ | operation | categorical |

**Censoring.** A request's reuse episode is right-censored when the trace window
ends before the key is requested again, or when its TTL expires first. Formally
the observed lifetime is $\tilde\tau_i = \min(\tau_i, \tau_i^{\text{TTL}}, T_{\text{end}} - t_i)$
with event indicator $\delta_i \in \{0,1\}$. Only $\delta_i = 1$ means the reuse
distance was actually observed. See [Right-censoring](papers/right-censoring.md).

## Data tiers

| Tier | Content | Size on disk | Command |
|---|---|---|---|
| T0 `samples` | 54 sample files, one per cluster | ~1 GB clone | `make data-samples` |
| T1 | clusters 45, 26, 10 (three smallest) | ~4 GB | `make data-t1` |
| T2 | cluster 52, full shard | 87 GB | `make data-t2` |

`scripts/fetch_traces.sh` is resumable (`curl -C -`), verifies each file with
`zstd -t`, appends to `data/raw/traces/CHECKSUMS.sha256`, and refuses to start a
transfer that would exceed `BUDGET_GB` (default 150).

## Worked micro-example

Six raw rows shaped like the real schema, timestamps in seconds:

| ts | key | key_size | value_size | client | op | ttl |
|---|---|---|---|---|---|---|
| 0 | `a:1` | 3 | 10 | c1 | get | 0 |
| 1 | `b:1` | 3 | 20 | c1 | get | 0 |
| 2 | `a:1` | 3 | 10 | c2 | get | 0 |
| 3 | `c:1` | 3 | 30 | c2 | set | 60 |
| 4 | `b:1` | 3 | 20 | c1 | get | 0 |
| 5 | `d:1` | 3 | 40 | c3 | get | 0 |

Derived quantities, computed from these rows:

| Quantity | Value |
|---|---|
| $N$ | 6 |
| $U$ (distinct keys) | 4 |
| $s_1$ (key + value) | 13 bytes |
| $\kappa = U/N$ | 0.6667 |
| gaps for `a:1` | 2 s |
| gaps for `b:1` | 3 s |
| writes | 1 of 6, so TTL is non-zero for 0.167 of rows |
| namespaces | `a`, `b`, `c`, `d` |

The key column maps to the toy trace A B A C B D used throughout the wiki: the
last two digits are the key's namespace, so this table *is* the toy trace with
sizes attached. That is deliberate, it keeps the pages consistent.

## Where this lives in the code

`src/mrc/data.py` (download, decompress, schema enforcement, dtype downcast to
`polars`; T0/T1/T2 tier selection), `scripts/fetch_traces.sh` (acquisition).
