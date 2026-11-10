# Data manual

Everything in this document was verified against the sources cited at the bottom.
Numbers marked *measured* were produced on this machine on 2026-11-10.

## 1. Where the data comes from

`twitter/cache-trace` publishes anonymized request traces from 54 in-memory cache
clusters (Twemcache and Pelikan) in production, collected in March 2020, each one
week long. The full corpus is **2.8 TB compressed / 14 TB uncompressed**, so the
project never downloads it whole.

| Mirror | Status | Use |
|---|---|---|
| `ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/` | live | primary, one file per cluster, `clusterN.sort.zst` |
| SNIA trace repository (`iotta.snia.org/tracetypes/17`) | live | fallback, split into smaller files |
| Baidu pan | live, Chinese UI | fallback for Asia |
| Storj | **dead since Dec 2020** | do not use |

*Measured* sizes on the CMU mirror: 54 files, 1.9 TB total. Smallest: cluster10 1.0 GB,
cluster26 1.3 GB, cluster45 1.7 GB, cluster3 4.5 GB. Recommended by the dataset authors
for miss-ratio work: cluster52, cluster17, cluster18, cluster24, cluster44, cluster45,
cluster29. cluster52 is 87 GB.

## 2. Tiers actually used

| Tier | Content | Size | Purpose | Gates |
|---|---|---|---|---|
| T0 | `samples/2020Mar/cluster001..054` from the GitHub repo | 3.8 GB clone, 11 to 94 MB per file | parser, unit tests, cross-cluster shape checks | `make test` |
| T1 | cluster45, cluster26, cluster10 (full shards) | 4.0 GB compressed | model development, hazard calibration | M1, M2 |
| T2 | cluster52 full shard | 87 GB compressed | the reported curves | M3, M4 |

Disk: 754 GB free on `/home`. T0 + T1 + T2 fits with room for intermediates, but
intermediates are deleted per stage.

## 3. Acquisition

```bash
make data-samples     # T0
make data-t1          # T1
make data-t2          # T2, resumable, hours
```

`scripts/fetch_traces.sh` is the only supported entry point. It is resumable
(`curl -C -`), retries five times, verifies every file with `zstd -t`, appends a
sha256 to `data/raw/traces/CHECKSUMS.sha256`, and refuses to start when free space
drops below 5 GB.

Throughput *measured* from this machine: about 2 MB/s per connection, and parallel
range requests on the same host did not exceed that in a 180 s probe. Budget
accordingly: T2 takes many hours.

To inspect a shard without decompressing it to disk:

```bash
zstd -dc data/raw/traces/cluster45.sort.zst | head -n 1000000 > data/interim/c45.head.csv
```

## 4. Trace format

Plain CSV, one request per row, no header:

| Position | Column | Type | Meaning |
|---|---|---|---|
| 0 | `timestamp` | int64 | seconds since the start of the trace |
| 1 | `key` | string | anonymized key; namespaces preserved, delimiter varies by workload |
| 2 | `key_size` | int32 | bytes |
| 3 | `value_size` | int32 | bytes |
| 4 | `client_id` | int32 | anonymized frontend service |
| 5 | `operation` | enum | get, gets, set, add, replace, cas, append, prepend, delete, incr, decr |
| 6 | `ttl` | int64 | seconds, 0 when the request is not a write |

Object size used by the cache model is `key_size + value_size`. Keeping both raw
columns is what allows the byte-weighted curve later.

## 5. Parsing contract

`src/mrc/data/trace.py` is the only reader. It uses pyarrow's CSV reader over a
zstd input stream, then dictionary-encodes keys once over the whole column. Two
things matter and both were bugs once:

1. **Do not** ask pyarrow for `pa.dictionary(...)` column types at parse time. The
   reader processes CSV in blocks and each block gets its own dictionary, so the
   index space is per-block and the resulting key ids are garbage. Read the key as
   `pa.string()` and call `.dictionary_encode()` on the combined column.
2. Per-request size must be read per row, never indexed by key id.

Memory *measured*: 1e6 requests of cluster45 occupy about 26 MB of key codes,
timestamps and sizes plus the key dictionary (173,874 distinct keys per 200k rows
in cluster45, that ratio is workload specific).

Frozen artifact schema (`data/interim/*.npz`, one file per shard and tier):

| Array | dtype | Meaning |
|---|---|---|
| `R` | int64 | distinct-key reuse distance, -1 for compulsory |
| `B` | int64 | distinct-byte reuse distance, -1 for compulsory |
| `gap` | int64 | requests since previous access, -1 for compulsory |
| `freq` | int64 | 1-based occurrence index of the key |
| `censor` | bool | the key is never requested again in the observed window |
| `compulsory` | bool | first access of the key |
| `key`, `obj_size`, `op`, `ts`, `client`, `ttl` | as read | request attributes |
| `n_keys` | int64 | size of the key vocabulary |

Change this schema only by a commit that updates `docs/DATA.md`, `src/mrc/cli.py`
and every consumer in the same commit.

## 6. Reuse distance

For request `i` to key `k_i` with previous access at `p(i)`, `R_i` is the number
of **distinct** keys requested strictly between `p(i)` and `i`; `B_i` is the sum
of their sizes. Both are computed in one pass with two Fenwick trees over
last-occurrence positions: mark 1 (and the size) at the most recent position of
each key, then the distinct count in `(p, i)` is the prefix difference. `O(n log n)`,
numba, `nogil`.

Gate: bit-identical to `mrc/data/reference.py::naive_reuse` on the T0 fixtures
(`tests/test_smoke.py`). The object-count curve is additionally bit-identical to
an exact LRU replay; the byte curve is close but not identical, because objects
larger than the remaining space break the "still resident" argument. Measured on
200k cluster45 requests: byte curve vs replay within `6.5e-5`; on small synthetic
traces the gap reaches a few percent, which is why ground truth is simulation, not
the curve.

## 7. Censoring at the data layer

A request is *right-censored* when its key is never requested again before the log
ends. Those rows carry `censor=True` and must not be used as positive labels: the
event they would encode has not been observed. Two rules:

1. The last occurrence of every key in the window is censored by construction.
2. Drop the final `W` requests of a shard as a washout window when fitting, with
   `W` at least the 99th percentile of the reuse distance, otherwise the tail is
   systematically mislabelled as censored.

## 8. Live and external data

Only when a fresh workload is genuinely needed. Bulk corpora always come from the
canonical mirrors above; scraping is for the long tail (a tool's documentation, a
published table, an ecosystem benchmark).

Tooling: **Scrapling** (Firecrawl is disabled on this machine). Rules:

- one request per second per host, robots.txt respected;
- cache every response under `data/raw/scraped/<host>/<sha256>.html`;
- record provenance in `data/raw/scraped/manifest.json`: `url`, `fetched_at` (UTC),
  `sha256`, `selector`, `notes`;
- check the source licence before any redistribution; a scraped table that ends up
  in the report needs a citation, not just a file.

```python
from scrapling.fetchers import FetcherSession

with FetcherSession(impersonate="chrome") as s:
    page = s.get("https://example.org/benchmarks", timeout=30)
rows = page.css("table.results tr")
```

## 9. Known blocker: libcachesim 0.3.5 pybind layer

Observed 2026-11-10, python 3.13, `libcachesim==0.3.5` wheel (manylinux):

- `LRU.process_trace(reader)` returns `(1.0, 1.0)` for every capacity, including on a
  synthetic trace with 200 keys where the true miss ratio at capacity 200 is 0.005.
- Manual replay through the same object double counts: after one `insert` the cache
  reports `get_n_obj() == 2`, and `get(req)` returns `False` for a key just inserted.

```python
import libcachesim as lc
r = lc.TraceReader("syn.txt", lc.TraceType.PLAIN_TXT_TRACE)
c = lc.LRU(cache_size=10); req = r.read_one_req()
c.insert(req); print(c.get_n_obj())   # 2, expected 1
print(c.get(req))                    # False, expected True
```

Consequence: the pybind cache cannot serve as ground truth. Ground truth is our own
numba LRU replay, cross-checked against the naive `OrderedDict` reference. The
`libcachesim` dependency stays installed for its C trace reader and for the day the
bug is fixed; `make test` must be extended with a version check before it is trusted
again.

## 10. Sources

- Dataset repository and format: <https://github.com/twitter/cache-trace> (fetched 2026-11-10).
- Mirror index and file sizes: <https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/> (fetched 2026-11-10).
- Per-cluster statistics: `stat/2020Mar.md` in the repository.
- Paper behind the traces: Yang, Yue, Vinayak, OSDI 2020.
- Licence: CC-BY 4.0.
