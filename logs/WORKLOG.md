## 2026-11-10T19:25:41Z · data · cluster52 span, op mix and TTL semantics resolved

**Tasked:** resolve whether the kqps figure and the row count agree, and pin the TTL
column's meaning before feature design.

**Actual:**
- Last row starts at t = 554,161 s = 6.41 days. 13,426,570,607 / 554,161 = 24,229 req/s
  against the published 24.25 kqps, a 0.09% gap. The row count and the stat table agree,
  so neither is a misread.
- Op mix on the first 2M rows: get 0.892, add 0.045, gets 0.029, cas 0.026, delete 0.008,
  set ~0.
- `ttl` is 0 on every get/gets/delete and non-zero on every add/cas/set: 141,421 writes,
  141,421 non-zero TTLs, exact complements. 0 means "read carried no TTL", not missing.
- Non-zero TTLs sit in three buckets (1 d, 12 h, 14 d) whose conditional shares
  0.64 / 0.07 / 0.28 reproduce the published `1d:0.65, 14d:0.27, 12h:0.07`. Recovering a
  published statistic from raw bytes is the parser's independent correctness check.
- Consequence recorded in docs: raw `log(ttl)` is a trap because the 0 spike encodes the
  operation; use a TTL category plus op column.

**Failed / limitations:**
- Op/TTL shares come from a time-prefix, not a random sample. Anything that needs exact
  whole-trace shares must be computed by a streaming pass, budgeted at about 10 minutes.

## 2026-11-10T19:14:11Z · data · cluster52 request count

**Tasked:** exact row count for T2 bookkeeping.

**Actual:**
- `zstd -dc | wc -l` = 13,426,570,607 requests in 653 GB uncompressed (48.63 bytes/row,
  10 min 13 s).
- Cross-check against the published stat table: 13.43e9 requests at the documented
  24.25 kqps implies a 6.4-day trace. Timestamp span pending (`tail -1` running); if the
  span is not about 6.4 days then either the kqps figure or the row count is being
  misread, and that has to be resolved before any window is defined.

**Failed / limitations:**
- Counting rows costs a full 10-minute decompress; there is no cheap index. Cache the
  number in docs rather than recomputing per run.

## 2026-11-10T19:03:56Z · data · cluster52 downloaded and verified: T0 + T1 + T2 complete

**Tasked:** finish T2 acquisition with integrity proof.

**Actual:**
- 8 parallel ranges, 66 minutes wall clock, one `Recv failure: Connection reset by peer` on
  a range that resumed from its own byte count (no restart).
- Part boundaries were exact to the byte: seven parts at 11,635,878,608 and the last at
  11,635,878,603, total 93,087,028,859 bytes = the server's Content-Length.
- `zstd -t`: OK, 652,984,944,787 bytes uncompressed, 615 s.
- sha1 `d709465e236199d7d3fddc1681cb828cdb85e72a` equals the mirror's published line for
  `cluster52.sort.zst`. sha256 `d65867a6f2bad96ef1fca9663a7189ee1dcced2814346250261275b51c7fb40e`
  recorded in `data/raw/traces/CHECKSUMS.sha256`.
- Parser sanity: 7 fields per row (time, key, key size, value size, client id, op, TTL flag),
  timestamps ascending from 0, 100,000 lines counted on a slice.
- Every cardinality claim in docs/DATA.md now matches disk: T1 = 4.33 GB, T2 = 93.09 GB
  compressed / 652.98 GB uncompressed.

**Failed / limitations:**
- `zstd -t` on 93 GB costs 10 minutes and `sha256sum` another 4. Verification of T2 is not
  free; run it once and trust the recorded digests afterwards.
- Exact row count for cluster52 is still pending (background `zstd -dc | wc -l`).

## 2026-11-10T17:33:49Z · data · root cause of the slow download: degraded Wi-Fi, not the mirror

**Tasked:** explain cluster52, find a faster source.

**Actual:**
- Wi-Fi was associated at 54 Mbit/s legacy (802.11a, HE-MCS absent) with -37 dBm signal, on an
  AP advertising 1170 Mbit/s. `nmcli connection down/up` re-associated at 286.7 Mbit/s
  (802.11ax, HE-MCS 11, NSS 2).
- Before the fix, every remote host measured about 1 MB/s: Cloudflare CDN 1.05, HF 0.94,
  CMU 0.71. After the fix, 4 parallel CDN streams gave 13.4 MB/s.
- Re-ran the parallel-range test on a healthy link: 8 ranges from the CMU mirror sustain
  21.9 MB/s aggregate on cluster52. The mirror limits per connection (2 to 3 MB/s), not per
  client. ETA went from 11 h to 1.0 h.
- Retracted the earlier "mirror throttles per client" conclusion: it was measured entirely on
  the crippled link, which is a textbook measurement artifact. docs/DATA.md now carries both
  link states in the table so the mistake is visible, not overwritten.
- `fetch_parallel.sh` promoted to the default T2 path; added a guard that refuses part files
  whose size exceeds their range (the failure mode when NPARTS changes between runs).
- cluster52 download seeded from the 9.6 GB already fetched as part_0, no bytes re-downloaded.

**Failed / limitations:**
- One wasted 25-minute attempt (8 parallel over the bad link) plus 165 MB of duplicate ranges.
- No alternative source found worth using: no HuggingFace mirror exists, SNIA needs a
  registration form, Baidu pan needs an account. The CMU mirror is fine once the link is.

## 2026-11-10T17:17:10Z · data · parallel fetch measured slower, T1 complete

**Tasked:** check whether the parallel downloader beats a single stream before spending 11 h.

**Actual:**
- cluster45, cluster26, cluster10 downloaded and verified (zstd + upstream sha1 + sha256):
  1.87 GB, 1.38 GB, 1.08 GB.
- cluster52 is 93.09 GB, not 87 GB (the 87 GB figure in the docs was decimal vs binary confusion).
- Measured: one curl stream 2.03 to 2.32 MB/s; 8 parallel ranges 1.44 MB/s aggregate;
  4 parallel ranges 1.07 MB/s. The mirror throttles per client. `fetch_parallel.sh` kept but
  marked measured-slower and not wired into `make`.
- Resume of cluster52 from 7.5 GB continues at 2.32 MB/s, ETA about 10 h.
- No HuggingFace mirror of this corpus exists (datasets API search: no match for
  cache-trace / twemcache / cluster52), so the HF CLI is not applicable; the corpus is only
  on the CMU mirror, SNIA and Baidu.

**Failed / limitations:**
- First parallel attempt wasted about 4 minutes and 165 MB of duplicate ranges, deleted.
- Nothing else failed. Download is the only long pole left.

## 2026-11-10T15:48:50Z · data · retraction and integrity upgrade

**Tasked:** audit the delegation batch's data claims, harden acquisition.

**Actual:**
- libcachesim 0.3.5 re-probed: `insert`/`get` work correctly (`get_n_obj()==1`, `get()` True),
  but `process_trace` returns `(0.210032, 0.210032)` at capacities 10, 100, 1000, 10000 and
  100000 on the same 1M-request trace, where our simulation gives 0.9967 down to 0.8707.
- Retracted the "insert double counts" report: it does not reproduce; it was my probe misusing
  the API. docs/DATA.md section 9 rewritten with the narrower blocker plus the retraction.
- Found a silent data-integrity trap: `pyarrow.csv.read_csv` defaults to `has_header=True`, so on
  these headerless shards it eats the first request and names columns after its values
  (999,999 vs 1,000,000 rows on the cluster045 sample). Documented as parser pitfall 3.
- End-to-end ingest measured: 0.40 s per 1M requests (86 MB/s, 2.48 M rows/s), peak RSS 387 MB;
  reuse distance 2.58 M req/s. Bare `pyarrow.csv` parse: 355 MB/s.
- Acquisition now verifies the mirror's published sha1 (`data/raw/traces/UPSTREAM.sha1`) on top of
  zstd integrity, before recording sha256. Tamper detection tested on a synthetic file.
- Running download restarted on the new script; cluster45 resumed from 1.62 GB of 1.74 GB.

**Failed / limitations:**
- Patching a script that bash was still reading killed the in-flight job (bash re-reads by byte
  offset). Rule learned: kill the job first, then edit, then relaunch. Resume made it cheap.
- `process_trace` remains unusable, so libcachesim stays out of the ground-truth path.

# WORKLOG

Append-only. Newest entry on top. UTC timestamps.

## 2026-11-10T15:39:17Z · scaffold · repo initialized

**Tasked:** stand up the project from an empty `uv init` directory: environment, data,
statistical skeleton, experiment pipeline, telemetry, docs, wiki, report template, agents.

**Actual:**
- uv env on python 3.13 with lightgbm 4.7, numpy 2.5, pyarrow 26, polars 2.1, duckdb 2.0,
  numba 0.68, libcachesim 0.3.5, optuna 5.0, mlflow 3.18, matplotlib 3.11, pytest, ruff.
- `src/mrc/`: trace reader, Fenwick reuse distance (numba), naive references, MRC curves,
  discrete hazard model, Optuna wrapper, frozen plot style, CLI (`inspect|reuse|mrc|fit|plots`).
- Gates pass: reuse distance bit-identical to the naive reference on synthetic and real
  traces; object-count MRC bit-identical to an exact LRU replay at 4 capacities.
- Measured on 200k cluster45 requests: KM baseline MAE 0.0018 vs empirical MRC,
  GBDT MAE 0.0057 (5 features, untuned).
- Data: T0 samples cloned (3.8 GB), T1 cluster45 in flight, T2 cluster52 queued.
- Report (13 pages: title, TOC, 8 body pages, references, 2 appendix pages), slides,
  wiki (27 pages, mkdocs strict build passes), five docs, kanban board `mrc` with five cards.

**Failed / limitations:**
- libcachesim 0.3.5 pybind cache is broken (`process_trace` returns 1.0 always, `insert`
  double counts). Ground truth moved to our own replay. Documented in docs/DATA.md.
- `AGENTS.md` write is gated by an approval prompt that timed out twice. Rules staged in
  `docs/AGENTS_RULES.md`; promote to `AGENTS.md` once approved.
- Byte-capacity stack distance is exact in the integer case and approximate in bytes
  (measured 6.5e-5 on real data, percent-level on tiny synthetic traces). Ground truth for
  reported numbers is simulation.
- 87 GB T2 shard not finished; download resumed in background.

**Next:** T1 artifacts -> hazard features -> MLflow runs -> Optuna -> figures -> report tables.
## 2026-11-10T21:23Z · audit · setup complete, science not started

- Promoted `AGENTS.md` from the staged `docs/AGENTS_RULES.md` copy (commit 752e6c2); the
  approval gate that blocked the filename is gone.
- Fixed a dangling dependency: `t_8cb919d5` (M2 figures) listed archived `t_5cd843df` as a
  parent, which would have pinned it in todo forever. Unlinked; it now waits only on
  `t_e7bb2b4a`.
- Audit verdict: repo scaffolding is complete and green (`pytest` 10/10, tree clean, 11
  commits). No scientific artifact exists yet: `artifacts/mlflow` and `artifacts/optuna`
  empty, `artifacts/tables` empty, `data/processed` empty, one figure and it is from the
  44 MB GitHub sample, and every table/figure in report and slides is a placeholder.
