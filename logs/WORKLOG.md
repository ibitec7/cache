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