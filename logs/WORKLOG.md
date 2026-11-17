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

## 2026-11-10T22:27Z · plan · mrc-plan expands the master plan into an eight-phase set

Commit: none for this work (planning documents are uncommitted at the time of writing).

Produced, all under planner ownership:
- `.hermes/plans/mrc/00-index.md` and eight phase plans, A through H.
- `docs/GATES.md` (one row per gate), `docs/CARDS.md` (29 card specs), `docs/OPEN_QUESTIONS.md`.
- `logs/DECISIONS.md` (master-plan audit plus the seven required decision blocks).

Audit, sections 1.1 to 1.3 checked against HEAD `3d27271`:
- Reproduced exactly: `10 passed`, module line counts, cluster sizes, `cluster45` sha1, the 13.06 / 8.80 /
  4.26 / 86.94 % rates, and the 21,174,922 / 745,784 / 3.73 / 28.39x row counts from `data/interim/s45.npz`.
- Does not reproduce: the four `R` percentile rows in master §1.2 (5,921 / 60,358 / 129,739 / 162,099
  measured; 2,812 / 34,379 / 90,134 / 131,873 claimed). No downstream number depends on them.
- 26 defects logged, M-1 to M-26. The load-bearing ones: `N_BINS`/`N_CAPACITIES` are requested, not
  realised (31 bins, 22 capacities); the `CAP_QUANTILE` = `b_{M-1} - 1` reading is self-contradictory and
  was resolved to `b_{M-1} = edges[-2]` = 837; labels carry int key codes so two of the twelve features
  would have been constant; A1's cap test, A2's hand table, D5's and E3's monotonicity assertions cannot
  pass as written; the baselines and `run_one` both violated the train/test protocol; `libcachesim`'s
  constructor takes `obj_size` first, so the master's positional call raises.

Worked: the frozen constants, the DAG, and the exact-first framing needed no change. Failed nothing that
blocks a phase; every discrepancy has a resolution recorded rather than a silent correction.

## 2026-11-11T10:15Z · contract · rtilde ceiling raised from 1023 to 2^21-1

Changed: `CAP_QUANTILE` in `src/mrc/config.py`, 1023 -> 2_097_151. Evidence: 5M-request prefixes,
`compute_reuse(..., cap=10**9)` on cluster45/26/10, reuse distance percentiles p50/p90/p99/p99.9 =
4/208853/1008186/1725692, 472/7938/17709/51990, 1/4/8/14. The old ceiling put 90 % of cluster45 and
about 99 % of cluster26 mass in the overflow bin, leaving the hazard model no resolution.
Consequence: `rtilde` now spans the observable range; grouped risk-set rows stay bounded by
N_BINS+1 per request, so the row budget is unchanged. Stale artefacts: none existed.

## 2026-11-11T10:55Z · method · the hazard target is the backward reuse distance

Measured on cluster45 (5M prefix, cached labels): a hazard fitted on the forward event
(`delta = eligible and next access inside the window`) reaches test MAE 0.170 with bias +0.170,
because 43 % of requests are cold and the forward-event distribution is not the backward distribution
the MRC curve is a survival function of. The same model, same features, fitted on the backward
distance (`rtilde = min(R, cap)`, event = `R <= cap`, only 38 of 2.84 M eligible requests censored at
the ceiling) reaches test MAE 0.005. The forward formulation is therefore not used by the pipeline.
`compute_reuse` still returns `gamma`, `delta` and `censor` so the ablation and the report can quantify
the difference; nothing on the critical path consumes them.

Consequence: MRC(C) = p_c + (1 - p_c) * E[S_R(bin(C-1) | x)] with S_R estimated from the conditional
distribution of the observed backward reuse distance. The only censoring is at the observation
ceiling. The 1M sample confirms the ordering: GBDT test MAE 0.0022 against Kaplan-Meier 0.0183.

## 2026-11-11T11:40Z · method · capacity grid aligned to bin upper edges

The reported grid was log-spaced independently of the hazard bins. A binned hazard is a step
function, so evaluating it inside a wide bin charges the model for the bin width. Measured on
cluster26 at C=5999 (inside a bin spanning the trace's reuse cliff): truth 0.473, prediction 0.327,
max error 0.147. Re-evaluating the same fitted booster on the bin upper edges gives test MAE 0.0055
and max 0.0206; cluster45 improves from 0.0046 to 0.0003, cluster10 is unchanged at 0.0077.
The grid is now `capacity_grid_from_edges(edges, 24)`, edges[1:-2], which keeps every capacity below
the overflow edge and makes `S_b` and `P(R >= e_{b+1})` the same event.

## 2026-11-17T05:52Z · prereg · cluster52 as the fourth-trace falsification test

Commit: working tree (uncommitted at the time of writing).

Pre-registered before any cluster52 model result existed. Design numbers only, from
`data/processed/cluster52/labels.manifest.json` (5,000,000 requests, first prefix of
the 93.09 GB shard): 932,119 distinct keys, eligible 4,067,881, p_compulsory 0.1864,
so the hit-ratio ceiling is 0.814.

The four existing traces split into two regimes: cluster45 has a long reuse tail
(R p99 1.01M) where the marginal curve must fail, cluster26 has dense reuse
(R p99 17.7k) where it nearly succeeds. cluster52 has 5.4 requests per key, i.e.
dense reuse, so it belongs to the cluster26 regime.

Predictions, stated before the fit:
1. Ordering holds: hazard MAE < Kaplan-Meier MAE < hit-count MAE on cluster52.
2. Margin is cluster26-like, not cluster45-like: hazard/KM MAE ratio under 3, against
   1.95 on cluster26 and 74 on cluster45. A ratio above 10 falsifies it.
3. Provisioning ratio for the hazard stays inside [0.9, 1.1] at 60, 80 and 95 % of
   the ceiling. Outside that band at any target is the falsifier named in the report.
4. Kaplan-Meier will be close at the 60 % target and worse at 95 %, because a short
   tail makes the level shift matter only high on the curve.

Protocol is identical to the other three traces: 5M-request prefix, 33 bins from the
observed p99.9, capacity grid on bin upper edges, 70/85/100 contiguous blocks, one
seed. Nothing about the model, grid, or features is tuned on cluster52.

## 2026-11-17T07:05Z · result · cluster52 scored: the ordering holds, my regime prediction did not

Commit: working tree (this entry's commit is the one that follows).

Cluster52 scored with the frozen protocol, no retuning: 5M-request prefix, 33 bins,
24-point grid, 70/85/100 blocks, seed 7, `MRC_THREADS=4`. Test-block MAE hazard
0.00027, Kaplan-Meier 0.02481, hit-count 0.11342 (`artifacts/tables/cluster52_metrics.json`).

Scored against the four predictions in the 05:52Z entry:

| # | Prediction | Outcome |
|---|---|---|
| 1 | hazard < KM < hit-count | holds |
| 2 | hazard/KM MAE ratio under 3, falsified above 10 | **falsified: 90.9** |
| 3 | hazard provisioning ratio inside [0.9, 1.1] | holds: 1.005, 1.004, 1.009 |
| 4 | KM close at the 60% target, worse at 95% | fails: 0.657, 0.563, 0.522, wrong end of the curve |

Why prediction 2 missed: I used requests-per-key (5.4) as a proxy for reuse density.
The measured statistic is R p99 = 488,644 against 932,119 keys, so the tail reaches
52% of the key space, statistically indistinguishable from cluster45's 47%. Requests
per key measures how often a key recurs, not how far into the key space a single
reuse reaches; only the second determines whether a marginal curve can work. The
proxy was the error, not the model.

Prediction 3 is the one that matters for the report, and it held at every target on
every trace, including the held-out one.

Also this session:

- **`NUM_THREADS` was hardcoded in four places** (`pipeline.DEFAULT_PARAMS`, two in
  `cli.py`, `eval/baselines.py`) despite `config.py` claiming to be the single source.
  Consequence: `MRC_THREADS` never reached LightGBM, and 12 threads on this box ran
  cluster10 in 15+ minutes against 54 s at 4. Measured, 3 rounds on 4.5M rows: 12
  threads 12.1 s, 8 threads 7.5 s, 4 threads 4.4 s. All four sites now read
  `config.NUM_THREADS`; the default is 8 and a background run sets 4.
- **Determinism is thread-count-conditional.** A same-thread rerun of cluster10 is
  bit-identical (0 differing fields). Changing the count moves the last bits:
  cluster10 gbdt MAE 0.005170615270493756 -> 0.005170615270493765, 2e-15 relative,
  below print precision. `deterministic=True` fixes the order for a given count, not
  across counts. The thread count is recorded in every artifact's `meta.params`.
- **Four-panel figures** now use a 2x2 grid (`_panels`); four abreast at 7.0 in left
  1.7 in per panel and the legend collided with the tick labels. `plot_importance`
  refuses a cluster without a SHAP artifact instead of drawing a blank panel.
- **Tables and figures default to `reported_clusters()`** (glob over the metrics
  artifacts), so a new trace appears in every table without editing a list.
- Other lanes' uncommitted work reviewed and kept: `read_trace` streams with an early
  stop at `limit` (93 GB shard: 29 s, 4.3 GB peak, bit-identical labels on cluster45),
  `predict_survival` chunks its grid, `pyproject.toml` drops nine unused dependencies
  (verified: no import of any of them in `src/`, `scripts/`, `tests/`), `configs/`
  removed (its constants live in `src/mrc/config.py`), `docs/GATES.md` paths updated
  to `artifacts/truth/`.
- Body is back to 10 pages after moving the residual figure and the noise table to
  the appendix; 15 pages total, 0 LaTeX errors, 0 overfull boxes, 88 tests pass.

## 2026-11-17T07:22Z · result · the four-trace ablation, and it reproduces exactly

Commit: this entry's commit.

`scripts/method_ablation.py` now defaults to `reported_clusters()`, so cluster52
entered the nine-estimator table. Re-running all four also served as an exact-first
gate: the three original traces reproduced their committed rows with **0 differing
fields**, on 4 threads against the original run's 12.

The number worth keeping: removing the timing family multiplies error by **234 on
the held-out cluster52**, against 51 on cluster45 and 5.4 on cluster10. On cluster26
it changes nothing (1.05). The mechanism claim now has an out-of-sample leg.

Table header was hardcoded to three columns while the cells looped over `results`,
so a fourth cluster would have produced five columns under a three-column spec;
the header and the column count are now derived. `--reuse` rebuilds the table from
the cached JSON so a formatting fix costs no refit.

## 2026-11-17T07:45Z · cleanup · retire the pre-registered calibration path

Commit: this entry's commit.

The appendix says calibration (Brier/IBS) was pre-registered, never reported, and
that the figure code was removed. The sentence was committed before the deletion
happened, so for one commit the report described a state the tree did not have.
`src/mrc/plots/reliability.py` and `tests/plots/test_reliability.py` are now gone;
the only references anywhere were the module's own definitions and docs/CARDS.md
H2 plus docs/GATES.md G-H2, which are removed with it. The IBS *metric* stays in
`mrc.eval.metrics` and its 17 tests still pass; only the reliability diagram is
retired.

Three audit rows were also stale in ways unrelated to that deletion:
`G-G3-ablate` ran `pytest tests/test_pipeline.py`, which tests the grid and target,
not the ablation; it now runs `scripts/method_ablation.py` with the reproduction
requirement. `G-H1-mrc` expected `2 passed` from `tests/plots` (3 after the
deletion), `G-H4-report` and `G-H4-slides` carried 13 and 18 pages (15 and 19).
CARDS E2 claimed `6 passed` for the metrics suite (17).

Verified after: report 15 pages, body 10, References p11, 0 errors, 0 overfull,
0 undefined references; slides 19 pages, 0 errors; wiki builds `--strict` in 0.4 s;
ruff clean on src/tests/scripts; mypy clean on 31 files; full suite passes.
