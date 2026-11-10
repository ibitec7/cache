# Risks

Scored before any experiment runs. Likelihood and impact are qualitative
(low, medium, high). Every mitigation names the artifact or command that proves
it worked. No probability numbers are invented.

## Risk table

| # | Risk | Likelihood | Impact | Early warning signal | Mitigation |
|---|---|---|---|---|---|
| R1 | 87 GB T2 download plus decode expansion exhausts the 754 GB free volume | medium | high | `make disk-check` fails; free space drops below the configured floor during decode | Stream decode and delete the `.zst` after the parquet is validated; keep T2 on a separate volume if one is available; hard floor in `Makefile` aborts before a stage starts; T1 only until M3 passes |
| R2 | libcachesim 0.3.5 pybind is defective: `process_trace` returns miss ratio 1.0 regardless of capacity, `insert()` double-counts objects | high (observed) | high | adapter returns exactly 1.0 for every capacity; object count exceeds distinct keys in the trace | Do not use libcachesim as ground truth. Primary truth is our exact LRU simulator plus numba stack distance. Adapter runs only as a third opinion, must pass two defect probes, and any output it produces is labelled untrusted in tables |
| R3 | Pure Python simulation runs at ~200k requests/s, so a 1e9-request shard needs hours per pass and cannot be swept over capacities | high | high | timed probe shows < 1M req/s; a single MRC sweep exceeds the stage timeout | numba `njit` LRU stack with a Fenwick tree, target throughput TBD but must clear one shard pass inside the stage budget; subsample requests for tuning, full pass only for final curves; parallelise across clusters, not within a shard |
| R4 | Reuse-distance computation cost: order-statistics over hundreds of millions of distinct keys, memory grows with distinct objects | high | high | RSS climbs with distinct keys; first pass on cluster52 exceeds memory | Compressed object ids, Fenwick tree over ids, chunked external processing, per-shard checkpoints; measure on T1 and extrapolate before touching T2; if T2 cannot fit, report T1 curves and state the limit |
| R5 | Right-censoring bias: the log ends mid-stream, so long reuses are unobserved; naive dropping under-estimates the curve | high | high | estimated miss ratio at large capacity falls below the LRU simulator on data with known tail | Censored likelihood (zero rows up to the last fully observed bin); treat the final window as censored, not as non-reuse; quantify bias on T1 by hiding the last 10% of the trace and comparing against full-trace truth |
| R6 | Hazard binning sensitivity: bin count and log spacing shift the knee and the calibration | medium | medium | metric table changes materially between adjacent bin counts | Pre-register a small grid of bin counts and spacings in `configs/`; report the chosen setting with a sensitivity appendix; isotonic projection so survival stays monotone |
| R7 | Person-period expansion explodes the training table (up to B rows per reference) | high | medium | label row count exceeds 40x request count; LightGBM dataset does not fit in RAM | Cap bins at risk, sample at-risk rows with inverse-probability weights, use LightGBM `weight` column; verify weighted hazard matches the unsampled hazard on a small shard |
| R8 | Trace vintage (Mar 2020) limits generalisation | certain | medium | reviewers ask for 2024+ traffic; namespace mix looks dated | Frame as a method paper with a dated dataset; cross-cluster holdout (train on 3 clusters, test on the 4th) as the generalisation test; state the caveat where the result is claimed, not only in limitations |
| R9 | Times New Roman is absent on Linux; the paper must still match the course template | certain (known) | low | LaTeX font warning; `pdffonts` shows a substitute | Use Nimbus Roman (metric-identical) through `style.py` and the LaTeX font package; verify with `pdffonts`; do not install msttcorefonts and do not silently substitute a different metric |
| R10 | Page limit 8-10 pages is exceeded or under-filled | medium | medium | `make paper` reports page count outside [8,10] | Page budget fixed at M0 (6 pages body, 2 pages figures/tables, references); move derivations to appendix; compile from M0 onward, not at the end |
| R11 | MLflow or Optuna state is lost or runs are not reproducible | medium | medium | two runs with the same config hash give different metrics; tracking URI unreachable | One run per (trace, config) with config hash as tag; seeds fixed and logged; model artifact plus params stored; `make eval-t1` refuses a run whose hash is unknown |
| R12 | Namespace parsing is inconsistent, so a namespace feature leaks label information or collapses to noise | medium | medium | one namespace covers > 90% of requests; namespace split changes between clusters | Normalise by delimiter rules documented in `features.py`; treat unparsed keys as an explicit `unknown` level; ablation with and without namespace in the metric table |
| R13 | TTL and explicit deletes break the LRU-only assumption behind the MRC identity | high | medium | model systematically over-predicts hits for short-TTL namespaces | Include TTL as a feature; measure the gap against the LRU simulator; report the TTL-heavy subset separately rather than correcting it away |

## Detail on the two blocking risks

### R2 libcachesim 0.3.5

Two defects were observed before this plan was written: `process_trace` returns
a miss ratio of 1.0 for every capacity, and `insert()` double-counts objects.
Both are in the pybind layer, so neither can be patched from Python. The project
therefore treats libcachesim as unusable for MRC ground truth.

Consequences for the design:

- Ground truth is our own exact LRU simulator, cross-checked against a brute
  force oracle on 1e5 requests.
- The adapter stays in the tree so the claim is reproducible: two probe tests
  assert the defects are still present, and the adapter raises when they are
  detected. If a future version fixes them, the probes fail and the card
  reopens.
- No table in the paper reports a libcachesim number as truth. If a number is
  shown at all, the caption says which implementation produced it.

### R3 simulation throughput

At the measured ~200k requests/s, one 1e9-request shard is about 1.4 hours per
pass in pure Python, before any capacity sweep. A sweep of 20 capacities would
be a day per shard, which rules out iteration.

Plan:

1. numba `njit` the LRU stack with a Fenwick tree over compressed object ids,
   single pass, O(log M) per request.
2. Benchmark on 1e7 requests, extrapolate, and record the number on the card.
3. If the target is missed, fall back to a sampled request stream (every k-th
   request) for tuning and keep the exact pass only for the final curve.
4. Parallelise across clusters with one process per shard, never across a single
   shard, because the stack is sequential.

The fallback changes what is claimed, so it is a plan decision recorded at M2,
not a silent optimisation at M4.

## Assumptions that would invalidate the plan

| Assumption | If false |
|---|---|
| T2 fits on disk alongside T1 after streaming decode | report T1 curves and T2 on a larger volume later; M4 scope shrinks to T1 |
| Exact stack distance is computable on T2 in bounded memory | use sampled windows and state the sampling regime |
| Censoring is confined to the trace tail, not to per-object truncation | the censored likelihood still applies, but the at-risk set must be rebuilt per object |
| TTL expiry can be treated as a feature rather than a competing risk | switch to a competing-risks hazard, which changes the estimator and the M3 gate |
