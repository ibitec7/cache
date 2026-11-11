# Decisions

Append only. One line per decision:

```
UTC date | decision | evidence | reversible?
```

A decision is any choice a future reader would otherwise re-litigate (master section 7.1). Evidence is
either an artefact path plus field, or a command whose output is quoted. A number that was not read
from an artefact or produced by a recorded command does not appear here.

All lines below were recorded in the planning session that began 2026-11-11 local (HKT); the stamped
time is that session's UTC clock reading, `date -u` -> `2026-11-10T22:27Z`.

---

## master-plan audit

Scope: master sections 1.1, 1.2, 1.3 verified against the repo at HEAD `3d27271` ("log: audit of
setup completeness"), working tree clean apart from untracked `.hermes/`.

### Verified, no discrepancy

| Claim (master) | Check | Result |
|---|---|---|
| `pytest -q` -> 10 passed | `env -u PYTHONPATH uv run pytest -q` | `..........` 10 passed |
| module line counts: reuse 114, trace 122, reference 51, eval/mrc 58, hazard 136, search 41, style 64, cli 114 | `wc -l src/mrc/**/*.py` | all match exactly |
| public API of `reuse`, `trace`, `reference`, `mrc`, `hazard`, `search`, `style`, `cli` | `grep -n "^def "`, `sed` on each module | matches the table, including `naive_lru_mrc(keys, sizes, capacity, by_bytes=False)` and `capacity_grid(max_capacity, points=24)` |
| cluster sizes 1.87 / 1.38 / 1.08 / 93.09 GB | `du -sb data/raw/traces/*` | 1872045622 / 1377035307 / 1079558382 / 93087028859 bytes |
| traces sha1-verified against `UPSTREAM.sha1` | `sha1sum data/raw/traces/cluster45.sort.zst` vs `grep cluster45 data/raw/traces/UPSTREAM.sha1` | both `10a0ef3ca35d8644374b00e35caf9d1216c76a19` |
| §1.2 rates: 13.06 % eligible, 8.80 % events, 4.26 % censored-eligible, 86.94 % cold | computed on `data/interim/s45.npz` fields `freq`, `censor`, `R` | 13.06 / 8.80 / 4.26 / 86.94 exactly |
| §1.2 row counts: 21,174,922 per-step, 745,784 grouped, 105.87 and 3.73 per request, 28.39x | computed on `data/interim/s45.npz` with `bin_edges(percentile(rtilde\|eligible, 99.9), 32)`, cap 1023 | 21,174,922 / 745,784 / 105.87 / 3.73 / 28.39 exactly |
| D1 no `gamma`, no `cap` | `src/mrc/data/reuse.py:96-114` | `compute_reuse(keys, sizes, n_keys)`, returns `R,B,gap,freq,censor,compulsory`; no `gamma`, no `delta`, no `rtilde`, no `cap` |
| D2 censored rows expanded to backward distance | `src/mrc/cli.py:77-79`, `src/mrc/models/hazard.py:47-66` | `b = assign_bins(R, edges)` includes censored rows; `event = ~censor & (b >= 0)`; `expand_risk_set` takes `bins` and never reads `censor` |
| D3 row-order split | `src/mrc/models/hazard.py:78` | `cut = int(n * (1 - valid_frac))` over expanded rows, default `valid_frac=0.2` |
| D4 five coarse features | `src/mrc/cli.py:96-101` | `log_obj_size, op, is_write, log_ttl, client_bucket` |
| D5 no truth/eval/tables/figures | `data/processed/` empty, `artifacts/tables/` empty, `cli.py` `plots` prints "no figures yet" | confirmed; only `artifacts/figures/mrc_cluster045_sample.{pdf,png}` exist, from the 44 MB GitHub sample |
| D6 `predict_mrc` clips | `src/mrc/models/hazard.py:104-113` | `kmax`, then `np.clip(bins_c, 0, kmax)`; no raise |

### Discrepancies

| ID | Master claim | Measured | Verdict |
|---|---|---|---|
| M-1 | §1.2: `R` percentiles 50/90/99/99.9 = 2,812 / 34,379 / 90,134 / 131,873 | `data/interim/s45.npz`, `freq > 1`: 5,921 / 60,358 / 129,739 / 162,099 | unreproducible. Also checked `B` and `gap` over eligible, and `R` over all requests (percentile of `R` with `R=-1` in the sample): none match. The four percentile rows are unsourced; nothing downstream depends on them (bins come from p99.9 of `rtilde`, which is 1023 after the cap) |
| M-2 | §2 and §7.2.1: `N_BINS = 32`, "the frozen 24-point grid" | `bin_edges(1023, 32)` returns 32 edges, i.e. 31 bins; `identified_grid(..., 24)` returns 22 capacities on cluster045 because `np.unique` collapses rounded `geomspace` duplicates | `N_BINS` and `N_CAPACITIES` are requested resolutions, not realised counts. B3's own test asserts `n_capacities == len(capacities)`, which forces the code reading. Resolution: realised counts are recorded in every manifest and caption; no new constant |
| M-3 | §2: `CAP_QUANTILE = 1023` is "`b_{M-1} - 1`"; §3 module map calls the last edge the overflow edge; D5 raises at `capacities >= edges[-2]` | edges tail = `[684, 837, 1023]` on cluster045, so `edges[-2] = 837` and `edges[-1] = 1023` | self-contradictory. Adopted reading R2: `b_{M-1} := edges[-2]`, identified capacities satisfy `C <= b_{M-1} - 1 = 836`, the overflow bin is `[837, 1023]` once the D1 off-by-one fix lands, and `CAP_QUANTILE = 1023` is the ceiling on `rtilde`. This is the only reading under which the frozen D5 guard and the frozen D1 docstring are both true |
| M-4 | D1..D6 do not list an `assign_bins` off-by-one | `np.searchsorted([0,1,2,4,8], 8, side="right") - 1 = 4` with `nbins = 4`; index 4 does not exist | real, unlisted. Master fixes it inside D1 step 4 (`np.clip(..., -1, len(edges) - 2)`), so no new task; the reviewer must see the fix in the D1 diff before D5's guard is trusted |
| M-5 | §1.1: "12 commits" | `git log --oneline \| wc -l` = 13 | cosmetic; the cited HEAD is correct |
| M-6 | §2 cache key: "sha256 in a `manifest.json` beside each artefact" | F1 writes `features.npz` with no manifest | contract gap. Resolution: F1 must also write `features.manifest.json` (sha256, `n_rows`, `n_cols`, `FEATURE_VERSION`, feature names). This is an application of the frozen contract, not a new constant |
| M-7 | §7.3 lists ten edges | F1 calls `predict_mrc`, which raises until D5 lands, but no `D5 -> F1` edge is listed; likewise `A1/A2 -> A3`, `B2 -> B3`, `D3 -> D4`, `E2 -> E3`, `G2 -> G3`, `H2/H3 -> H4`, `F1 -> F2`, `F3 -> F4` | the ten edges stay frozen as the cross-phase set. Intra-phase order is encoded in the phase plans, and Phase F's gate table makes D5's test a precondition of F1 |
| M-8 | §7.2.7: `mrc-data` owns Phases A to D | D1 to D5 write `src/mrc/models/**`, which `AGENTS.md` assigns to `mrc-run` | collision. Resolution: `AGENTS.md` file-level ownership wins, D1 to D5 go to `mrc-run`. Unowned new files assigned: `src/mrc/features.py` and `src/mrc/gt/**` to `mrc-data`; `src/mrc/pipeline.py`, `src/mrc/tuning/**`, `src/mrc/interp/**` to `mrc-run`; `src/mrc/report_tables.py` and `scripts/fill_results.py` to `mrc-write`. `src/mrc/cli.py` and `Makefile` are shared and serialised, one stage per commit |
| M-9 | plan gates are `make check`, `make labels`, `make truth`, `make train`, `make tune` | `docs/MILESTONES.md` cites `make schema-check`, `make truth-t1`, `make train-t1`, `make eval-t1`, `make refs-check`, `make disk-check` and `docs/contracts/requests_schema.json`; `grep -c` for those targets in `Makefile` = 0 and `docs/contracts/` does not exist | `MILESTONES.md` is historical. `docs/GATES.md` is authoritative. Not edited here (not a planner-owned file); surfaced to the user |
| M-10 | `docs/RISKS.md` R2: libcachesim "`insert()` double-counts objects" | `docs/DATA.md` §9 retracts exactly that observation ("does not reproduce on this host and was a misuse of the API in the probe") | stale hedge in `RISKS.md`. Flagged, not edited |
| M-11 | §1.1: `STATISTICAL_SKELETON.md` "71 symbols" | `sed -n '609,687p' docs/STATISTICAL_SKELETON.md \| grep -c "^|"` = 73 rows (table header included) | cosmetic |
| M-12 | H4 expects report body 8 to 10 pages, zero placeholders | `pdfinfo report/main.pdf` -> `Pages: 13`; `grep -rn "PLACEHOLDER\|figplaceholder" report/sections \| wc -l` = 5 | H4 starts from 13 pages with 5 placeholders. Deleting placeholders is not sufficient; H4 must also fit the body, which is what its gate measures |
| M-13 | frozen consequence 1: bin-level grouped hazard is canonical | `configs/model/hazard.yaml` first line: "Variant A: one row per request" and `num_threads: 0` | stale config, unused by every task in the plan. `pipeline._params()` is authoritative. Flagged, not edited |
| M-14 | B1 sketch: `lc.Request(int(k), int(k))` | `libcachesim 0.3.5`; `dir(lc.Request)` = `obj_id, obj_size, op, ttl, clock_time, valid, hv, next_access_vtime`; `lc.Request(1, 16)` raises `TypeError: incompatible constructor arguments`, supported signature is `Request(obj_size=1, op=ReqOp, valid=True, obj_id=0, ...)`; `lc.Request(obj_id=5, obj_size=1)` works and a 5-request replay at `cache_size=2` gives 2 hits, 3 misses | B1's positional call is wrong in two ways (first positional is `obj_size`, second is `op`, which rejects an int). B1 must use keywords, `lc.Request(obj_id=int(k), obj_size=1)`. Recorded so B1 does not burn its 10-minute budget on the constructor |
| M-15 | A1 test `test_rtilde_is_capped_at_the_overflow_edge` asserts `rtilde.max() == 4` on `k2 = np.repeat(np.arange(50), 2)` with `cap=4` | command: `compute_reuse(np.repeat(np.arange(50), 2), ones(100, int64), 50)` -> 50 eligible requests, `max R` over eligible = 0, so `min(R, 4)` max = 0 | the assertion cannot hold under any correct implementation: consecutive repeats have `R = 0`. Corrected array `np.concatenate([np.arange(50), np.arange(50)])` gives `max R` = 49 and `min(49, 4)` = 4. Phase A plan carries the corrected test |
| M-16 | A2 hand-derivation: "Eligible requests = 5 (A@2, A@4, B@5, C@7)" and `sorted(R[delta]) == [1, 2, 3]` on `ABACABDC`, with the instruction "if `[1, 2, 3]` fails, the invariant is wrong, not the test" | command: `compute_reuse` on `ABACABDC` -> `R = [-1,-1,1,-1,1,2,-1,3]`, `freq = [1,1,2,1,3,2,1,2]`, `censor = [F,F,F,F,T,T,F,T]`; eligible = 4, delta = 1, `R[delta] = [1]`; with `cap=7`, `rtilde = [-1,-1,1,-1,3,2,-1,0]` | the master's arithmetic is wrong under its own scheme: it lists four indices and calls them five, counts keys from before the previous occurrence (giving A@4 = 2 by including B at index 1), and counts D at index 6 for B at index 5. Separately, skeleton §2.3's authoritative trace is `ABACABCB` over three keys, not the master's substituted four-key string, and skeleton §4.4 censors only the cold requests while master D1 censors every last occurrence. Phase A plan replaces A2's assertions with the measured table and adds the skeleton trace as a second case (eligible 5, delta 2, `R[delta] = [1, 2]`) |
| M-17 | A3's `build_labels` copies `ts, key, obj_size, op, client, ttl` only; C1 assumes `labels["key"].dtype == object` for namespace coding and key size | `data/interim/s45.npz` field `key` is int32 `(200000,)` and there is no `key_vocab` in the file; `src/mrc/data/trace.py` documents `key` as int32 codes into `key_vocab`, `Trace.key_vocab` holds the strings, and `obj_size` is key_size + value_size | on real data C1 takes the integer-key branch, so `ns_code` = 0 and `key_size` = 8.0 for every row: `ns_code` is constant, `log_key_size` is constant, `log_val_size` is wrong by `(len(key) - 8)`, and the `no_namespace` and `no_sizes` ablations are vacuous. Fix: A3 adds `key_ns` (int32, deterministic code of the first `:`-component) and `key_len` (int32, key string length) computed from `Trace.key_vocab`; C1 consumes them when `key` is integer-coded. Additive, pickle-free |
| M-18 | D4's `fit_grouped` derives `n_req = int(rows["eligible_index"].max()) + 1` and splits on `rows["request_index"]`, which indexes the eligible subset | `src/mrc/models/risk_set.py` per master D1: `req = np.repeat(np.arange(b.shape[0]), counts)` over the **kept** array, while `eligible_index = np.flatnonzero(keep)` indexes all requests; master D3's `block_masks(n)` splits all requests | two different boundaries for the same frozen rule ("contiguous blocks 70/15/15 by request index"). Resolution: `fit_grouped` takes `n_requests` and splits original indices, `orig = eligible_index[request_index]`, through `block_masks(n_requests)`. One boundary function for hazard and baselines |
| M-19 | E1 "note for the implementer": "if your implementation disagrees, the implementation is right and the test constant is wrong" | arithmetic check of `test_mre_ignores_capacities_below_the_reporting_floor`: kept points are C=1 (err 0) and C=10 (err 0.1/0.5 = 0.2), mean = 0.1, `mre_n` = 2 | the test constants are correct and the note is wrong. A worker following the note would rewrite a correct test. The plan states the arithmetic and forbids editing the constants |
| M-20 | D5's test: `assert np.all(np.diff(curve) >= -1e-9)` with the comment "non-increasing in capacity" | `predict_mrc` returns `p_comp + (1-p_comp) * S[:, bins_c].mean(0)`; `S` is non-increasing in step by construction, so the curve is non-increasing in capacity and `diff <= 0` | inverted assertion: it fails against a correct implementation. Corrected to `<= 1e-9` in the Phase D plan |
| M-21 | E3's `test_km_curve_is_monotone_and_in_range`: `assert np.all(np.diff(c) >= -1e-12)` | `km_curve` returns `p_comp + (1-p_comp) * S[bins_c]` with `S = np.cumprod(1-h)` non-increasing and `bins_c` non-decreasing in capacity | same inversion. Corrected to `<= 1e-12` in the Phase E plan; the bound check `c >= p_comp - 1e-9` is correct as written |
| M-22 | E3's baselines take `(X, rtilde, delta)` and both fit and evaluate on those arrays | master E3 signatures and bodies: `single_row_ablation` trains on `X[keep]` and predicts on `X`; `hitcount_regression` trains on `Xe` and predicts on `X` | if F1 passes the test block, every baseline trains on the block it is scored on, which is the one protocol violation the plan exists to avoid, and it flatters the baselines. Resolution: optional `eval_X` / `eval_rtilde` / `eval_delta` keywords; the caller fits on train and scores on test |
| M-23 | F1 scores `curve_metrics(curve_gbdt, mrc_true, caps)` with `curve_gbdt` averaged over all `X` and `mrc_true` from the whole prefix | master F1 body: `curve_gbdt = model.predict_mrc(X, caps, edges, p_compulsory=p_comp)`; `mrc_true` from `truth.npz` | the reported number is a whole-prefix number, while master §5's acceptance threshold is "`gbdt.mae` on the test block". Two denominators are mixed. Resolution: F1 computes `mrc_curve(R[mask], caps)` per block and reports `valid` and `test` metrics separately; the test block is scored once |
| M-24 | F1 calls `integrated_brier(np.tile(mrc_true, (X.shape[0], 1)), ...)` | master F1 body: the first argument to `integrated_brier` is the truth curve tiled over requests, i.e. the truth is scored against itself | IBS would measure the data, not the model, and would come out near zero by construction. Resolution: IBS takes per-request predicted hit probabilities, `1 - S_R(bin(C-1) | x)`, from the fitted model |
| M-25 | §2 "nothing else may hardcode these" | master F3's `suggest` writes `"num_threads": 12` as a literal rather than `NUM_THREADS` | cosmetic violation of the frozen constant rule; the plan requires `NUM_THREADS` |
| M-26 | A3 creates `src/mrc/config.py`, but A1's `compute_reuse` already needs `CAP_QUANTILE` as a default argument | master A1 sketch writes `cap: int = 1023` inline (a literal), while §2 forbids a literal outside `config.py`; A1 precedes A3 in the DAG | ordering conflict between two frozen rules. Resolution: `config.py` lands in A1, verbatim, and A3's commit no longer adds it (message kept verbatim, body states the move) |

---

## Decision log

### Primary metric (§7.2.1)

```
2026-11-10T22:27Z | primary metric = validation-block MAE of MRC(C) against Mattson stack-distance ground truth, on the frozen capacity grid, one scalar for Optuna; MAX, bias, MRE, ABC, IBS reported beside it | master §2 row "Primary metric" and §7.2.1; STATISTICAL_SKELETON.md §4.1 defines MRC(C) = (1/N) sum_i [1 - F_R(C-1|x_i)]; src/mrc/eval/mrc.py:41 mrc_error returns exactly mae/max_abs/bias/mre | no, this is the frozen contract |
2026-11-10T22:27Z | validation block, not test block, is the Optuna objective | STATISTICAL_SKELETON.md §6.5 "test block is evaluated once"; master §6 risk row "Optuna overfits the validation block" mitigation | no |
2026-11-10T22:27Z | the grid is "N_CAPACITIES requested, realised count reported"; realised count on cluster045 is 22 | command: identified_grid(n_keys=173874, overflow_edge=837, points=24) -> 22 unique ints, min 1, max 836, strictly increasing (audit M-2) | no, but the realised count is per cluster |
```

### Identification limit (§7.2.2)

```
2026-11-10T22:27Z | b_{M-1} = 837 for cluster045, therefore no capacity at or above 837 is identified and the reportable grid tops at 836 | command: p99.9 of rtilde over eligible = 1023; bin_edges(1023, 32)[-3:] = [684, 837, 1023]; so edges[-2] = 837. Provenance: data/interim/s45.npz, fields R (int64, 200000), freq (int64), censor (bool), 200000 requests, n_keys 173874. This is NOT a label artefact: data/processed/ is empty, so the number is provisional and is re-derived from data/processed/<cluster>/labels.manifest.json at A4 | yes, superseded by A4 and by every other cluster |
2026-11-10T22:27Z | b_{M-1} is read as edges[-2] (last edge below the overflow bin), not edges[-1] | audit M-3; the frozen D5 guard raises at capacities >= edges[-2] | no |
2026-11-10T22:27Z | b_{M-1} for cluster26, cluster10, cluster52 = TO BE MEASURED, produced by task A4 (`mrc labels`) then B3 (`mrc truth`), recorded in each labels.manifest.json | data/processed/ contains no label artefact at audit time | yes |
```

### Cost model (§7.2.3)

```
2026-11-10T22:27Z | 3.73 grouped rows per trace request is the cost basis | measured on data/interim/s45.npz: 745,784 grouped rows / 200,000 requests = 3.7289 (matches master §1.2) | no |
2026-11-10T22:27Z | 5,000,000 requests -> 18,650,000 grouped rows, under MAX_GROUPED_ROWS = 60,000,000, so the LABEL_REQUESTS window stands unchanged | command: 5e6 * 3.73 = 18,650,000; guard constant from src/mrc/config.py (frozen, master §2) | no |
2026-11-10T22:27Z | 20,000,000 requests -> 74,600,000 grouped rows, which BREACHES MAX_GROUPED_ROWS = 60,000,000, so the cluster52 scale run is HALVED to 10,000,000 requests -> 37,300,000 rows (62 % of the guard) | command: 2e7 * 3.73 = 74,600,000 > 6e7; 1e7 * 3.73 = 37,300,000. Master §7.2.3 authorises exactly this ("if either exceeds MAX_GROUPED_ROWS, halve the window and record why") | yes, the halving is per run |
2026-11-10T22:27Z | memory lower bound per grouped row = 66 bytes: X float32 x 12 = 48, bin int16 = 2, y float32 = 4, w float32 = 4, request_index int64 = 8 | command: 4*12+2+4+4+8 = 66, from the dtypes fixed in the master's D1 sketch. 5 M -> 1.23 GB, 10 M -> 2.46 GB, 20 M -> 4.92 GB | no |
2026-11-10T22:27Z | the LightGBM Dataset overhead is excluded from the 66 bytes and is TO BE MEASURED; task F1 records RSS during the sample45 run | no artefact measures it; the master's F1 section has no memory measurement | yes |
```

### Capacity grid freeze and transfer (§7.2.4)

```
2026-11-10T22:27Z | one grid for all clusters, top = min over clusters of (b_{M-1}) - 1, requested resolution N_CAPACITIES = 24 | master §2 row "Capacity grid"; identified_grid applies the same formula to every cluster | no |
2026-11-10T22:27Z | provisional frozen grid on cluster045: 22 points, [1, 2, 3, 4, 6, 8, 10, 14, 19, 25, 33, 45, 60, 80, 108, 145, 194, 259, 348, 466, 624, 836], max G = 836 < b_{M-1} = 837 | command: np.unique(np.round(np.geomspace(1, 836, 24)).astype(int64)) printed in full; monotonicity checked with np.all(np.diff > 0) -> True | yes, recomputed when the first label artefact exists |
2026-11-10T22:27Z | transfer check to cluster26, cluster10, cluster52 = TO BE MEASURED at A4's gate G-LABELS-MULTI; if any cluster has b_{M-1} < 837 the grid is truncated to that cluster's limit and the freeze is re-recorded here | no label artefacts exist for those clusters; data/raw/traces/ has the three shards at the sizes listed in the audit | yes |
```

### Ablation list (§7.2.5), frozen before any tuning run

```
2026-11-10T22:27Z | six feature-group ablations, decided before tuning: all, no_namespace (ns_code), no_sizes (log_obj_size, log_key_size, log_val_size), no_timing (log_ttl, has_ttl, log_gap, log_age), no_popularity (log_freq), no_client (client_bucket) | master G3 GROUPS dict, verbatim; feature names from master C1 FEATURE_NAMES (12 columns) | no, changing it after a tuning run invalidates the comparison |
2026-11-10T22:27Z | the structural ablation reported in the paper is E3's single_row (one row per request, event-only label), which is skeleton §8.3 row 1 | src/mrc/eval/baselines.py per master E3 docstring "quantifies what the expansion buys (STATISTICAL_SKELETON.md 6.1)" | no |
2026-11-10T22:27Z | skeleton §8.3 rows 2 and 4 (censoring dropped, random split) are NOT implemented by any master task and are therefore NOT claimed; they become an appendix note only if the user adds a card | master G3 objective cites §8.3 but its GROUPS covers feature families only; grep of master task list finds no task producing either ablation | yes, a new card would restore them |
```

### Open questions (§6), resolutions and defaults

```
2026-11-10T22:27Z | Q1 byte-capacity curve = appendix, object-capacity curve = primary headline result | master §6 risk row "Byte-MRC is not exact" and §2 "Ground truth ... exact for object capacities"; docs/DATA.md WORKLOG note "byte stack distance approximate (6.5e-5 on real data)" | yes, a user answer before H1 reverses it |
2026-11-10T22:27Z | Q2 cluster52 20 M run = NOT required for the headline claim; the robustness claim rests on the 5 M window on four clusters; the cluster52 run is 10 M (cost guard, above) and is a sensitivity appendix only if time remains | cost projection breach above; master §6 mitigation lists it as a "sensitivity check" | yes |
2026-11-10T22:27Z | Q3 report leads with the accuracy result, interpretability as the second contribution; figure order H1 (MRC) then G figures; abstract first sentence is the accuracy claim | the frozen primary metric is accuracy MAE (this log, first line), and every gate in master §5 is an accuracy gate | yes, a user answer before H4 reverses it |
```

### Plan-set decisions

```
2026-11-10T22:27Z | the ten edges of §7.3 are the board's cross-phase set; intra-phase order is carried by the phase plan task order, and Phase F's gate table makes D5 a precondition of F1 | audit M-7 | yes |
2026-11-10T22:27Z | ownership follows AGENTS.md globs, not the master's phase-to-profile summary | audit M-8 | yes |
2026-11-10T22:27Z | F1 must write features.manifest.json in addition to the master's sketch | audit M-6, frozen contract "sha256 in a manifest.json beside each artefact" | no |
2026-11-10T22:27Z | B1 must construct requests with keywords, `lc.Request(obj_id=k, obj_size=1)` | audit M-14, measured ctor signature | no |
2026-11-10T22:27Z | B3 must derive the grid from the fit-time bin edges (`overflow_edge = edges[-2]`), not from `2**ceil(log2(rtilde.max()))+1` = 1025 | audit M-3; the master's formula yields capacities up to 1024, which `predict_mrc` rejects, so the truth curve would be unscoreable | no |
2026-11-10T22:27Z | no kanban card is created or modified in this planning session; docs/CARDS.md is the specification and the board is written by the user or the dispatcher | task constraint | yes |
```

### Test corrections forced by the audit

```
2026-11-10T22:27Z | the canonical censoring scheme is last-occurrence, delta = eligible AND NOT is_last, so a request whose key never recurs after it is censored even if it has a previous access | master A1 code and docstring; measured on ABACABDC: censor = [F,F,F,F,T,T,F,T] | no |
2026-11-10T22:27Z | STATISTICAL_SKELETON.md §4.4's event table is illustrative of a horizon where only cold requests are censored, and is NOT the scheme the pipeline uses; A2 must not borrow its event counts | measured divergence on ABACABCB: skeleton counts 5 events, the last-occurrence scheme counts 2 | no |
2026-11-10T22:27Z | A1's fourth test array is corrected to np.concatenate([np.arange(50), np.arange(50)]) so the cap is exercised (R = 49, cap 4 -> rtilde 4) | audit M-15, computed | yes |
2026-11-10T22:27Z | A2's assertions are replaced by the measured table: on ABACABDC, eligible = 4, delta = 1, R[delta] = [1], rtilde(cap 7) = [-1,-1,1,-1,3,2,-1,0]; a second case on the skeleton trace ABACABCB gives eligible = 5, delta = 2, R[delta] = [1,2], rtilde(cap 3) = [-1,-1,1,-1,2,2,1,0] | audit M-16, computed; the master's [1,2,3] is retracted | yes |
```

### Feature-input decision

```
2026-11-10T22:27Z | labels.npz gains two additive int32 columns, key_ns (deterministic code of the first ':'-component of the key string) and key_len (key string length in bytes), both computed in build_labels from Trace.key_vocab; labels.manifest.json gains feature_source: "key_vocab"; C1 consumes them when labels["key"] is not an object array | audit M-17; src/mrc/data/trace.py documents key as int32 codes into key_vocab; data/interim/s45.npz key dtype int32 with no vocab column | yes, additive |
2026-11-10T22:27Z | labels.npz is never written with a pickled object array, so load_labels needs no allow_pickle; key strings stay out of the cache | the master's load_labels calls np.load without allow_pickle, and a pickle-free cache is the only version that loads under the frozen function | no |
```

### Pipeline contract decisions

```
2026-11-10T22:27Z | the split is one function, block_masks(n_requests) on original request indices, and fit_grouped takes n_requests | audit M-18; the frozen contract says "contiguous blocks 70/15/15 by request index" | no |
2026-11-10T22:27Z | baselines take optional eval_X / eval_rtilde / eval_delta so the caller can fit on train and score on test | audit M-22; master E3 fits and scores on the same arrays | yes, the keywords default to the master's behaviour |
2026-11-10T22:27Z | run_one reports validation and test metrics separately, each against mrc_curve(R[mask], caps), and scores the test block once | audit M-23; master §5 threshold "gbdt.mae on test block" cannot be computed from the master's F1 body | no |
2026-11-10T22:27Z | IBS is computed from per-request predicted hit probabilities 1 - S_R(bin(C-1)|x), never from the truth curve | audit M-24; the master's F1 passes the truth as the prediction | no |
2026-11-10T22:27Z | every config constant is read from mrc.config, including NUM_THREADS inside the Optuna suggest function | audit M-25; master §2 "nothing else may hardcode these" | no |
2026-11-10T22:27Z | src/mrc/config.py is created in A1 verbatim, not in A3, so compute_reuse's default cap is the frozen constant rather than a literal | audit M-26; A1 must run before A3 in the DAG | no, the file is frozen on creation |
```
