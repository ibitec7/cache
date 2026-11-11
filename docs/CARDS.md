# Card specifications

One card per master task, 29 cards in 8 phases. This file is the specification for the board; **no card is
created or modified on the kanban board by this planning session** (task constraint). A dispatcher or the
user writes the board from this table.

`PY` below means `env -u PYTHONPATH uv run`. Every command assumes `cd /home/ibrahim/uni/cache`.
Full command text, counts and failure semantics: `docs/GATES.md`. Task bodies: the phase plans under
`.hermes/plans/mrc/`.

Test counts are the ones the phase plans produce, which include the tests added or corrected where the
master's own test cannot pass as written (audit findings M-15 to M-24 in `logs/DECISIONS.md`). The master's
smaller counts apply to its smaller test files.

Dependency notation: `X (edge)` is one of the ten frozen cross-phase edges from master §7.3;
`X (intra)` is an intra-phase ordering edge recorded in `logs/DECISIONS.md` audit M-7; `-` means none.

Numeric acceptance criteria are only those the master states. Where a value is not yet knowable the cell
says `TO BE MEASURED` and names the task that produces it.

---

## Phase A: labels (owner `mrc-data`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| A1 | add `gamma`, `delta`, `rtilde` to `compute_reuse` with a `cap` argument | `mrc-data` | A | - | `PY pytest tests/data/test_labels_bound.py -q` then `PY pytest -q` | `5 passed`, then `15 passed` | `rtilde == -1` exactly for ineligible requests; `rtilde <= cap` everywhere; `delta` is a subset of `eligible`; the five bound tests pass |
| A2 | pin semantics on two hand-derived micro-examples | `mrc-data` | A | A1 (intra) | `PY pytest tests/data/test_labels_bound.py -q` | `7 passed` | on `ABACABDC` with `cap=7`: eligible 4, `delta.sum() == 1`, `R[delta] == [1]`, `rtilde == [-1,-1,1,-1,3,2,-1,0]`; on `ABACABCB` with `cap=3`: eligible 5, `delta.sum() == 2` |
| A3 | label cache, manifest, `key_ns`/`key_len`, and `src/mrc/config.py` | `mrc-data` | A | A1, A2 (intra) | `PY pytest tests/data/test_label_cache.py -q` | `2 passed` | round-trip is array-identical; no pickled array; manifest carries `n_requests`, `cap`, 64-hex `sha256`, `eligible`, `events`, `p_compulsory`, `feature_source`, `git_sha`, `created_utc`; `config.py` matches master §2 verbatim |
| A4 | `mrc labels` command and `make labels` | `mrc-data` | A | A3 (intra) | `PY python -m mrc.cli labels data/raw/cache-trace/samples/2020Mar/cluster045 --cluster sample45 --limit 1000000`; rerun the identical command; rerun with `--limit 500000` | `wrote ...`, `E` within 25 % of 130,600, `D` within 25 % of 88,000; rerun prints `cache hit` in under 0.2 s; the `--limit 500000` run is a miss | artefact plus manifest exist; the cache key is `(source, limit, cap)`; `configs/data/t1.yaml` sets `limit_requests: 5000000` and `cap_quantile: 1023` |

## Phase B: ground truth (owner `mrc-data`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| B1 | libCacheSim single-request replay cross-check | `mrc-data` | B | - | `PY pytest tests/gt/test_replay.py -q` | `1 passed` | `replay_mrc` equals `naive_lru_mrc` with `atol=0.0`; requests built as `lc.Request(obj_id=k, obj_size=1)` (audit M-14). May be deleted after a documented 10-minute failure without blocking anything |
| B2 | exact MRC from stack distance, and the identification-limited grid | `mrc-data` | B | A1 to A3 | `PY pytest tests/gt -q` | `4 passed` (`3 passed` if B1 was deleted) | `mrc_from_labels` equals `naive_lru_mrc` under `array_equal`; `identified_grid` is strictly increasing, `min >= 1`, `max < overflow_edge`, and unique-collapsed |
| B3 | truth artefact, `mrc truth`, `make truth` | `mrc-data` | B | A4 (edge 1), B2 (intra) | `PY python -m mrc.cli truth sample45` then print `capacities` and `mrc` from the npz | `K` strictly increasing capacities, `K` non-increasing ratios in `[0, 1]`, `mrc[0] > 0.95`, `K = manifest.n_capacities` (22 on cluster045, audit M-2) | `truth.npz` plus manifest; grid derived from the fit-time edges (`overflow_edge = edges[-2]`), not from `2**ceil(log2(rtilde.max()))+1` (audit M-3); `b_M_minus_1` recorded |

## Phase C: features (owner `mrc-data`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| C1 | `build_features` with a fixed column order | `mrc-data` | C | A4 (edge 2) | `PY pytest tests/test_features.py -q` | `5 passed` | `X` is float32 `(n, 12)`, all finite, `names == list(FEATURE_NAMES)`, `cat_idx` indexes `ns_code`, `op`, `is_write`; integer-key labels without `key_ns`/`key_len` raise; no column is a function of `R` |
| C2 | namespace coding, both paths agreeing | `mrc-data` | C | C1 (intra) | `PY pytest tests/test_features.py -q` | `7 passed` | `q:q:1:*` shares one code; the string path and the cached column path induce the same partition; no code is 0 for a parseable key |
| C3 | 200 k-row throughput gate | `mrc-data` | C | C2 (intra) | `PY pytest tests/test_features.py -q` | `8 passed` | `build_features` on 200,000 object-key rows completes under 2.0 s; the budget is not raised |

## Phase D: estimator (owner `mrc-run`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| D1 | grouped risk-set rows; `assign_bins` clip fix | `mrc-run` | D | C1 (edge 3) | `PY pytest tests/models/test_risk_set.py -q` | `2 passed` | one row per (request, bin at risk); `y` sums to the event count; `w >= 1`; `assign_bins` returns at most `len(edges) - 2` for the top value (audit M-4) |
| D2 | grouped and per-step cover the same events | `mrc-run` | D | D1 (intra) | `PY pytest tests/models/test_risk_set.py -q` | `3 passed` | identical `eligible_index` sets, identical `y.sum()`, `n_rows` differ; `per_step_rows` honours both frozen guards |
| D3 | contiguous request-block split masks | `mrc-run` | D | D1 (intra) | `PY pytest tests/models/test_blocks.py -q` | `2 passed` | masks are contiguous, ordered, disjoint, and cover every request once; boundaries at `int(n*0.70)` and `int(n*0.85)` |
| D4 | `fit_grouped` on grouped rows with block splits | `mrc-run` | D | D1 (edge 4), D2, D3 (intra) | `PY pytest tests/models/test_hazard_model.py -q` | `2 passed` | survival non-increasing in step, in `(0, 1]`; `validation_logloss` recorded; the split uses original request indices through `block_masks(n_requests)` (audit M-18) |
| D5 | `predict_mrc` raises past the identification limit | `mrc-run` | D | D4 (intra) | `PY pytest tests/models -q` | `8 passed` | `ValueError` for any capacity `>= edges[-2]` (837 on cluster045); curve non-increasing in capacity; no capacity clip remains |

## Phase E: evaluation (owner `mrc-run`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| E1 | metric set with hand-checked values | `mrc-run` | E | D5 (intra) | `PY pytest tests/eval/test_metrics.py -q` | `4 passed` | `mae`, `max_abs`, `bias`, `mre`, `mre_n`, `abc`, `n` on a hand example; MRE excludes truth below `mre_floor`; ABC log-normalised; a perfect prediction gives zeros |
| E2 | Graf IPCW weights and integrated Brier score | `mrc-run` | E | E1 (intra) | `PY pytest tests/eval/test_metrics.py -q` | `6 passed` | a censored request below capacity gets zero weight; IBS is 0 for perfect predictions; `S_Gamma` is evaluated at `t^-` |
| E3 | KM, single-row and hit-count baselines | `mrc-run` | E | E2 (intra) | `PY pytest tests/eval/test_baselines.py -q` | `4 passed` | all three return curves on the grid; KM non-increasing in capacity and bounded in `[p_compulsory, 1]`; no baseline trains on its evaluation block (audit M-22) |

## Phase F: pipeline and telemetry (owner `mrc-run`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| F1 | `run_one` orchestration with stage caching | `mrc-run` | F | D4 (edge 5), D5, E1 (edge 8), C3 | `PY pytest tests/test_pipeline.py -q` | `2 passed` | five `Stages`; `artifacts/tables/<cluster>_metrics.json` holds `valid` and `test` blocks with four models each; `ibs.test > 0`; `features.manifest.json` written with `labels_sha256`; `cap=CAP_QUANTILE` on every save |
| F2 | MLflow run is idempotent and observable | `mrc-run` | F | F1 (intra) | `PY pytest tests/test_pipeline.py -q` | `3 passed` | one run per call; `params_cluster` set; `mlflow_run_id` written into the metric json |
| F3 | Optuna on cached artefacts, retire `models/search.py` | `mrc-run` | F | F1 (edge 6) | `PY pytest tests/tuning/test_search.py -q`; then `git rm src/mrc/models/search.py && PY pytest -q` | `1 passed`; then all pass | three trials persisted and reloadable; `best_value` equals the minimum; `MemoryError` prunes; `NUM_THREADS` not a literal (audit M-25) |
| F4 | `train` and `tune` commands, `make check` | `mrc-run` | F | F3 (intra) | `PY python -m mrc.cli train sample45 --source data/raw/cache-trace/samples/2020Mar/cluster045 --limit 1000000`; then `make check`; then `PY python -m mrc.cli tune sample45 --trials 5` | five stages, four models per block, `mae` in `[0, 1]`, at most 4 minutes; ruff silent, mypy `Success`, pytest pass; a `best mae` line | end-to-end run on the 1 M sample; `make check` exits 0; the study persists and extends |

## Phase G: interpretability (owner `mrc-run`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| G1 | permutation importance on the validation block | `mrc-run` | G | F1 (edge 7) | `PY pytest tests/interp/test_importance.py -q` | `1 passed` | the informative column ranks first with `drop > 0.2`; one row per feature; every row has `feature`, `drop`, `std`; the test mask is never passed in |
| G2 | SHAP mean-absolute contributions | `mrc-run` | G | G1 (intra) | `PY pytest tests/interp -q` | `2 passed` | `shap_summary` writes a json with `top` of length `top`, drops the bias column, and names the informative feature first |
| G3 | feature-group ablation matrix | `mrc-run` | G | G2 (intra) | `PY pytest tests/test_pipeline.py -q` | `4 passed` | the six frozen groups; `all` keeps `X.shape`; `no_sizes` drops exactly 3 columns; an unknown column raises `KeyError` |

## Phase H: figures, tables, report (owner `mrc-write`)

| ID | Title | Owner | Phase | Depends on | Gate command | Expected output | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| H1 | MRC comparison figure | `mrc-write` | H | F4 (intra), B3 (artefact) | `PY pytest tests/plots -q` | `2 passed` | a PDF over 1000 bytes produced through `fig_double` and `apply_style`; no title; more than four curves raises `ValueError` |
| H2 | reliability diagram | `mrc-write` | H | H1 (intra) | `PY pytest tests/plots -q` | `3 passed` | bin fractions sum to 1; the PDF exists; the diagonal is drawn; at most one annotation |
| H3 | metric tables, markdown and booktabs | `mrc-write` | H | G3 (edge 9), E3 (intra) | `PY pytest tests/test_report_tables.py -q` | `1 passed` | `metrics_table` has one row per model with `model, MAE, MAX, bias, MRE, n`; `booktabs` emits `\toprule`/`\bottomrule` and a `% generated` first line; generated `.tex` is never hand-edited |
| H4 | fill report, slides and wiki from artefacts | `mrc-write` | H | H1 (edge 10), H2, H3 (intra) | `grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections wiki/docs/results.md`; then the report, slides and wiki builds | no output from grep; a `Pages:` line with body 8 to 10 from `main.toc`; slides `Pages: 18`; `Documentation built in ... seconds` | every number traces to `artifacts/tables/*_metrics.json`; every MRC caption states the window and `b_{M-1}`; all three builds exit 0 |

---

## Board-writing rules

- One card per row. The card body is the master task text for that ID, verbatim, plus the gate line from
  `docs/GATES.md` and the corrections listed in the phase plan.
- Parent links are the ten frozen edges plus the intra-phase edges in the "Depends on" column. A card whose
  parent is archived is not written; it is reported to the user.
- Numeric acceptance thresholds that the master places on cards (master §5): `gbdt.mae` on the test block
  at most 0.02; `gbdt.mae` strictly below `km.mae` on at least 3 of 4 clusters; `gbdt.mae` below
  `hitcount.mae`; `max_abs` at most 0.10 wherever truth is at least 0.01; `|bias|` at most 0.01. These go
  on the cards whose gates produce the numbers (F4, H4), never in prose.
- A card with no runnable gate is a question. Send it to `docs/OPEN_QUESTIONS.md` instead.
- Two failed worker runs on the same card escalate to the user with the exact error text.
