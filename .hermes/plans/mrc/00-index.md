# MRC hazard pipeline: plan-set index

Master plan: `.hermes/plans/2026-11-11_052903-mrc-hazard-pipeline.md` (2399 lines, read in full
2026-11-11). This set expands it into eight executable phase plans. Where this set and the master
disagree, the disagreement is logged in `logs/DECISIONS.md` under "master-plan audit" and the
resolution is stated there. Nothing here overrides a frozen contract.

Planner profile: `mrc-plan`. Planner writes planning documents only, never `src/`, `tests/`,
`scripts/`, `report/`, `slides/`, `configs/`, `Makefile`.

---

## 1. Goal

Estimate a cache miss-ratio curve `MRC(C)` from Twitter production traces by fitting a LightGBM
discrete-time hazard model over reuse distance, validate it against exact Mattson stack-distance
ground truth, tune it with Optuna, record it in MLflow, explain it with permutation/SHAP/ablation
artefacts, and fill an 8 to 10 page report. Five cached stages, each a pure function of its inputs
plus a hashed manifest, orchestrated by `src/mrc/pipeline.py`.

Definition of the primary object (frozen, master §2 and §7.2.1):

```
MRC(C) = p_compulsory + (1 - p_compulsory) * E_x[ S_R(bin(C - 1) | x) ]
S_R(k | x) = prod_{b=0..k} (1 - h_b(x))
h_b(x)      = P(R in bin b | R >= b, X = x)
```

Symbols: `C` capacity in objects on the frozen grid, `R` reuse distance in distinct keys,
`x` the 12-column feature vector, `p_compulsory` the fraction of requests with no previous access,
`S_R` survival, `h_b` discrete hazard in bin `b`.

---

## 2. Frozen contracts (verbatim from master section 2, not to be edited downstream)

Constants live in one file. Nothing else may hardcode them; a change is a commit on this file plus a
`logs/WORKLOG.md` line.

```python
# src/mrc/config.py
"""Frozen run constants. One source of truth; nothing else may hardcode these."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw" / "traces"
PROCESSED = DATA / "processed"
ARTIFACTS = REPO / "artifacts"
MLFLOW_URI = f"sqlite:///{ARTIFACTS / 'mlflow' / 'mlflow.db'}"
OPTUNA_URI = f"sqlite:///{ARTIFACTS / 'optuna' / 'study.db'}"

LABEL_REQUESTS = 5_000_000      # prefix per cluster for labels / features / truth
SCALE_REQUESTS = 20_000_000     # cluster52 robustness run only
CAP_QUANTILE = 1023             # rtilde cap == b_{M-1} - 1 <= 1023
N_BINS = 32
N_BINS_SEARCH = (8, 16, 24, 32, 48, 64)
N_CAPACITIES = 24

SPLIT_TRAIN = 0.70              # contiguous request blocks
SPLIT_VALID = 0.85              # valid [0.70, 0.85), test [0.85, 1.0)

MAX_GROUPED_ROWS = 60_000_000
PER_STEP_MAX_REQUESTS = 200_000

SEED = 7
NUM_THREADS = 12
```

| Contract | Value | Rationale |
|---|---|---|
| Window | first 5,000,000 requests per cluster | 87 % cold rate; 5 M yields ~650 k eligible rows, enough for a 12-feature GBDT |
| Ground truth | Mattson stack distance from `R`, exact for object capacities | skeleton §7.1; `libCacheSim` aggregate path is broken (`docs/DATA.md` §9), single-request replay retained as a cross-check |
| Capacity grid | `capacity_grid(n_keys, 24)`, truncated so `max G < b_{M-1}` | one grid for all clusters, curves comparable across traces |
| Bins | `bin_edges(percentile(rtilde, 99.9), 32)` | log-spaced to match the log-spaced grid; overflow bin is the identification limit |
| Split | contiguous blocks 70 / 15 / 15 by request index | no key leakage; test block touched once |
| Primary metric | validation-block `MAE` of MRC | one scalar for Optuna; `MAX`, `bias`, `MRE`, `ABC`, `IBS` reported beside it |
| Determinism | `SEED=7`, `NUM_THREADS=12`, `deterministic=True`, `force_row_wise=True` | bit-reproducible on this box |
| Cache key | sha256 in a `manifest.json` beside each artefact + the config values that produced it | rerun is a hit; config drift is a miss, never silent staleness |

Reading note on `N_BINS = 32` and `N_CAPACITIES = 24`: both are *requested resolutions*. The frozen
`bin_edges` and `capacity_grid` implementations apply `np.unique` to rounded log spacing, so the
realised counts can be smaller. Measured on `data/interim/s45.npz`: 31 bins and 22 capacities (see
`logs/DECISIONS.md`, audit finding M-2). Every manifest, table and caption reports the realised
count. No new constant is introduced for this.

---

## 3. Module map (after the plan, from master section 3)

| Path | Role | State before Phase A |
|---|---|---|
| `src/mrc/config.py` | frozen constants | absent; created in A1 (moved from A3, audit M-26) |
| `src/mrc/data/reuse.py` | reuse distance kernel | present, 114 lines, no `cap`/`gamma`/`delta`/`rtilde` |
| `src/mrc/data/labels.py` | label cache + manifest | absent (A3) |
| `src/mrc/data/reference.py` | naive OrderedDict reference | present, 51 lines |
| `src/mrc/features.py` | design matrix builder | absent (C1) |
| `src/mrc/gt/replay.py` | libCacheSim single-request cross-check | absent (B1) |
| `src/mrc/gt/stackdist.py` | exact MRC from labels | absent (B2) |
| `src/mrc/gt/write.py` | truth artefact + manifest | absent (B3) |
| `src/mrc/models/risk_set.py` | grouped + per-step expansion | absent (D1) |
| `src/mrc/models/hazard.py` | `HazardModel`, block splits, identification guard | present, 136 lines, defects D2/D3/D6 |
| `src/mrc/eval/mrc.py` | empirical MRC, `capacity_grid` | present, 58 lines |
| `src/mrc/eval/metrics.py` | MAE/MAX/bias/MRE/ABC/IBS with Graf weights | absent (E1, E2) |
| `src/mrc/eval/baselines.py` | KM, single-row, hit-count regression | absent (E3) |
| `src/mrc/pipeline.py` | `run_one`, `Stages`, ablation matrix | absent (F1, G3) |
| `src/mrc/tuning/search.py` | Optuna study on cached artefacts | absent (F3, replaces `models/search.py`) |
| `src/mrc/interp/importance.py` | permutation, SHAP | absent (G1, G2) |
| `src/mrc/plots/curve.py`, `src/mrc/plots/reliability.py` | figures | absent (H1, H2) |
| `src/mrc/report_tables.py` | markdown + booktabs writers | absent (H3) |
| `src/mrc/cli.py` | `inspect`, `reuse`, `mrc`, `fit`, `plots` | present, 114 lines; gains `labels`, `truth`, `train`, `tune` |
| `src/mrc/plots/style.py` | frozen style, `SINGLE_COL=3.35`, `DOUBLE_COL=7.0` | present, 64 lines, frozen |

---

## 4. Dependency DAG (master section 7.3)

```
A1 A2 A3 A4  →  B2 B3  →  C1 C2 C3  →  D1 D2 D3 D4 D5
                                             ├→ E1 E2 E3
                                             ├→ F1 F2 → F3 F4
                                             └→ G1 G2 G3
                                                       └→ H1 H2 H3 H4
B1 (libCacheSim cross-check) is off the critical path: it may run any time and may fail without blocking.
```

The ten explicit edges to encode on the board, verbatim from master §7.3:

| # | Edge | Why |
|---|---|---|
| 1 | `A4 → B3` | the truth artefact consumes a cached label set |
| 2 | `A4 → C1` | features join onto label columns |
| 3 | `C1 → D1` | risk-set rows carry the design matrix |
| 4 | `D1 → D4` | `fit_grouped` consumes `grouped_rows` output |
| 5 | `D4 → F1` | `run_one` calls `fit_grouped` |
| 6 | `F1 → F3` | the Optuna objective reuses `run_one`'s stages and `_params` |
| 7 | `F1 → G1` | importance is scored on the validation block a fitted run produced |
| 8 | `E1 → F1` | `run_one` writes the metric table |
| 9 | `G3 → H3` | the ablation matrix is a table row set |
| 10 | `H1 → H4` | the report embeds the primary figure |

Implied intra-phase edges the board must also honour when ordering work (not part of the ten, see
audit finding M-7): `A1 → A3`, `A2 → A3` (tests pin semantics before the cache), `A3 → A4`,
`B2 → B3`, `D3 → D4`, `D5 → F1` (F1 calls `predict_mrc`, which raises until D5 lands),
`F1 → F2`, `F3 → F4`, `E2 → E3` (baselines report IBS-adjacent metrics), `H2 → H4`, `H3 → H4`,
`G2 → G3`.

`B1` has no edge and no dependant. If it fails, delete it and note it in `logs/WORKLOG.md`; the
stack-distance curve stays the ground truth.

---

## 5. Ownership matrix (AGENTS.md section 7 plus recorded amendments)

Phase-to-profile mapping follows master §7.2.7 (`mrc-data` A to D, `mrc-run` E to G, `mrc-write` H).
File-level globs follow `AGENTS.md`, which is canonical. Where the master's phase mapping would put a
file under a profile that does not own it, this matrix wins and the amendment is logged (audit
finding M-8).

| Profile | Owns (exact globs) | Never touches |
|---|---|---|
| `mrc-plan` | `.hermes/plans/**`, `docs/GATES.md`, `docs/OPEN_QUESTIONS.md`, `docs/CARDS.md`, `logs/DECISIONS.md` | `src/`, `tests/`, `scripts/`, `configs/`, `report/`, `slides/`, `Makefile` |
| `mrc-data` | `src/mrc/data/**`, `src/mrc/features.py`, `src/mrc/gt/**`, `src/mrc/config.py`, `configs/data/**`, `scripts/fetch_traces.sh`, `data/**` | `src/mrc/models/**`, `src/mrc/eval/**`, `report/`, `slides/` |
| `mrc-run` | `src/mrc/models/**`, `src/mrc/eval/**`, `src/mrc/pipeline.py`, `src/mrc/tuning/**`, `src/mrc/interp/**`, `artifacts/**` | `src/mrc/data/**`, `src/mrc/gt/**`, `src/mrc/features.py`, `report/`, `slides/` |
| `mrc-write` | `src/mrc/plots/**`, `src/mrc/report_tables.py`, `scripts/fill_results.py`, `report/**`, `slides/**`, `wiki/**`, `README.md` | `src/mrc/data/**`, `src/mrc/models/**`, `src/mrc/eval/**` |

Shared surfaces, one writer per commit, tasks touching them are serialised by the dispatcher:

| Path | Owner of record | Rule |
|---|---|---|
| `src/mrc/cli.py` | the profile that owns the stage being wired (`mrc-data` for `labels`/`truth`, `mrc-run` for `train`/`tune`) | one command per commit, never two stages in one commit |
| `Makefile` | same as the command it exposes | one target per commit |
| `tests/**` | the profile that owns the mirrored module path | `tests/data/**` and `tests/gt/**` and `tests/test_features.py` to `mrc-data`; `tests/models/**`, `tests/eval/**`, `tests/tuning/**`, `tests/interp/**`, `tests/test_pipeline.py` to `mrc-run`; `tests/plots/**`, `tests/test_report_tables.py` to `mrc-write`; `tests/test_smoke.py` frozen |

Frozen files, no profile may edit without stopping and asking the user: `src/mrc/plots/style.py`,
`report/bibliography/refs.bib`, the npz schemas in `docs/DATA.md`, and `src/mrc/config.py` after A3
lands it.

---

## 6. Global gates

Every code task, without exception, ends with:

```bash
cd /home/ibrahim/uni/cache
env -u PYTHONPATH uv run pytest -q              # expected: all pass, 0 failed
env -u PYTHONPATH uv run ruff check src tests   # expected: no output
```

Two additional global gates:

| Gate | Command | Expected | When |
|---|---|---|---|
| `G0-env` | `env -u PYTHONPATH uv run pytest -q` | `10 passed` (measured 2026-11-11, baseline) | before A1 |
| `G-CHECK` | `make check` | ruff silent, `Success: no issues found` from mypy, pytest all pass | F4 onward, and at the whole-project definition of done |

`env -u PYTHONPATH` is mandatory on every command: Hermes injects a Python 3.14 site-packages that
shadows pydantic and breaks mlflow.

Full gate registry: `docs/GATES.md`.

---

## 7. Phase plans and one-line acceptance

| Phase | Plan | Owner | Tasks | Acceptance in one line |
|---|---|---|---|---|
| A labels | `01-phase-a-labels.md` | `mrc-data` | A1 A2 A3 A4 | `compute_reuse` returns `gamma`/`delta`/`rtilde` and a cached label set with a sha256 manifest re-loads bit-identically |
| B ground truth | `02-phase-b-truth.md` | `mrc-data` | B1 B2 B3 | exact object-MRC equals LRU replay with `atol=0.0` and a truth artefact exists with a non-increasing curve in [0,1] |
| C features | `03-phase-c-features.md` | `mrc-data` | C1 C2 C3 | a 12-column float32 matrix, all finite, fixed order, 200 k rows in under 2 s |
| D estimator | `04-phase-d-estimator.md` | `mrc-run` | D1 D2 D3 D4 D5 | grouped rows equal per-step support, block splits are contiguous, `predict_mrc` raises past the identification limit |
| E evaluation | `05-phase-e-evaluation.md` | `mrc-run` | E1 E2 E3 | five metrics with hand-checked values and three baselines returning curves on the frozen grid |
| F pipeline and telemetry | `06-phase-f-pipeline.md` | `mrc-run` | F1 F2 F3 F4 | `run_one` writes every stage and a metrics json, one MLflow run per cluster, Optuna study persists |
| G interpretability | `07-phase-g-interpretability.md` | `mrc-run` | G1 G2 G3 | the informative column ranks first, SHAP json is written, six ablation groups have the right column counts |
| H figures, tables, report | `08-phase-h-report.md` | `mrc-write` | H1 H2 H3 H4 | no `PLACEHOLDER` anywhere, booktabs tables generated, report body 8 to 10 pages |

---

## 8. How a fresh worker uses this set

1. Read this index. Nothing else, first.
2. Read the phase plan for the phase your card belongs to. It carries the input and output contract
   you must satisfy and the module interface you must implement.
3. Read your card in `docs/CARDS.md`. The card body is the master task text; the card row carries the
   gate and the acceptance criterion.
4. Run the gate command from `docs/GATES.md` **before** writing anything, and record the baseline in
   your handoff. A gate that already passes means the work is already done or the test is wrong; stop
   and say which.
5. Run the TDD steps in order: failing test, exact command, expected failure text, implementation,
   pass command, expected pass text. Do not reorder.
6. Commit with the message given verbatim in the task. One task, one commit.
7. Run the global gates. Then write your handoff as `tasked -> actual -> left -> next steps` in the
   card comment.
8. A worker that needs a contract change stops and asks. It does not edit a frozen file, and it does
   not edit a file outside its globs.

---

## 9. Out of scope, per phase

| Phase | Out of scope (a worker must refuse these) |
|---|---|
| A | binning, capacity grids, feature construction, any model code, `data/processed/` artefacts for clusters other than the one named |
| B | `libCacheSim` aggregate path (`process_trace`), byte-capacity claims beyond "approximate", any use of the truth curve as a training input |
| C | row filtering, target construction, lightgbm parameters, reordering or renaming columns, `FEATURE_VERSION` bumps |
| D | tuning, telemetry, plots, per-step expansion above 200,000 requests, any capacity at or above the overflow edge |
| E | model fitting, Optuna, figures, tables in `report/` |
| F | new metrics, new features, figures, report text; MLflow model registry; remote tracking servers |
| G | retraining with different features, test-block access, per-run tuning |
| H | any hand-typed number in `report/`, `slides/`, `wiki/`; editing generated `report/tables/*.tex`; editing `style.py`; editing `refs.bib` |

---

## 10. Whole-project definition of done (master section 7.5, verbatim commands)

All of these must hold simultaneously:

```bash
make check                                                        # 0
env -u PYTHONPATH uv run python -m mrc.cli train cluster45        # writes artifacts/tables/cluster45_metrics.json
env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 60   # study.db has 60 trials
grep -rn "PLACEHOLDER" report/sections                            # no output
cd report && latexmk -pdf main.tex && pdfinfo main.pdf | grep Pages     # body 8-10 pp
cd ../wiki && mkdocs build --strict                                # 0
git status --porcelain                                            # clean
```

Numeric acceptance thresholds live on the cards (`docs/CARDS.md`), not in prose: `gbdt.mae` on the
test block at most 0.02; `gbdt.mae` strictly below `km.mae` on at least 3 of 4 clusters;
`gbdt.mae` below `hitcount.mae`; `max_abs` at most 0.10 wherever truth is at least 0.01;
`|bias|` at most 0.01; reuse recompute bit-identical to `naive_reuse`.
