# Gate registry

One row per acceptance command. This file is authoritative for expected output. `docs/MILESTONES.md` is
historical and names targets that do not exist (audit finding M-9 in `logs/DECISIONS.md`).

Test counts are the counts produced by the phase plans, which are the master's test files plus the tests
the plans add or correct where the master's test cannot pass as written (audit findings M-15 to M-24).
Where a count differs from the master plan, the master's number is for its own, smaller test file; the
phase plan states which tests were added and why.

Every command assumes `cd /home/ibrahim/uni/cache`. `env -u PYTHONPATH` is mandatory: Hermes injects a
Python 3.14 site-packages that shadows pydantic and breaks mlflow.

"Blocks" lists the task IDs that may not start, or may not be marked done, until the gate passes.

Two gates are marked `planner-added`. They apply an existing master rule (the §7.2.4 transfer check, the
frozen style contract, the frozen determinism contract) to a command the master states in prose only. No
new threshold, file name or constant is introduced by them.

## Phase A: labels (owner `mrc-data`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G0-env` | `env -u PYTHONPATH uv run pytest -q` | `10 passed` | A1 |
| `G-A1-bound` | `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` | `5 passed` | A2, A3 |
| `G-A1-suite` | `env -u PYTHONPATH uv run pytest -q` | `15 passed` | A2 |
| `G-A2-micro` | `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` | `7 passed` | A3, B2, C1 |
| `G-A3-cache` | `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` | `2 passed` | A4, B3 |
| `G-A4-real` | `env -u PYTHONPATH uv run python -m mrc.cli labels data/raw/cache-trace/samples/2020Mar/cluster045 --cluster sample45 --limit 1000000` | `wrote .../data/processed/sample45/labels.npz: 1000000 requests, E eligible, D events; manifest labels.manifest.json`, with `E` within 25 % of 130,600 and `D` within 25 % of 88,000 (rates 13.06 % and 8.80 %, measured on `data/interim/s45.npz`) | A4 commit, B3, C1 |
| `G-A4-hit` | same command, rerun | `cache hit .../data/processed/sample45/labels.npz (1000000 requests)`, wall time under 0.2 s | A4 commit |
| `G-LABELS-MULTI` | `for c in cluster45 cluster26 cluster10; do env -u PYTHONPATH uv run python -m mrc.cli labels data/raw/traces/$c.sort.zst --cluster $c --limit 5000000; done` | three `wrote` lines, each `5000000 requests`, three manifests with `n_requests = 5000000` and a recorded `cap` | B3, grid freeze (planner-added, source master §7.2.4) |

## Phase B: ground truth (owner `mrc-data`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-B1-replay` | `env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q` | `1 passed` | nothing (off the critical path) |
| `G-B2-stackdist` | `env -u PYTHONPATH uv run pytest tests/gt -q` | `4 passed` (`3 passed` if B1 was deleted after a documented failure) | B3, D5 |
| `G-B3-artifact` | `env -u PYTHONPATH uv run pytest tests/gt/test_truth_artifact.py -q` | `1 passed` | B3 commit |
| `G-B3-curve` | `env -u PYTHONPATH uv run python -m mrc.cli truth sample45` then `env -u PYTHONPATH uv run python -c "import numpy as np; z=np.load('data/processed/sample45/truth.npz'); print(z['capacities'].tolist()); print(np.round(z['mrc'],4).tolist())"` | `K` strictly increasing capacities and `K` non-increasing ratios in `[0, 1]` with `mrc[0] > 0.95`, where `K = truth.manifest.json.n_capacities` (22 on cluster045 by audit M-2; the master's "24" is the requested resolution) | E1, F1, H1 |
| `G-GRID-TRANSFER` | `env -u PYTHONPATH uv run python -c "import json,glob; [print(p, json.load(open(p))['max_capacity']) for p in sorted(glob.glob('data/processed/*/truth.manifest.json'))]"` | one line per labelled cluster; every `max_capacity < min over clusters of b_M_minus_1`; the single shared grid is re-frozen in `logs/DECISIONS.md` with its realised count | D1, H1 (planner-added, source master §7.2.4) |

## Phase C: features (owner `mrc-data`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-C1-features` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `5 passed` | C2, D1 |
| `G-C2-namespace` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `7 passed` | C3, G3 |
| `G-C3-throughput` | `env -u PYTHONPATH uv run pytest tests/test_features.py -q` | `8 passed` | D1, F1 |

## Phase D: estimator (owner `mrc-run`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-D1-risk` | `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` | `2 passed` (after the `assign_bins` clip fix in the same commit) | D2, D4 |
| `G-D2-agree` | `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` | `3 passed` | D4 |
| `G-D3-blocks` | `env -u PYTHONPATH uv run pytest tests/models/test_blocks.py -q` | `2 passed` | D4 |
| `G-D4-fit` | `env -u PYTHONPATH uv run pytest tests/models/test_hazard_model.py -q` | `2 passed` | D5, F1 |
| `G-D5-guard` | `env -u PYTHONPATH uv run pytest tests/models -q` | `8 passed`, with the `DID NOT RAISE` failure replaced by a real `ValueError` | F1, E3, H1 |

## Phase E: evaluation (owner `mrc-run`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-E1-metrics` | `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` | `4 passed` | E2, F1 |
| `G-E2-ipcw` | `env -u PYTHONPATH uv run pytest tests/eval/test_metrics.py -q` | `6 passed` | E3, F1 |
| `G-E3-baselines` | `env -u PYTHONPATH uv run pytest tests/eval/test_baselines.py -q` | `4 passed` | F1, H3 |

## Phase F: pipeline and telemetry (owner `mrc-run`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-F1-pipeline` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `2 passed` | F2, G1 |
| `G-F2-mlflow` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `3 passed` | F3 |
| `G-F3-study` | `env -u PYTHONPATH uv run pytest tests/tuning/test_search.py -q` | `1 passed` | F4 |
| `G-F3-remove` | `git rm src/mrc/models/search.py && env -u PYTHONPATH uv run pytest -q` | all pass, 0 failed, nothing imports `mrc.models.search` | F4 |
| `G-F4-train` | `env -u PYTHONPATH uv run python -m mrc.cli train sample45 --source data/raw/cache-trace/samples/2020Mar/cluster045 --limit 1000000` | JSON with `"stages": ["labels","truth","features","fit","eval"]`, four models per block, every `mae` in `[0, 1]`, wall clock at most 4 minutes | F4 commit, G1, H1 |
| `G-F4-check` | `make check` | ruff silent, `Success: no issues found`, pytest all pass, exit 0 | F4 commit, Phase H |

## Phase G: interpretability (owner `mrc-run`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-G1-perm` | `env -u PYTHONPATH uv run pytest tests/interp/test_importance.py -q` | `1 passed` | G2 |
| `G-G2-shap` | `env -u PYTHONPATH uv run pytest tests/interp -q` | `2 passed` | G3, H4 |
| `G-G3-ablate` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `4 passed` | H3, H4 |

## Phase H: figures, tables, report (owner `mrc-write`)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-STYLE` | `env -u PYTHONPATH uv run pytest tests/test_smoke.py::test_style_contract -q` | `1 passed` | H1, H2, H4 (planner-added, frozen style contract) |
| `G-H1-mrc` | `env -u PYTHONPATH uv run pytest tests/plots -q` | `2 passed` | H2, H4 |
| `G-H2-reliability` | `env -u PYTHONPATH uv run pytest tests/plots -q` | `3 passed` | H4 |
| `G-H3-tables` | `env -u PYTHONPATH uv run pytest tests/test_report_tables.py -q` | `1 passed` | H4 |
| `G-H4-placeholders` | `grep -rn "PLACEHOLDER\|figplaceholder\|pending" report/sections slides/sections wiki/docs/results.md` | no output (exit 1) | H4 commit |
| `G-H4-report` | `cd report && latexmk -pdf -interaction=nonstopmode main.tex && pdfinfo main.pdf \| grep Pages` | a line `Pages: N`; body length 8 to 10 pages read from `main.toc` (page of References minus page of the first section), because `pdfinfo` counts the appendix. Baseline at audit: `Pages: 13` | H4 commit |
| `G-H4-slides` | `cd slides && latexmk -pdf -interaction=nonstopmode main.tex && pdfinfo main.pdf \| grep Pages` | `Pages: 18` | H4 commit |
| `G-H4-wiki` | `cd wiki && env -u PYTHONPATH uv run --group viz mkdocs build --strict 2>&1 \| tail -2` | `Documentation built in ... seconds`, exit 0 | H4 commit |

## Whole project (any profile)

| Gate | Exact command | Expected output | Blocks |
|---|---|---|---|
| `G-CHECK` | `make check` | ruff silent, `Success: no issues found`, pytest all pass, exit 0 | every task from F4 onward, and the whole-project definition of done |
| `G-DOD` | `make check` | exit 0 | done |
| `G-DOD` | `env -u PYTHONPATH uv run python -m mrc.cli train cluster45` | writes `artifacts/tables/cluster45_metrics.json` | done |
| `G-DOD` | `env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 60` | `artifacts/optuna/study.db` holds 60 trials | done |
| `G-DOD` | `grep -rn "PLACEHOLDER" report/sections` | no output | done |
| `G-DOD` | `cd report && latexmk -pdf main.tex && pdfinfo main.pdf \| grep Pages` | body 8 to 10 pages | done |
| `G-DOD` | `cd ../wiki && mkdocs build --strict` | exit 0 | done |
| `G-DOD` | `git status --porcelain` | no output | done |

## Gate semantics that hold everywhere

- A test count higher than expected means another task's tests landed early. Pass, and note it in the
  handoff. A count lower than expected means a test file is missing or a test was skipped: stop.
- `KeyError`, `ModuleNotFoundError`, `ImportError`, `AttributeError` and `DID NOT RAISE` are the expected
  pre-implementation failures in Phases A to F. Any other failure text means the test itself is wrong;
  re-derive by hand from `docs/STATISTICAL_SKELETON.md` §2.3 before touching the implementation.
- A gate that passes before the task is implemented means either the task is already done or the test is
  vacuous. Stop and report which.
- No gate is satisfied by a number typed into a document. Numbers come from `artifacts/tables/*.json`.
