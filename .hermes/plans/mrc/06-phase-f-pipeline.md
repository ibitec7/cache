# Phase F: pipeline and telemetry

**Phase:** F. **Owner profile:** `mrc-run`. **Defects closed:** D5's remaining half (no end-to-end run, no
telemetry, no tuning entry point).
**Upstream:** `04-phase-d-estimator.md` (edge 5, `D4 -> F1`; D5's guard is a precondition),
`05-phase-e-evaluation.md` (edge 8, `E1 -> F1`), `03-phase-c-features.md` (C3 throughput).
**Downstream:** `07-phase-g-interpretability.md` (edge 7, `F1 -> G1`), `08-phase-h-report.md` (all figures
and tables read `artifacts/tables/`).

**Estimated wall time:** 4 tasks at 2 to 5 minutes each (master §4), plus the 1 M-request end-to-end run,
whose master bound is 4 minutes (F4 step 3), plus a 5-trial Optuna smoke run.
**Compute:** 12 threads (`NUM_THREADS`) for LightGBM, no GPU. Peak memory is the grouped row block:
5,000,000 requests x 3.73 rows x 66 bytes = 1.23 GB of arrays (`logs/DECISIONS.md` cost projection),
before LightGBM's internal Dataset. `MAX_GROUPED_ROWS = 60,000,000` is the hard guard.

---

## 1. Input and output contract

### Inputs

| Input | Path | Provenance |
|---|---|---|
| shard or sample | `--source` | `data/raw/traces/cluster45.sort.zst` or `data/raw/cache-trace/samples/2020Mar/cluster045` |
| cached stages | `data/processed/<cluster>/{labels,truth,features}.npz` + manifests | Phases A, B, C, D |
| frozen constants | `src/mrc/config.py` | `LABEL_REQUESTS`, `N_BINS`, `N_CAPACITIES`, `MAX_GROUPED_ROWS`, `SEED`, `NUM_THREADS`, `MLFLOW_URI`, `OPTUNA_URI` |

### Outputs

| Artefact | Path | Producer |
|---|---|---|
| stage caches | `data/processed/<cluster>/labels.npz`, `truth.npz`, `features.npz` + three manifests | F1 |
| metric table | `artifacts/tables/<cluster>_metrics.json` | F1 |
| MLflow run | `artifacts/mlflow/mlflow.db`, experiment `mrc` | F1 (via `_log_mlflow`) |
| Optuna study | `artifacts/optuna/study.db`, study `hazard` | F3, F4 |

`artifacts/tables/<cluster>_metrics.json`:

```
cluster            str
capacities         list[int]        strictly increasing, every value < b_M_minus_1
b_M_minus_1        int              edges[-2] of the fit-time edges
bins               int              len(edges) - 1
p_compulsory       float            (~eligible).mean() on the whole prefix
limit              int|null
eligible           int
rows               int              grouped rows used for the fit
mrc_true           {"valid": [...], "test": [...]}     mrc_curve(R[mask], capacities)
curves             {"gbdt": {"valid": [...], "test": [...]},
                    "km": {...}, "single_row": {...}, "hitcount": {...}}
metrics            {"valid": {model: metric dict}, "test": {model: metric dict}}
ibs                {"valid": float, "test": float}
seconds            float
git_sha            str
created_utc        str
```

Each metric dict is the `curve_metrics` output: `mae`, `max_abs`, `bias`, `mre`, `mre_n`, `abc`, `n`.

Invariants:

- `I-F1` every stage writes its artefact only if its own manifest is absent or its inputs changed; a rerun
  with unchanged inputs reuses the file and reports the stage as a cache hit.
- `I-F2` `features.npz` has a manifest with `sha256`, `labels_sha256`, `feature_version`, `names`
  (audit M-6). A features file whose `labels_sha256` differs from the current labels manifest is a cache
  miss.
- `I-F3` the reported `test` metrics are computed once, against `mrc_curve(R[test_mask], capacities)`, and
  never used for selection (audit M-23).
- `I-F4` `ibs` is computed from per-request predicted hit probabilities, `1 - S_R(bin(C-1) | x)`, not from
  the truth curve (audit M-24).
- `I-F5` every metric value written to JSON is a finite Python `float` or `int`; a `NaN` (possible for
  `mre`) is written as `null` and `mre_n` is 0.
- `I-F6` one MLflow run per `(cluster, limit, bins)` triple; `params_cluster` is set; the run id is
  recorded in the metric json as `mlflow_run_id`.
- `I-F7` the Optuna objective reads only cached artefacts; it never calls `build_labels`, `read_trace` or
  `written_truth`.
- `I-F8` `run_one` never writes outside `processed/<cluster>/` and `artifacts/`.

---

## 2. Module interface contract

```python
# src/mrc/pipeline.py
class Stages(str, Enum):
    """Stage names used in the returned dict and in the cache checks."""
    LABELS = "labels"; TRUTH = "truth"; FEATURES = "features"; FIT = "fit"; EVAL = "eval"

def _params(num_boost_round: int = 400) -> dict:
    """LightGBM parameters with the frozen seed, thread count and determinism flags."""

def run_one(cluster: str, *, source, processed: Path = PROCESSED, artifacts: Path = ARTIFACTS,
            mlflow_uri: str | None = None, limit: int | None = None, bins: int = N_BINS,
            num_boost_round: int = 400, n_capacities: int = N_CAPACITIES,
            n_requests: int | None = None) -> dict:
    """Run labels -> truth -> features -> fit -> eval for one cluster, caching each stage."""

def _log_mlflow(uri: str, cluster: str, model, out: dict, names: list[str],
                cat_idx: list[int]) -> str:
    """Log params, per-block metrics and the curve json; return the run id."""

def ablate(X: np.ndarray, names: list[str], groups: dict = GROUPS) -> dict[str, np.ndarray]:
    """Return {group: X with the dropped columns removed} (G3 owns the call site)."""

GROUPS: dict[str, tuple[str, ...]]
```

Callers of `run_one`: `src/mrc/cli.py` `train` (F4), `tests/test_pipeline.py`. Caller of `_log_mlflow`:
`run_one`. Caller of `ablate`: G3's tests and the ablation driver.

```python
# src/mrc/tuning/search.py
def suggest(trial: optuna.Trial) -> dict:
    """Sample one hazard hyperparameter set, including the bin count as `_n_bins`."""

def objective_factory(score_fn) -> Callable[[optuna.Trial], float]:
    """Turn score(params) -> float into an Optuna objective that prunes on MemoryError."""

def make_study(storage: str, name: str = "hazard") -> optuna.Study:
    """Create or load the persistent minimisation study with a seeded TPE sampler."""
```

Callers: `src/mrc/cli.py` `tune` (F4), `tests/tuning/test_search.py`.

```python
# src/mrc/cli.py
def train(cluster: str, source: str | None, limit: int | None, bins: int) -> None:
    """Run labels -> truth -> features -> fit -> eval for one cluster."""

def tune(cluster: str, trials: int, timeout: int) -> None:
    """Tune hazard hyperparameters on one cached cluster; score = validation-block MAE."""
```

`n_requests` is added by this plan so `run_one` can pass the true request count to `fit_grouped`
(audit M-18). It defaults to `limit` or to the label array length.

---

## 3. Ordered tasks

### F1: `run_one` orchestration with stage caching

**Purpose:** one function that runs the chain for a cluster, so a bad tail costs one stage.
**Files:** create `src/mrc/pipeline.py`, `tests/test_pipeline.py`.
**Depends on:** D4 (edge 5), D5, E1 (edge 8), C3.

**Five corrections carried into this task**

1. `save_labels(..., cap=int(lab["rtilde"].max()))` in the master's body passes the *observed maximum* as
   the cap, so a label set built with `CAP_QUANTILE` would be re-saved with a different cap and the
   identification limit would silently move. Pass `cap=CAP_QUANTILE` and assert that
   `labels["rtilde"].max() <= CAP_QUANTILE`.
2. `features.npz` gets a manifest (`I-F2`), which the master's body omits.
3. Metrics are computed per block against `mrc_curve(R[mask], capacities)` (`I-F3`), not against the
   whole-prefix truth.
4. `ibs` uses the model's per-request hit probabilities (`I-F4`).
5. `build_features` must be called on labels that carry `key_ns`/`key_len`; `run_one` does not rebuild
   labels when a cache exists, so a labels file written before Phase A3's extension is a cache miss, not
   a silent constant.

**TDD steps**

1. Failing test. The master's `test_run_one_writes_every_stage`, plus two:

```python
def test_run_one_reports_both_blocks_and_a_real_ibs(tmp_path):
    src = _tiny_csv(tmp_path)
    res = run_one("tiny", source=src, processed=tmp_path / "proc", artifacts=tmp_path / "art",
                  mlflow_uri=None, limit=4000, bins=12, num_boost_round=30, n_capacities=8)
    m = json.loads((tmp_path / "art" / "tables" / "tiny_metrics.json").read_text())
    assert set(m["metrics"]) == {"valid", "test"}
    assert m["ibs"]["test"] > 0.0          # the truth scored against itself would be 0
    assert m["b_M_minus_1"] > max(m["capacities"])
    assert "features" in m and m["features"]["feature_version"] == 1
```

2. Verify failure: `ModuleNotFoundError: No module named 'mrc.pipeline'`.
3. Implement. Start from the master's body and apply the five corrections. The stage cache check is:

```python
def _cache_ok(man_path: Path, artefact: Path, **expected) -> bool:
    """True when the manifest exists, its sha256 matches, and every expected key/value agrees."""
```

   `run_one` builds the test mask once with `block_masks(len(lab["R"]))`, computes
   `mrc_true[block] = mrc_curve(lab["R"][mask], caps)`, and fits with
   `fit_grouped(rows, edges, names, num_boost_round=..., n_requests=len(lab["R"]))`.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` -> `2 passed`.

**Gate:** `G-F1-pipeline` (`2 passed`).
**Commit:** `feat(pipeline): run_one with per-stage caching, per-block metrics and a real IBS`
**Rollback:** `git revert <sha>`; the CLI has no `train` yet, so nothing user-facing breaks.
**Done when:** five stages appear, the metric json has both blocks, `ibs.test > 0`, and every stage artefact
has a manifest.

---

### F2: MLflow logging is idempotent and observable

**Purpose:** every reported number maps to a run id.
**Files:** append to `tests/test_pipeline.py`.
**Depends on:** F1.

**TDD steps**

1. Failing test, as master F2 gives it, plus one assertion on the run id being written into the json:

```python
def test_mlflow_run_id_is_recorded_in_the_metrics_json(tmp_path):
    import mlflow
    src = _tiny_csv(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_one("tiny", source=src, processed=tmp_path / "p", artifacts=tmp_path / "a",
            mlflow_uri=uri, limit=4000, bins=12, num_boost_round=20, n_capacities=8)
    m = json.loads((tmp_path / "a" / "tables" / "tiny_metrics.json").read_text())
    mlflow.set_tracking_uri(uri)
    df = mlflow.search_runs(experiment_names=["mrc"])
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == m["mlflow_run_id"]
```

2. Verify failure: `KeyError: 'mlflow_run_id'`.
3. Implement `_log_mlflow` per master F1, returning `run_id`, with metric names
   `f"{block}_{model}_{metric}"`, and write the id into `out` before the json dump (which then happens
   after logging, so the order in `run_one` changes: log, then dump).
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` -> `3 passed`.

**Gate:** `G-F2-mlflow` (`3 passed`).
**Commit:** `test(pipeline): MLflow run carries the cluster, both blocks and its run id`
**Rollback:** `git revert <sha>`; telemetry is additive.
**Done when:** one run per call, `params_cluster` set, and the run id in the json.

---

### F3: Optuna on cached artefacts

**Purpose:** tune the hazard without recomputing labels, truth or features.
**Files:** create `src/mrc/tuning/__init__.py`, `src/mrc/tuning/search.py`; delete
`src/mrc/models/search.py`; create `tests/tuning/test_search.py`.
**Depends on:** F1 (edge 6).

**TDD steps**

1. Failing test, as master F3 gives it.
2. Verify failure: `ModuleNotFoundError: No module named 'mrc.tuning'`.
3. Implement `suggest`, `objective_factory`, `make_study` per master F3, with `NUM_THREADS` instead of
   the literal `12` (audit M-25). Keep `_n_bins` in the returned dict; the caller pops it.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/tuning/test_search.py -q` -> `1 passed`.
5. Remove the superseded module: `git rm src/mrc/models/search.py`, then
   `env -u PYTHONPATH uv run pytest -q` -> all pass, nothing imports `mrc.models.search`.

**Gate:** `G-F3-study` (`1 passed`), `G-F3-remove` (all pass after the removal).
**Commit:** `feat(tuning): Optuna study over cached artefacts, replaces models/search.py`
**Rollback:** `git revert <sha>` restores `models/search.py`; no study.db is deleted.
**Done when:** three trials persist and reload, `best_value` is the minimum, and the old module is gone.

---

### F4: `train` and `tune` commands, and `make check`

**Purpose:** the two commands the definition of done names, and the one gate that matters.
**Files:** modify `src/mrc/cli.py`, `Makefile` (shared surfaces).
**Depends on:** F3.

**TDD steps**

1. Failing evidence: `make check` with no target -> `make: *** No rule to make target 'check'`.
2. Implement `train` and `tune` per master F4. In `tune`, the objective must:
   - pop `_n_bins` from a **copy** of params, since `objective_factory` reads it afterwards;
   - cache `grouped_rows` per bin count (the master does this);
   - score validation-block MAE only: `mrc_curve(lab["R"][valid_mask], caps)` as truth and the curve
     averaged over `X[valid_mask]`;
   - never touch the test block.
3. Make targets `train`, `tune`, `check`, and `.PHONY` additions, verbatim from master F4.
4. Verify on the sample:

```bash
env -u PYTHONPATH uv run python -m mrc.cli train sample45 \
  --source data/raw/cache-trace/samples/2020Mar/cluster045 --limit 1000000
```

   Expected: JSON with `"stages": ["labels","truth","features","fit","eval"]`, four metric dicts per block,
   every `mae` in `[0, 1]`, wall clock at most 4 minutes.
5. Verify the gate: `make check` -> ruff silent, `Success: no issues found`, pytest all pass.
6. Verify tuning persists:

```bash
env -u PYTHONPATH uv run python -m mrc.cli tune sample45 --trials 5
```

   Expected: a `best mae` line and a non-empty `artifacts/optuna/study.db`. The 60-trial run is the
   definition-of-done command, not this gate.

**Gate:** `G-F4-train`, `G-F4-check`.
**Commit:** `feat(cli): train and tune commands, make check gate`
**Rollback:** `git revert <sha>`; cached artefacts stay valid.
**Done when:** both gates pass and `tune` leaves a study that a second invocation extends rather than
duplicates.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-F1-pipeline` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `2 passed` | a stage did not write, or the metric json lacks a block |
| `G-F2-mlflow` | `env -u PYTHONPATH uv run pytest tests/test_pipeline.py -q` | `3 passed` | MLflow is unreachable, or the run is not tagged with the cluster |
| `G-F3-study` | `env -u PYTHONPATH uv run pytest tests/tuning/test_search.py -q` | `1 passed` | the study does not persist or the objective cannot be built |
| `G-F3-remove` | `git rm src/mrc/models/search.py && env -u PYTHONPATH uv run pytest -q` | all pass | something still imports the removed module |
| `G-F4-train` | `env -u PYTHONPATH uv run python -m mrc.cli train sample45 --source data/raw/cache-trace/samples/2020Mar/cluster045 --limit 1000000` | five stages, four models per block, `mae` in `[0, 1]`, at most 4 minutes | the run is slower than the budget (look at grouped row count first), or a metric is `null` |
| `G-F4-check` | `make check` | ruff silent, mypy `Success`, pytest pass | a type error or lint error in a new module |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| Stale stage cache | `features.npz` from an older labels file would be reused silently | `I-F2` compares `labels_sha256`; a mismatch is a miss |
| Cap drifting between saves | the master re-saves labels with `cap=int(rtilde.max())` | `I-F1` and correction 1: `cap=CAP_QUANTILE` always, with an assertion |
| Test block contaminating selection | metrics computed on all rows, or tuning scored on test | `I-F3`, and `tune` scores the validation block only |
| IBS that measures nothing | the master scores the truth against itself | `I-F4` and a test asserting `ibs.test > 0` |
| Memory blow-up at 5 M | 18.65 M grouped rows, 1.23 GB of arrays plus the LightGBM Dataset | `MAX_GROUPED_ROWS` raises with the projected count before allocation; F1 records RSS in the metric json's `seconds` neighbourhood as a card comment, and the run is halved per `logs/DECISIONS.md` if it breaches |
| Optuna mutating params | `params.pop("_n_bins")` inside the objective mutates the dict the factory still reads | pop from a copy; the factory reads `_n_bins` before calling `score_fn` |
| MLflow sqlite lock | 12 threads plus repeated runs on one file | one run per call, no parallel `train` invocations on the same `MLFLOW_URI`; the dispatcher never runs two F4 cards at once |
| `_params` drift from the frozen contract | a worker "improves" the parameters | `_params` is the only source; the reviewer diffs it against master F1's dict |
| Feature-set drift after Phase G | G3 edits `pipeline.py` while F1 owns it | same owner (`mrc-run`), one card at a time, G3 lands after F4's gate |

---

## 6. Reviewer checklist

- [ ] `run_one` writes exactly five stage artefacts and no file outside `data/processed/` and
      `artifacts/`.
- [ ] `cap=CAP_QUANTILE` on every `save_labels` call, with the assertion present.
- [ ] `features.manifest.json` exists and records `labels_sha256` and `feature_version`.
- [ ] Metrics exist for `valid` and `test`, each against `mrc_curve(R[mask], caps)`.
- [ ] `ibs` is built from model hit probabilities; no `np.tile(mrc_true, ...)` anywhere.
- [ ] `NaN` never reaches JSON; `mre_n` accompanies `mre`.
- [ ] `_params()` matches master F1's dict exactly, including `deterministic` and `force_row_wise`.
- [ ] `NUM_THREADS`, `SEED`, `N_BINS`, `N_CAPACITIES` are imported, never literal.
- [ ] The `tune` objective reads cached artefacts only (`I-F7`).
- [ ] `make check` passes and `Makefile` gained exactly `train`, `tune`, `check` plus `.PHONY` entries.

---

## 7. Handoff note

Before `07-phase-g-interpretability.md` starts (edge 7):

1. `run_one` produces `artifacts/tables/<cluster>_metrics.json` with both blocks and a `mlflow_run_id`.
2. The fitted `HazardModel` is reachable for G1 without refitting: `run_one` returns
   `{"stages", "metrics", "artifact", "model", "rows", "masks"}` so G1 can score the validation block.
   If returning the model is impractical, G1 refits from the cached artefacts with the frozen `_params`,
   and the handoff must say which.
3. `make check` is green, so Phase G's new modules inherit a clean lint and type baseline.
4. `artifacts/optuna/study.db` exists with at least 5 trials, proving the study persists.

Before `08-phase-h-report.md`:

5. The metric json is the only source for every number in the report, so its schema in section 1 is frozen
   once H3 starts reading it.
