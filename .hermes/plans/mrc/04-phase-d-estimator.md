# Phase D: the estimator

**Phase:** D. **Owner profile:** `mrc-run`. **Defects closed:** D2 (censored rows expanded to their backward
distance), D3 (row-order split leaks keys), D6 (`predict_mrc` clips instead of raising), plus the
unlisted `assign_bins` off-by-one (audit M-4).
**Upstream:** `03-phase-c-features.md` (edge 3, `C1 -> D1`), `02-phase-b-truth.md` (grid and
`b_{M-1}`). **Downstream:** `05-phase-e-evaluation.md`, `06-phase-f-pipeline.md` (edge 5, `D4 -> F1`;
and D5's guard is a precondition of F1), `08-phase-h-report.md`.

**Estimated wall time:** 5 tasks at 2 to 5 minutes each (master §4), dominated by D4's LightGBM fits
(60 and 20 boosting rounds on synthetic rows). D2 expands about 21 M rows on the master's measured
ratio (105.9 per request) for 200,000 requests.
**Compute:** 4 to 12 threads depending on the test fixture (`num_threads` in each fixture), no GPU.
`MAX_GROUPED_ROWS = 60,000,000` is the memory guard; the 66 byte per row lower bound in
`logs/DECISIONS.md` puts 18.65 M rows (5 M requests) at 1.23 GB of array data.

---

## 1. Input and output contract

### Inputs

| Input | Shape and dtype | Source |
|---|---|---|
| `X` | float32 `(n, 12)` | `build_features` (C1) |
| `rtilde` | int64 `(n,)`, `-1` for ineligible | labels (A3) |
| `delta` | bool `(n,)` | labels (A3) |
| `edges` | int64 `(M+1,)` | `bin_edges(int(percentile(rtilde[eligible], 99.9)), N_BINS)`; `M` realised as `len(edges)-1`, 31 on cluster045 (audit M-2) |
| `n_requests` | int | `len(R)` from the labels; the split boundary needs it |

`b_{M-1} = int(edges[-2])` is the identification limit: measured 837 on cluster045. Nothing below the
split may use `edges[-1]` as an identified step.

### Outputs

`grouped_rows` returns a dict, arrays row-aligned:

| Key | dtype | Shape | Invariant |
|---|---|---|---|
| `X` | float32 | `(n_rows, n_features)` | row `k` belongs to `request_index[k]` |
| `bin` | int16 | `(n_rows,)` | `0 <= bin <= M-1`, non-decreasing within a request |
| `y` | float32 | `(n_rows,)` | exactly one `1` per observed event, `0` on every row of a censored request |
| `w` | float32 | `(n_rows,)` | bin width, clipped at 1, `w >= 1` |
| `request_index` | int64 | `(n_rows,)` | index into the **eligible** subset, non-decreasing |
| `eligible_index` | int64 | `(n_eligible,)` | original request indices, strictly increasing |
| `n_rows` | int | scalar | `== int((bin+1).sum())` |

`per_step_rows` returns `X`, `step` (int32), `y` (float32), `request_index`, `eligible_index`, `n_rows`.
`block_masks(n)` returns three bool arrays of shape `(n,)`.
`HazardModel.fit_grouped` returns `self` and sets `self.booster`, `self.edges`, `self.feature_names`,
`self.validation_logloss`.
`HazardModel.predict_mrc(X, capacities, edges, p_compulsory)` returns float64 `(len(capacities),)`.

Invariants:

- `I-D1` every eligible request appears, and only eligible requests appear, at the row set level:
  `len(eligible_index) == int((rtilde >= 0).sum())`.
- `I-D2` `y.sum() == int(delta.sum())`, and each request has at most one positive row.
- `I-D3` the split is by **original request index** with boundaries `int(n_requests * SPLIT_TRAIN)` and
  `int(n_requests * SPLIT_VALID)`, computed by `block_masks(n_requests)`. Train indices are all strictly
  below every validation index, which are all strictly below every test index (audit M-18).
- `I-D4` `S_R` is non-increasing in step and lies in `(0, 1]`.
- `I-D5` `predict_mrc` raises `ValueError` for any capacity `>= b_{M-1} = edges[-2]`. No `np.clip` on a
  capacity.
- `I-D6` no row from the test block is read during fitting or early stopping. Early stopping watches the
  validation block only.
- `I-D7` `per_step_rows` refuses when either the request count exceeds `PER_STEP_MAX_REQUESTS` or the
  projected row count exceeds `MAX_GROUPED_ROWS`.
- `I-D8` a refit with the same `SEED`, `num_threads`, `deterministic=True`, `force_row_wise=True`
  produces predictions equal to the previous fit under `np.array_equal` (frozen contract row
  "Determinism").

---

## 2. Module interface contract

```python
# src/mrc/models/risk_set.py
def grouped_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray,
                 edges: np.ndarray, n_features: int) -> dict:
    """Expand eligible requests into one row per (request, bin at risk); canonical estimator input."""

def per_step_rows(X: np.ndarray, rtilde: np.ndarray, delta: np.ndarray,
                  edges: np.ndarray | None = None) -> dict:
    """Exact per-step expansion; ablation only, refuses above the frozen guards."""
```

Callers of `grouped_rows`: `HazardModel.fit_grouped` (D4), `src/mrc/cli.py` `tune` (F4),
`src/mrc/pipeline.py::run_one` (F1), tests. Callers of `per_step_rows`: D2's agreement test and the
ablation table only.

```python
# src/mrc/models/hazard.py
def block_masks(n: int, train: float | None = None, valid: float | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous request-index masks for train / valid / test."""

class HazardModel:
    validation_logloss: float | None

    def fit_grouped(self, rows: dict, edges: np.ndarray, feature_names: list[str],
                    num_boost_round: int = 400, n_requests: int | None = None) -> "HazardModel":
        """Fit on grouped rows with a contiguous original-request-index split."""

    def predict_survival(self, X: np.ndarray, steps: np.ndarray) -> np.ndarray:
        """S_R(k | x) for every request at every step k in `steps` (sorted)."""

    def predict_mrc(self, X: np.ndarray, capacities: np.ndarray, edges: np.ndarray,
                    p_compulsory: float = 0.0) -> np.ndarray:
        """MRC(C) = p_comp + (1 - p_comp) * E_x[S_R(bin(C - 1) | x)]; raises at the identification limit."""
```

Callers of `fit_grouped`: `run_one` (F1), `cli.py` `tune` (F4), `tests/models/test_hazard_model.py`.
Callers of `predict_mrc`: `run_one` (F1), `cli.py` `tune` (F4), `eval/baselines.py` (E3 does not call it,
but `single_row_ablation` mirrors its algebra), `tests`.
`fit` (the existing method) stays until F1 removes its last caller, which is `cli.py fit`.

Two signature additions relative to the master, both forced and both recorded:

- `fit_grouped(..., n_requests: int | None = None)`: the master derives `n_req` from
  `rows["eligible_index"].max() + 1`, which is the eligible-subset bound, not the request count, and
  therefore gives a different boundary than `block_masks(n)`. `n_requests` lets both use one boundary
  function (audit M-18). If omitted, `fit_grouped` falls back to `eligible_index.max() + 1` and records
  that it did.
- `per_step_rows(..., edges: np.ndarray | None = None)`: unused by the row construction (per-step rows
  carry `step`, not `bin`), accepted so the ablation can be called through the same adapter. If it is
  never needed, drop it rather than inventing a use.

---

## 3. Ordered tasks

### D1: grouped risk-set rows, and the `assign_bins` clip

**Purpose:** the canonical estimator input, at 28.4x fewer rows than the per-step expansion (measured on
cluster045: 745,784 vs 21,174,922 for 200,000 requests).
**Files:** create `src/mrc/models/risk_set.py`, `tests/models/test_risk_set.py`; modify
`src/mrc/models/hazard.py` (`assign_bins`).
**Depends on:** C1 (edge 3).

**TDD steps**

1. Failing test, as master D1 gives it (two tests).
2. Verify failure:
   `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.models.risk_set'`.
3. Implement `grouped_rows` and `per_step_rows` per master D1, plus:
   - `assign_bins` must return `np.clip(..., -1, len(edges) - 2)` for the top value. Measured: with
     `edges = [0,1,2,4,8]`, `searchsorted(..., 8, "right") - 1 = 4`, and `nbins = 4`, so index 4 does not
     exist. The master's second test asserts `== 3`; the fix is required before D5's guard can be trusted.
   - `per_step_rows` raises `MemoryError` when `total > MAX_GROUPED_ROWS` as well as when the request
     count exceeds `PER_STEP_MAX_REQUESTS` (`I-D7`). The request guard alone does not bound rows: at
     `cap = 1023` the row count per request can reach 1024.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` -> `2 passed`.

**Gate:** `G-D1-risk` (`2 passed`).
**Commit:** `feat(risk-set): grouped expansion as canonical, per-step as bounded ablation`
(The `assign_bins` fix is in the same commit and named in its body.)
**Rollback:** `git revert <sha>`; `assign_bins` returns to `len(edges)-1` and D5 must not land on top of it.
**Done when:** both tests pass and `assign_bins` never returns `len(edges)-1`.

---

### D2: grouped and per-step cover the same events

**Purpose:** prove the grouped likelihood is an approximation of the same observation scheme, not a
different one.
**Files:** `tests/models/test_risk_set.py` (append).
**Depends on:** D1.

**TDD steps**

1. Failing test, as master D2 gives it.
2. Verify failure: `1 failed` (`assert set(...) == set(...)` fires before the fix only if D1 is wrong;
   with a correct D1 this is a characterisation run and passes first time).
3. Implementation: none.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` -> `3 passed`.

**Gate:** `G-D2-agree` (`3 passed`).
**Commit:** `test(risk-set): grouped and per-step cover the same events`
**Rollback:** `git revert <sha>`.
**Done when:** identical `eligible_index` sets, identical `y.sum()`, and `n_rows` demonstrably different.

---

### D3: contiguous time-block split masks

**Purpose:** one definition of the split, by request index, that every stage uses.
**Files:** modify `src/mrc/models/hazard.py`; create `tests/models/test_blocks.py`.
**Depends on:** D1 (the rows it splits exist).

**TDD steps**

1. Failing test, as master D3 gives it (two tests).
2. Verify failure: `ImportError: cannot import name 'block_masks'`.
3. Implement `block_masks` exactly as master D3 gives it, importing `SPLIT_TRAIN` and `SPLIT_VALID` from
   `mrc.config` inside the function (the master's sketch does this; keep it, it avoids a circular import
   at module load).
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/models/test_blocks.py -q` -> `2 passed`.

**Gate:** `G-D3-blocks` (`2 passed`).
**Commit:** `feat(hazard): contiguous time-block split masks, fixes row-order leakage`
**Rollback:** `git revert <sha>`; `fit` still uses the row cut and nothing new depends on the masks.
**Done when:** masks are contiguous, ordered, disjoint and total.

---

### D4: `fit_grouped` on grouped rows with block splits

**Purpose:** fit the hazard on (request, bin) rows, with early stopping on the validation block only.
**Files:** modify `src/mrc/models/hazard.py`.
**Depends on:** D1 (edge 4), D2, D3.

**TDD steps**

1. Failing test, as master D4 gives it (`_synthetic` plus the survival-shape assertions), with one
   addition that pins `I-D3`:

```python
def test_split_boundaries_use_original_request_indices():
    X, rtilde, delta, edges = _synthetic(6_000)
    rows = grouped_rows(X, rtilde, delta, edges, 4)
    m = HazardModel(params={"objective": "binary", "verbosity": -1, "num_threads": 2,
                           "seed": 7, "deterministic": True, "force_row_wise": True},
                    nbins=len(edges) - 1)
    m.fit_grouped(rows, edges, [f"f{i}" for i in range(4)],
                  num_boost_round=10, n_requests=6_000)
    tr, va, te = block_masks(6_000)
    assert int(tr.sum()) == 4_200 and int(va.sum()) == 900 and int(te.sum()) == 900
    # the fit records which boundary function it used
    assert m.split_n == 6_000
```

2. Verify failure: `AttributeError: 'HazardModel' object has no attribute 'fit_grouped'`.
3. Implement. The master's `fit_grouped` body, with the split changed from
   `req < i_tr` on the eligible-subset index to the original request index (`I-D3`):

```python
        orig = rows["eligible_index"][rows["request_index"]]
        tr, va, _ = block_masks(n_req)
        tr, va = tr[orig], va[orig]
```

   where `n_req = n_requests if n_requests is not None else int(rows["eligible_index"].max()) + 1`. Also
   add the dataclass field `validation_logloss: float | None = None` and `split_n: int | None = None`.
   Early stopping is on the validation block only; the test block is never passed to `lgb.train`.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/models/test_hazard_model.py -q` -> `2 passed`.
5. Determinism check (`I-D8`), planner-added because the frozen contract claims bit reproducibility:

```bash
env -u PYTHONPATH uv run python -c "
import numpy as np, lightgbm as lgb
print(lgb.__version__)
"
```

   and in the test file, fit twice on the same synthetic rows with the same params and assert
   `np.array_equal(p1, p2)` on `predict_mrc` output. If it fails, the cause is thread count or
   `force_row_wise`; fix the parameter, never the assertion.

**Gate:** `G-D4-fit` (`2 passed`).
**Commit:** `feat(hazard): fit_grouped with contiguous original-request-index validation split`
**Rollback:** `git revert <sha>`; `fit_grouped` disappears and F1 cannot start.
**Done when:** the survival is monotone and in `(0,1]`, `validation_logloss` is recorded, and the split
boundaries are the ones `block_masks` produces.

---

### D5: `predict_mrc` raises past the identification limit

**Purpose:** stop the model from inventing a number where the data cannot identify it (D6).
**Files:** modify `src/mrc/models/hazard.py`; append to `tests/models/test_hazard_model.py`.
**Depends on:** D4.

**TDD steps**

1. Failing test, as master D5 gives it, with one corrected assertion (audit M-20). The master writes
   `assert np.all(np.diff(curve) >= -1e-9)` and labels it "non-increasing in capacity"; the curve **is**
   non-increasing, so the sign must be `<=`:

```python
    assert np.all(np.diff(curve) <= 1e-9)           # non-increasing in capacity
```

   With the master's sign the test fails against a correct implementation, which would send a worker
   hunting a survival bug that does not exist.
2. Verify failure: `Failed: DID NOT RAISE`.
3. Implement the master's `predict_mrc` replacement. Note the guard is `capacities.max() >= top` with
   `top = int(edges[-2])`, which is the M-3 reading: the last edge below the overflow bin. Keep
   `assign_bins(capacities - 1, edges)` without a clip on the capacity itself; clipping the resulting
   bin index to the valid step range is fine because the capacity is already below the limit.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/models -q` -> `8 passed`
   (`risk_set` 3 + `blocks` 2 + `hazard_model` 3).
5. Boundary evidence, recorded in the card comment: on cluster045, `edges[-2] = 837`, so capacity 836 is
   accepted and 837 raises. This is the number every figure caption must state (master §7.2.2).

**Gate:** `G-D5-guard` (`8 passed`).
**Commit:** `fix(hazard): raise past the identification limit instead of clipping`
**Rollback:** `git revert <sha>`; the clip returns and every reported number at large capacity becomes an
extrapolation.
**Done when:** no `np.clip` remains on a capacity path, and the `ValueError` message names the capacity
and the edge.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-D1-risk` | `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` | `2 passed` | the row count, the label placement or the bin width is wrong; `assign_bins` returning `len(edges)-1` is the usual cause |
| `G-D2-agree` | `env -u PYTHONPATH uv run pytest tests/models/test_risk_set.py -q` | `3 passed` | grouped and per-step disagree on which requests or which events exist, so one of them drops censored rows |
| `G-D3-blocks` | `env -u PYTHONPATH uv run pytest tests/models/test_blocks.py -q` | `2 passed` | the boundaries are off by one or the masks overlap |
| `G-D4-fit` | `env -u PYTHONPATH uv run pytest tests/models/test_hazard_model.py -q` | `2 passed` | the survival is not monotone (the product form is being computed over unsorted steps) or the split uses the wrong index |
| `G-D5-guard` | `env -u PYTHONPATH uv run pytest tests/models -q` | `7 passed` | `predict_mrc` still clips; every curve above capacity 836 is then a fabricated number |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| Censored requests leaking their backward distance into the risk set | D2's original bug: `expand_risk_set` ignores `censor` | grouped rows are built from `rtilde`, which is `min(gamma, cap)` for censored requests; D1's second test asserts the overflow bin, D2 asserts the event count |
| Split boundary disagreement between hazard and baselines | master's `fit_grouped` splits the eligible subset, `block_masks` splits all requests | `I-D3` and the `n_requests` keyword force one boundary function; D4's added test pins the counts 4,200 / 900 / 900 on 6,000 |
| Row explosion in the ablation | `per_step_rows` guards requests (200,000) but not rows, and one request can contribute 1,024 rows | `I-D7` adds the `MAX_GROUPED_ROWS` guard to `per_step_rows` |
| Memory on the 5 M window | 18.65 M grouped rows at 66 bytes each is 1.23 GB of array data, before LightGBM's internal Dataset | `MAX_GROUPED_ROWS` raises with the projected count; the cost projection is in `logs/DECISIONS.md`; F1 measures RSS |
| Early stopping on the test block | passing `dvalid` built from test rows would make the reported test number optimistic | `I-D6`; the test asserts only two masks are used, and the reviewer reads the `lgb.train` call |
| Nondeterministic fits | LightGBM with `num_threads > 1` can vary unless `deterministic` and `force_row_wise` are set | `I-D8` plus the double-fit check; the frozen contract already fixes both flags |
| Overflow bin treated as identified | the final bin absorbs every capped distance, so survival below it is not identified | D5 raises at `edges[-2]`; B3's grid stops at `edges[-2]-1` |

---

## 6. Reviewer checklist

- [ ] `expanded` rows are built from `rtilde`, never from `R`.
- [ ] `y` has at most one positive row per request, and `y.sum()` equals `delta.sum()`.
- [ ] `w` is the bin width clipped at 1, identical for all rows in the same bin.
- [ ] `assign_bins` clips to `len(edges) - 2`, and the clip is visible in the diff.
- [ ] The split uses `block_masks(n_requests)` and original request indices, not the eligible subset.
- [ ] The test block is not passed to `lgb.train` in any form.
- [ ] `predict_mrc` has no capacity clip; the `ValueError` message names both numbers.
- [ ] `validation_logloss` is read from `best_score` and stored on the model.
- [ ] Two fits with the same parameters give identical predictions.
- [ ] `fit` (the old method) is untouched except for the new dataclass fields, because `cli.py fit` still
      calls it until F1.

---

## 7. Handoff note

Before `05-phase-e-evaluation.md` and `06-phase-f-pipeline.md`:

1. `grouped_rows` and `per_step_rows` exist with the schemas in section 1, and `assign_bins` clips.
2. `block_masks(n)` is the only split definition; `fit_grouped` accepts `n_requests` and uses it.
3. `predict_mrc` raises at `edges[-2]`; F1 must therefore build its capacity grid from the truth manifest,
   which is already truncated to `b_{M-1} - 1` (Phase B).
4. The measured identification limit `b_{M-1} = 837` on cluster045 is quoted in the handoff so Phase F and
   Phase H captions cannot silently drop it.
5. `validation_logloss` is on the model and is what MLflow logs as the fit's own metric, separate from the
   MRC MAE that Optuna minimises.
