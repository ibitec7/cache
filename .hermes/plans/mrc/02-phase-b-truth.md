# Phase B: exact ground truth

**Phase:** B. **Owner profile:** `mrc-data`. **Defects closed:** D5 in part (the ground-truth artefact
that did not exist; the evaluation half of D5 is Phase E).
**Upstream:** `01-phase-a-labels.md` (edge 1, `A4 -> B3`). **Downstream:** `05-phase-e-evaluation.md`
(gates compare against this curve), `06-phase-f-pipeline.md` (`run_one` writes truth), `08-phase-h-report.md`
(`H1` plots it).

**Estimated wall time:** 3 tasks at 2 to 5 minutes each (master §4), plus B1's bounded 10-minute
libCacheSim attempt (master B1 step 4) and B3's real runs on the four label sets. B2 is pure numpy over
the label columns.
**Compute:** one process, no GPU. `mrc_curve` is a sort plus a search per capacity: O(n log n) on
5,000,000 int64 values, once per cluster.

B1 is off the critical path. It may run at any time, and it may fail without blocking B2, B3 or any
other phase.

---

## 1. Input and output contract

### Inputs

| Input | Path | Provenance |
|---|---|---|
| label sets | `data/processed/<cluster>/labels.npz` | produced by A4; columns per `01-phase-a-labels.md` §1 |
| label manifests | `data/processed/<cluster>/labels.manifest.json` | `cap`, `n_keys`, `n_requests`, `eligible` |
| frozen constants | `src/mrc/config.py` | `N_CAPACITIES`, `N_BINS` |

B2 and B3 read `R` (object reuse distance, int64, `-1` for compulsory) and, for the approximate byte
variant only, `B` and `obj_size`. They never read `rtilde`: the truth curve is a function of the trace,
not of the model's truncation.

### Outputs

`data/processed/<cluster>/truth.npz` and `data/processed/<cluster>/truth.manifest.json`.

| Array | dtype | Shape | Invariant |
|---|---|---|---|
| `capacities` | int64 | `(K,)` | strictly increasing, `min >= 1`, `max <= b_{M-1} - 1` where `b_{M-1} = truth.manifest.json.b_M_minus_1` |
| `mrc` | float64 | `(K,)` | non-increasing in `capacities`, every value in `[0, 1]`, `mrc[-1] >= p_compulsory` |

`K` is the realised grid size, not necessarily `N_CAPACITIES`: `identified_grid` applies `np.unique`
to rounded log spacing, and on cluster045 that yields 22 points for 24 requested (audit M-2). Every
reader takes `K` from the array, never from the constant.

`truth.manifest.json`, required fields:

```
capacities        list[int]  the realised grid
n_capacities      int        == len(capacities)
grid_points_requested int    the N_CAPACITIES value passed in
max_capacity      int        == capacities[-1]
b_M_minus_1       int        the last finite bin edge, edges[-2] (added by this plan, audit M-3)
overflow_edge     int        the edge the grid was truncated below
n_requests        int        == labels["R"].shape[0]
p_compulsory      float      == (labels["R"] < 0).mean()
source_labels     str        path of the labels.npz consumed
labels_sha256     str        64 hex, the value from labels.manifest.json
git_sha           str        short HEAD at write time
created_utc       str        ISO 8601 UTC, seconds
```

Invariants beyond the array table:

- `I-B1` `mrc` equals the exact Mattson stack-distance curve: `mrc_from_labels(labels, caps)` must equal
  `[naive_lru_mrc(keys, sizes, int(c)) for c in caps]` exactly (`array_equal`, not `allclose`) on data
  where both are computable.
- `I-B2` `capacities` is strictly increasing and never reaches the overflow bin, so no reported capacity
  is the extrapolation D6 refuses.
- `I-B3` the manifest recomputes: `n_capacities == len(capacities)`, `max_capacity == capacities[-1]`,
  `n_requests == len(R)`.
- `I-B4` file-byte equality is not an invariant (same zip-timestamp reason as `I-A8`).

---

## 2. Module interface contract

```python
# src/mrc/gt/replay.py
def replay_mrc(keys: np.ndarray, sizes: np.ndarray, capacities: Sequence[int],
               by_bytes: bool = False) -> np.ndarray:
    """Miss ratio at each capacity by single-request LRU replay through libcachesim."""
```

Callers: `tests/gt/test_replay.py` only. Nothing in the pipeline imports it, by design: the ground truth
must not depend on a library the docs mark untrustworthy.

```python
# src/mrc/gt/stackdist.py
def mrc_from_labels(labels: dict, capacities: Sequence[int], by_bytes: bool = False) -> np.ndarray:
    """Miss ratio at each capacity from a label dict produced by compute_reuse (exact for objects)."""

def identified_grid(n_keys: int, overflow_edge: int, points: int = N_CAPACITIES) -> np.ndarray:
    """Log-spaced object capacities strictly below the overflow bin edge."""
```

Callers of `mrc_from_labels`: `written_truth` (B3), `tests/gt/test_stackdist.py`.
Callers of `identified_grid`: `written_truth` (B3), `src/mrc/pipeline.py` (F1 imports it for its truth
stage).

```python
# src/mrc/gt/write.py
def written_truth(out: str | Path, labels: dict, *, n_keys: int, overflow_edge: int,
                  points: int = N_CAPACITIES, source_labels: str | Path = "") -> Path:
    """Write the ground-truth MRC artefact plus its manifest; return the manifest path."""
```

Callers: `src/mrc/cli.py` `truth`, `src/mrc/pipeline.py::run_one` (F1). The `source_labels` keyword is
added by this plan so the manifest can record provenance; it defaults to empty so the master's three
call sites keep working.

```python
# src/mrc/cli.py
def truth(cluster: str, points: int) -> None:
    """Write the ground-truth MRC curve for a cached label set."""
```

Non-goals: no `process_trace` (broken per `docs/DATA.md` §9), no byte curve in the primary artefact, no
use of the truth curve as a training input, no capacity at or above `b_{M-1}`.

---

## 3. Ordered tasks

### B1: libCacheSim prefix cross-check

**Purpose:** an independent miss-ratio computation, so `I-B1` is corroborated by a second code path
rather than by the same numba kernel plus a different aggregation.
**Files:** create `src/mrc/gt/__init__.py` (empty), `src/mrc/gt/replay.py`, `tests/gt/test_replay.py`.
**Depends on:** nothing.

**TDD steps**

1. Failing test, as master B1 gives it:

```python
import numpy as np
from mrc.data.reference import naive_lru_mrc
from mrc.gt.replay import replay_mrc


def test_replay_matches_naive_on_toy():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 20, 500)
    sizes = rng.integers(8, 64, 500)
    caps = np.array([1, 2, 5, 10, 50])
    assert np.allclose(replay_mrc(keys, sizes, caps),
                       [naive_lru_mrc(keys, sizes, int(c)) for c in caps], atol=0.0)
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.gt'`.
3. Implement. The master's sketch constructs `lc.Request(int(k), int(k))`. Measured on this host
   (`libcachesim 0.3.5`): that call raises `TypeError: incompatible constructor arguments`, because the
   first positional parameter is `obj_size` and the second is `op`, which does not accept an `int`.
   The working form is the keyword form. Use it, and set `obj_size=1` so the replay is a unit-size
   cache, matching `naive_lru_mrc(..., by_bytes=False)`:

```python
def _run_one(keys: np.ndarray, capacity: int) -> float:
    """Miss ratio of one unit-size LRU cache replayed over `keys`."""
    import libcachesim as lc

    cache = lc.LRU(cache_size=int(capacity))
    misses = 0
    for k in keys:
        req = lc.Request(obj_id=int(k), obj_size=1)
        if not cache.get(req):
            misses += 1
            cache.insert(req)
    return misses / len(keys)
```

   Measured behaviour of the keyword form on this host: keys `[5, 6, 5, 7, 5]` at `cache_size=2` gives
   2 hits and 3 misses, which is the LRU answer.
4. Run: `env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q` -> `1 passed`. If the constructor
   still cannot be made to work within 10 minutes, delete the module and the test, and record the
   failure and the exact error in `logs/WORKLOG.md`. Ground truth does not depend on this task.
5. Optional corroboration, not a gate: replay the 1 M-request `sample45` prefix at three capacities and
   compare to `truth.npz`. Record the numbers in the card comment, labelled untrusted if they disagree.

**Gate:** `G-B1-replay` (`1 passed`).
**Commit:** `feat(gt): libCacheSim single-request replay as an independent cross-check`
**Rollback:** `git rm -r src/mrc/gt/replay.py tests/gt/test_replay.py`; nothing imports it.
**Done when:** the toy test passes with `atol=0.0`, or the module is deleted with a WORKLOG entry naming
the error.

---

### B2: exact MRC from stack distance, and the identification-limited grid

**Purpose:** the real ground truth: `MRC(C) = P(R >= C)` under LRU, exact for object capacities.
**Files:** create `src/mrc/gt/stackdist.py`, `tests/gt/test_stackdist.py`.
**Depends on:** A1 to A3 (needs `compute_reuse` with `cap` and the label columns); A2 pins the distance
semantics this task inherits.

**TDD steps**

1. Failing test, as master B2 gives it, with one added assertion on the realised count:

```python
import numpy as np
from mrc.data.reference import naive_lru_mrc
from mrc.data.reuse import compute_reuse
from mrc.gt.stackdist import mrc_from_labels, identified_grid


def test_objects_mrc_is_exact_against_replay():
    rng = np.random.default_rng(1)
    keys = rng.integers(0, 30, 800)
    sizes = rng.integers(8, 64, 800)
    d = compute_reuse(keys, sizes, 30, cap=4095)
    caps = np.array([1, 2, 4, 8, 16, 64])
    got = mrc_from_labels(d, caps, by_bytes=False)
    want = np.array([naive_lru_mrc(keys, sizes, int(c)) for c in caps])
    assert np.array_equal(got, want)


def test_grid_respects_the_identification_limit():
    caps = identified_grid(n_keys=1_000_000, overflow_edge=1024, points=12)
    assert caps.max() < 1024
    assert caps.min() >= 1
    assert np.all(np.diff(caps) > 0)


def test_grid_is_unique_collapsed_not_padded():
    caps = identified_grid(n_keys=173_874, overflow_edge=837, points=24)
    assert caps.max() == 836
    assert len(caps) <= 24
    assert len(caps) == len(np.unique(caps))
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/gt/test_stackdist.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.gt.stackdist'`.
3. Implement exactly as master B2 gives it. `mrc_from_labels` delegates to `mrc_curve`, which already
   sends compulsory requests to a sentinel so they miss at every capacity; do not re-implement that.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/gt/test_stackdist.py -q` -> `3 passed`
   (`pytest tests/gt -q` -> `4 passed` with B1 landed).

**Gate:** `G-B2-stackdist` (`4 passed` with B1, `3 passed` without).
**Commit:** `feat(gt): exact stack-distance MRC and identification-limited grid`
**Rollback:** `git revert <sha>`; B3 does not exist yet.
**Done when:** `array_equal` holds against `naive_lru_mrc` on the toy, and the grid never reaches the
overflow edge.

---

### B3: the truth artefact, `mrc truth`, `make truth`

**Purpose:** one curve per cluster, with provenance, that E1, F1 and H1 all read.
**Files:** create `src/mrc/gt/write.py`, `tests/gt/test_truth_artifact.py`; modify `src/mrc/cli.py`,
`Makefile` (shared surfaces).
**Depends on:** A4 (edge 1, `A4 -> B3`), B2.

**Two corrections carried into this task**

1. The master's test asserts `meta["n_capacities"] == meta["capacities"].__len__()`. Lists have no
   `__len__` attribute, so the test raises `AttributeError` before it can fail on anything scientific.
   The master calls this "the one deliberate trap"; it is written as `len(...)` here. The commit body
   records the deviation.
2. The master's CLI derives the grid from
   `edge = int(2 ** np.ceil(np.log2(max(lab["rtilde"].max(), 2)))) + 1`, which on cluster045 gives 1025
   and therefore capacities up to 1024. `predict_mrc` (D5) raises for any capacity `>= edges[-2] = 837`,
   so that grid would produce a truth curve the estimator cannot be scored on. The audit decision
   (M-3) is that the grid comes from the same edges the fit uses: `overflow_edge = int(edges[-2])`, so
   `max G = 836`.

**TDD steps**

1. Failing test:

```python
import json
import numpy as np

from mrc.gt.write import written_truth


def test_truth_artifact_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    labels = {"R": rng.integers(-1, 500, 4000), "B": rng.integers(-1, 500, 4000),
              "obj_size": rng.integers(8, 64, 4000), "n_keys": np.array([2000])}
    man = written_truth(tmp_path / "c45.npz", labels, n_keys=2000, overflow_edge=1024)
    meta = json.loads(man.read_text())
    assert meta["n_capacities"] == len(meta["capacities"])
    assert meta["b_M_minus_1"] == 1024
    z = np.load(tmp_path / "c45.npz")
    assert z["mrc"].shape == z["capacities"].shape
    assert np.all(np.diff(z["mrc"]) <= 1e-12)
    assert 0.0 <= float(z["mrc"].min()) and float(z["mrc"].max()) <= 1.0
    assert z["capacities"].max() <= meta["b_M_minus_1"] - 1
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/gt/test_truth_artifact.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.gt.write'`.
3. Implement `written_truth` per master B3, plus the manifest fields listed in section 1
   (`b_M_minus_1`, `grid_points_requested`, `source_labels`, `labels_sha256`, `git_sha`,
   `created_utc`). Then the CLI:

```python
    lab = load_labels(PROCESSED / cluster / "labels.npz")
    edges = bin_edges(int(np.percentile(lab["rtilde"][lab["eligible"]], 99.9)), N_BINS)
    proven = PROCESSED / cluster / "labels.manifest.json"
    written_truth(PROCESSED / cluster / "truth.npz", lab,
                  n_keys=int(lab["n_keys"][0]), overflow_edge=int(edges[-2]),
                  points=points, source_labels=proven)
```

   The Makefile target is the master's `truth:` line verbatim.
4. Verify:
   `env -u PYTHONPATH uv run python -m mrc.cli truth sample45`, then
   `env -u PYTHONPATH uv run python -c "import numpy as np; z=np.load('data/processed/sample45/truth.npz'); print(z['capacities'].tolist()); print(np.round(z['mrc'],4).tolist())"`
   Expected: `K` strictly increasing capacities, `K` non-increasing ratios in `[0, 1]`, `mrc[0] > 0.95`,
   `K` equal to the manifest's `n_capacities` (22 on cluster045 per audit M-2; the master's "24" is the
   requested resolution).
5. Run on the four label sets from `G-LABELS-MULTI`, then the transfer check `G-GRID-TRANSFER`.

**Gate:** `G-B3-artifact` (`1 passed`), `G-B3-curve`, `G-GRID-TRANSFER`.
**Commit:** `feat(gt): truth artefact with manifest, CLI and make target`
**Rollback:** `git revert <sha>` and delete `data/processed/*/truth.*`; no consumer exists until E1.
**Done when:** the artefact exists for all four clusters, every manifest records `b_M_minus_1`, and the
single shared grid is re-frozen in `logs/DECISIONS.md` with its realised count.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G-B1-replay` | `env -u PYTHONPATH uv run pytest tests/gt/test_replay.py -q` | `1 passed` | the replay and the naive reference disagree, so either the capacity semantics or the request construction is wrong; the naive reference is the arbiter |
| `G-B2-stackdist` | `env -u PYTHONPATH uv run pytest tests/gt -q` | `4 passed` | `mrc_from_labels` is not exactly the LRU curve; check that compulsory requests go through `mrc_curve`'s sentinel and are not dropped |
| `G-B3-artifact` | `env -u PYTHONPATH uv run pytest tests/gt/test_truth_artifact.py -q` | `1 passed` | the manifest is incomplete or the curve is not monotone |
| `G-B3-curve` | `env -u PYTHONPATH uv run python -m mrc.cli truth sample45` and the print command above | `K` capacities, non-increasing `mrc`, `mrc[0] > 0.95` | `mrc[0] <= 0.95` means compulsory requests are being counted as hits, which is a `mrc_curve` misuse, not a data problem |
| `G-GRID-TRANSFER` | `env -u PYTHONPATH uv run python -c "import json,glob; [print(p, json.load(open(p))['max_capacity']) for p in sorted(glob.glob('data/processed/*/truth.manifest.json'))]"` | one line per cluster, every `max_capacity` below that cluster's `b_M_minus_1` | two clusters imply different grids, so curves are not comparable and the frozen "one grid" contract is broken |

---

## 5. Phase risks and failure modes

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| API drift in libCacheSim | the constructor signature differs from the master's sketch (measured: positional form raises) | B1 uses keywords, has a 10-minute bound, and may be deleted; no other task imports it |
| Grid wider than the estimator's identified range | master's `2**ceil(log2(...))+1` yields 1025, giving capacities up to 1024 while `predict_mrc` refuses `>= 837` | B3 derives `overflow_edge` from the fit-time `edges[-2]`, and the test asserts `max <= b_M_minus_1 - 1` |
| Byte MRC presented as exact | `mrc_curve(..., own_size=...)` ignores that an object larger than the cache cannot be admitted | the primary artefact is object-capacity only; `by_bytes` exists but B3 never writes it; OPEN_QUESTIONS Q1 governs any byte figure |
| Realised grid shorter than requested | `np.unique` collapses rounded `geomspace` duplicates (24 requested, 22 realised on cluster045) | every reader takes `K` from the array; the manifest records both `n_capacities` and `grid_points_requested`; B2 has an explicit test for it |
| Grid that changes when labels change | grid derived from `percentile(rtilde, 99.9)`, which depends on `limit` | the manifest records `source_labels` and `labels_sha256`, so a grid drawn from different labels is detectable; the transfer check compares all clusters at the same `limit` |
| Silent use of the truth curve as a feature | a worker tempted to add `mrc_true` to the design matrix | out-of-scope list in `00-index.md` §9 and the reviewer checklist |

---

## 6. Reviewer checklist

- [ ] `mrc_from_labels` is exact (`array_equal`) against `naive_lru_mrc`, not "close".
- [ ] The artefact contains only object capacities; no `by_bytes=True` call reaches `written_truth`.
- [ ] `capacities.max() <= b_M_minus_1 - 1`, and `b_M_minus_1` is recorded, not recomputed at read time.
- [ ] `K` is read from the array everywhere; no code assumes 24.
- [ ] The manifest's `p_compulsory` equals `(R < 0).mean()` on the same labels that produced the curve.
- [ ] `written_truth` records `source_labels` and `labels_sha256`.
- [ ] Nothing in `src/mrc/gt/` imports `libcachesim` outside `replay.py`.
- [ ] If B1 was deleted, `logs/WORKLOG.md` names the exact error and the date.
- [ ] `truth.npz` is not written under `data/raw/` and not committed (`.gitignore` covers
      `data/processed/`; verify with `git status --porcelain`).

---

## 7. Handoff note

Before `03-phase-c-features.md` starts, only one item is required (C depends on A4, not on B):

1. `data/processed/sample45/labels.npz` exists with the manifest fields from Phase A.

Before `05-phase-e-evaluation.md` and `06-phase-f-pipeline.md`:

2. `truth.npz` exists for every cluster that will be evaluated, each with `b_M_minus_1` in its manifest.
3. The shared grid is frozen in `logs/DECISIONS.md` with its realised count and the rule
   `max G <= min over clusters (b_{M-1}) - 1`.
4. B1 is either green or deleted with a WORKLOG line. Its state does not block anything, and the handoff
   must say which of the two it is so no reader assumes a corroborating path exists.
