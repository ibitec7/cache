# Phase A: labels with an honest censoring bound

**Phase:** A. **Owner profile:** `mrc-data`. **Defects closed:** D1 (no censoring bound `Gamma_i`).
**Upstream:** `00-index.md` (contracts, ownership, global gates).
**Downstream:** `02-phase-b-truth.md` (edge 1, `A4 -> B3`), `03-phase-c-features.md` (edge 2, `A4 -> C1`).

**Estimated wall time:** 4 tasks at the master's per-task budget of 2 to 5 minutes (master §4 opening
line), so 8 to 20 minutes of agent work, plus one real-data label run whose kernel is measured at
2.58 M req/s (master §1.1, `reuse.py` row), i.e. sub-second of kernel time on a 1 M-request prefix; the
run is dominated by decoding a 34,557,882 byte sample (`ls -la
data/raw/cache-trace/samples/2020Mar/cluster045`).
**Compute:** one process, no GPU, no parallelism. Memory is the label columns only: 11 int64 columns at
5,000,000 rows x 8 B = 440 MB, plus 4 shallow columns.

---

## 1. Input and output contract

### Inputs

| Input | Path | Provenance |
|---|---|---|
| development sample | `data/raw/cache-trace/samples/2020Mar/cluster045` | 34,557,882 bytes, plain text, comma separated, no header; column order fixed by `src/mrc/data/trace.py` `COLUMNS` |
| tier-1 shard | `data/raw/traces/cluster45.sort.zst` | 1,872,045,622 bytes, sha1 `10a0ef3ca35d8644374b00e35caf9d1216c76a19` matches `data/raw/traces/UPSTREAM.sha1` (verified at audit) |
| frozen constants | `src/mrc/config.py` | created by A3, verbatim from master §2 |

### Outputs

`data/processed/<cluster>/labels.npz` and `data/processed/<cluster>/labels.manifest.json`.

`labels.npz`, every array shape `(n,)` unless stated. `n` is the number of requests in the prefix.

| Column | dtype | Invariant |
|---|---|---|
| `R` | int64 | `R >= -1`; `R == -1` iff the request is a first access |
| `B` | int64 | `B >= -1`; `B == -1` iff `R == -1` |
| `gap` | int64 | requests since the previous access, `-1` iff `R == -1` |
| `freq` | int64 | 1-based occurrence index, `freq >= 1` |
| `gamma` | int64 | distinct keys requested strictly after the request, `0 <= gamma <= n_keys` |
| `eligible` | bool | `freq > 1`; exactly the requests that may enter a risk set |
| `delta` | bool | observed event: `eligible & ~is_last`; `delta` implies `eligible` |
| `rtilde` | int64 | `-1` iff not `eligible`; else `min(R, cap)` when `delta`, `min(gamma, cap)` otherwise; always `<= cap` |
| `compulsory` | bool | `R == -1` |
| `ts` | int64 | non-decreasing in the shard |
| `key` | int32 | int codes into the trace vocabulary, unchanged from `Trace.key` |
| `obj_size` | int64 | key_size + value_size, bytes |
| `op` | int8 | code into `trace.OPS` |
| `client` | int32 | client id |
| `ttl` | int64 | seconds, `0` when the request carries none |
| `key_ns` | int32 | **added by this plan** (audit M-17): deterministic code of the first `:`-component of the key string, `>= 1` for a non-empty key, `0` reserved for unknown |
| `key_len` | int32 | **added by this plan**: byte length of the key string, `>= 1` |
| `n_keys` | int64, shape `(1,)` | number of distinct key codes in the prefix |

`labels.manifest.json` fields, all required:

```
source            str   the path exactly as passed to the CLI
npz               str   the artefact path
sha256            str   64 hex chars, sha256 of the .npz file bytes
n_requests        int   == len(R)
n_keys            int   == int(n_keys[0])
limit             int|null  the prefix length used
cap               int   the rtilde ceiling used
eligible          int   == int(eligible.sum())
events            int   == int(delta.sum())
p_compulsory      float == float((~eligible).mean()), 0 <= p_compulsory <= 1
feature_source    str   "key_vocab" (added by this plan, audit M-17)
git_sha           str   short HEAD at write time
created_utc       str   ISO 8601, seconds precision, UTC
```

Invariants the artefact must satisfy, and which the tests assert:

- `I-A1` `delta ⊆ eligible` and `compulsory ∩ eligible = {}`.
- `I-A2` `rtilde == -1` exactly where `~eligible`.
- `I-A3` `rtilde[i] == min(R[i], cap)` where `delta[i]`, else `min(gamma[i], cap)` where `eligible[i]`.
- `I-A4` `gamma` counts keys strictly after `i`, so `gamma[n-1] == 0` always.
- `I-A5` the manifest's `n_requests`, `eligible`, `events`, `p_compulsory` recompute exactly from the
  arrays.
- `I-A6` a rerun with the same `(source, limit, cap)` triple is a cache hit, and a rerun with a
  different `limit` or `cap` is a miss.
- `I-A7` `key_ns` and `key_len` are functions of the key string only, never of the target.
- `I-A8` file-byte equality is NOT an invariant: `np.savez_compressed` writes zip entries with
  timestamps, so two runs on identical data produce different file bytes and different `sha256`.
  Reproducibility claims are made on arrays, via `np.array_equal`, never on the file hash.

---

## 2. Module interface contract

New and changed public surface. One-line docstrings are the contract; the implementation may not widen
them.

```python
# src/mrc/data/reuse.py
def suffix_distinct(keys: np.ndarray, n_keys: int) -> np.ndarray:
    """distinct[i] = number of distinct key codes in keys[i:]; distinct[n] = 0."""

def compute_reuse(keys: np.ndarray, sizes: np.ndarray, n_keys: int,
                  cap: int = CAP_QUANTILE) -> dict[str, np.ndarray]:
    """Reuse distance, censoring bound and the truncation the model consumes (R, B, gap, freq,
    gamma, eligible, delta, rtilde, compulsory, censor)."""
```

Callers of `compute_reuse`: `src/mrc/data/labels.py::build_labels`, `tests/test_smoke.py` (existing,
keeps its 2-argument call working through the default), `src/mrc/cli.py` `reuse` command (unchanged).

```python
# src/mrc/data/labels.py
LABEL_KEYS: tuple[str, ...]     # R, B, gap, freq, gamma, rtilde, eligible, delta, compulsory, key_ns, key_len
REQUEST_KEYS: tuple[str, ...]   # ts, key, obj_size, op, client, ttl

def build_labels(path: str | Path, limit: int | None = None,
                 cap: int = CAP_QUANTILE) -> dict[str, np.ndarray]:
    """Read a shard prefix and return label columns plus the request columns they join to."""

def save_labels(out: str | Path, labels: dict, *, source: str | Path, limit: int | None,
                cap: int) -> Path:
    """Write the npz and its manifest; return the manifest path."""

def load_labels(path: str | Path) -> dict[str, np.ndarray]:
    """Load a label npz without pickling; keys are the array names."""
```

Callers of `build_labels`: `src/mrc/cli.py` `labels`, `src/mrc/pipeline.py::run_one` (F1).
Callers of `save_labels`: same two. Callers of `load_labels`: `cli.py` `truth` (B3), `run_one` (F1),
`cli.py` `tune` (F4).

```python
# src/mrc/cli.py
def labels(path: str, cluster: str, limit: int | None, cap: int) -> None:
    """Compute and cache reuse-distance labels for one shard."""
```

`key_ns` / `key_len` are produced inside `build_labels` from `Trace.key_vocab`; the helper that does it
is private (`_key_codes`), so it is not part of the frozen interface, but its output columns are.

Explicit non-goals of this phase: no binning, no capacity grid, no feature matrix, no model code, no
`data/processed/` write for a cluster whose input path was not named on the command line.

---

## 3. Ordered tasks

### A1: add `gamma`, `delta`, `rtilde` to `compute_reuse`

**Purpose:** give every request an honest horizon, so a censored request can enter a risk set at the
bound the trace actually supports.
**Files:** create `src/mrc/config.py` (verbatim from master §2, moved from A3 by this plan so the default
argument can be the frozen constant), modify `src/mrc/data/reuse.py`; create
`tests/data/test_labels_bound.py`.
**Depends on:** nothing.

**TDD steps**

1. Failing test, `tests/data/test_labels_bound.py`. Five tests. Tests 1, 2, 3 are the master's. Test 4
   is corrected (audit M-15): the master's array `np.repeat(np.arange(50), 2)` gives every eligible
   request `R = 0`, so a `rtilde.max() == 4` assertion cannot pass under any correct implementation.
   Test 5 is new and pins `gamma` on the artefact that exists today.

```python
import numpy as np
from mrc.data.reuse import compute_reuse


def test_gamma_counts_distinct_keys_after_each_request():
    k = np.array([0, 1, 0, 2, 1, 3])
    d = compute_reuse(k, np.ones(6, dtype=np.int64), 4, cap=10)
    assert d["gamma"].tolist() == [4, 4, 3, 2, 1, 0]


def test_delta_true_only_when_next_reference_is_inside_the_window():
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["eligible"].tolist() == [False, False, True, True]
    assert d["delta"].tolist() == [False, False, True, False]


def test_rtilde_uses_forward_bound_when_censored():
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["rtilde"][2] == 1
    assert d["rtilde"][3] == 0
    assert d["rtilde"][0] == -1


def test_rtilde_is_capped_at_the_overflow_edge():
    # corrected: consecutive repeats have R = 0 and never exercise the cap (audit M-15)
    k2 = np.concatenate([np.arange(50), np.arange(50)])
    d2 = compute_reuse(k2, np.ones(100, dtype=np.int64), 50, cap=4)
    assert int(d2["R"][d2["eligible"]].max()) == 49
    assert int(d2["rtilde"].max()) == 4


def test_gamma_is_zero_at_the_last_request_and_never_negative():
    rng = np.random.default_rng(0)
    k3 = rng.integers(0, 40, 2000).astype(np.int64)
    d3 = compute_reuse(k3, np.ones(2000, dtype=np.int64), 40, cap=64)
    assert int(d3["gamma"][-1]) == 0
    assert d3["gamma"].min() >= 0
    assert d3["rtilde"].max() <= 64
```

2. Verify failure:
   `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q`
   Expected: `5 failed`, with `KeyError: 'gamma'` and `TypeError: compute_reuse() got an unexpected
   keyword argument 'cap'`.

3. Implement: `src/mrc/config.py` first and verbatim from master §2 (audit M-26: it lands here, not in A3,
   because `compute_reuse`'s default cannot be a literal). Then `suffix_distinct` exactly as the master
   gives it (numba `njit(cache=True, nogil=True)`, `seen` as `np.zeros(n_keys, dtype=np.bool_)`, reverse
   scan); replace `compute_reuse`, keeping the master's body but with `cap: int = CAP_QUANTILE` imported
   from `mrc.config`. Keep `censor` in the returned dict (`eligible & is_last`) for the existing
   `cli.py fit` caller.
4. Verify pass:
   `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` -> `5 passed`
   `env -u PYTHONPATH uv run pytest -q` -> `15 passed` (10 baseline + 5 new)

**Gate:** `G-A1-bound` (`5 passed`), `G-A1-suite` (`15 passed`).
**Commit:** `feat(labels): add censoring bound, delta and rtilde to compute_reuse`
**Rollback:** `git revert <sha>`; no artefact depends on this commit yet, and the old signature is
restored exactly.
**Done when:** both gates pass and `compute_reuse` accepts `cap` while the existing 2-argument call in
`tests/test_smoke.py` still passes unchanged.

---

### A2: pin the semantics on a hand-derived micro-example

**Purpose:** make the three quantities checkable by hand, so a later refactor cannot silently redefine
censoring.
**Files:** `tests/data/test_labels_bound.py` (append). Test only.
**Depends on:** A1.

**Correction carried into this task (audit M-16).** The master's `ABACABDC` derivation is arithmetically
wrong under its own scheme: it lists four eligible indices and calls them five, it counts keys from
before the previous occurrence, and it counts `D` at index 6 for `B` at index 5. The master also cites
skeleton §2.3, whose trace is `ABACABCB` over three keys, not `ABACABDC` over four, and whose §4.4 event
table censors only cold requests while this pipeline censors every last occurrence. Measured by command:

```
ABACABDC  R = [-1,-1,1,-1,1,2,-1,3]  freq = [1,1,2,1,3,2,1,2]  censor = [F,F,F,F,T,T,F,T]
          eligible 4  delta 1  R[delta] = [1]  gamma = [4,4,4,4,3,2,1,0]
          rtilde(cap 7) = [-1,-1,1,-1,3,2,-1,0]
ABACABCB  R = [-1,-1,1,-1,1,2,2,1]  freq = [1,1,2,1,3,2,2,3]  censor = [F,F,F,F,T,F,T,T]
          eligible 5  delta 2  R[delta] = [1,2]  gamma = [3,3,3,3,2,2,1,0]
          rtilde(cap 3) = [-1,-1,1,-1,2,2,1,0]
```

Derivation for `ABACABDC`, distinct keys strictly inside `(p(i), i)`: index 2 `A` sees `{B}` = 1;
index 4 `A` sees `{C}` = 1; index 5 `B` sees `{A, C}` = 2; index 7 `C` sees `{A, B, D}` = 3. Eligible
requests are indices 2, 4, 5, 7. Indices 4, 5, 7 are each the last occurrence of their key, so they are
censored and only index 2 is an event.

**TDD steps**

1. Failing test (append):

```python
def test_four_key_micro_example_matches_the_hand_table():
    key = {"A": 0, "B": 1, "C": 2, "D": 3}
    k = np.array([key[c] for c in "ABACABDC"], dtype=np.int64)
    d = compute_reuse(k, np.ones(8, dtype=np.int64), 4, cap=7)
    assert d["R"].tolist() == [-1, -1, 1, -1, 1, 2, -1, 3]
    assert int(d["eligible"].sum()) == 4
    assert int(d["delta"].sum()) == 1
    assert d["R"][d["delta"]].tolist() == [1]
    assert d["rtilde"].tolist() == [-1, -1, 1, -1, 3, 2, -1, 0]


def test_three_key_skeleton_trace_censors_every_last_occurrence():
    k = np.array([0, 1, 0, 2, 0, 1, 2, 1], dtype=np.int64)   # A B A C A B C B, K = 3
    d = compute_reuse(k, np.ones(8, dtype=np.int64), 3, cap=3)
    assert int(d["eligible"].sum()) == 5
    assert int(d["delta"].sum()) == 2
    assert sorted(d["R"][d["delta"]].tolist()) == [1, 2]
    assert d["rtilde"].tolist() == [-1, -1, 1, -1, 2, 2, 1, 0]
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` -> `2 failed`
   if A1 is not yet landed; if A1 is landed, this step is a characterisation step and must pass on the
   first run. Expected pass on a correct A1: `7 passed`.
3. Implementation: none.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` -> `7 passed`.

**Gate:** `G-A2-micro` (`7 passed`: the master's five bound tests plus A2's two micro-examples).
**Commit:** `test(labels): pin rtilde against two hand-derived micro-examples`
**Rollback:** `git revert <sha>`; test-only, nothing consumes it.
**Done when:** both arrays above are asserted literally, and no assertion borrows skeleton §4.4's event
counts.

---

### A3: the label cache, its manifest, and `src/mrc/config.py`

**Purpose:** make the label stage a cache with a verifiable identity, and freeze the constants in one
file.
**Files:** `src/mrc/config.py` already exists from A1 (verify it is unchanged and do not re-add it),
create `src/mrc/data/labels.py`, create `tests/data/test_label_cache.py`.
**Depends on:** A1, A2.

**TDD steps**

1. Failing test:

```python
import json
import numpy as np
from mrc.data.labels import build_labels, load_labels, save_labels


def _toy_csv(tmp_path):
    p = tmp_path / "toy.csv"
    p.write_text("0,aaa,10,0,1,get,0\n1,bbb,10,0,1,get,0\n2,aaa,10,0,2,get,0\n"
                 "3,ccc,10,0,1,get,0\n4,aaa,10,0,1,get,0\n")
    return p


def test_build_save_load_roundtrip(tmp_path):
    src = _toy_csv(tmp_path)
    lab = build_labels(src, limit=None, cap=7)
    assert lab["R"].shape == (5,)
    assert lab["rtilde"].dtype == np.int64
    out = tmp_path / "labels.npz"
    man = save_labels(out, lab, source=src, limit=None, cap=7)
    meta = json.loads(man.read_text())
    assert meta["n_requests"] == 5 and meta["cap"] == 7
    assert len(meta["sha256"]) == 64
    assert np.array_equal(load_labels(out)["rtilde"], lab["rtilde"])


def test_cache_columns_are_pickle_free_and_carry_key_geometry(tmp_path):
    lab = build_labels(_toy_csv(tmp_path), limit=None, cap=7)
    assert lab["key"].dtype == np.int32
    assert lab["key_ns"].dtype == np.int32 and lab["key_len"].dtype == np.int32
    assert lab["key_ns"].min() >= 1            # "aaa", "bbb", "ccc" all parse
    assert lab["key_len"].tolist() == [3, 3, 3, 3, 3]
    # the cache must load without allow_pickle
    out = tmp_path / "l.npz"
    save_labels(out, lab, source="toy", limit=None, cap=7)
    assert not any(load_labels(out)[k].dtype == object for k in load_labels(out))
```

2. Verify failure: `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` ->
   `ModuleNotFoundError: No module named 'mrc.data.labels'`.
3. Implement: `config.py` exactly as master §2, no edits. `labels.py` per the master's sketch plus the
   two added columns:

```python
def _key_codes(vocab: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic first-component code and string length per request, from the key vocabulary."""
```

   `_key_codes` builds a code table once per distinct vocabulary entry (first `:`-component, truncated
   to 32 characters as master C1 does, `0` reserved for unknown), then indexes it with `keys`. It is
   O(n_keys + n), not O(n) string hashing. `build_labels` adds `key_ns`, `key_len` to `LABEL_KEYS` and
   the manifest gains `feature_source: "key_vocab"`.
4. Verify pass: `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` -> `2 passed`.

**Gate:** `G-A3-cache` (`2 passed`).
**Commit:** `feat(data): label cache with sha256 manifest and frozen config constants`
(the message is the master's verbatim; `config.py` landed in A1 per audit M-26, and the commit body says so)
**Rollback:** `git revert <sha>`; `labels.py` disappears and A4 cannot run, which is the intended
failure mode.
**Done when:** the manifest carries every field listed in section 1, `load_labels` never needs
`allow_pickle`, and `config.py` is byte-identical to master §2.

---

### A4: `mrc labels` and `make labels`

**Purpose:** one command that produces the artefact, with a cache check that cannot serve a stale file.
**Files:** modify `src/mrc/cli.py` (shared surface), `Makefile` (shared surface), `configs/data/t1.yaml`.
**Depends on:** A3.

**TDD steps**

1. Failing evidence: `make labels` with no target -> `make: *** No rule to make target 'labels'`.
   This task has no unit test of its own; its gate is the real run. Record the pre-state in the commit
   body.
2. Implement the command as master A4 gives it, plus `feature_source` in the echo line is not needed.
   Makefile target and `.PHONY` update as master gives them.
3. `configs/data/t1.yaml`: `limit_requests: 5000000`, add `cap_quantile: 1023`.
4. Verify on a real shard:

```bash
env -u PYTHONPATH uv run python -m mrc.cli labels \
  data/raw/cache-trace/samples/2020Mar/cluster045 --cluster sample45 --limit 1000000
```

   Expected: one line `wrote .../data/processed/sample45/labels.npz: 1000000 requests, E eligible, D
   events; manifest labels.manifest.json` with `E` within 25 % of 130,600 and `D` within 25 % of 88,000
   (rates 13.06 % and 8.80 % measured on `data/interim/s45.npz`, master §1.2, reproduced at audit).
5. Verify the cache: rerun the identical command. Expected `cache hit
   .../labels.npz (1000000 requests)` in under 0.2 s. Then rerun with `--limit 500000` and confirm it is
   a miss, not a hit.

**Gate:** `G-A4-real`, `G-A4-hit`.
**Commit:** `feat(cli): labels command with manifest cache check and make target`
**Rollback:** `git revert <sha>`; delete `data/processed/sample45/` by hand, since artefacts are not
tracked.
**Done when:** both gates pass, the cache-hit path is demonstrated, and the miss path is demonstrated.

---

## 4. Gate table

| Gate | Exact command | Expected output | What a failure means |
|---|---|---|---|
| `G0-env` | `env -u PYTHONPATH uv run pytest -q` | `10 passed` | the baseline is broken before the phase started; stop, do not attribute it to A1 |
| `G-A1-bound` | `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` | `5 passed` | `gamma` or `cap` is missing, or `is_last` is wrong; re-derive from the five tests, do not edit them |
| `G-A1-suite` | `env -u PYTHONPATH uv run pytest -q` | `15 passed` | a caller of `compute_reuse` broke on the new signature; `tests/test_smoke.py` uses the 2-argument call and must still pass |
| `G-A2-micro` | `env -u PYTHONPATH uv run pytest tests/data/test_labels_bound.py -q` | `7 passed` | the hand table and the kernel disagree; the kernel is wrong, because the table was derived independently by command at audit time |
| `G-A3-cache` | `env -u PYTHONPATH uv run pytest tests/data/test_label_cache.py -q` | `2 passed` | the manifest is incomplete, the npz needs pickle, or `key_ns`/`key_len` are absent |
| `G-A4-real` | `env -u PYTHONPATH uv run python -m mrc.cli labels data/raw/cache-trace/samples/2020Mar/cluster045 --cluster sample45 --limit 1000000` | `wrote ...`, `E` near 130,600, `D` near 88,000 | rates far from 13.06 % / 8.80 % mean the trace reader changed column order, or censoring is inverted |
| `G-A4-hit` | same command, rerun | `cache hit ... (1000000 requests)`, under 0.2 s | the cache key ignores a parameter, so stale labels can be served |
| `G-LABELS-MULTI` | `for c in cluster45 cluster26 cluster10; do env -u PYTHONPATH uv run python -m mrc.cli labels data/raw/traces/$c.sort.zst --cluster $c --limit 5000000; done` | three `wrote` lines, `5000000 requests` each | the 5 M window is not reachable on a shard, which invalidates the frozen `LABEL_REQUESTS` contract and must be escalated, not worked around |

---

## 5. Phase risks and failure modes, with the mitigation built into the tasks

| Risk | Mechanism | Mitigation in this design |
|---|---|---|
| Censoring bound computed over the wrong horizon | `gamma` over the whole shard while the model trains on a 1 M prefix silently inflates the identified range | every stage reads the same `limit` from the manifest; A4's cache key includes `limit`; `I-A6` tests the miss path |
| `rtilde` semantics drift between phases | B2's `mrc_from_labels` reads `R`, D1 reads `rtilde`; if A1's censored branch used `R` the two would disagree silently | A2 pins `rtilde` literally, including the censored branch (`rtilde[7] == 0`) |
| File-hash reproducibility trap | `np.savez_compressed` stamps zip entries with the wall clock, so identical data gives different bytes and different `sha256` | `I-A8` states the invariant and the reviewer checklist forbids file-hash equality claims; tests use `np.array_equal` |
| Two of the twelve features being vacuous | the trace stores int key codes, so the master's object-array branch in C1 never runs on real data | A3 adds `key_ns` and `key_len` (audit M-17) and A3's second test asserts them on a real column layout |
| Python loop over 5 M requests for `last_occ` | master's sketch loops in Python for `last_occ`, about 1.5 s at 5 M rows, and `is_last[...]` is O(n_keys) | budget allows it: the reuse kernel dominates nothing here, and the loop is bounded by n. If A4's wall clock on 1 M exceeds 5 s, replace with `np.maximum.at` and record the change in the commit body |
| Writing a cluster directory not named on the command line | the CLI derives the path from `--cluster` | out-of-scope list in `00-index.md` §9; the reviewer checks the diff for a second path |

---

## 6. Reviewer checklist

- [ ] No hardcoded `1023` in `reuse.py` or `labels.py`; `cap` comes from `mrc.config.CAP_QUANTILE`.
- [ ] `gamma` counts strictly after the request (`gamma[-1] == 0`), never including the request itself.
- [ ] `delta` is a subset of `eligible`; no request can be an event and compulsory at once.
- [ ] `rtilde` is `-1` exactly on non-eligible requests; no silent clipping of negative values into 0.
- [ ] `key_ns` and `key_len` are computed from the vocabulary only; a shuffled target would not change
      them.
- [ ] The npz loads without `allow_pickle`, and no object-dtype array is written.
- [ ] The manifest recomputes from the arrays: `n_requests`, `eligible`, `events`, `p_compulsory`.
- [ ] The cache key is `(source, limit, cap)` and a changed `limit` produces a miss.
- [ ] No test asserts equality between two file hashes.
- [ ] `configs/data/t1.yaml` change is `limit_requests: 5000000` and `cap_quantile: 1023`, nothing else.
- [ ] `git status --porcelain` shows only `src/mrc/config.py`, `src/mrc/data/reuse.py`,
      `src/mrc/data/labels.py`, `src/mrc/cli.py`, `Makefile`, `configs/data/t1.yaml`, and the two test
      files.

---

## 7. Handoff note

Before `02-phase-b-truth.md` starts, all of the following are true:

1. `data/processed/sample45/labels.npz` and `labels.manifest.json` exist, and the manifest's
   `n_requests` is 1000000.
2. `data/processed/{cluster45,cluster26,cluster10}/labels.*` exist at `limit = 5000000`, so the truth
   stage can be run on four label sets rather than one.
3. `src/mrc/config.py` exists and is byte-identical to master §2. It is now frozen: a change is a commit
   on that file plus a `logs/WORKLOG.md` line.
4. `cap` is recorded for every cluster. B3 derives its grid from the fit-time edges and not from
   `2**ceil(log2(rtilde.max()))+1` (audit M-3), so B3 needs `cap` to reconstruct the same edges.
5. The provisional identification limit from the audit (`b_{M-1} = 837` on cluster045, provenance
   `data/interim/s45.npz`) is replaced in `logs/DECISIONS.md` by the value read from
   `labels.manifest.json` for each cluster.

Phase C must not start before item 1 holds, because `C1` joins onto the label columns (edge 2).
