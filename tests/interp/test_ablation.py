"""Feature-family ablation matrix (Phase G3)."""
import numpy as np
import pytest

from mrc.interp.importance import GROUPS, ablation_matrix
from mrc.models.hazard import block_masks

NAMES = [
    "log_gap",
    "log_freq",
    "ns_code",
    "log_obj_size",
    "log_key_size",
    "log_val_size",
    "noise",
]
CAPACITIES = np.array([1, 2, 4, 8, 16], dtype=np.int64)
EDGES = np.array([0, 1, 2, 4, 8, 16, 32], dtype=np.int64)
TEST_GROUPS: dict[str, tuple[str, ...]] = {
    "all": (),
    "no_signal": ("log_gap",),
    "no_sizes": ("log_obj_size", "log_key_size", "log_val_size"),
    "no_noise": ("noise",),
}


def _problem(n: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """A synthetic problem where reuse distance is a function of one column."""
    rng = np.random.default_rng(3)
    signal = rng.normal(size=n)
    X = rng.normal(size=(n, len(NAMES))).astype(np.float32)
    X[:, NAMES.index("log_gap")] = signal.astype(np.float32)
    _tr, va, _te = block_masks(n)
    X[va, NAMES.index("noise")] = 0.0        # no signal at all on the validation block
    rtilde = np.clip(np.round(4.0 + 6.0 * signal), 0, 20).astype(np.int64)
    delta = np.ones(n, dtype=bool)
    return X, rtilde, delta, n


def _linear_fit_predict(
    X: np.ndarray,
    names: list[str],
    rtilde: np.ndarray,
    delta: np.ndarray,
    edges: np.ndarray,
    capacities: np.ndarray,
    n_requests: int,
) -> np.ndarray:
    """Least squares for rtilde on the train block, MRC from the fitted distances."""
    tr, va, _te = block_masks(n_requests)
    A = np.column_stack([X[tr], np.ones(int(tr.sum()), dtype=np.float32)])
    w, *_ = np.linalg.lstsq(A.astype(np.float64), rtilde[tr].astype(np.float64), rcond=None)
    Av = np.column_stack([X[va], np.ones(int(va.sum()), dtype=np.float32)])
    rhat = Av @ w
    return np.array([np.mean(rhat >= c) for c in capacities], dtype=np.float64)


def test_groups_are_the_six_frozen_families() -> None:
    assert set(GROUPS) == {
        "all",
        "no_namespace",
        "no_sizes",
        "no_timing",
        "no_popularity",
        "no_client",
    }
    assert GROUPS["all"] == ()
    assert set(GROUPS["no_namespace"]) == {"ns_code"}
    assert set(GROUPS["no_sizes"]) == {"log_obj_size", "log_key_size", "log_val_size"}
    assert set(GROUPS["no_timing"]) == {"log_ttl", "has_ttl", "log_gap", "log_age"}
    assert set(GROUPS["no_popularity"]) == {"log_freq"}
    assert set(GROUPS["no_client"]) == {"client_bucket"}


def test_ablation_zeroes_exactly_the_named_columns() -> None:
    X, rtilde, delta, n_requests = _problem()
    seen: list[np.ndarray] = []

    def record(
        Xg: np.ndarray,
        names: list[str],
        rt: np.ndarray,
        d: np.ndarray,
        e: np.ndarray,
        caps: np.ndarray,
        n: int,
    ) -> np.ndarray:
        seen.append(Xg.copy())
        return np.full(int(caps.size), 0.5)

    rows = ablation_matrix(
        record, X, NAMES, rtilde, delta, EDGES, CAPACITIES, n_requests, groups=TEST_GROUPS
    )
    assert [mat.shape for mat in seen] == [X.shape] * len(TEST_GROUPS)
    for mat, group in zip(seen, TEST_GROUPS, strict=True):
        drop = TEST_GROUPS[group]
        for j, name in enumerate(NAMES):
            if name in drop:
                assert np.all(mat[:, j] == 0.0), (group, name)
            else:
                assert np.array_equal(mat[:, j], X[:, j]), (group, name)
    assert len(rows) == len(TEST_GROUPS)
    assert set(rows[0]) == {"group", "mae", "max_abs", "bias", "dropped_features"}
    sizes = next(row for row in rows if row["group"] == "no_sizes")
    assert sizes["dropped_features"] == ["log_obj_size", "log_key_size", "log_val_size"]


def test_full_model_has_the_lowest_mae() -> None:
    X, rtilde, delta, n_requests = _problem()
    rows = ablation_matrix(
        _linear_fit_predict, X, NAMES, rtilde, delta, EDGES, CAPACITIES, n_requests,
        groups=TEST_GROUPS,
    )
    assert rows[0]["group"] == "all"
    assert [row["mae"] for row in rows] == sorted(row["mae"] for row in rows)
    mae = {row["group"]: row["mae"] for row in rows}
    assert mae["no_signal"] > mae["all"] + 0.01      # the signal family earns its place
    assert mae["no_noise"] == pytest.approx(mae["all"], abs=1e-12)


def test_unknown_feature_raises_key_error_before_any_fit() -> None:
    X, rtilde, delta, n_requests = _problem(n=60)
    calls: list[tuple] = []

    def never(
        Xg: np.ndarray,
        names: list[str],
        rt: np.ndarray,
        d: np.ndarray,
        e: np.ndarray,
        caps: np.ndarray,
        n: int,
    ) -> np.ndarray:
        calls.append((Xg.shape, tuple(names)))
        return np.zeros(int(caps.size))

    with pytest.raises(KeyError):
        ablation_matrix(
            never, X, NAMES, rtilde, delta, EDGES, CAPACITIES, n_requests,
            groups={"bogus": ("not_a_column",)},
        )
    assert calls == []
