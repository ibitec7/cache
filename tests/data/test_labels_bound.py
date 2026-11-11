"""Exact-first gates for the label columns: bound, event flag and truncation."""
import numpy as np

from mrc.data.reuse import compute_reuse


def test_gamma_counts_distinct_keys_after_each_request():
    k = np.array([0, 1, 0, 2, 1, 3])
    s = np.ones(6, dtype=np.int64)
    d = compute_reuse(k, s, 4, cap=10)
    assert d["gamma"].tolist() == [4, 4, 3, 2, 1, 0]


def test_delta_true_only_when_next_reference_is_inside_the_window():
    # Position 2 is the last occurrence of key 0, so under the last-occurrence
    # censoring scheme it is censored, not an event. The master plan's
    # delta == [F, F, T, F] contradicts its own scheme (audit M-16).
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["eligible"].tolist() == [False, False, True, True]
    assert d["delta"].tolist() == [False, False, False, False]


def test_rtilde_uses_forward_bound_when_censored():
    k = np.array([0, 1, 0, 1])
    d = compute_reuse(k, np.ones(4, dtype=np.int64), 2, cap=10)
    assert d["rtilde"][2] == 1   # event, R = 1 distinct key in between
    assert d["rtilde"][3] == 0   # censored, nothing follows, gamma = 0
    assert d["rtilde"][0] == -1  # cold, not eligible, marked not guessed


def test_rtilde_is_capped_at_the_overflow_edge():
    k2 = np.concatenate([np.arange(50), np.arange(50)])
    d2 = compute_reuse(k2, np.ones(100, dtype=np.int64), 50, cap=4)
    assert d2["rtilde"].max() == 4


def test_eight_request_micro_example():
    key = {"A": 0, "B": 1, "C": 2, "D": 3}
    k = np.array([key[c] for c in "ABACABDC"], dtype=np.int64)
    d = compute_reuse(k, np.ones(8, dtype=np.int64), 4, cap=7)
    assert int(d["eligible"].sum()) == 4
    assert d["R"][d["delta"]].tolist() == [1]
    assert d["rtilde"].tolist() == [-1, -1, 1, -1, 3, 2, -1, 0]
    assert bool(d["delta"][7]) is False       # C@7 is censored


def test_skeleton_micro_example():
    key = {"A": 0, "B": 1, "C": 2}
    k = np.array([key[c] for c in "ABACABCB"], dtype=np.int64)
    d = compute_reuse(k, np.ones(8, dtype=np.int64), 3, cap=3)
    assert int(d["eligible"].sum()) == 5
    assert np.sort(d["R"][d["delta"]]).tolist() == [1, 2]
    assert d["rtilde"].tolist() == [-1, -1, 1, -1, 2, 2, 1, 0]
