"""Hand-checked arithmetic for the curve metrics (Phase E1)."""
import numpy as np

from mrc.eval.metrics import abc, curve_metrics


def test_curve_metrics_on_a_hand_example():
    caps = np.array([1, 10, 100, 1000])
    truth = np.array([1.0, 0.8, 0.4, 0.2])
    pred = np.array([1.0, 0.9, 0.4, 0.1])
    m = curve_metrics(pred, truth, caps)
    assert abs(m["mae"] - 0.05) < 1e-12        # (0 + .1 + 0 + .1) / 4
    assert abs(m["max_abs"] - 0.1) < 1e-12
    assert abs(m["bias"] - 0.0) < 1e-12        # +.1, 0, -.1 cancel
    assert m["n"] == 4


def test_mre_ignores_capacities_below_the_reporting_floor():
    caps = np.array([1, 10, 1000])
    truth = np.array([1.0, 0.5, 0.004])        # below 0.01 -> excluded
    pred = np.array([1.0, 0.6, 0.5])
    m = curve_metrics(pred, truth, caps, mre_floor=0.01)
    assert abs(m["mre"] - 0.1) < 1e-12          # mean(0, 0.1 / 0.5) over the two kept points
    assert m["mre_n"] == 2


def test_abc_is_log_normalised():
    caps = np.array([1, 10, 100])
    truth = np.array([1.0, 1.0, 1.0])
    pred = np.array([1.0, 1.0, 1.0])
    assert abc(pred, truth, caps) == 0.0
    pred2 = np.array([1.0, 1.2, 1.0])
    assert abs(abc(pred2, truth, caps) - 0.1) < 1e-12


def test_perfect_prediction_gives_zero_error_and_bias():
    caps = np.array([1, 10, 100])
    truth = np.array([0.9, 0.5, 0.1])
    m = curve_metrics(truth, truth, caps)
    assert m["mae"] == 0.0 and m["max_abs"] == 0.0 and m["bias"] == 0.0 and m["abc"] == 0.0
    assert m["n"] == 3


def test_metrics_are_plain_python_scalars():
    caps = np.array([1, 10])
    m = curve_metrics(np.array([1.0, 0.5]), np.array([1.0, 0.4]), caps)
    assert type(m["mae"]) is float and type(m["max_abs"]) is float
    assert type(m["bias"]) is float and type(m["mre"]) is float and type(m["abc"]) is float
    assert type(m["mre_n"]) is int and type(m["n"]) is int
