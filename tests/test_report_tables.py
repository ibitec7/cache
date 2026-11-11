"""Table contract: markdown rows and a booktabs body, both from the metric dict."""
from __future__ import annotations

import pytest

from mrc.report_tables import booktabs, metrics_table

METRICS = {
    "gbdt": {"mae": 0.0123, "max_abs": 0.0312, "bias": -0.0004, "mre": 0.0219, "n": 24},
    "km": {"mae": 0.0501, "max_abs": 0.1101, "bias": 0.0102, "mre": 0.0903, "n": 24},
    "hitcount": {"mae": 0.0709, "max_abs": 0.2011, "bias": 0.0301, "mre": None, "n": 24},
}


def test_metrics_table_has_one_row_per_model():
    metrics = {"gbdt": {"mae": 0.01, "max_abs": 0.03, "bias": 0.0, "mre": 0.02},
               "km": {"mae": 0.05, "max_abs": 0.11, "bias": 0.01, "mre": 0.09}}
    md = metrics_table("cluster45", metrics)
    assert md.count("\n|") == 3
    tex = booktabs(metrics, caption="T1 error", label="tab:t1")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "gbdt" in tex


def test_metrics_table_rows_and_columns_exactly():
    md = metrics_table("cluster45", METRICS)
    lines = md.splitlines()
    assert lines[0] == "| model | MAE | MAX | bias | MRE | n |"
    assert lines[1] == "|---|---|---|---|---|---|"
    assert lines[2] == "| gbdt | 0.0123 | 0.0312 | -0.0004 | 0.0219 | 24 |"
    assert lines[3] == "| km | 0.0501 | 0.1101 | 0.0102 | 0.0903 | 24 |"
    assert lines[4] == "| hitcount | 0.0709 | 0.2011 | 0.0301 | n/a | 24 |"
    assert md.count("\n|") == 4
    assert "cluster45" in md
    assert "artifacts/tables/cluster45_metrics.json" in md


def test_booktabs_writes_the_file_and_keeps_the_generated_header(tmp_path):
    path = tmp_path / "t1_metrics.tex"
    tex = booktabs(METRICS, caption="Test-block error", label="tab:t1", path=path)
    assert path.exists()
    assert path.read_text() == tex
    lines = tex.splitlines()
    assert lines[0].startswith("% generated")
    assert "\\caption{Test-block error}" in tex
    assert "\\label{tab:t1}" in tex
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in tex
    assert "gbdt & 0.0123 & 0.0312 & -0.0004 & 0.0219 & 24 \\\\" in tex
    assert "hitcount & 0.0709 & 0.2011 & 0.0301 & n/a & 24 \\\\" in tex


def test_booktabs_without_a_path_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tex = booktabs(METRICS, caption="c", label="tab:x")
    assert "\\midrule" in tex
    assert list(tmp_path.iterdir()) == []


def test_an_empty_metric_dict_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        metrics_table("cluster45", {})
    with pytest.raises(ValueError, match="empty"):
        booktabs({}, caption="c", label="tab:x")
