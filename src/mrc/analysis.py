"""Applied analysis: what the curve is for.

The metrics say how close a predicted curve is to the truth. The question an
operator asks is different: how much capacity do I need for 90 % hits, and what
does the model's answer cost me if it is wrong? That is a relative capacity
error, and it is the quantity the provisioning table reports.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from mrc.config import ARTIFACTS

TABLE_DIR = ARTIFACTS / "tables"

# Single-file runs report `cluster45`; a windowed run reports `cluster45_w0_5000000`.
# The suffix is what tells them apart, so it is defined once, here, for the readers
# that must not mix the two (`pipeline.run_one` builds it, this filters it out).
_WINDOW_TAG = re.compile(r"_w\d+_\d+$")

# Report column order. Traces are added to the tables as their artifacts appear, so
# a new trace needs no edit in any script; this order only fixes the columns.
REPORT_ORDER = ("cluster45", "cluster26", "cluster52", "cluster10")


def reported_clusters(table_dir: Path = TABLE_DIR) -> tuple[str, ...]:
    """Clusters with a metrics artifact, in report order.

    `sample45` is excluded by the name filter: it is the small validation sample, not
    a reported trace. Window artefacts (`cluster45_w0_5000000_metrics.json`, written
    by `mrc train --store` and `scripts/sweep.py`) are excluded for the same reason:
    they are per-window runs of a trace, not the trace's reported row.
    """
    have = {p.name.removesuffix("_metrics.json")
            for p in table_dir.glob("cluster*_metrics.json")}
    have = {name for name in have if not _WINDOW_TAG.search(name)}
    ordered = tuple(c for c in REPORT_ORDER if c in have)
    return ordered + tuple(sorted(have - set(ordered)))


def load_metrics(cluster: str) -> dict:
    path = TABLE_DIR / f"{cluster}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `mrc train {cluster}` first")
    return json.loads(path.read_text())


def provisioning_rows(cluster: str) -> list[dict]:
    """One row per (model, target): capacity, ratio to the true capacity, MAE."""
    payload = load_metrics(cluster)
    rows = []
    for model, entries in payload["metrics"]["provisioning"].items():
        mae = payload["metrics"]["test"][model]["mae"]
        for entry in entries:
            rows.append({
                "cluster": cluster, "model": model, "fraction": entry.get("fraction"),
                "ceiling": entry.get("ceiling"), "target_hit": entry["target_hit"],
                "capacity_true": entry["capacity_true"],
                "capacity_pred": entry["capacity_pred"],
                "ratio": entry["ratio"], "test_mae": mae,
            })
    return rows


def provisioning_table(clusters: list[str]) -> str:
    """Markdown table of capacity ratios; ratio > 1 is over-provisioning."""
    header = ("| cluster | model | fraction of ceiling | target hit | ceiling | "
              "capacity (truth) | capacity (model) | ratio | test MAE |\n"
              "|---|---|---|---|---|---|---|---|---|")
    lines = [header]
    for cluster in clusters:
        try:
            rows = provisioning_rows(cluster)
        except FileNotFoundError:
            lines.append(f"| {cluster} | - | - | - | - | - | - | - | not run |")
            continue
        for row in rows:
            ratio = row["ratio"]
            ratio_txt = "n/a" if not np.isfinite(ratio) else f"{ratio:.3f}"
            fraction = row.get("fraction")
            ceiling = row.get("ceiling")
            lines.append(
                f"| {cluster} | {row['model']} | "
                f"{'-' if fraction is None else f'{fraction:.2f}'} | "
                f"{row['target_hit']:.3f} | "
                f"{'-' if ceiling is None else f'{ceiling:.3f}'} | "
                f"{row['capacity_true']:.0f} | {row['capacity_pred']:.0f} | "
                f"{ratio_txt} | {row['test_mae']:.5f} |"
            )
    return "\n".join(lines)


def metric_rows(cluster: str, block: str = "test") -> dict[str, dict]:
    """Per-model metric dicts on one block, ordered as the headline table."""
    payload = load_metrics(cluster)
    order = ("gbdt", "km", "hitcount")
    metrics = payload["metrics"][block]
    return {name: metrics[name] for name in order if name in metrics}


def write_tables(clusters: list[str], block: str = "test") -> list[Path]:
    """Write one markdown table and one booktabs table per cluster, plus a combined md."""
    from mrc.report_tables import booktabs, metrics_table

    tex_dir = Path(ARTIFACTS).parent / "report" / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    out_md = TABLE_DIR / "metrics.md"
    parts = ["# Per-cluster accuracy\n"]
    written = []
    for cluster in clusters:
        if not (TABLE_DIR / f"{cluster}_metrics.json").exists():
            continue
        rows = metric_rows(cluster, block)
        parts.append(f"## {cluster} ({block} block)\n")
        parts.append(metrics_table(cluster, rows))
        tex = booktabs(rows, caption=f"{cluster}: MRC error on the {block} block",
                       label=f"tab:{cluster}-{block}", path=tex_dir / f"{cluster}_metrics.tex")
        written.append(tex_dir / f"{cluster}_metrics.tex")
        assert tex
    out_md.write_text("\n".join(parts))
    written.append(out_md)
    return written
