"""Applied analysis: what the curve is for.

The metrics say how close a predicted curve is to the truth. The question an
operator asks is different: how much capacity do I need for 90 % hits, and what
does the model's answer cost me if it is wrong? That is a relative capacity
error, and it is the quantity the provisioning table reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrc.config import ARTIFACTS

TABLE_DIR = ARTIFACTS / "tables"


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
