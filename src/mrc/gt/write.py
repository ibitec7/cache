"""The ground-truth MRC artefact and its provenance manifest.

One curve per cluster, read by the evaluation, pipeline and figure stages. The
manifest records the identification limit (`b_M_minus_1`) and the labels that
produced the grid, because a grid drawn from different labels is a different
curve even at the same capacities.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mrc.config import N_CAPACITIES, REPO
from mrc.gt.stackdist import capacity_grid_from_edges, identified_grid, mrc_from_labels

MANIFEST_SUFFIX = ".manifest.json"


def _git_sha() -> str:
    """Short HEAD of the repo at write time, empty when git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _labels_sha256(source: str | Path) -> str:
    """The sha256 recorded beside a label cache, empty when it is unreadable."""
    if not str(source):
        return ""
    path = Path(source)
    sidecar = path.with_name(path.stem + MANIFEST_SUFFIX)
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    sha = meta.get("sha256", "")
    return sha if isinstance(sha, str) else ""


def written_truth(
    out: str | Path,
    labels: Mapping[str, np.ndarray],
    *,
    n_keys: int,
    overflow_edge: int,
    points: int = N_CAPACITIES,
    source_labels: str | Path = "",
    edges: np.ndarray | None = None,
) -> Path:
    """Write the ground-truth MRC artefact plus its manifest; return the manifest path.

    `overflow_edge` is b_{M-1} = `bin_edges(...)[-2]`, the same limit the fit
    uses, so every reported capacity is one the hazard model may predict at.
    The curve is object-capacity only: `by_bytes` never reaches this function.
    """
    out = Path(out)
    capacities = (capacity_grid_from_edges(edges, points) if edges is not None
                  else identified_grid(n_keys, overflow_edge, points))
    mrc = np.asarray(mrc_from_labels(labels, capacities, by_bytes=False), dtype=np.float64)
    r = np.asarray(labels["R"])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, capacities=capacities.astype(np.int64), mrc=mrc)

    source = str(source_labels)
    manifest = {
        "capacities": capacities.tolist(),
        "n_capacities": int(capacities.size),
        "grid_points_requested": int(points),
        "max_capacity": int(capacities[-1]),
        "b_M_minus_1": int(overflow_edge),
        "overflow_edge": int(overflow_edge),
        "n_requests": int(r.shape[0]),
        "p_compulsory": float((r < 0).mean()),
        "source_labels": source,
        "labels_sha256": _labels_sha256(source_labels),
        "git_sha": _git_sha(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    man = out.with_name(out.stem + MANIFEST_SUFFIX)
    man.write_text(json.dumps(manifest, indent=2) + "\n")
    return man
