"""Command line entry point: inspect | reuse | mrc | fit | plots."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import numpy as np

from mrc.data.trace import read_trace, summary
from mrc.data.reuse import compute_reuse
from mrc.eval.mrc import capacity_grid, mrc_curve, mrc_error


@click.group()
def main() -> None:
    """Cache hit-ratio curves from reuse distance."""


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--limit", type=int, default=None)
def inspect(path: str, limit: int | None) -> None:
    """Summarise a trace shard."""
    t = read_trace(path, limit=limit)
    click.echo(json.dumps(summary(t), indent=2))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--limit", type=int, default=None)
@click.option("--out", type=click.Path(), required=True)
def reuse(path: str, limit: int | None, out: str) -> None:
    """Compute reuse distance and write a compressed npz."""
    t = read_trace(path, limit=limit)
    d = compute_reuse(t.key, t.obj_size, t.n_keys)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, key=t.key, obj_size=t.obj_size, op=t.op, ts=t.ts,
                        client=t.client, ttl=t.ttl, n_keys=t.n_keys,
                        **{k: v for k, v in d.items()})
    click.echo(f"wrote {out}: {len(t)} requests, {t.n_keys} distinct keys")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--points", type=int, default=24)
@click.option("--by", "mode", type=click.Choice(["objects", "bytes"]), default="objects")
def mrc(path: str, points: int, mode: str) -> None:
    """Empirical MRC curve from a reuse-distance file."""
    z = np.load(path)
    if mode == "objects":
        caps = capacity_grid(int(z["n_keys"]), points)
        curve = mrc_curve(z["R"], caps)
    else:
        caps = capacity_grid(int(np.percentile(z["obj_size"], 99.9) * 200), points)
        curve = mrc_curve(z["B"], caps, own_size=z["obj_size"])
    for c, m in zip(caps, curve):
        click.echo(f"{c}\t{m:.6f}")


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--bins", type=int, default=32)
@click.option("--limit", type=int, default=200_000)
def fit(path: str, bins: int, limit: int) -> None:
    """Fit the discrete hazard model and report MRC error against the empirical curve."""
    import lightgbm as lgb
    import mlflow
    from mrc.models.hazard import (HazardModel, assign_bins, bin_edges, km_hazard)

    z = np.load(path)
    R = z["R"][:limit]
    max_d = int(np.percentile(R[R >= 0], 99.9))
    edges = bin_edges(max_d, bins)
    b = assign_bins(R, edges)
    event = ~z["censor"][:limit] & (b >= 0)
    X = np.column_stack([
        np.log1p(z["obj_size"][:limit]), z["op"][:limit].astype(np.float32),
        (z["ttl"][:limit] > 0).astype(np.float32), np.log1p(np.clip(z["ttl"][:limit], 0, None)),
        np.log1p(z["client"][:limit] % 512).astype(np.float32),
    ]).astype(np.float32)
    names = ["log_obj_size", "op", "is_write", "log_ttl", "client_bucket"]

    caps = capacity_grid(int(z["n_keys"]), 16)
    truth = mrc_curve(R, caps)
    h_km, _, p_comp = km_hazard(R, bins)
    S = np.cumprod(1 - h_km)
    bins_c = np.clip(assign_bins(caps - 1, edges), 0, len(S) - 1)
    km_curve = p_comp + (1 - p_comp) * S[bins_c]

    params = {"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
              "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.9,
              "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 0, "seed": 7}
    model = HazardModel(params=params, nbins=len(edges))
    model.fit(X, b, event, edges, names)
    pred = model.predict_mrc(X, caps, edges, p_compulsory=p_comp)

    click.echo(f"capacities {caps.tolist()}")
    click.echo(f"empirical  {np.round(truth, 5).tolist()}")
    click.echo(f"kaplan-meier {np.round(km_curve, 5).tolist()}")
    click.echo(f"gbdt       {np.round(pred, 5).tolist()}")
    click.echo(f"KM   error {mrc_error(km_curve, truth)}")
    click.echo(f"GBDT error {mrc_error(pred, truth)}")


@main.command()
def plots() -> None:
    """Regenerate the standard figures from artifacts."""
    click.echo("no figures yet: run `mrc fit` first, then extend src/mrc/plots/curve.py")


if __name__ == "__main__":
    main()
