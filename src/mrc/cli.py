"""Command line entry point: inspect | labels | features | truth | train | tune | plots."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

from mrc.config import CAP_QUANTILE, HELD_OUT, LABEL_REQUESTS, N_BINS, NUM_THREADS, RAW, REPO
from mrc.data.reuse import compute_reuse
from mrc.data.trace import read_trace, summary
from mrc.eval.mrc import capacity_grid, mrc_curve

SAMPLE_DIR = REPO / "data" / "raw" / "cache-trace" / "samples" / "2020Mar"


def default_source(cluster: str) -> str:
    """Resolve a cluster name to a shard, preferring the tier-1 file."""
    for candidate in (RAW / f"{cluster}.sort.zst", RAW / f"{cluster}.zst",
                      SAMPLE_DIR / cluster, SAMPLE_DIR / f"cluster0{cluster[-2:]}"):
        if candidate.exists():
            return str(candidate)
    raise click.ClickException(f"no shard for {cluster!r}; pass --source")


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
@click.option("--out", type=click.Path(), required=True)
@click.option("--limit", type=int, default=None)
@click.option("--rows-per-part", type=int, default=5_000_000)
@click.option("--row-group", type=int, default=1_000_000)
def store(path: str, out: str, limit: int | None, rows_per_part: int,
          row_group: int) -> None:
    """Convert a CSV shard into zstd parquet parts and a manifest."""
    from mrc.data.store import write_store

    meta = write_store(path, out, limit=limit, rows_per_part=rows_per_part,
                       row_group_rows=row_group)
    mb = sum(p["bytes"] for p in meta["parts"]) / 1e6
    click.echo(f"wrote {out}: {meta['rows']} requests in {len(meta['parts'])} parts, "
               f"{mb:.1f} MB")


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
                        client=t.client, ttl=t.ttl, n_keys=t.n_keys, **d)
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
@click.option("--cluster", required=True)
@click.option("--limit", type=int, default=LABEL_REQUESTS)
@click.option("--cap", type=int, default=CAP_QUANTILE)
def labels(path: str, cluster: str, limit: int | None, cap: int) -> None:
    """Compute and cache reuse-distance labels for one shard."""
    from mrc.pipeline import ensure_labels, labels_path

    lab = ensure_labels(cluster, path, limit=limit, cap=cap)
    click.echo(f"wrote {labels_path(cluster)}: {lab['R'].shape[0]} requests, "
               f"{int(lab['eligible'].sum())} eligible, {int(lab['delta'].sum())} events")


@main.command()
@click.argument("cluster")
def features(cluster: str) -> None:
    """Build and cache the 12-column design matrix for a labelled cluster."""
    from mrc.data.labels import load_labels
    from mrc.pipeline import ensure_features, labels_path

    lab = load_labels(labels_path(cluster))
    X, names, _ = ensure_features(cluster, lab)
    click.echo(f"features {X.shape} for {cluster}: {', '.join(names)}")


@main.command()
@click.argument("cluster")
@click.option("--bins", type=int, default=N_BINS)
def truth(cluster: str, bins: int) -> None:
    """Write the ground-truth MRC curve for a cached label set."""
    from mrc.data.labels import load_labels
    from mrc.pipeline import cluster_edges, ensure_truth, labels_path, truth_path

    lab = load_labels(labels_path(cluster))
    edges = cluster_edges(lab, bins)
    caps, curve = ensure_truth(cluster, lab, edges)
    click.echo(f"wrote {truth_path(cluster)}: {len(caps)} capacities "
               f"up to {int(caps[-1])}, b_M-1={int(edges[-2])}")
    click.echo(f"mrc {np.round(curve, 4).tolist()}")


@main.command()
@click.argument("cluster")
@click.option("--source", type=str, default=None)
@click.option("--store", type=click.Path(exists=True, file_okay=False), default=None,
              help="parquet store root; run one request window from it")
@click.option("--start", type=int, default=0, help="first request index in the store")
@click.option("--window", type=int, default=5_000_000,
              help="requests in the window (default: 5M, clipped to the store)")
@click.option("--limit", type=int, default=LABEL_REQUESTS)
@click.option("--rounds", type=int, default=400)
@click.option("--bins", type=int, default=N_BINS)
@click.option("--rows", type=int, default=20_000_000)
@click.option("--tuned/--no-tuned", default=False, help="use the best Optuna trial")
@click.option("--mlflow/--no-mlflow", default=True)
@click.option("--keep-cache/--no-keep-cache", default=True,
              help="write the per-window label, feature and truth arrays")
def train(cluster: str, source: str | None, store: str | None, start: int,
          window: int | None, limit: int, rounds: int, bins: int,
          rows: int, tuned: bool, mlflow: bool, keep_cache: bool) -> None:
    """Run labels, truth, features, fit and metrics for one cluster or store window."""
    from mrc.data.store import StoreShard
    from mrc.pipeline import run_one

    if store and source:
        raise click.UsageError("pass either --source or --store, not both")
    run_limit: int | None = limit
    if store:
        source_obj: str | StoreShard = StoreShard(store, start, window)
        run_limit = None
    else:
        source_obj = source or default_source(cluster)
    params = None
    if tuned:
        params = best_params(cluster)
    payload = run_one(cluster, source_obj, limit=run_limit,
                      params=params, num_boost_round=rounds, nbins=bins,
                      row_budget=rows, with_mlflow=mlflow, keep_cache=keep_cache)
    for block in ("valid", "test"):
        for name, metrics in payload["metrics"][block].items():
            click.echo(f"{block:5s} {name:5s} mae={metrics['mae']:.5f} "
                       f"max={metrics['max_abs']:.5f} bias={metrics['bias']:+.5f} "
                       f"mre={metrics['mre']:.4f}")
    click.echo(f"wrote artifacts/tables/{cluster}_metrics.json")


def best_params(cluster: str) -> dict:
    """Best trial of the cluster's study, as fitting params."""
    import optuna

    from mrc.config import OPTUNA_URI
    from mrc.tuning.search import study_name

    if cluster == HELD_OUT:
        raise click.ClickException("cluster52 is held out; tune on a training trace")
    try:
        study = optuna.load_study(study_name=study_name(cluster), storage=OPTUNA_URI)
    except KeyError as exc:
        raise click.ClickException(
            f"no Optuna study for {cluster}; run `mrc tune {cluster}` first") from exc
    params = dict(study.best_params)
    params.update({"objective": "binary", "metric": "binary_logloss", "verbosity": -1,
                   "bagging_freq": 1, "num_threads": NUM_THREADS, "seed": 7,
                   "deterministic": True, "force_row_wise": True})
    return params


@main.command()
@click.argument("cluster")
@click.option("--trials", type=int, default=20)
@click.option("--rounds", type=int, default=300)
@click.option("--rows", type=int, default=8_000_000)
@click.option("--timeout", type=float, default=None)
@click.option("--source", type=str, default=None)
@click.option("--limit", type=int, default=LABEL_REQUESTS)
@click.option("--mlflow/--no-mlflow", default=True)
def tune(cluster: str, trials: int, rounds: int, rows: int, timeout: float | None,
         source: str | None, limit: int, mlflow: bool) -> None:
    """Optuna search on cached artefacts; objective is validation MAE."""
    from mrc.data.labels import load_labels
    from mrc.pipeline import (
        cluster_edges,
        ensure_features,
        ensure_labels,
        ensure_truth,
        labels_path,
        prepare,
    )
    from mrc.tuning.search import run_study

    ensure_labels(cluster, source or default_source(cluster), limit=limit)
    lab = load_labels(labels_path(cluster))
    X, names, _ = ensure_features(cluster, lab)
    edges = cluster_edges(lab, N_BINS)
    truth = ensure_truth(cluster, lab, edges)
    prep = prepare(cluster, lab, X, names, truth, row_budget=rows)
    study = run_study(prep, trials, cluster=cluster, num_boost_round=rounds,
                      timeout=timeout, with_mlflow=mlflow)
    click.echo(f"study {study.study_name}: {len(study.trials)} trials, "
               f"best mae={study.best_value:.5f}")
    click.echo(json.dumps(study.best_params, indent=1))


@main.command()
@click.argument("cluster")
@click.option("--requests", type=int, default=600_000)
@click.option("--rounds", type=int, default=200)
@click.option("--repeats", type=int, default=5)
def interpret(cluster: str, requests: int, rounds: int, repeats: int) -> None:
    """Permutation importance, SHAP and feature-family ablation for one cluster."""
    from mrc.interp.driver import run_interpret

    payload = run_interpret(cluster, n_requests=requests, rounds=rounds, n_repeats=repeats)
    for row in payload["features"][:6]:
        click.echo(f"{row['feature']:14s} drop={row['drop']:.5f} +/- {row['std']:.5f}")
    click.echo("ablation (mae, ascending):")
    for row in payload["ablation"]:
        click.echo(f"  {row['group']:14s} mae={row['mae']:.5f}")
    click.echo(f"wrote artifacts/tables/{cluster}_importance.json")


@main.command()
@click.argument("cluster")
@click.option("--source", type=str, default="", help="provenance string for the run")
def log(cluster: str, source: str) -> None:
    """Re-log a committed metrics artefact to MLflow, without refitting."""
    from mrc.pipeline import TABLE_DIR, log_mlflow

    payload = json.loads((TABLE_DIR / f"{cluster}_metrics.json").read_text())
    meta = payload["meta"]
    artifacts = [TABLE_DIR / f"{cluster}_metrics.json", TABLE_DIR / f"{cluster}_curves.npz"]
    model = meta.get("model_path")
    if model:
        artifacts.append(Path(model))
    log_mlflow(cluster, meta["params"], payload["metrics"], artifacts, meta,
                source=source or str(meta.get("source", "")),
                params_source="default", timings={})


@main.command()
@click.option("--clusters", default=None, help="comma-separated; default: every reported trace")
def analysis(clusters: str | None) -> None:
    """Capacity-provisioning analysis from the cached metrics artefacts."""
    from mrc.analysis import provisioning_table, reported_clusters
    clusters = clusters or ",".join(reported_clusters())

    click.echo(provisioning_table(clusters.split(",")))


@main.command()
@click.option("--clusters", default=None, help="comma-separated; default: every reported trace")
def plots(clusters: str | None) -> None:
    """Regenerate every figure and table from the cached metrics artefacts."""
    from mrc.analysis import reported_clusters, write_tables
    from mrc.plots.report_figures import make_all

    names = (clusters or ",".join(reported_clusters())).split(",")
    for path in make_all(names):
        click.echo(f"wrote {path}")
    for path in write_tables(names):
        click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
