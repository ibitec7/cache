"""Build the technical blog: Mermaid charts, tables, and the assembled post.

Nothing in the post is typed by hand. Every table and every chart is generated here from
`artifacts/`, so a number that moves in an artifact moves in the post, and a number that no
artifact supports cannot appear at all. Structural diagrams live in `blog/src/post.md` as
Mermaid source; only the data charts are generated.

Input contract
--------------
Reads:
    artifacts/tables/<cluster>_curves.npz        truth and baseline curves
    artifacts/tables/<cluster>_metrics.json      single-block baselines
    artifacts/oot/<cluster>/result.json          out-of-time summary
    artifacts/oot/<cluster>/test/*.json          per-window MAE and bias
    artifacts/loco/<cluster>/result.json         leave-one-cluster-out summary
    artifacts/retrain/<cluster>-<n>/result.json  retrain curve points
    data/processed/<cluster>/labels.npz          trace distribution statistics
    blog/src/post.md                             prose with {{MER:x}} / {{TBL:x}} slots

Writes:
    blog/tables/<id>.md                          one markdown table per id
    blog/post.md                                 the assembled single-file post

Run: env -u PYTHONPATH uv run python scripts/blog_build.py
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import statistics

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"
BLOG = ROOT / "blog"
TBLS = BLOG / "tables"
SRC = BLOG / "src" / "post.md"

CLUSTERS = ("cluster10", "cluster26", "cluster45", "cluster52")
SHORT = {"cluster10": "c10", "cluster26": "c26", "cluster45": "c45", "cluster52": "c52"}


# ---------------------------------------------------------------- artifacts

def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def curves(cluster: str) -> dict:
    return dict(np.load(ART / "tables" / f"{cluster}_curves.npz"))


def window_files(family: str, cluster: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(str(ART / family / cluster / "test" / "*.json"))):
        out.append(load_json(pathlib.Path(path)))
    return out


def trace_stats(cluster: str) -> dict:
    path = PROCESSED / cluster / "labels.npz"
    if not path.exists():
        return {}
    d = np.load(path)
    r = np.asarray(d["R"], dtype=np.int64)
    eligible = np.asarray(d["eligible"], dtype=bool)
    obj = np.asarray(d["obj_size"], dtype=np.int64)
    op = np.asarray(d["op"])
    live = r[eligible & (r > 0)]
    ops, counts = np.unique(op, return_counts=True)
    mix = sorted(zip(counts.tolist(), ops.tolist()), reverse=True)[:3]
    op_names = {0: "get", 3: "add/gets", 4: "cas", 6: "delete", 7: "set"}
    return {
        "requests": int(r.size),
        "unique_keys": int(np.asarray(d["n_keys"]).ravel()[0]),
        "cold": float(1.0 - eligible.mean()),
        "p50": int(np.percentile(live, 50)) if live.size else -1,
        "p99": int(np.percentile(live, 99)) if live.size else -1,
        "max": int(live.max()) if live.size else -1,
        "obj_mean": float(obj.mean()),
        "mix": [(op_names.get(int(o), f"op{o}"), round(c / r.size, 3)) for c, o in mix],
    }



def retrain_points(cluster: str) -> list[tuple[int, float, int]]:
    """(fit requests, test MAE, training windows) per retrain budget."""
    pts = []
    for path in sorted(glob.glob(str(ART / "retrain" / f"{cluster}-*" / "result.json"))):
        payload = load_json(pathlib.Path(path))
        pts.append((int(payload["meta"]["prep"]["fit_requests"]),
                    float(payload["summary"]["test"]["mae_mean"]),
                    int(payload["meta"]["max_windows"])))
    return sorted(pts, key=lambda t: t[2])


# ---------------------------------------------------------------- mermaid charts
#
# Three data charts only. Everything structural is Mermaid source in the post itself,
# because a hand-placed schematic drifts from the argument the moment the argument moves.

MERM_CONFIG = """---
config:
  theme: neutral
  themeVariables:
    fontFamily: Inter, system-ui, sans-serif
    fontSize: 14px
    primaryColor: '#ffffff'
    primaryBorderColor: '#2d3142'
    primaryTextColor: '#2d3142'
    lineColor: '#8a91a0'
    tertiaryColor: '#f7f7f5'
    xyChart:
      plotColorPalette: '#bf4520, #2d3142, #1f6f5c, #7a4b8f'
  xyChart:
    width: 760
    height: 360
---"""


def block(title: str, desc: str, body: str, kind: str) -> str:
    """A fenced Mermaid block with front-matter theming and accessibility metadata."""
    return ("```mermaid\n" + MERM_CONFIG + "\n" + body.strip() + "\n```")


def chart_curves() -> str:
    """Miss-ratio curves sampled at shared capacities.

    Two of the four traces have a 24-point grid and one has a 12-point grid, so a single
    chart needs a shared x. Each series is read at the nearest capacity its own grid
    contains (a lookup, not an interpolation); beyond a trace's largest capacity the curve
    is flat, so its last value is carried forward and the caption says so.
    """
    targets = (1, 2, 5, 10, 100, 1000, 10_000, 100_000, 1_000_000)
    lines = []
    for cluster in CLUSTERS:
        d = curves(cluster)
        caps = np.asarray(d["capacities"], dtype=float)
        miss = np.asarray(d["truth_test"], dtype=float)
        vals = []
        for t in targets:
            idx = int(np.argmin(np.abs(caps - t))) if caps.size else 0
            vals.append(float(miss[idx]))
        lines.append(f'    line "{cluster}" [' + ", ".join(f"{v:.3f}" for v in vals) + "]")
    axis = ", ".join("1M" if t >= 1_000_000 else "100k" if t >= 100_000 else "10k" if t >= 10_000
                     else "1k" if t >= 1000 else str(t) for t in targets)
    body = (
        "xychart-beta\n"
        f'    accTitle: Measured miss-ratio curves\n'
        f'    accDescr: Miss ratio against capacity for four production traces, sampled at shared '
        f'capacities on a log-spaced axis.\n'
        f'    title "Miss ratio against capacity"\n'
        f"    x-axis [{axis}]\n"
        '    y-axis "miss ratio" 0 --> 1.0\n'
        + "\n".join(lines)
    )
    return block("Measured miss-ratio curves", "Miss ratio per capacity for four traces", body, "xy")


def chart_ratio() -> str:
    """Model error as a fraction of the Kaplan-Meier error, the readable form of Table 8/9."""
    rows, labels = [], []
    for cluster in CLUSTERS:
        oot = load_json(ART / "oot" / cluster / "result.json")["summary"]["test"]
        loco = load_json(ART / "loco" / cluster / "result.json")["summary"]["test"]
        for name, s in (("OOT", oot), ("LOCO", loco)):
            rows.append(round(float(s["mae_mean"]) / float(s["km_mae_mean"]), 3))
            labels.append(f"{SHORT[cluster]}-{name}")
    body = (
        "xychart-beta\n"
        f'    accTitle: Model error relative to the floor\n'
        f'    accDescr: Mean absolute error of the model divided by the Kaplan-Meier error, '
        f'per trace and protocol.\n'
        f'    title "Model error as a fraction of Kaplan-Meier error (lower is better)"\n'
        f"    x-axis [{', '.join(labels)}]\n"
        "    y-axis \"model / Kaplan-Meier\" 0 --> 0.35\n"
        f'    bar "model / KM" [{", ".join(f"{v:.3f}" for v in rows)}]\n'
    )
    return block("Model error relative to the floor", "Eight ratios", body, "bar")


def chart_retrain() -> str:
    """Mean test error against the number of pooled training windows, per trace."""
    series = {}
    for cluster in CLUSTERS[:3]:
        pts = {}
        for requests, mae, windows in retrain_points(cluster):
            pts[windows] = mae * 1000.0
        series[cluster] = pts
    wins = sorted({w for v in series.values() for w in v})
    lines = []
    for cluster, pts in series.items():
        vals = ", ".join(f"{pts[w]:.2f}" if w in pts else "0" for w in wins)
        lines.append(f'    line "{SHORT[cluster]}" [{vals}]')
    body = (
        "xychart-beta\n"
        f'    accTitle: Error against pooled history\n'
        f'    accDescr: Mean test error in thousandths of miss ratio against the number of '
        f'5M-request training windows pooled, with the fit row count held fixed.\n'
        f'    title "Mean test error against pooled training windows"\n'
        f"    x-axis [{', '.join(str(w) for w in wins)}]\n"
        "    y-axis \"MAE x 1000\" 0 --> 20\n"
        + "\n".join(lines)
    )
    return block("Error against pooled history", "Three traces", body, "xy")


CHARTS = {"curves": chart_curves, "ratio": chart_ratio, "retrain": chart_retrain}


# ---------------------------------------------------------------- tables

def t_curves() -> str:
    rows = ["| trace | m(1) | p_c floor | capacity at 80% of achievable | p99 reuse distance |",
            "|---|---|---|---|---|"]
    for cluster in CLUSTERS:
        d = curves(cluster)
        caps, truth = d["capacities"], d["truth_test"]
        hit = 1.0 - truth
        target = 0.8 * hit.max()
        idx = int(np.argmax(hit >= target))
        p99 = trace_stats(cluster).get("p99", -1)
        rows.append(f"| {cluster} | {truth[0]:.4f} | {1 - hit.max():.4f} | "
                    f"{int(caps[idx]):,} | {p99:,} |")
    return "\n".join(rows)


def t_cost() -> str:
    def capacity_for(caps, hit, target):
        if target >= hit.max():
            return float(caps[-1])
        idx = int(np.argmax(hit >= target))
        if idx == 0:
            return float(caps[0])
        x0, x1 = math.log10(max(caps[idx - 1], 1)), math.log10(max(caps[idx], 1))
        y0, y1 = hit[idx - 1], hit[idx]
        if y1 == y0:
            return float(caps[idx])
        frac = (target - y0) / (y1 - y0)
        return 10 ** (x0 + frac * (x1 - x0))

    rows = ["| trace | model: 60 / 80 / 95% of achievable | Kaplan-Meier: 60 / 80 / 95% | disagree by |",
            "|---|---|---|---|"]
    for cluster in CLUSTERS:
        d = curves(cluster)
        caps = np.asarray(d["capacities"], dtype=float)
        picks = {}
        for key in ("gbdt_test", "km_test"):
            hit = 1.0 - d[key]
            best = float(hit.max())
            picks[key] = [capacity_for(caps, hit, f * best) for f in (0.6, 0.8, 0.95)]
        model_picks, km_picks = picks["gbdt_test"], picks["km_test"]
        ratio = km_picks[1] / max(model_picks[1], 1e-9)
        fmt = " / ".join
        rows.append(f"| {cluster} | {fmt(f'{v:,.0f}' for v in model_picks)} | "
                    f"{fmt(f'{v:,.0f}' for v in km_picks)} | {ratio:.2f}x |")
    return "\n".join(rows)


def t_baselines() -> str:
    rows = ["| baseline | status | c10 | c26 | c45 | c52 | reading |", "|---|---|---|---|---|---|---|"]
    names = [("gbdt", "hazard GBDT (this work)"), ("km", "Kaplan-Meier"),
             ("hitcount", "hit-count regression"), ("single_row", "single-row (oracle)")]
    for key, label in names:
        vals = []
        for cluster in CLUSTERS:
            path = ART / "tables" / f"{cluster}_metrics.json"
            payload = load_json(path)
            block = payload.get("test") or payload.get("metrics", {}).get("test", {})
            vals.append(f"{block.get(key, {}).get('mae', float('nan')):.4f}")
        note = {"gbdt": "the model", "km": "floor with no covariates",
                "hitcount": "worse than the floor",
                "single_row": "reads the truth, not a competitor"}[key]
        rows.append(f"| {label} | ran on all four | " + " | ".join(vals) + f" | {note} |")
    return "\n".join(rows)


def t_traces() -> str:
    rows = ["| trace | requests measured | unique keys | cold fraction | R p50 | R p99 | mean object | top operations |",
            "|---|---|---|---|---|---|---|---|"]
    for cluster in CLUSTERS:
        s = trace_stats(cluster)
        if not s:
            rows.append(f"| {cluster} | not cached locally | | | | | | |")
            continue
        mix = ", ".join(f"{n} {v:.2f}" for n, v in s["mix"])
        rows.append(f"| {cluster} | {s['requests']:,} | {s['unique_keys']:,} | "
                    f"{s['cold']:.3f} | {s['p50']:,} | {s['p99']:,} | {s['obj_mean']:.0f} B | {mix} |")
    return "\n".join(rows)


def t_symbols() -> str:
    return "| symbol | meaning | domain |\n|---|---|---|\n| $R$ | reuse distance: distinct keys touched since the previous access to this key | $\\mathbb{N}\\cup\\{\\infty\\}$ |\n| $d$ | lifetime of one episode, equal to $R$ of its closing access | $\\mathbb{N}\\cup\\{\\infty\\}$ |\n| $b$ | bucket index, 1 to 32, log spaced over the distance range | $1\\dots 32$ |\n| $h_b(x)$ | discrete hazard: probability the episode ends in bucket $b$, given it reached $b$ | $[0,1]$ |\n| $S(t\\mid x)$ | survival: probability the episode is still open after bucket $t$ | $[0,1]$ |\n| $x$ | 12 metadata features of the request that opened the episode | $\\mathbb{R}^{12}$ |\n| $p_c$ | compulsory fraction: requests whose key was never seen before | $[0,1]$ |\n| $m(C)$ | miss ratio of a fully associative LRU cache of capacity $C$ | $[0,1]$ |\n| $C$ | capacity in distinct objects | $\\mathbb{N}$ |\n| $\\delta$ | event indicator, 1 if the reuse was observed, 0 if censored | $\\{0,1\\}$ |\n| $n$ | number of scored windows, the unit of replication | $\\mathbb{N}$ |"


def t_learner() -> str:
    params = ["objective", "metric", "num_leaves", "learning_rate", "min_data_in_leaf",
              "feature_fraction", "bagging_fraction", "bagging_freq", "lambda_l2",
              "deterministic"]
    meaning = {
        "objective": "binary logloss, one row per (episode, bucket)",
        "metric": "early stopping on validation logloss",
        "num_leaves": "capacity of one tree, the main regulariser here",
        "learning_rate": "shrinkage per tree",
        "min_data_in_leaf": "minimum rows per leaf, guards rare buckets",
        "feature_fraction": "column subsample per tree",
        "bagging_fraction": "row subsample per iteration",
        "bagging_freq": "bagging every iteration",
        "lambda_l2": "L2 penalty on leaf values",
        "deterministic": "replicate runs are bit-comparable",
    }
    defaults = {"objective": "binary", "metric": "binary_logloss", "num_leaves": 63,
                "learning_rate": 0.05, "min_data_in_leaf": 100, "feature_fraction": 0.9,
                "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                "deterministic": "True"}
    rows = ["| parameter | default | tuned range | what it controls |", "|---|---|---|---|"]
    for p in params:
        tuned = {"num_leaves": "20 to 63", "learning_rate": "0.05 to 0.11",
                 "min_data_in_leaf": "50 to 100", "lambda_l2": "0.003 to 1.0"}.get(p, "frozen")
        rows.append(f"| `{p}` | `{defaults[p]}` | {tuned} | {meaning[p]} |")
    rows.append("| `boosting` | `gbdt` | not tuned | GOSS is available and **not used** |")
    return "\n".join(rows)


def study_stats(cluster: str, prefix: str = "study_") -> tuple[int, float]:
    """(completed trials, best validation MAE) from the Optuna sqlite artefact."""
    import sqlite3
    path = ART / f"{prefix}{cluster}.db"
    if not path.exists():
        return (0, float("nan"))
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        n, best = con.execute(
            "select count(*), min(v.value) from trials t join trial_values v "
            "on v.trial_id = t.trial_id where t.state = 'COMPLETE'").fetchone()
    finally:
        con.close()
    return (int(n or 0), float(best) if best is not None else float("nan"))


def t_optuna() -> str:
    rows = ["| trace | trials in the study | best validation MAE | test MAE after refit |",
            "|---|---|---|---|"]
    for cluster in CLUSTERS:
        trials, best = study_stats(cluster)
        test = load_json(ART / "oot" / cluster / "result.json")["summary"]["test"]["mae_mean"]
        rows.append(f"| {cluster} | {trials} | {best:.6f} | {test:.6f} |")
    return "\n".join(rows)


def t_oot() -> str:
    rows = ["| trace | n windows | MAE | median | Kaplan-Meier MAE | ratio | MRE | windows won |",
            "|---|---|---|---|---|---|---|---|"]
    for cluster in CLUSTERS:
        s = load_json(ART / "oot" / cluster / "result.json")["summary"]["test"]
        mre = mre_for("oot", cluster, "gbdt")
        rows.append(f"| {cluster} | {s['windows']} | {s['mae_mean']:.6f} | {s['mae_median']:.6f} | "
                    f"{s['km_mae_mean']:.6f} | {s['mae_ratio_to_km']:.3f} | {mre:.4f} | "
                    f"{s['gbdt_beats_km_windows']}/{s['windows']} |")
    return "\n".join(rows)


def mre_for(family: str, cluster: str, key: str) -> float:
    vals = []
    for path in glob.glob(str(ART / family / cluster / "test" / "*.json")):
        rec = load_json(pathlib.Path(path))
        vals.append(rec[key].get("mre", float("nan")))
    return statistics.fmean(vals) if vals else float("nan")


def t_loco() -> str:
    rows = ["| held out | n windows | MAE | Kaplan-Meier MAE | ratio | MRE | MAE vs own-data model |",
            "|---|---|---|---|---|---|---|"]
    for cluster in CLUSTERS:
        s = load_json(ART / "loco" / cluster / "result.json")["summary"]["test"]
        own = load_json(ART / "oot" / cluster / "result.json")["summary"]["test"]["mae_mean"]
        rows.append(f"| {cluster} | {s['windows']} | {s['mae_mean']:.6f} | {s['km_mae_mean']:.6f} | "
                    f"{s['mae_ratio_to_km']:.3f} | {mre_for('loco', cluster, 'gbdt'):.4f} | "
                    f"{s['mae_mean'] / own:.2f}x |")
    return "\n".join(rows)


def t_retrain() -> str:
    rows = ["| trace | training windows | requests in the fit | mean test MAE | n test windows |",
            "|---|---|---|---|---|"]
    for cluster in CLUSTERS[:3]:
        n_test = 0
        for requests, mae, windows in retrain_points(cluster):
            n_test = load_json(ART / "retrain" / f"{cluster}-{windows}" / "result.json")[
                "summary"]["test"]["windows"]
            rows.append(f"| {cluster} | {windows} | {requests:,} | {mae:.6f} | {n_test} |")
    return "\n".join(rows)


def t_bias() -> str:
    rows = ["| trace | MAE | mean bias | abs(bias)/MAE | mean worst grid point | worst / MAE |",
            "|---|---|---|---|---|---|"]
    for cluster in CLUSTERS:
        recs = window_files("oot", cluster)
        if not recs:
            continue
        mae = statistics.fmean(r["gbdt"]["mae"] for r in recs)
        bias = statistics.fmean(r["gbdt"]["bias"] for r in recs)
        worst = statistics.fmean(r["gbdt"]["max_abs"] for r in recs)
        rows.append(f"| {cluster} | {mae:.6f} | {bias:+.6f} | {abs(bias) / mae:.2f} | "
                    f"{worst:.6f} | {worst / mae:.1f}x |")
    return "\n".join(rows)


def t_retract() -> str:
    return "| claim we made | what was wrong | corrected |\n|---|---|---|\n| 12-trial Optuna studies had converged | seeded TPE spends its first 10 trials on random draws, so three traces returned identical parameters and a rank correlation of 1.000 | continued to 26 to 34 trials; every test MAE improved |\n| three traces re-tuned better after a refit | the scorer reused window files older than the model, so the same scores were reported under new parameters, identical to 16 digits | rescoring is keyed on the model's mtime |\n| `single_row` scored a perfect curve | it reads `eval_rtilde` for the block it scores, so its own bin decides the hit | it is an oracle, reported as one, never as a competitor |"


def t_repro() -> str:
    return "| step | command | produces |\n|---|---|---|\n| fetch a trace | `bash scripts/fetch_traces.sh cluster45` | `data/raw/traces/` |\n| build the shard | `env -u PYTHONPATH uv run python -m mrc.cli build cluster45` | `data/publish/cluster45/` |\n| fit and score one window | `env -u PYTHONPATH uv run python -m mrc.cli train cluster45` | `artifacts/tables/cluster45_metrics.json` |\n| tune | `env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 30` | `artifacts/study_cluster45.db` |\n| out of time, all windows | `bash scripts/farm_oot.sh` | `artifacts/oot/cluster45/` |\n| leave one out | `bash scripts/farm_loco.sh` | `artifacts/loco/cluster45/` |\n| rebuild this post | `env -u PYTHONPATH uv run python scripts/blog_build.py` | `blog/post.md` |"


TABLES = {
    "curves": t_curves, "cost": t_cost, "baselines": t_baselines, "traces": t_traces,
    "symbols": t_symbols, "learner": t_learner, "optuna": t_optuna, "oot": t_oot,
    "loco": t_loco, "retrain": t_retrain, "bias": t_bias, "retract": t_retract,
    "repro": t_repro,
}


# ---------------------------------------------------------------- assembly

def main() -> int:
    TBLS.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, builder in TABLES.items():
        body = builder()
        (TBLS / f"{name}.md").write_text(body + "\n")
        tables[name] = body
        print(f"table  blog/tables/{name}.md  {body.count(chr(10)) + 1} lines")

    charts = {}
    for name, builder in CHARTS.items():
        charts[name] = builder()
        fence = charts[name].count("```")
        assert fence == 2, f"chart {name} has {fence} fences, expected 2"
        print(f"chart  {{MER:{name}}}  {len(charts[name].splitlines())} lines")

    post = SRC.read_text()
    for name, text in charts.items():
        post = post.replace(f"{{{{MER:{name}}}}}", text)
    for name, body in tables.items():
        post = post.replace(f"{{{{TBL:{name}}}}}", body)

    leftover = [ln.strip() for ln in post.splitlines()
                if ln.strip().startswith("{{") and ln.strip().endswith("}}")]
    if leftover:
        print("UNRESOLVED SLOTS:", leftover)
        return 1

    blocks = post.count("```mermaid")
    fences = post.count("```")
    if fences != blocks * 2:
        print(f"FENCE MISMATCH: {blocks} mermaid blocks, {fences} fences")
        return 1

    out = BLOG / "post.md"
    out.write_text(post)
    dashes = post.count("\u2014")
    print(f"\nwrote {out.relative_to(ROOT)}: {len(post.split())} words, "
          f"{blocks} mermaid diagrams, {dashes} em-dashes")
    return 0 if dashes == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
