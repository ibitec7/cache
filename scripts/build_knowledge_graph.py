"""Build `.ua/knowledge-graph.json` from the repository and the measured artifacts.

The graph is not a hand-written summary. Every code node is parsed from the source, every
import edge is resolved from the AST, and every claim node carries a value read out of
`artifacts/` at build time, so a stale graph cannot disagree with a measured result
without the build failing.

Input contract
--------------
Reads:
    src/mrc/**/*.py, scripts/*.py     source files, function and class nodes
    wiki/docs/*.md, docs/*.md         document nodes
    artifacts/{oot,loco,retrain}/*/result.json   claim nodes
    artifacts/{oot,loco}/<c>/test/*.json         per-window bias, for the offset claim

Writes:
    .ua/knowledge-graph.json          nodes, edges, layers, tour, project metadata

Node ids use the type as prefix (`file:`, `function:`, `class:`, `document:`, `concept:`,
`claim:`), which is what the dashboard and `/understand-explain` expect.

Run: `env -u PYTHONPATH uv run python scripts/build_knowledge_graph.py`
"""
from __future__ import annotations

import ast
import datetime as dt
import glob
import json
import pathlib
import statistics
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
UA_DIR = ROOT / (".understand-anything" if (ROOT / ".understand-anything").is_dir() else ".ua")
CAPACITY_KEYWORDS = ("if", "for", "while", "and", "or", "try", "with", "comprehension")


def rel(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def head(text: str, limit: int = 200) -> str:
    """First sentence of a docstring, collapsed to one line."""
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    cut = flat.find(". ")
    return flat[: cut + 1] if 0 < cut < limit else flat[:limit]


def complexity(node: ast.AST) -> str:
    branches = sum(isinstance(n, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp,
                                  ast.IfExp, ast.ExceptHandler)) for n in ast.walk(node))
    return "simple" if branches <= 2 else "moderate" if branches <= 8 else "complex"


def module_of(name: str, known: set[str]) -> str | None:
    """Map a dotted import to the repo-relative module path that defines it."""
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        cand = "/".join(parts[:cut])
        for prefix in ("src/", ""):
            if f"{prefix}{cand}.py" in known:
                return f"{prefix}{cand}.py"
            init = f"{prefix}{cand}/__init__.py"
            if init in known:
                return init
    return None


def scan_code() -> tuple[list[dict], list[dict], dict[str, ast.Module]]:
    paths = sorted([*ROOT.glob("src/mrc/**/*.py"), *ROOT.glob("scripts/*.py")])
    paths = [p for p in paths if "__pycache__" not in p.as_posix()]
    known = {rel(p) for p in paths}
    nodes: list[dict] = []
    trees: dict[str, ast.Module] = {}
    for path in paths:
        source = path.read_text()
        tree = ast.parse(source)
        trees[rel(path)] = tree
        doc = ast.get_docstring(tree) or ""
        nodes.append({
            "id": f"file:{rel(path)}",
            "type": "file",
            "name": path.name,
            "filePath": rel(path),
            "summary": head(doc) or f"{path.name}, {len(source.splitlines())} lines",
            "tags": ["code", path.parent.name],
            "complexity": complexity(tree),
            "languageNotes": "python" if path.suffix == ".py" else None,
        })
        for item in tree.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nodes.append({
                    "id": f"function:{rel(path)}:{item.name}",
                    "type": "function",
                    "name": item.name,
                    "filePath": rel(path),
                    "summary": head(ast.get_docstring(item) or "")
                    or f"function {item.name}, lines {item.lineno} to {item.end_lineno}",
                    "tags": ["code", "function"],
                    "complexity": complexity(item),
                })
            elif isinstance(item, ast.ClassDef):
                nodes.append({
                    "id": f"class:{rel(path)}:{item.name}",
                    "type": "class",
                    "name": item.name,
                    "filePath": rel(path),
                    "summary": head(ast.get_docstring(item) or "")
                    or f"class {item.name}, lines {item.lineno} to {item.end_lineno}",
                    "tags": ["code", "class"],
                    "complexity": complexity(item),
                })
    for path in sorted(ROOT.glob("scripts/*.sh")):
        lines = path.read_text().splitlines()
        note = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("#")), "")
        nodes.append({
            "id": f"file:{rel(path)}",
            "type": "file",
            "name": path.name,
            "filePath": rel(path),
            "summary": head(note) or f"{path.name}, {len(lines)} lines",
            "tags": ["code", "shell", "orchestration"],
            "complexity": "moderate",
            "languageNotes": "shell",
        })
    edges: list[dict] = []
    for path_rel, tree in trees.items():
        for node in nodes:
            if node["type"] in ("function", "class") and node["filePath"] == path_rel:
                edges.append({"source": f"file:{path_rel}", "target": node["id"],
                              "type": "contains", "direction": "out", "weight": 1.0})
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module)
                imported.update(f"{node.module}.{a.name}" for a in node.names)
        for name in sorted(imported):
            target = module_of(name, known)
            if target and target != path_rel:
                edges.append({"source": f"file:{path_rel}", "target": f"file:{target}",
                              "type": "imports", "direction": "out", "weight": 0.7})
    return nodes, edges, trees


def scan_docs() -> tuple[list[dict], list[dict]]:
    nodes: list[dict] = []
    edges: list[dict] = []
    code_paths = {n["filePath"] for n in NODES if n["type"] == "file"}
    for path in sorted([*ROOT.glob("wiki/docs/**/*.md"), *ROOT.glob("docs/*.md")]):
        text = path.read_text()
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), rel(path))
        nodes.append({
            "id": f"document:{rel(path)}",
            "type": "document",
            "name": title,
            "filePath": rel(path),
            "summary": f"wiki page '{title}', {len(text.splitlines())} lines",
            "tags": ["documentation", "wiki" if rel(path).startswith("wiki/") else "docs"],
            "complexity": "simple",
        })
        for code in sorted(code_paths):
            if f"`{code}`" in text and not code.startswith("artifacts/"):
                edges.append({"source": f"document:{rel(path)}", "target": f"file:{code}",
                              "type": "documents", "direction": "out", "weight": 0.5})
    return nodes, edges


CONCEPTS = [
    ("reuse-distance",
     ("Backward reuse distance R: the number of distinct objects touched since the last "
     "access to the same key. A cache of capacity C holds the last C distinct objects, so "
     "a request hits exactly when R < C."),
     ["file:src/mrc/data/reuse.py", "file:src/mrc/data/reference.py",
      "document:wiki/docs/reuse-distance.md"]),
    ("mrc-curve",
     ("Miss-ratio curve m(C) = P(R >= C), the fraction of requests that miss in a fully "
     "associative LRU cache of capacity C."),
     ["file:src/mrc/eval/mrc.py", "file:src/mrc/gt/stackdist.py",
      "document:wiki/docs/ground-truth.md"]),
    ("discrete-hazard",
     ("Discrete-time survival model: the binned reuse distance is a survival time, each "
     "request contributes one row per bin it was at risk in, and the miss ratio follows "
     "from the survival function. Right censoring enters through the risk set, not a "
     "discarded row."),
     ["file:src/mrc/models/hazard.py", "document:wiki/docs/hazard-model.md"]),
    ("risk-set-expansion",
     ("One training row per (request, bin at risk), label 1 only in the bin where the "
     "reuse event happened. The expansion is what makes censored requests usable and is "
     "the dominant cost of the fit."),
     ["file:src/mrc/models/hazard.py", "document:wiki/docs/hazard-model.md"]),
    ("censoring",
     ("A request whose next access has not happened by the end of the observation window "
     "is right-censored: it survived past that bin and contributes no event."),
     ["file:src/mrc/models/hazard.py", "document:wiki/docs/papers/right-censoring.md"]),
    ("kaplan-meier-floor",
     ("Kaplan-Meier on the binned reuse distance without covariates. The floor the "
     "covariates must beat, not a strawman: censoring is handled correctly."),
     ["file:src/mrc/eval/baselines.py", "document:wiki/docs/papers/kaplan-meier-1958.md"]),
    ("hitcount-baseline",
     ("The practitioner's shortcut: regress log(1+frequency), then convert the predicted "
     "frequency to a reuse-distance proxy through training deciles. Frequency is not a "
     "distance, so the conversion is a marginal and loses to the covariate-free floor."),
     ["file:src/mrc/eval/baselines.py"]),
    ("single-row-oracle",
     ("A baseline that reads eval_rtilde for the block it scores, so its own bin decides "
     "the hit and the curve reconstructs the truth. It reports MAE 0.000000 by "
     "construction and is an oracle, not a competitor."),
     ["file:src/mrc/eval/baselines.py", "document:wiki/docs/generalization.md"]),
    ("out-of-time",
     ("Train on the first 60% of a trace, score every 5M-request window of the last 25%. "
     "Distribution and grid fixed, only time moves, so drift is isolated."),
     ["document:wiki/docs/generalization.md", "file:scripts/oot.py"]),
    ("leave-one-cluster-out",
     ("Train on three traces, score the fourth. The new-cluster test. Confounds model "
     "transfer with distribution shift, which is why it is reported beside out-of-time "
     "and never alone."),
     ["document:wiki/docs/generalization.md", "file:scripts/farm_loco.sh"]),
    ("retrain-curve",
     ("Test error against training volume with hyper-parameters frozen, so the data axis "
     "is not confounded with the search. Answers how much local data a new trace needs."),
     ["document:wiki/docs/generalization.md", "file:scripts/farm_retrain.sh"]),
    ("block-bootstrap-interval",
     ("Confidence interval for a per-window mean, resampling contiguous blocks of requests "
     "rather than single requests, because requests inside a window are autocorrelated "
     "and an i.i.d. bootstrap reports an interval that is too narrow."),
     ["file:src/mrc/eval/metrics.py"]),
    ("stale-window-invalidation",
     ("A per-window score file older than model.json describes a model that no longer "
     "exists. Reusing it reports one model's numbers under another model's parameters, "
     "so the scorer rescoring is keyed on mtime."),
     ["file:scripts/oot_parallel.py", "document:wiki/docs/experiments.md"]),
    ("optuna-startup-phase",
     ("Seeded TPE spends its first 10 trials on random draws, so a study stopped at 12 "
     "trials reports capacity ordering rather than a search: three traces returned "
     "identical parameters and a Spearman correlation of 1.000. Report a study only past "
     "trial 20."),
     ["file:src/mrc/tuning/search.py", "document:wiki/docs/experiments.md"]),
]


def build_concepts() -> tuple[list[dict], list[dict]]:
    file_ids = {n["id"] for n in NODES}
    nodes: list[dict] = []
    edges: list[dict] = []
    for slug, definition, attaches in CONCEPTS:
        nodes.append({
            "id": f"concept:{slug}",
            "type": "concept",
            "name": slug.replace("-", " "),
            "summary": definition,
            "tags": ["concept", "method"],
            "complexity": "simple",
        })
        for target in attaches:
            if target not in file_ids:
                raise SystemExit(f"concept {slug} attaches to missing id {target}")
            if target in file_ids:
                edges.append({"source": f"concept:{slug}", "target": target,
                              "type": "related", "direction": "out", "weight": 0.5})
    return nodes, edges


def window_claims(family: str, cluster: str) -> dict | None:
    files = sorted(glob.glob(str(ROOT / f"artifacts/{family}/{cluster}/test/*.json")))
    if not files:
        return None
    biases, maes = [], []
    for path in files:
        record = json.loads(pathlib.Path(path).read_text())
        biases.append(record["gbdt"]["bias"])
        maes.append(record["gbdt"]["mae"])
    return {"n_files": len(files), "bias": statistics.fmean(biases),
            "mae": statistics.fmean(maes)}


def build_claims() -> tuple[list[dict], list[dict], dict[str, str]]:
    """One claim node per reported number, valued from the artifact at build time."""
    nodes: list[dict] = []
    edges: list[dict] = []
    file_ids = {n["id"] for n in NODES}
    headline: dict[str, str] = {}

    for family, label in (("oot", "out-of-time"), ("loco", "leave-one-cluster-out")):
        for result in sorted(glob.glob(str(ROOT / f"artifacts/{family}/*/result.json"))):
            source = pathlib.Path(result)
            payload = json.loads(source.read_text())
            cluster = payload["meta"]["cluster"]
            summary = payload["summary"]["test"]
            win = window_claims(family, cluster)
            ratio = summary["mae_ratio_to_km"]
            text = (
                f"{label} on {cluster}: mean test MAE {summary['mae_mean']:.6f} over "
                f"{summary['windows']} windows, median {summary['mae_median']:.6f}, "
                f"Kaplan-Meier {summary['km_mae_mean']:.6f} (ratio {ratio:.3f}), model "
                f"wins {summary['gbdt_beats_km_windows']}/{summary['windows']} windows"
            )
            if win:
                text += (f"; mean signed bias {win['bias']:+.2e} against MAE "
                         f"{win['mae']:.6f}, |bias|/MAE {abs(win['bias']) / win['mae']:.2f}")
            if "mae_ci95_block_bootstrap" in summary:
                lo, hi = summary["mae_ci95_block_bootstrap"]
                text += f"; block-bootstrap 95% interval [{lo:.6f}, {hi:.6f}]"
            ident = f"claim:{family}-{cluster}"
            nodes.append({
                "id": ident, "type": "claim", "name": f"{label} {cluster}",
                "filePath": rel(source), "summary": text + ".",
                "tags": ["claim", "result", family], "complexity": "simple",
            })
            edges.append({"source": ident, "target": "file:" + rel(source),
                          "type": "related", "direction": "out", "weight": 0.5})
            doc = "document:wiki/docs/generalization.md"
            if doc in file_ids:
                edges.append({"source": doc, "target": ident, "type": "documents",
                              "direction": "out", "weight": 0.5})
            if family == "oot":
                headline[cluster] = f"{summary['mae_mean']:.6f}"

    for result in sorted(glob.glob(str(ROOT / "artifacts/retrain/*/result.json"))):
        source = pathlib.Path(result)
        payload = json.loads(source.read_text())
        summary = payload["summary"]["test"]
        name = source.parent.name
        ident = f"claim:retrain-{name}"
        nodes.append({
            "id": ident, "type": "claim", "name": f"retrain {name}",
            "filePath": rel(source),
            "summary": (f"retrain point {name}: mean test MAE {summary['mae_mean']:.6f} "
                        f"over {summary['windows']} windows, Kaplan-Meier "
                        f"{summary['km_mae_mean']:.6f}, fixed hyper-parameters."),
            "tags": ["claim", "result", "retrain"], "complexity": "simple",
        })
        edges.append({"source": ident, "target": "file:" + rel(source),
                      "type": "related", "direction": "out", "weight": 0.5})

    for family in ("oot", "loco"):
        nodes.append({
            "id": f"claim:{family}-protocol",
            "type": "claim",
            "name": f"{family} protocol",
            "filePath": "scripts/farm_oot.sh" if family == "oot" else "scripts/farm_loco.sh",
            "summary": (
                "Windows are 5,000,000 contiguous requests; the split is [0.60, 0.75] in "
                "request-index space, so train is the first 60%, validation the next 15%, "
                "test the last 25%; the booster is frozen before any test window opens "
                "and every window is scored into its own JSON file."
            ),
            "tags": ["claim", "protocol"], "complexity": "simple",
        })
    return nodes, edges, headline


LAYERS = [
    ("data", "Trace ingest, reuse distance, labels, store",
     ("src/mrc/data/",)),
    ("features", "Causal feature matrix from request metadata only",
     ("src/mrc/features.py",)),
    ("ground-truth", "Exact Mattson stack distance and its writer",
     ("src/mrc/gt/",)),
    ("models", "Hazard model and risk-set expansion",
     ("src/mrc/models/",)),
    ("eval", "Curves, metrics, baselines",
     ("src/mrc/eval/",)),
    ("tuning", "Optuna search and frozen configuration",
     ("src/mrc/tuning/", "src/mrc/config.py")),
    ("pipeline", "End-to-end training entry point and the command line",
     ("src/mrc/pipeline.py", "src/mrc/cli.py", "src/mrc/__init__.py")),
    ("orchestration", "Windowed fit, score, reduce and the experiment farm",
     ("scripts/",)),
    ("reporting", "Analysis, tables, plots",
     ("src/mrc/analysis.py", "src/mrc/report_tables.py", "src/mrc/plots/")),
    ("interpretability", "Permutation and SHAP importance",
     ("src/mrc/interp/",)),
    ("knowledge", "This wiki and the project docs",
     ("wiki/", "docs/")),
]


def build_layers() -> list[dict]:
    layers: list[dict] = []
    for slug, description, prefixes in LAYERS:
        ids = [n["id"] for n in NODES
               if n["type"] in ("file", "document") and n["filePath"].startswith(prefixes)]
        if ids:
            layers.append({"id": f"layer:{slug}", "name": slug, "description": description,
                           "nodeIds": ids})
    return layers


def build_tour(headline: dict[str, str]) -> list[dict]:
    steps = [
        ("Problem and target", "document:wiki/docs/problem.md", "document:wiki/docs/index.md"),
        ("Reuse distance and the exact curve", "document:wiki/docs/reuse-distance.md",
         "file:src/mrc/data/reuse_distance.py", "file:src/mrc/eval/mrc.py"),
        ("The hazard model", "document:wiki/docs/hazard-model.md",
         "file:src/mrc/models/hazard.py", "file:src/mrc/models/risk_set.py"),
        ("Baselines it has to beat", "document:wiki/docs/papers/kaplan-meier-1958.md",
         "file:src/mrc/eval/baselines.py"),
        ("Protocol: out of time, leave one out, retrain",
         "document:wiki/docs/generalization.md", "file:scripts/oot.py",
         "file:scripts/oot_parallel.py"),
        ("Measured results and what they do not prove",
         "document:wiki/docs/results.md", "document:wiki/docs/generalization.md"),
    ]
    existing = {n["id"] for n in NODES}
    tour = []
    for order, step in enumerate(steps, start=1):
        ids = [i for i in step[1:] if i in existing]
        tour.append({"order": order, "title": step[0],
                     "description": f"Read {step[0].lower()} in the project wiki, then the code it names.",
                     "nodeIds": ids})
    return tour


def import_edges_dedupe(edges: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out = []
    for edge in edges:
        key = (edge["source"], edge["target"], edge["type"])
        if key not in seen:
            seen.add(key)
            out.append(edge)
    return out


def main() -> int:
    global NODES
    code_nodes, code_edges, _trees = scan_code()
    NODES = code_nodes
    doc_nodes, doc_edges = scan_docs()
    NODES = code_nodes + doc_nodes
    concept_nodes, concept_edges = build_concepts()
    NODES = NODES + concept_nodes
    claim_nodes, claim_edges, headline = build_claims()
    NODES = NODES + claim_nodes

    edges = import_edges_dedupe(code_edges + doc_edges + concept_edges + claim_edges)
    known = {n["id"] for n in NODES}
    edges = [e for e in edges if e["source"] in known and e["target"] in known]

    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    graph = {
        "version": "1.0.0",
        "project": {
            "name": "cache",
            "description": "Miss-ratio curves from reuse distance on Twitter cache traces: a "
                           "discrete-time hazard model against exact stack distance and "
                           "covariate-free baselines, evaluated out-of-time and "
                           "leave-one-cluster-out on four production traces.",
            "languages": ["python", "markdown"],
            "frameworks": ["lightgbm", "optuna", "numpy", "mkdocs"],
            "analyzedAt": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "gitCommitHash": commit,
        },
        "nodes": NODES,
        "edges": edges,
        "layers": build_layers(),
        "tour": build_tour(headline),
    }

    problems: list[str] = []
    ids = [n["id"] for n in NODES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate node ids")
    for node in NODES:
        for field in ("id", "type", "name", "summary", "tags"):
            if not node.get(field):
                problems.append(f"{node.get('id')} missing {field}")
    assigned: dict[str, str] = {}
    for layer in graph["layers"]:
        for node_id in layer["nodeIds"]:
            if node_id in assigned:
                problems.append(f"{node_id} in two layers")
            assigned[node_id] = layer["id"]
    for node in NODES:
        if node["type"] in ("file", "document") and node["id"] not in assigned:
            problems.append(f"{node['id']} not in any layer")
    if problems:
        print("VALIDATION FAILED")
        for problem in problems[:20]:
            print(" ", problem)
        return 1

    UA_DIR.mkdir(parents=True, exist_ok=True)
    out = UA_DIR / "knowledge-graph.json"
    out.write_text(json.dumps(graph, indent=1) + "\n")
    kinds: dict[str, int] = {}
    for node in NODES:
        kinds[node["type"]] = kinds.get(node["type"], 0) + 1
    print(f"wrote {rel(out)}: {len(NODES)} nodes {kinds}, {len(edges)} edges, "
          f"{len(graph['layers'])} layers, {len(graph['tour'])} tour steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
