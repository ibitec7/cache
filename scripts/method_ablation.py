"""One fixed-protocol comparison of the hazard model against ablations and alternatives.

Every row is fitted and scored on the same subsample, split and capacity grid, so the
differences are attributable to the variant and not to a change of protocol. The
subsample is 1M requests per trace because nine variants times three traces at the
full row budget would cost hours; the ordering is what the table is for.
"""
from __future__ import annotations

import json

import numpy as np

from mrc.analysis import reported_clusters
from mrc.config import ARTIFACTS, N_CAPACITIES, REPO, SEED
from mrc.data.labels import load_labels
from mrc.eval.baselines import fit_predict_hitcount, fit_predict_single_row
from mrc.eval.metrics import curve_metrics
from mrc.gt.stackdist import capacity_grid_from_edges, mrc_from_labels
from mrc.interp.importance import GROUPS
from mrc.pipeline import (
    DEFAULT_PARAMS,
    cluster_edges,
    ensure_features,
    fit,
    km_curve,
    labels_path,
    predict_block,
    prepare,
)

N_REQUESTS = 1_000_000
ROUNDS = 200
ROW_BUDGET = 5_000_000
# Default: every trace with a metrics artefact, so a new trace needs no edit here.
CLUSTERS = reported_clusters()
TABLE_DIR = ARTIFACTS / "tables"
TEX_DIR = REPO / "report" / "tables"


def _subsample(labels: dict, n: int) -> tuple[dict, np.ndarray | None]:
    total = int(np.asarray(labels["R"]).shape[0])
    if n >= total:
        return labels, None
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(total, size=n, replace=False))
    sub = {k: np.asarray(v)[idx] for k, v in labels.items()
           if np.ndim(v) > 0 and len(v) == total}
    sub["n_keys"] = np.asarray(labels["n_keys"])
    return sub, idx


def _eval_arrays(prep, labels: dict) -> dict:
    test = prep.masks[2] & prep.eligible
    freq = np.asarray(labels["freq"])
    return {
        "X": prep.X[test], "rtilde": prep.rtilde[test], "delta": prep.delta[test],
        "freq": freq[test], "pc": prep.p_compulsory["test"],
    }


def run_cluster(cluster: str) -> dict:
    labels = load_labels(labels_path(cluster))
    X, names, _ = ensure_features(cluster, labels)
    edges = cluster_edges(labels)
    caps = capacity_grid_from_edges(edges, N_CAPACITIES)
    truth = mrc_from_labels(labels, caps)
    lab, idx = _subsample(labels, N_REQUESTS)
    Xs = X if idx is None else X[idx]
    prep = prepare(cluster, lab, Xs, names, (caps, truth), row_budget=ROW_BUDGET,
                   edges=edges)
    test_truth = prep.truth_blocks["test"]
    ev = _eval_arrays(prep, lab)
    train = prep.masks[0] & prep.eligible
    freq = np.asarray(lab["freq"])
    fit_X, fit_rt, fit_delta = prep.X[train], prep.rtilde[train], prep.delta[train]
    fit_freq = freq[train]

    rows: dict[str, dict] = {}
    index = {n: j for j, n in enumerate(names)}
    for group, dropped in GROUPS.items():
        Xg = Xs.copy()
        if dropped:
            Xg[:, [index[f] for f in dropped]] = 0.0
        prep_g = prepare(cluster, lab, Xg, names, (caps, truth),
                         row_budget=ROW_BUDGET, edges=edges)
        model = fit(prep_g, DEFAULT_PARAMS, num_boost_round=ROUNDS)
        pred = predict_block(model, prep_g, "test")
        rows[group] = curve_metrics(pred, test_truth, caps)

    rows["kaplan_meier"] = curve_metrics(km_curve(prep, "test"), test_truth, caps)
    hit = fit_predict_hitcount(fit_X, fit_freq, caps, rtilde=fit_rt,
                               eval_X=ev["X"], eval_freq=ev["freq"],
                               p_compulsory=ev["pc"])
    rows["hitcount"] = curve_metrics(hit, test_truth, caps)
    single = fit_predict_single_row(fit_X, fit_rt, fit_delta, edges, caps,
                                    eval_X=ev["X"], eval_rtilde=ev["rtilde"],
                                    eval_delta=ev["delta"], p_compulsory=ev["pc"])
    rows["single_row_oracle"] = curve_metrics(single, test_truth, caps)
    print(cluster, {k: round(v["mae"], 5) for k, v in rows.items()}, flush=True)
    return {"cluster": cluster, "n_requests": len(lab["R"]), "rounds": ROUNDS,
            "rows": rows}


def main() -> None:
    results = [run_cluster(c) for c in CLUSTERS]
    payload = {"meta": {"n_requests": N_REQUESTS, "rounds": ROUNDS,
                        "row_budget": ROW_BUDGET, "block": "test"},
               "clusters": results}
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "method_ablation.json").write_text(json.dumps(payload, indent=1) + "\n")

    order = ["all", "no_timing", "no_sizes", "no_popularity", "no_client",
             "no_namespace", "kaplan_meier", "hitcount", "single_row_oracle"]
    label = {"all": "Hazard, all 12 features", "no_timing": "Hazard, no timing family",
             "no_sizes": "Hazard, no size family", "no_popularity": "Hazard, no popularity",
             "no_client": "Hazard, no client", "no_namespace": "Hazard, no token bucket",
             "kaplan_meier": "Kaplan-Meier (marginal)", "hitcount": "Hit-count regression",
             "single_row_oracle": "Single-row, oracle$^\\dagger$"}
    lines = ["% generated by scripts/method_ablation.py; do not edit",
             "\\begin{table}[t]", "\\centering", "\\footnotesize",
             ("\\caption{One protocol, nine estimators. Each cell is test-block MAE over MAX "
             "on a 1M-request subsample, 200 boosting rounds, identical split and capacity "
             "grid. $\\dagger$The single-row ablation uses the scored request's own distance, "
             "so it is an oracle, not a deployable estimator; it bounds what the risk set buys.}"),
             "\\label{tab:methods}",
             "\\begin{tabular}{@{}l r r r@{}}", "\\toprule",
             "Estimator & cluster45 & cluster26 & cluster10 \\\\", "\\midrule"]
    for key in order:
        cells = []
        for res in results:
            m = res["rows"][key]
            cells.append(f"{m['mae']:.5f} / {m['max_abs']:.5f}")
        lines.append(f"{label[key]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    (TEX_DIR / "method_ablation.tex").write_text("\n".join(lines) + "\n")
    print("wrote", TABLE_DIR / "method_ablation.json", "and", TEX_DIR / "method_ablation.tex")


if __name__ == "__main__":
    main()
