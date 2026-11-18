"""One table per experiment family, read straight from the per-window JSONs.

Every number here comes from a `result.json` written by `scripts/oot_parallel.py
reduce`; nothing is transcribed by hand. Two families are recognised, by path:

    artifacts/oot/<cluster>/result.json    out-of-time, one trace per row
    artifacts/loco/<cluster>/result.json   leave-one-cluster-out, one held-out trace per row

Usage:
    uv run python scripts/collect_results.py                    # both, markdown
    uv run python scripts/collect_results.py --json             # machine-readable
    uv run python scripts/collect_results.py --root artifacts/oot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrc.config import REPO


def load(root: Path) -> list[dict]:
    rows = []
    for f in sorted(root.glob("*/result.json")):
        d = json.loads(f.read_text())
        meta = d.get("meta", {})
        for block in ("valid", "test"):
            s = d.get("summary", {}).get(block)
            if not s:
                continue
            rows.append({
                "run": f.parent.name, "block": block,
                "windows": s["windows"],
                "mae_mean": s["mae_mean"], "mae_median": s["mae_median"],
                "mae_iqr": s["mae_iqr"], "ci95": s["mae_ci95_block_bootstrap"],
                "km_mean": s["km_mae_mean"], "ratio": s["mae_ratio_to_km"],
                "beats_km": s["gbdt_beats_km_windows"],
                "train_windows": meta.get("prep", {}).get("pool_requests"),
                "edges_top": meta.get("prep", {}).get("edges_last"),
                "fit_s": (meta.get("prep", {}).get("seconds")
                          if meta.get("prep", {}).get("seconds") is not None
                          else meta.get("fit_seconds")),
                "trials": meta.get("trials"),
                "stores": [s2["name"] for s2 in meta.get("stores", [])],
            })
    return rows


def fmt(rows: list[dict], title: str) -> str:
    header = ("| run | block | windows | MAE mean | MAE median | IQR | 95% CI (block boot) "
              "| KM mean | ratio to KM | beats KM | grid top | fit s |")
    out = [f"### {title}", "", header, "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        iqr = f"{r['mae_iqr'][0]:.5f} to {r['mae_iqr'][1]:.5f}"
        ci = ("n/a" if r["ci95"][0] != r["ci95"][0]
              else f"{r['ci95'][0]:.5f} to {r['ci95'][1]:.5f}")
        out.append(f"| {r['run']} | {r['block']} | {r['windows']} | {r['mae_mean']:.6f} "
                   f"| {r['mae_median']:.6f} | {iqr} | {ci} | {r['km_mean']:.6f} "
                   f"| {r['ratio']:.3f} | {r['beats_km']}/{r['windows']} "
                   f"| {r['edges_top']} | {r['fit_s']} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="one result family, instead of both")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.root:
        rows = load(Path(args.root))
        print(json.dumps(rows, indent=1) if args.json else fmt(rows, args.root))
        return

    fams = [("out-of-time (OOT)", REPO / "artifacts" / "oot"),
            ("leave-one-cluster-out (LOCO)", REPO / "artifacts" / "loco")]
    out = []
    for title, root in fams:
        rows = load(root)
        if rows:
            out.append(json.dumps(rows, indent=1) if args.json else fmt(rows, title))
    print("\n\n".join(out) if out else "no result.json found under artifacts/")


if __name__ == "__main__":
    main()
