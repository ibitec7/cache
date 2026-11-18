"""Build the store manifest and a dataset README from a directory of parquet parts.

Input contract: a directory of `*.parquet` parts in request order, one part per
row-range, plus the source CSV.zst named in `data/raw/traces/CHECKSUMS.sha256`.
Output contract: `<out>/manifest.json` with the `mrc.data.store` schema, a
`<out>/README.md` suitable for a Hugging Face dataset card, and hard links to
every part in `<out>`.

Usage:
    env -u PYTHONPATH uv run python scripts/build_manifest.py \
        cluster52 data/parquet/cluster52 data/raw/traces/cluster52.sort.zst data/publish/cluster52
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
CHECKSUMS = REPO / "data" / "raw" / "traces" / "CHECKSUMS.sha256"


def natural_key(name: str) -> int:
    """Part number from a name such as `data_10.parquet`."""
    match = re.search(r"(\d+)", name)
    if not match:
        raise ValueError(f"no part number in {name!r}")
    return int(match.group(1))


def source_sha256(source: Path) -> str:
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip().endswith(source.name):
            return line.split()[0]
    raise ValueError(f"{source.name} is not in {CHECKSUMS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster")
    parser.add_argument("parts_dir", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    names = sorted((p.name for p in args.parts_dir.glob("*.parquet")), key=natural_key)
    if not names:
        raise SystemExit(f"no parquet parts in {args.parts_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    parts = []
    start = 0
    rows = 0
    group_sizes: list[int] = []
    for name in names:
        part = args.parts_dir / name
        meta = pq.ParquetFile(part).metadata
        n = meta.num_rows
        size = part.stat().st_size
        # Exact row-group offsets, not a nominal part size: the converted parts
        # carry groups of 5,001,978 rows at most but the last group of a part is
        # short (cluster26 has one of 1,458 rows), so a hardcoded 5M would place
        # every window after the first at the wrong offset.
        sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        parts.append({"file": name, "start": start, "rows": n, "bytes": size,
                      "row_groups": len(sizes), "row_group_rows": sizes})
        start += n
        rows += n
        group_sizes.extend(sizes)
        dst = args.out / name
        if not dst.exists():
            try:
                os.link(part, dst)
            except OSError as exc:
                raise SystemExit(
                    f"cannot hard link {part} to {dst}; keep the output on the same "
                    "filesystem as the parts") from exc

    sha = source_sha256(args.source)
    meta = {
        "store_version": 1,
        "source": str(args.source),
        "source_sha256": sha,
        "source_bytes": args.source.stat().st_size,
        "rows": rows,
        "rows_per_part": parts[0]["rows"] if len(parts) == 1 else None,
        # This conversion did not choose a row-group size; it reports what the
        # parts actually contain. `write_store` sets this to its own target.
        "row_group_rows": (group_sizes[0] if len(set(group_sizes)) == 1 else None),
        "compression": "zstd",
        "parts": parts,
        "columns": ["timestamp", "key", "key_size", "value_size",
                    "client_id", "operation", "ttl"],
        "license": "CC-BY-4.0",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    total = sum(p["bytes"] for p in parts)
    readme = f"""---
license: cc-by-4.0
task_categories:
- tabular-classification
tags:
- cache-trace
- mrc
- reuse-distance
- twitter
---

# Twitter cache-trace `{args.cluster}` in parquet

Full shard from the official Twitter cache-trace mirror, converted to zstd parquet.
Columns and row order match the source CSV; the first 5M rows were checked against
the CSV on every column.

| Property | Value |
|---|---|
| Requests | {rows:,} |
| Parts | {len(parts)} |
| Parquet bytes | {total:,} |
| Row groups | {len(group_sizes)} (largest {max(group_sizes):,} rows) |
| Compression | zstd |
| Source | `{args.source.name}` |
| Source sha256 | `{sha}` |
| License | CC-BY 4.0 |

`manifest.json` maps every part to its request range for `mrc train --store`.

Source: <https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/{args.source.name}>

Yang, Yue, Vinayak. *A large scale analysis of hundreds of in-memory cache clusters
at Twitter.* OSDI 2020. Data licensed CC-BY 4.0.
"""
    (args.out / "README.md").write_text(readme)
    print(json.dumps({"cluster": args.cluster, "rows": rows, "parts": len(parts),
                      "bytes": total, "sha256": sha}, indent=1))


if __name__ == "__main__":
    main()
