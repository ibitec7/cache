| step | command | produces |
|---|---|---|
| fetch a trace | `bash scripts/fetch_traces.sh cluster45` | `data/raw/traces/` |
| build the shard | `env -u PYTHONPATH uv run python -m mrc.cli build cluster45` | `data/publish/cluster45/` |
| fit and score one window | `env -u PYTHONPATH uv run python -m mrc.cli train cluster45` | `artifacts/tables/cluster45_metrics.json` |
| tune | `env -u PYTHONPATH uv run python -m mrc.cli tune cluster45 --trials 30` | `artifacts/study_cluster45.db` |
| out of time, all windows | `bash scripts/farm_oot.sh` | `artifacts/oot/cluster45/` |
| leave one out | `bash scripts/farm_loco.sh` | `artifacts/loco/cluster45/` |
| rebuild this post | `env -u PYTHONPATH uv run python scripts/blog_build.py` | `blog/post.md` |
