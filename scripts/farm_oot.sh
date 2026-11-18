#!/usr/bin/env bash
# OOT on the farm: one fit+study process per cluster, then a window fan-out.
# Detached, resumable: each phase skips work whose output exists.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cache" || exit 1
mkdir -p artifacts/oot logs

CLUSTERS="${CLUSTERS:-cluster10 cluster26 cluster45}"
TRIALS="${TRIALS:-12}"
ROUNDS="${ROUNDS:-200}"
MAXWIN="${MAXWIN:-20}"
SCORERS="${SCORERS:-24}"

# One sqlite file per cluster: three concurrent Optuna writers on the same file
# fail on commit, which is how the first launch of this script died.
echo "[$(date -u +%H:%M:%SZ)] phase 1: fits + studies, $CLUSTERS"
pids=()
for c in $CLUSTERS; do
  env -u PYTHONPATH uv run python scripts/oot_parallel.py fit --cluster "$c" \
    --store "$c:data/publish/$c" --max-windows "$MAXWIN" --rounds "$ROUNDS" \
    --trials "$TRIALS" --score-sample 200000 --row-budget 20000000 \
    --study-storage "sqlite:///$HOME/cache/artifacts/study_$c.db" \
    --out "artifacts/oot/$c" > "logs/fit_$c.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; echo "fit pid $p exit $?"; done
echo "[$(date -u +%H:%M:%SZ)] phase 1 done"

echo "[$(date -u +%H:%M:%SZ)] phase 2: score valid + test, ${SCORERS} workers each"
pids=()
for c in $CLUSTERS; do
  for blk in valid test; do
    env -u PYTHONPATH uv run python scripts/oot_parallel.py score \
      --model "artifacts/oot/$c/model.json" --store "data/publish/$c" \
      --block "$blk" --workers "$SCORERS" --force > "logs/score_${c}_$blk.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]}"; do wait "$p"; echo "score pid $p exit $?"; done
echo "[$(date -u +%H:%M:%SZ)] phase 2 done"

for c in $CLUSTERS; do
  env -u PYTHONPATH uv run python scripts/oot_parallel.py reduce \
    --model "artifacts/oot/$c/model.json" > "logs/reduce_$c.log" 2>&1
  echo "$c: $(python3 -c "import json;d=json.load(open('artifacts/oot/$c/result.json'));print(json.dumps(d['summary'].get('test',{}))[:300])" 2>/dev/null)"
done
echo FARM_OOT_DONE
