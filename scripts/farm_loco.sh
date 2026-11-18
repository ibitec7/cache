#!/usr/bin/env bash
# LOCO on the farm: four folds, each trained on the other three clusters.
#
# `--max-windows 12` caps each store's train and validation blocks, so the pooled
# fit rows come from all three training clusters without streaming 1611 cluster52
# windows per fold. The test block is never capped: `score` enumerates every window
# of the held-out store, because the per-window distribution is the result.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cache" || exit 1
mkdir -p artifacts/loco logs

CLUSTERS="${CLUSTERS:-cluster10 cluster26 cluster45 cluster52}"
TRIALS="${TRIALS:-6}"
ROUNDS="${ROUNDS:-200}"
MAXWIN="${MAXWIN:-12}"
SCORERS="${SCORERS:-24}"

store_args() {  # every cluster except the held-out one
  local held="$1" out=""
  for c in $CLUSTERS; do
    [ "$c" = "$held" ] && continue
    out="$out --store $c:data/publish/$c"
  done
  printf '%s' "$out"
}

echo "[$(date -u +%H:%M:%SZ)] LOCO fits, held-out = $CLUSTERS"
pids=()
for held in $CLUSTERS; do
  # shellcheck disable=SC2046
  env -u PYTHONPATH uv run python scripts/oot_parallel.py fit --cluster "loco-$held" \
    $(store_args "$held") --max-windows "$MAXWIN" --rounds "$ROUNDS" \
    --trials "$TRIALS" --score-sample 200000 --row-budget 20000000 \
    --study-storage "sqlite:///$HOME/cache/artifacts/study_loco_$held.db" \
    --out "artifacts/loco/$held" > "logs/loco_fit_$held.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; echo "loco fit pid $p exit $?"; done
echo "[$(date -u +%H:%M:%SZ)] LOCO fits done"

for held in $CLUSTERS; do
  env -u PYTHONPATH uv run python scripts/oot_parallel.py score \
    --model "artifacts/loco/$held/model.json" --store "data/publish/$held" \
    --block test --workers "$SCORERS" --force > "logs/loco_score_$held.log" 2>&1 &
done
wait
echo "[$(date -u +%H:%M:%SZ)] LOCO scoring done"

for held in $CLUSTERS; do
  env -u PYTHONPATH uv run python scripts/oot_parallel.py reduce \
    --model "artifacts/loco/$held/model.json" > "logs/loco_reduce_$held.log" 2>&1
  echo "loco-$held: $(python3 -c "import json;d=json.load(open('artifacts/loco/$held/result.json'));print(json.dumps(d['summary'].get('test',{}))[:240])" 2>/dev/null)"
done
echo FARM_LOCO_DONE
