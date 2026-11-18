#!/usr/bin/env bash
# Retrain curve: how much trace before the out-of-time pass works at all.
#
# One axis only, the number of 5M-request training windows. Hyper-parameters are
# NOT re-tuned per budget (`--trials 0`, DEFAULT_PARAMS): tuning each point would
# confound "more data" with "a better search", and the point of the curve is the
# data budget. The test block is the same last-quarter windows at every budget, so
# the points are comparable row by row.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/cache" || exit 1
mkdir -p artifacts/retrain logs

CLUSTERS="${CLUSTERS:-cluster10 cluster26 cluster45 cluster52}"
BUDGETS="${BUDGETS:-1 5 12}"     # windows of 5M requests: 5M, 25M, 60M
WINDOW="${WINDOW:-5000000}"
SCORERS="${SCORERS:-20}"
# Same test windows at every budget, capped because the three small clusters have
# at most 12 of them and one window is too noisy to separate budgets (cluster45's
# per-window test MAE spans 0.0007 to 0.0035). Cluster52 is excluded from this
# family: 672 windows x 3 budgets is 35 minutes of I/O for the same curve.
LIMIT="${LIMIT:-12}"

for n in $BUDGETS; do
  pids=()
  for c in $CLUSTERS; do
    # train block = first n windows; valid keeps 15% of what is left; test = last 25%
    total=$(python3 -c "import json;print(json.load(open('data/publish/$c/manifest.json'))['rows'])")
    frac=$(python3 -c "print(min(0.6, $n*$WINDOW/$total))")
    out="artifacts/retrain/$c-$n"
    (
      env -u PYTHONPATH uv run python scripts/oot_parallel.py fit --cluster "retrain-$c-$n" \
        --store "$c:data/publish/$c" --window "$WINDOW" --split "$frac" 0.75 \
        --max-windows "$n" --trials 0 --rounds 200 --score-sample 200000 \
        --row-budget 20000000 --study-storage "sqlite:///$HOME/cache/artifacts/study_retrain.db" \
        --out "$out" > "logs/retrain_fit_$c-$n.log" 2>&1 \
      && env -u PYTHONPATH uv run python scripts/oot_parallel.py score \
        --model "$out/model.json" --store "data/publish/$c" --block test --workers "$SCORERS" --limit "$LIMIT" --force \
        > "logs/retrain_score_$c-$n.log" 2>&1 \
      && env -u PYTHONPATH uv run python scripts/oot_parallel.py reduce \
        --model "$out/model.json" > "logs/retrain_reduce_$c-$n.log" 2>&1
      echo "retrain $c train_windows=$n exit $?"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "[$(date -u +%H:%M:%SZ)] budget $n windows done"
done
echo FARM_RETRAIN_DONE
