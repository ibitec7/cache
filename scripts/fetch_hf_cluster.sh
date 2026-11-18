#!/usr/bin/env bash
# Pull one cluster's parquet store straight from the Hub, no local Hub cache.
#
# Why not `hf download`: with --local-dir it stages every blob under
# <dir>/.cache/huggingface before placing it, so a 98 GB repo needs 196 GB of
# disk at peak. On the farm box that does not fit. Plain parallel curl writes the
# final path directly and resumes with -C -, which is all this needs.
#
#   ./scripts/fetch_hf_cluster.sh cluster52
#   WORKERS=16 ./scripts/fetch_hf_cluster.sh cluster26
set -euo pipefail

CLUSTER="${1:?usage: fetch_hf_cluster.sh <cluster>}"
REPO="ibitec/cache-trace-${CLUSTER}-parquet"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/publish/${CLUSTER}"
WORKERS="${WORKERS:-12}"
BASE="https://huggingface.co/datasets/${REPO}/resolve/main"

mkdir -p "$DEST"
log() { printf '[hf %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

curl -fsSL "https://huggingface.co/api/datasets/${REPO}/tree/main?recursive=1" \
  | python3 -c "import json,sys;[print(f['path'],f.get('size',0)) for f in json.load(sys.stdin) if f['type']=='file']" \
  > /tmp/hf_files.txt
log "$(wc -l < /tmp/hf_files.txt) files, $(awk '{s+=$2} END {printf "%.1f", s/1e9}' /tmp/hf_files.txt) GB"

fetch_one() {
  local path="$1" want="$2"
  local out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then return 0; fi
  curl -fsSL -C - -o "$out" "$BASE/$path" || { printf '[hf] FAILED %s\n' "$path"; return 1; }
  local got; got=$(stat -c%s "$out")
  [ "$got" = "$want" ] || { printf '[hf] SIZE MISMATCH %s %s != %s\n' "$path" "$got" "$want"; return 1; }
}
export -f fetch_one
export BASE DEST

# One path and one size per invocation: neither contains a space, so xargs can
# split on whitespace and no quoting has to survive a subshell.
awk '{print $1, $2}' /tmp/hf_files.txt \
  | xargs -P "$WORKERS" -n 2 bash -c 'fetch_one "$1" "$2"' _ 
log "done: $(ls "$DEST" | wc -l) top-level entries"
[ -f "$DEST/manifest.json" ] && log "manifest present" || log "NO MANIFEST"
