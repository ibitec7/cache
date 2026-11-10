#!/usr/bin/env bash
# Parallel, resumable download of one corpus shard via HTTP range requests.
#
# MEASURED SLOWER THAN A SINGLE STREAM ON THE CMU MIRROR: 1.44 MB/s aggregate across 8
# ranges versus 2.03 to 2.32 MB/s for one curl on cluster52. The mirror throttles per
# client, so do not use this for pdl.cmu.edu. Kept for a mirror that is per-connection
# limited. It splits the object into N byte ranges, resumes each from its existing part
# size, then concatenates. Verification is the caller's job (zstd -t + upstream sha1).
#
#   ./scripts/fetch_parallel.sh cluster52 8
#
# Output: $RAW/traces/<cluster>.sort.zst (parts removed on success)
set -uo pipefail

CLUSTER="${1:?usage: fetch_parallel.sh <cluster> [parts]}"
NPARTS="${2:-8}"
MIRROR="${MIRROR:-https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source}"
RAW="${RAW:-$(cd "$(dirname "$0")/.." && pwd)/data/raw}"
URL="$MIRROR/$CLUSTER.sort.zst"
DEST="$RAW/traces/$CLUSTER.sort.zst"
PARTS="$RAW/traces/$CLUSTER.parts"

mkdir -p "$RAW/traces" "$PARTS"

log() { printf '[par %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

[ -s "$DEST" ] && { log "$CLUSTER already present at $DEST"; exit 0; }

SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{v=$2} END{gsub(/\r/,"",v); print v}')
[ -n "${SIZE:-}" ] && [ "$SIZE" -gt 0 ] || { log "ERROR: cannot read Content-Length for $URL"; exit 1; }
CHUNK=$(( (SIZE + NPARTS - 1) / NPARTS ))
log "$CLUSTER size=$SIZE bytes, $NPARTS parts of $CHUNK"

fetch_part() {
  local i="$1" start end want have part
  start=$(( i * CHUNK ))
  end=$(( start + CHUNK - 1 ))
  [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
  want=$(( end - start + 1 ))
  part="$PARTS/part_$i"

  while :; do
    have=0
    [ -f "$part" ] && have=$(stat -c %s "$part")
    [ "$have" -ge "$want" ] && { log "part $i complete ($have/$want)"; return 0; }
    curl -fsSL -r "$(( start + have ))-$end" -o "$part.tmp" "$URL" \
      && cat "$part.tmp" >> "$part" && rm -f "$part.tmp"
    # a short write is normal (connection reset); loop until the part is full
  done
}

for i in $(seq 0 $(( NPARTS - 1 ))); do fetch_part "$i" & done
wait

TOTAL=0
for i in $(seq 0 $(( NPARTS - 1 ))); do TOTAL=$(( TOTAL + $(stat -c %s "$PARTS/part_$i") )); done
[ "$TOTAL" -eq "$SIZE" ] || { log "ERROR: assembled $TOTAL of $SIZE bytes"; exit 1; }

cat "$PARTS"/part_* > "$DEST"
rm -rf "$PARTS"
log "$CLUSTER assembled: $(du -h "$DEST" | cut -f1)"
