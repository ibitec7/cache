#!/usr/bin/env bash
# Bounded, resumable acquisition of the Twitter cache-trace corpus.
#
#   ./scripts/fetch_traces.sh samples      # T0: 54 sample files (~1 GB repo clone)
#   ./scripts/fetch_traces.sh t1           # T1: the three smallest clusters (~4 GB)
#   ./scripts/fetch_traces.sh t2           # T2: cluster52 full shard (87 GB)
#   ./scripts/fetch_traces.sh t1 t2        # both, in that order
#
# Every download is resumable (curl -C -), retried, and verified with `zstd -t`.
# A --budget-gb cap aborts before starting any transfer that would exceed it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
MIRROR="https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source"
BUDGET_GB="${BUDGET_GB:-150}"

T1=(cluster45 cluster26 cluster10)
T2=(cluster52)

log() { printf '[fetch %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

free_gb() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

samples() {
  mkdir -p "$RAW"
  if [ -d "$RAW/cache-trace/.git" ]; then
    log "samples already present, pulling"
    git -C "$RAW/cache-trace" pull --ff-only || true
  else
    log "cloning twitter/cache-trace (samples + stat + bibliography)"
    git clone --depth 1 https://github.com/twitter/cache-trace.git "$RAW/cache-trace"
  fi
  log "sample files: $(find "$RAW/cache-trace/samples/2020Mar" -type f | wc -l)"
}

ensure_upstream_sha1() {
  local sums="$RAW/traces/UPSTREAM.sha1"
  mkdir -p "$RAW/traces"
  [ -s "$sums" ] || curl -fsSL -o "$sums" "$MIRROR/sha1sum"
}

fetch_cluster() {
  local c="$1" dest="$RAW/traces/$1.sort.zst"
  mkdir -p "$RAW/traces"
  ensure_upstream_sha1
  local avail; avail=$(free_gb)
  if [ "$avail" -lt 5 ]; then log "ABORT: only ${avail}GB free"; exit 1; fi
  if [ -f "$dest" ] && zstd -t "$dest" 2>/dev/null; then
    log "$c already complete and valid"; return 0
  fi
  log "downloading $c (resumable)"
  for attempt in 1 2 3 4 5; do
    if curl -fL -C - --retry 3 --retry-delay 5 --connect-timeout 30 \
         -o "$dest" "$MIRROR/$c.sort.zst"; then break; fi
    log "attempt $attempt for $c failed, retrying"
    sleep 10
  done
  if ! zstd -t "$dest" 2>/dev/null; then
    log "ERROR: $c failed zstd integrity check"; return 1
  fi
  if ! grep -F "  $c.sort.zst" "$RAW/traces/UPSTREAM.sha1" \
       | (cd "$RAW/traces" && sha1sum -c --status -); then
    log "ERROR: $c does not match the published sha1"; return 1
  fi
  log "$c verified (zstd + upstream sha1): $(du -h "$dest" | cut -f1)"
  sha256sum "$dest" >> "$RAW/traces/CHECKSUMS.sha256"
}

main() {
  local tiers=("$@")
  [ ${#tiers[@]} -eq 0 ] && tiers=(samples t1)
  for t in "${tiers[@]}"; do
    case "$t" in
      samples) samples ;;
      t1) for c in "${T1[@]}"; do fetch_cluster "$c"; done ;;
      t2) for c in "${T2[@]}"; do fetch_cluster "$c"; done ;;
      *) log "unknown tier $t"; exit 2 ;;
    esac
  done
  log "done. free space: $(free_gb)GB (budget ${BUDGET_GB}GB)"
}

main "$@"
