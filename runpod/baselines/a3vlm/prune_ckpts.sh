#!/usr/bin/env bash
# Keep only the newest N A3VLM checkpoint dirs. Each one is ~103 GB (40 GB model shards +
# 63 GB consolidated optimizer state), so an unpruned 3-epoch run with per-500-iteration saves
# would need ~1.2 TB (measured: 135 GB per checkpoint, not the 103 GB the file sizes suggest --
# the "other"/rank-specific files add ~30 GB). Never touches a dir younger than $MIN_AGE_MIN
# minutes, so a checkpoint being written is safe.
#   KEEP=2 RUNS=/workspace/bl/runs/a3vlm bash prune_ckpts.sh
set -uo pipefail
# ROOT, not a single run dir: the smoke writes to runs/a3vlm_smoke and the real run to runs/a3vlm,
# and a pruner pointed at one silently lets the other grow (it did, to 612 GB, on 2026-09-13).
ROOT="${ROOT:-/workspace/bl/runs}"; KEEP="${KEEP:-2}"; MIN_AGE_MIN="${MIN_AGE_MIN:-8}"
while true; do
  for RUNS in "$ROOT"/*/; do
    [ -d "$RUNS" ] || continue
    mapfile -t dirs < <(ls -dt "$RUNS"epoch* 2>/dev/null)
    if [ "${#dirs[@]}" -gt "$KEEP" ]; then
      for d in "${dirs[@]:$KEEP}"; do
        if [ -z "$(find "$d" -maxdepth 1 -newermt "-${MIN_AGE_MIN} minutes" -print -quit 2>/dev/null)" ]; then
          echo "[$(date -u +%H:%M)] pruning $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
          rm -rf "$d"
        fi
      done
    fi
  done
  used=$(du -sb /workspace 2>/dev/null | cut -f1)
  echo "[$(date -u +%H:%M)] volume $((${used:-0}/1073741824)) GB, checkpoint dirs: $(ls -d "$ROOT"/*/epoch* 2>/dev/null | wc -l)"
  sleep 300
done
