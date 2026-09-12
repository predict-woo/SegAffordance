#!/usr/bin/env bash
# Keep only the newest N A3VLM checkpoint dirs. Each one is ~103 GB (40 GB model shards +
# 63 GB consolidated optimizer state), so an unpruned 3-epoch run with per-500-iteration saves
# would need ~900 GB. Runs alongside training; never touches a dir younger than $MIN_AGE_MIN
# minutes, so a checkpoint being written is safe.
#   KEEP=2 RUNS=/workspace/bl/runs/a3vlm bash prune_ckpts.sh
set -uo pipefail
RUNS="${RUNS:-/workspace/bl/runs/a3vlm}"; KEEP="${KEEP:-2}"; MIN_AGE_MIN="${MIN_AGE_MIN:-10}"
while true; do
  if [ -d "$RUNS" ]; then
    mapfile -t dirs < <(ls -dt "$RUNS"/epoch* 2>/dev/null)
    if [ "${#dirs[@]}" -gt "$KEEP" ]; then
      for d in "${dirs[@]:$KEEP}"; do
        if [ -z "$(find "$d" -maxdepth 1 -newermt "-${MIN_AGE_MIN} minutes" -print -quit 2>/dev/null)" ]; then
          echo "[$(date -u +%H:%M)] pruning $(basename "$d") ($(du -sh "$d" 2>/dev/null | cut -f1))"
          rm -rf "$d"
        fi
      done
    fi
    df_used=$(du -sb /workspace 2>/dev/null | cut -f1)
    echo "[$(date -u +%H:%M)] volume $((${df_used:-0}/1073741824)) GB, checkpoints: $(ls -d "$RUNS"/epoch* 2>/dev/null | wc -l)"
  fi
  sleep 300
done
