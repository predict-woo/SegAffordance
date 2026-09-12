#!/usr/bin/env bash
# Watch one baseline chain on an EU-FR-1 pod (volume root /workspace/bl); on CHAIN_DONE copy
# the small results to the volume's results dir and DELETE the pod; on a dead chain print
# CHAIN_DEAD and exit 1 (pod kept for debugging). One line per poll (10 min).
#   bash runpod/baselines/eufr/watch_chain.sh <pod-name> <run-name> "<chain pgrep pattern>"
set -uo pipefail
POD="$1"; RUN="$2"; PAT="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$HERE/pods.sh"
B=/workspace/bl; RUNS=$B/runs/$RUN
while true; do
  out=$(bash "$P" run "$POD" "if [ -f $RUNS/CHAIN_DONE ]; then echo DONE; elif pgrep -f '$PAT' >/dev/null; then echo RUNNING; else echo DEAD; fi; \
    grep -hoE 'Epoch: \[[0-9]+\] *\[[0-9/]+\]|iter: *[0-9]+|loss: *[0-9.]+' $B/logs/train_${RUN}.log 2>/dev/null | tail -2 | tr '\n' ' ' | cut -c1-160; \
    grep -E -m3 'Traceback|CUDA out of memory|Killed|ChildFailedError' $B/logs/train_${RUN}.log 2>/dev/null | head -3" 2>&1)
  state=$(echo "$out" | head -1); prog=$(echo "$out" | tail -n +2 | tr '\n' ' ' | cut -c1-300)
  echo "[$(date -u +%H:%M)] $POD/$RUN $state $prog"
  case "$state" in
    DONE)
      bash "$P" run "$POD" "mkdir -p $B/results/$RUN && cp -r $RUNS/eval $B/results/$RUN/ 2>/dev/null; cp $B/logs/train_${RUN}.log $B/logs/eval_*_${RUN}.log $B/results/$RUN/ 2>/dev/null; ls -la $B/results/$RUN"
      echo "CHAIN_DONE $POD/$RUN -> results copied on the volume"
      echo "NOTE: results live on volume bl-eufr; copy them off BEFORE deleting the pod."
      exit 0 ;;
    DEAD) echo "CHAIN_DEAD $POD/$RUN (pod kept)"; exit 1 ;;
  esac
  sleep 600
done
