#!/usr/bin/env bash
# Watch one baseline chain on one pod; on CHAIN_DONE copy the small results to
# the volume's results dir and DELETE the pod; on a dead chain print CHAIN_DEAD
# and exit 1 (pod kept for debugging). Emits one line per poll (10 min).
#   bash runpod/baselines/watch_chain.sh <pod-name> <run-name> "<chain pgrep pattern>"
set -uo pipefail
POD="$1"; RUN="$2"; PAT="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$HERE/pods.sh"
B=/workspace/datasets/baselines; RUNS=$B/runs/$RUN
while true; do
  out=$(bash "$P" run "$POD" "if [ -f $RUNS/CHAIN_DONE ]; then echo DONE; elif pgrep -f '$PAT' >/dev/null; then echo RUNNING; else echo DEAD; fi; \
    tail -c 4000 $B/logs/train_${RUN}.log 2>/dev/null | grep -E 'iter:|Epoch|loss' | tail -1 | cut -c1-160; grep -E -m3 'Traceback|CUDA out of memory|Killed' $B/logs/train_${RUN}.log 2>/dev/null | head -3" 2>&1)
  state=$(echo "$out" | head -1); prog=$(echo "$out" | tail -n +2 | tr '\n' ' ' | cut -c1-300)
  echo "[$(date -u +%H:%M)] $POD/$RUN $state $prog"
  case "$state" in
    DONE)
      bash "$P" run "$POD" "mkdir -p $B/results/$RUN && cp $RUNS/preds.jsonl $B/results/$RUN/ 2>/dev/null; cp $B/logs/train_${RUN}.log $B/logs/test_${RUN}.log $B/results/$RUN/ 2>/dev/null; cp $RUNS/config*.yaml $RUNS/eval_ckpt.txt $B/results/$RUN/ 2>/dev/null; ls $B/results/$RUN"
      echo "CHAIN_DONE $POD/$RUN -> results copied; deleting pod"
      bash "$P" delete "$POD"; exit 0 ;;
    DEAD) echo "CHAIN_DEAD $POD/$RUN (pod kept)"; exit 1 ;;
  esac
  sleep 600
done
