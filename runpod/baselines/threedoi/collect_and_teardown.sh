#!/usr/bin/env bash
# Wait for the 3DOI chain on a pod to finish, copy predictions, logs and the best-val checkpoint
# (~1.5 GB) to the MAIN volume via the dev pod, then delete the pod.
#   bash runpod/baselines/threedoi/collect_and_teardown.sh [pod-name] [--no-delete]
set -uo pipefail
POD="${1:-bl-3doi}"; NODELETE="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$HERE/../../external/common/pods.sh"
DEV=v0clt0iywmvoc5
DEV_IP=$(runpodctl ssh info $DEV 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])")
DEV_PORT=$(runpodctl ssh info $DEV 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['port'])")
SSH_OPTS="-i /root/.ssh/runpod_key -p $DEV_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
B=/workspace/datasets/baselines; RUNS=$B/runs/3doi
while true; do
  state=$(bash "$P" run "$POD" "if [ -f $RUNS/CHAIN_DONE ]; then echo DONE; elif pgrep -f 'train\.py --config-name sam_sf3d|threedoi_expor[t]' >/dev/null; then echo RUNNING; else echo IDLE; fi" 2>/dev/null | tail -1)
  echo "[$(date -u +%H:%M)] $POD: ${state:-unreachable}"
  if [ "$state" = "DONE" ]; then
    echo "== copying results + best checkpoint to the main volume"
    bash "$P" run "$POD" "mkdir -p /tmp/out && cp $RUNS/preds.jsonl /tmp/out/ && cp $RUNS/checkpoints/checkpoint_best.pth /tmp/out/ 2>/dev/null; cp $RUNS/config.yaml /tmp/out/ 2>/dev/null; \
      rsync -a -e 'ssh $SSH_OPTS' /tmp/out/ root@$DEV_IP:/workspace/datasets/baselines/results/3doi_runpod/ && \
      rsync -a -e 'ssh $SSH_OPTS' --include='*3doi*' --include='chain_3doi.log' --exclude='*' $B/logs/ root@$DEV_IP:/workspace/datasets/baselines/results/3doi_runpod/logs/ && echo COPIED"
    if [ "$NODELETE" != "--no-delete" ]; then bash "$P" delete "$POD"; echo "== pod deleted"; fi
    exit 0
  fi
  sleep 600
done
