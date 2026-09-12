#!/usr/bin/env bash
# Wait for the A3VLM chain to finish, copy the (small) results off the pod to the MAIN volume via
# the dev pod, then delete the pod. Written for the AP-JP-1 fallback pod, where both the pod
# ($18.36/h) and its volume ($0.15/GB-month) are worth releasing promptly.
#   bash runpod/baselines/a3vlm/collect_and_teardown.sh [pod-name] [--no-delete]
set -uo pipefail
POD="${1:-bl-a3vlm}"; NODELETE="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$HERE/../../external/common/pods.sh"
DEV=v0clt0iywmvoc5
DEV_IP=$(runpodctl ssh info $DEV 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])")
DEV_PORT=$(runpodctl ssh info $DEV 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['port'])")
SSH_OPTS="-i /root/.ssh/runpod_key -p $DEV_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
while true; do
  state=$(bash "$P" run "$POD" "if [ -f /workspace/bl/runs/a3vlm/CHAIN_DONE ]; then echo DONE; elif pgrep -f 'main_finetun[e]|eval_affordance_v[2]' >/dev/null; then echo RUNNING; else echo IDLE; fi" 2>/dev/null | tail -1)
  echo "[$(date -u +%H:%M)] $POD: ${state:-unreachable}"
  if [ "$state" = "DONE" ]; then
    echo "== copying results to the main volume"
    bash "$P" run "$POD" "rsync -a -e 'ssh $SSH_OPTS' /workspace/bl/runs/a3vlm/eval/ root@$DEV_IP:/workspace/datasets/baselines/results/a3vlm/ && \
      rsync -a -e 'ssh $SSH_OPTS' /workspace/bl/logs/ root@$DEV_IP:/workspace/datasets/baselines/results/a3vlm/logs/ && echo COPIED"
    # keep the final checkpoint metadata only (the 13B shards are ~40 GB and are not worth storing)
    bash "$P" run "$POD" "ls -la /workspace/bl/runs/a3vlm | tail -5"
    if [ "$NODELETE" != "--no-delete" ]; then
      bash "$P" delete "$POD"
      echo "== pod deleted. Remember: 'runpodctl network-volume rm 18bdh0pzec' (bl-apjp) once the"
      echo "   results are verified, and 'runpodctl network-volume rm 5pw4vigftc' (bl-eufr, unused)."
    fi
    exit 0
  fi
  sleep 600
done
