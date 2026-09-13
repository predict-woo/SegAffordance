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
    # Keep the trained MODEL (2 x 19.9 GB shards + tokenizer/config) on the main EU-RO volume: it
    # cost ~$400 to produce and re-evaluation or a 3rd-epoch continuation would need it. The 63 GB
    # optimizer state is dropped.
    echo "== copying the final model shards to the main volume (~40 GB)"
    bash "$P" run "$POD" "last=\$(ls -d /workspace/bl/runs/a3vlm/epoch* | sort -V | tail -1); echo \"final checkpoint: \$last\"; \
      rsync -a --info=progress2 -e 'ssh $SSH_OPTS' --include='*.model.pth' --include='tokenizer.model' --include='config.json' --include='meta.json' --exclude='*' \
      \$last/ root@$DEV_IP:/workspace/datasets/baselines/results/a3vlm/final_model/ 2>&1 | tail -2 && echo MODEL_COPIED"
    bash "$P" run "$POD" "ls -la /workspace/bl/runs/a3vlm | tail -5"
    if [ "$NODELETE" != "--no-delete" ]; then
      bash "$P" delete "$POD"
      echo "== pod deleted. The AP-JP-1 volume bl-apjp (18bdh0pzec, 700 GB, ~\$105/month) and the unused"
      echo "   EU-FR-1 volume bl-eufr (5pw4vigftc, 150 GB) should be deleted once results are verified:"
      echo "   runpodctl network-volume delete 18bdh0pzec ; runpodctl network-volume delete 5pw4vigftc"
    fi
    exit 0
  fi
  sleep 600
done
