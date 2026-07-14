#!/usr/bin/env bash
# Control the SegAffordance dev pod from the local machine.
#
#   bash runpod/dev.sh status          # pod state + hourly cost
#   bash runpod/dev.sh start           # start, wait for SSH, refresh ~/.ssh/config
#   bash runpod/dev.sh stop            # stop (billing stops, /workspace persists)
#   bash runpod/dev.sh ssh             # interactive shell on the pod
#   bash runpod/dev.sh run <cmd...>    # run a command in /workspace/SegAffordance
#   bash runpod/dev.sh sync            # mutagen sync status (~/dev/ethz-workspace)
#   bash runpod/dev.sh sync-reset      # recreate the mutagen session (after pod recreation)
set -euo pipefail

POD_NAME="segaffordance-dev"
HOST_ALIAS="segaff-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pod_id() {
  runpodctl pod list --all | python3 -c "
import json, sys
pods = [p for p in json.load(sys.stdin) if p['name'] == '$POD_NAME']
if not pods:
    sys.exit('no pod named $POD_NAME — create one (see runpod/README.md)')
print(pods[0]['id'])
"
}

wait_for_ssh() {
  local id="$1"
  for _ in $(seq 1 40); do
    if runpodctl ssh info "$id" 2>/dev/null | grep -q '"ip"'; then
      return 0
    fi
    sleep 15
  done
  echo "timed out waiting for SSH (pod may still be pulling the image)" >&2
  return 1
}

cmd="${1:-status}"
[ $# -gt 0 ] && shift

case "$cmd" in
  status)
    id="$(pod_id)"
    runpodctl pod get "$id" | python3 -c "
import json, sys
p = json.load(sys.stdin)
print(f\"{p['name']} ({p['id']}): {p['desiredStatus']}  \${p['costPerHr']}/hr\")
"
    ;;
  start)
    id="$(pod_id)"
    runpodctl pod start "$id" >/dev/null
    echo "started $id, waiting for SSH..."
    wait_for_ssh "$id"
    bash "$SCRIPT_DIR/update-ssh-config.sh" "$id"
    if ssh -o BatchMode=yes "$HOST_ALIAS" nvidia-smi --query-gpu=name --format=csv,noheader; then
      echo "ready: ssh $HOST_ALIAS"
    else
      echo "WARNING: pod started but has no GPU (stock was likely taken while" >&2
      echo "stopped). Consider delete + recreate, see runpod/README.md." >&2
      exit 1
    fi
    ;;
  stop)
    id="$(pod_id)"
    runpodctl pod stop "$id" >/dev/null
    echo "stopped $id (billing stopped, /workspace persists)"
    ;;
  ssh)
    exec ssh "$HOST_ALIAS"
    ;;
  run)
    [ $# -gt 0 ] || { echo "usage: dev.sh run <cmd...>" >&2; exit 2; }
    ssh -o BatchMode=yes "$HOST_ALIAS" "cd /workspace/SegAffordance 2>/dev/null || cd /workspace; $*"
    ;;
  sync)
    mutagen sync list ethz-workspace
    ;;
  sync-reset)
    mutagen sync terminate ethz-workspace 2>/dev/null || true
    mutagen sync create --name=ethz-workspace \
      --ignore="/datasets" \
      --ignore="/cache" \
      --ignore="/models" \
      --ignore="/checkpoints" \
      --ignore="/runs/wandb" \
      --ignore="**/__pycache__" \
      --ignore="*.pyc" \
      --ignore="*.pt" \
      --ignore="*.ckpt" \
      --ignore=".DS_Store" \
      "$HOME/dev/ethz-workspace" "$HOST_ALIAS:/workspace"
    ;;
  *)
    echo "usage: dev.sh {status|start|stop|ssh|run <cmd...>|sync|sync-reset}" >&2
    exit 2
    ;;
esac
