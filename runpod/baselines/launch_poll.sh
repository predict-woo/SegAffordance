#!/usr/bin/env bash
# Retry pod creation until stock lands, then run an on-pod command detached.
#   bash runpod/baselines/launch_poll.sh <pod-name> "<on-pod command>" [max_a100_attempts]
# Tries the A100/H100 SKUs (torch 2.1.1 cu121 image) every 3 min; after
# max_a100_attempts (default 10) also tries RTX PRO 6000 Server Edition with
# the image's newer torch (BL_ALLOW_BLACKWELL=1 required). Run detached:
#   nohup bash runpod/baselines/launch_poll.sh ... > log 2>&1 & disown
set -uo pipefail
NAME="$1"; CMD="$2"; MAXA="${3:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"; COMMON="$HERE/../external/common/pods.sh"
VOL="${BL_VOLUME:-bckt1t9uuf}"; export EXT_DC="${EXT_DC:-EU-RO-1}"
IMG_A100="runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"
IMG_BW="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
attempt=0
while true; do
  attempt=$((attempt+1)); echo "[$(date -u +%H:%M:%S)] attempt $attempt"
  id=$(runpodctl pod list 2>/dev/null | python3 -c "import json,sys; ps=json.load(sys.stdin); print(next((p['id'] for p in ps if p.get('name')=='$NAME'),''))")
  if [ -z "$id" ]; then
    for gpu in "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe"; do
      EXT_IMAGE=$IMG_A100 bash "$COMMON" gpu "$NAME" "$VOL" "$gpu" >/dev/null 2>&1 && break
    done
    id=$(runpodctl pod list 2>/dev/null | python3 -c "import json,sys; ps=json.load(sys.stdin); print(next((p['id'] for p in ps if p.get('name')=='$NAME'),''))")
    if [ -z "$id" ] && [ "$attempt" -ge "$MAXA" ] && [ "${BL_ALLOW_BLACKWELL:-0}" = "1" ]; then
      EXT_IMAGE=$IMG_BW bash "$COMMON" gpu "$NAME" "$VOL" "NVIDIA RTX PRO 6000 Blackwell Server Edition" >/dev/null 2>&1 || true
      id=$(runpodctl pod list 2>/dev/null | python3 -c "import json,sys; ps=json.load(sys.stdin); print(next((p['id'] for p in ps if p.get('name')=='$NAME'),''))")
    fi
  fi
  if [ -n "$id" ]; then
    read -r ip port <<<"$(runpodctl ssh info "$id" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ip',''), d.get('port',''))")"
    if [ -n "$port" ]; then
      echo "LANDED $NAME $id $ip:$port"
      bash "$COMMON" run "$NAME" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
      bash "$COMMON" run "$NAME" "nohup bash -lc $(printf '%q' "$CMD") > /workspace/datasets/baselines/logs/launch_$NAME.log 2>&1 < /dev/null & disown; echo started"
      echo "STARTED $NAME"; exit 0
    fi
    echo "pod $id exists, ssh not ready"
  fi
  sleep 180
done
