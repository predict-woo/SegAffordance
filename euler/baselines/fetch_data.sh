#!/usr/bin/env bash
# Pull the converted baseline datasets + SPHINX weights onto Euler scratch.
# Runs ON EULER (login node). Source = the RunPod dev pod, which holds the staged trees;
# SPHINX-1k comes straight from Hugging Face (38 GB) when the login node has internet.
#
#   bash fetch_data.sh <pod-ip> <pod-port>     # ip/port from `runpodctl ssh info` on the Mac
#
# Needs ~57 GB: 3doi 17 GB, a3vlm 1.8 GB, sphinx1k 38 GB.
set -euo pipefail
IP="${1:?pod ip}"; PORT="${2:?pod port}"
BASE="${EULER_BASE:-$SCRATCH/baselines}"
KEY="${RUNPOD_KEY:-$HOME/.ssh/runpod_key}"
mkdir -p "$BASE"/{data,ckpt,repos,runs,logs}
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
echo "== $(date -u) pulling datasets from $IP:$PORT"
rsync -a --info=progress2 -e "$SSH" "root@$IP:/workspace/datasets/baselines/stage/3doi/"  "$BASE/data/3doi/"
rsync -a --info=progress2 -e "$SSH" "root@$IP:/workspace/datasets/baselines/stage/a3vlm/" "$BASE/data/a3vlm/"
echo "== $(date -u) datasets done"; du -sh "$BASE"/data/*
# SPHINX-1k: prefer a direct HF download (faster than relaying through the pod)
mkdir -p "$BASE/ckpt/sphinx1k"; cd "$BASE/ckpt/sphinx1k"
HF=https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/resolve/main/finetune/mm/SPHINX/SPHINX-1k
if curl -sI --max-time 20 "$HF/config.json" | head -1 | grep -qE '200|302'; then
  for f in config.json meta.json tokenizer.model; do [ -s $f ] || curl -sL -o $f "$HF/$f"; done
  for f in consolidated.00-of-02.model.pth consolidated.01-of-02.model.pth; do
    [ -f .done_$f ] || { curl -L -C - -o $f "$HF/$f" && [ "$(stat -c %s $f)" = 19909875004 ] && touch .done_$f; } &
  done; wait
else
  echo "no HF access from this node; relaying via the pod"
  rsync -a --info=progress2 -e "$SSH" "root@$IP:/workspace/bl/ckpt/sphinx1k/" "$BASE/ckpt/sphinx1k/" || \
    echo "NOTE: sphinx1k lives on the EU-FR-1 volume, not this pod -- run this against the A3VLM pod instead"
fi
ls -la "$BASE/ckpt/sphinx1k"; echo "== $(date -u) fetch done"
