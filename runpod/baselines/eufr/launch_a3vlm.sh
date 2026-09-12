#!/usr/bin/env bash
# One-shot A3VLM launch on an EU-FR-1 pod that already exists (created by
# `runpod/baselines/eufr/pods.sh gpu bl-a3vlm <n>`), on volume bl-eufr which already holds
# SPHINX-1k and the images. Steps: push tooling -> env -> refresh the VQA JSONs from the dev
# pod -> full-width smoke -> full 3-epoch run, detached.
#
#   bash runpod/baselines/eufr/launch_a3vlm.sh [pod-name]
#
# The smoke runs at the REAL gpu count because A3VLM cannot be tested at a smaller one
# (sdp shards the optimizer over data-parallel ranks; DP=1 always OOMs).
set -euo pipefail
POD="${1:-bl-a3vlm}"
HERE="$(cd "$(dirname "$0")" && pwd)"; P="$HERE/pods.sh"; SEG="$(cd "$HERE/../../.." && pwd)"
DEV_IP=$(runpodctl ssh info v0clt0iywmvoc5 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])")
DEV_PORT=$(runpodctl ssh info v0clt0iywmvoc5 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['port'])")
echo "== $(date -u) launching A3VLM on $POD (dev pod $DEV_IP:$DEV_PORT)"

bash "$P" run "$POD" "mkdir -p /workspace/SegAffordance/{tools,runpod/baselines,config,experiments/baselines_sf3d} /workspace/bl/{scripts,logs}"
for d in tools/baselines_sf3d runpod/baselines/a3vlm config/baselines; do
  bash "$P" scp-to "$POD" "$SEG/$d" "/workspace/SegAffordance/$d"
done
bash "$P" scp-to "$POD" "$SEG/experiments/baselines_sf3d/splits.json" /workspace/SegAffordance/experiments/baselines_sf3d/splits.json
bash "$P" scp-to "$POD" "$SEG/runpod/baselines/a3vlm/setup_env.sh" /workspace/bl/scripts/setup_env.sh

echo "== env"
bash "$P" run "$POD" "cd /workspace/bl && bash scripts/setup_env.sh 2>&1 | grep -viE 'warning|notice|pycairo' | tail -5"

# The VQA JSONs are regenerated whenever the converter changes; the images are content-identical
# and rsync skips them. Pull the current ones from the dev pod.
echo "== refresh data"
bash "$P" run "$POD" "which rsync > /dev/null || { apt-get update -qq > /dev/null 2>&1; apt-get install -y -qq rsync > /dev/null 2>&1; }; which rsync"
bash "$P" scp-to "$POD" "$HOME/.runpod/ssh/runpodctl-ssh-key" /root/.ssh/runpod_key
bash "$P" run "$POD" "chmod 600 /root/.ssh/runpod_key; mkdir -p /workspace/bl/data/a3vlm; rsync -a --info=stats2 -e 'ssh -i /root/.ssh/runpod_key -p $DEV_PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' root@$DEV_IP:/workspace/datasets/baselines/stage/a3vlm/ /workspace/bl/data/a3vlm/ 2>&1 | tail -4; ls /workspace/bl/data/a3vlm | head -3; ls /workspace/bl/data/a3vlm/images | wc -l"

echo "== smoke (full width)"
bash "$P" run "$POD" "cd /workspace/bl && MODE=smoke bash /workspace/SegAffordance/runpod/baselines/a3vlm/chain.sh 2>&1 | tail -25"

echo "== if the smoke printed preds_*.jsonl line counts, start the real run with:"
echo "   bash $P run $POD \"cd /workspace/bl && (nohup env MODE=train bash /workspace/SegAffordance/runpod/baselines/a3vlm/chain.sh > logs/chain_a3vlm.log 2>&1 &)\""
