#!/bin/bash
# Per-sample SF3D metric probe for four more checkpoints (2026-09-15): dense_seed7, h1anchor, query, attnpool.
# ONE process with all four models resident (frames are read once from the network-volume LMDB, which is the
# bottleneck: four separate processes ran slower in aggregate and one was OOM-killed under the 31 GB cgroup cap).
# Output: experiments/20260913_joint4_decoder_l2anchor_dense/sf3d_per_sample_metrics_extra4.csv; append its rows
# (minus header) to sf3d_per_sample_metrics.csv afterwards.
# Run detached ON THE POD:  nohup bash runpod/mao_probe_extra.sh > experiments/mao_probe_extra.log 2>&1 &
# Marker: EXTRA4_DONE.
cd /workspace/SegAffordance
export PATH=/opt/venv/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache TRITON_CACHE_DIR=/root/triton_cache HF_HOME=/root/hfcache HF_HUB_OFFLINE=1
D=experiments/20260913_joint4_decoder_l2anchor_dense
DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
POOL=config/sf3d_test_decoder_rgb_scalefree.yaml
python tools/sf3d_mao_probe.py \
  --model dense_seed7 $DENSE experiments/20260913_joint4_decoder_l2anchor_dense_seed7/checkpoints/best-epoch19-sf3dval1.0205.ckpt \
  --model h1anchor $POOL experiments/20260912_joint4_decoder_h1anchor/checkpoints/best-epoch16-sf3dval1.1649.ckpt \
  --model query config/joint4_decoder_l2anchor_query.yaml experiments/20260913_joint4_decoder_l2anchor_query/checkpoints/best-epoch16-sf3dval1.1168.ckpt \
  --model attnpool config/joint4_decoder_l2anchor_attnpool.yaml experiments/20260913_joint4_decoder_l2anchor_attnpool/checkpoints/best-epoch19-sf3dval1.1630.ckpt \
  --out $D/sf3d_per_sample_metrics_extra4.csv 2>&1 | grep -v -i 'warning\|Loaded\|Item keys\|Splitting\|scenes\|LMDB\|load_state'
echo "EXTRA4_DONE $(date) rows $(wc -l < $D/sf3d_per_sample_metrics_extra4.csv)"
