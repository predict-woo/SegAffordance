#!/bin/bash
# One-off dev-pod chain (2026-09-14 night): wait for the per-sample probe, summarise it, then run the
# sign-corrected ARCTIC probes for five checkpoints, the hand-video panels (SF3D-only control vs final
# dense model), the phone photos, and the Fig. 4 panels (GT | OPDFormer-C oracle instance | ours).
# Run detached ON THE POD:  nohup bash runpod/night3_chain.sh > /workspace/SegAffordance/experiments/night3_chain.log 2>&1 &
# Markers: PROBE_DONE, ARCTIC_DONE, PANELS_DONE, PHONE_DONE, FIG4_DONE, CHAIN_DONE.
cd /workspace/SegAffordance
export PATH=/opt/venv/bin:$PATH
export TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache TRITON_CACHE_DIR=/root/triton_cache HF_HOME=/root/hfcache HF_HUB_OFFLINE=1
DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
CKD=experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt
CKC=experiments/20260913_sf3d_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval1.1144.ckpt
CKL=experiments/20260912_joint4_decoder_l2anchor/checkpoints/best-epoch17-sf3dval1.1495.ckpt
CKDIR=experiments/20260914_joint4_decoder_dense_directloss/checkpoints/best-epoch16-sf3dval0.7608.ckpt
CKS=experiments/20260914_joint4_decoder_dense_sampledtraj/checkpoints/best-epoch19-sf3dval1.1797.ckpt
CSV=experiments/20260913_joint4_decoder_l2anchor_dense/sf3d_per_sample_metrics.csv

while pgrep -f '/opt/venv/bin/python tools/sf3d_mao_probe.py' > /dev/null; do sleep 60; done
echo "PROBE_DONE $(date)"; wc -l $CSV
python tools/sf3d_mao_probe.py --summarize $CSV

echo "=== ARCTIC probe (sign-fixed, point offset) $(date)"
python tools/arctic_axis_probe.py --model dense $DENSE $CKD --model sf3d_only $DENSE $CKC \
  --model l2anchor config/sf3d_test_decoder_rgb_scalefree.yaml $CKL --model directloss $DENSE $CKDIR \
  --model sampledtraj $DENSE $CKS --out experiments/20260913_joint4_decoder_l2anchor_dense/arctic_axis_probe_signfix.csv \
  2>&1 | grep -v -i 'warning\|^[0-9]*/[0-9]*$\|Loaded\|Item keys\|Splitting\|scenes\|LMDB\|load_state'
echo "ARCTIC_DONE $(date)"

for ds in hoi4d epic arctic; do
  echo "=== panels $ds $(date)"
  python tools/hoi4d_predict_articulation.py --dataset $ds --model sf3d_only $DENSE $CKC --model dense $DENSE $CKD \
    --num 12 --per-seq 1 --scale 2 --out viz/20260914_handvideo_control_vs_dense/$ds 2>&1 | grep -E 'rendering|Error|Traceback|done'
done
echo "PANELS_DONE $(date)"

echo "=== phone $(date)"
python tools/predict_image.py --model sf3d_only $DENSE $CKC --model dense $DENSE $CKD \
  --case viz/20260911_iphone_probe/inputs/IMG_0876.jpg 'open door' --case viz/20260911_iphone_probe/inputs/IMG_0877.jpg 'open closet' \
  --case viz/20260911_iphone_probe/inputs/IMG_0874.jpg 'close laptop' --case viz/20260911_iphone_probe/inputs/IMG_0875.jpg 'push chair forward' \
  --out viz/20260914_iphone_control_vs_dense --f35 26 2>&1 | grep -E 'wrote|done|Error|Traceback'
echo "PHONE_DONE $(date)"

echo "=== fig4 $(date)"
python tools/sf3d_vis_baseline_vs_ours.py --model ours $DENSE $CKD \
  --baseline OPDFormer-C /workspace/datasets/baselines/results/opd_c_rgbd/preds.jsonl \
  --val-idx 1696 1773 529 4233 283 501 1140 1246 4193 615 --out viz/20260914_fig4_sf3d_gt_baseline_ours 2>&1 | grep -E 'wrote|done|Error|Traceback'
echo "FIG4_DONE $(date)"
echo "CHAIN_DONE $(date)"
