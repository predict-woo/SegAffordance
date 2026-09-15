#!/usr/bin/env bash
# Run the trained 3DOI checkpoint on the staged Fig. 5 hand-video frames (handvideo_stage.py threedoi)
# and export in the figure schema (line_2d everywhere, depth-plane lift where depth exists).
#   bash runpod/baselines/handvideo_3doi.sh [checkpoint]
set -euo pipefail
B=/workspace/datasets/baselines; H=$B/handvideo/3doi; R=$B/repos/3DOI/monoarti; VENV=/opt/venv_3doi; LOGS=$B/logs
CKPT="${1:-$B/results/3doi_runpod/checkpoint_best.pth}"
export TMPDIR=/workspace/tmp; mkdir -p /workspace/tmp $B/results/3doi_runpod
echo "== $(date -u) handvideo 3doi from $CKPT"
cd /workspace/SegAffordance
$VENV/bin/python tools/baselines_sf3d/run.py tools/baselines_sf3d/threedoi_export.py --repo $R --ckpt $CKPT \
  --data $H --out $H/preds_raw.jsonl --batch 2 --workers 4 > $LOGS/handvideo_3doi.log 2>&1 || { echo "export failed"; tail -25 $LOGS/handvideo_3doi.log; exit 1; }
tail -2 $LOGS/handvideo_3doi.log
$VENV/bin/python tools/baselines_sf3d/run.py tools/baselines_sf3d/handvideo_export.py threedoi --stage $H \
  --pred $H/preds_raw.jsonl --out $B/results/3doi_runpod/handvideo_preds.jsonl
echo "== $(date -u) HANDVIDEO_DONE 3doi"
