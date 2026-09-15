#!/usr/bin/env bash
# Run a trained OPDFormer / MOPD checkpoint on the staged Fig. 5 hand-video frames and export.
#   bash runpod/baselines/handvideo_opd.sh <run dir name> <variant c_rgbd|p_rgb|mopd_rgb>
# Uses the run's own repo (OPDMulti or MOPD), the 512x384 stage from handvideo_stage.py opd and the
# training pixel stats; keeps <stage>/out_<run>/inference/instances_predictions.pth for the figure.
set -euo pipefail
RUN="$1"; V="$2"
B=/workspace/datasets/baselines; H=$B/handvideo/opd; LOGS=$B/logs
export TMPDIR=/workspace/tmp; mkdir -p /workspace/tmp $B/results/$RUN
case $V in
  c_rgbd)   R=$B/repos/OPDMulti/opdformer; CFG=configs/opd_c_real.yaml; FMT=RGBD; SKIP=--skip-no-depth;;
  p_rgb)    R=$B/repos/OPDMulti/opdformer; CFG=configs/opd_p_real.yaml; FMT=RGB;  SKIP=;;
  mopd_rgb) R=$B/repos/MOPD/opdformer;     CFG=configs/opd_p_real.yaml; FMT=RGB;  SKIP=;;
  *) echo "bad variant $V"; exit 1;;
esac
read -r MEAN STD <<<"$(python - <<PY
import json
s = json.load(open("$H/stats.json")); n = 4 if "$FMT" == "RGBD" else 3
f = lambda xs: "[" + ",".join(f"{x:.6f}" for x in xs[:n]) + "]"
print(f(s["pixel_mean"]), f(s["pixel_std"]))
PY
)"
COMMON="MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN $MEAN MODEL.PIXEL_STD $STD DATALOADER.NUM_WORKERS 4 INPUT.MASK_FORMAT bitmask \
INPUT.IMAGE_SIZE 512 INPUT.MIN_SIZE_TRAIN (512,) INPUT.MAX_SIZE_TRAIN 512 INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TEST 512"
echo "== $(date -u) handvideo $RUN ($V) on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $R
rm -rf $H/out_$RUN
python evaluate_on_log.py --config-file $CFG --output-dir $H/out_$RUN --data-path $H/MotionDataset_h5 --input-format $FMT \
  --model_attr_path $H/obj_info.json --opts MODEL.WEIGHTS $B/runs/$RUN/model_final.pth $COMMON DATASETS.TEST "('MotionNet_test',)" \
  > $LOGS/handvideo_$RUN.log 2>&1 || { echo "evaluate failed"; tail -25 $LOGS/handvideo_$RUN.log; exit 1; }
ls -la $H/out_$RUN/inference/instances_predictions.pth
cd /workspace/SegAffordance
python tools/baselines_sf3d/run.py tools/baselines_sf3d/handvideo_export.py opd --stage $H \
  --pred $H/out_$RUN/inference/instances_predictions.pth --out $B/results/$RUN/handvideo_preds.jsonl $SKIP
echo "== $(date -u) HANDVIDEO_DONE $RUN"
