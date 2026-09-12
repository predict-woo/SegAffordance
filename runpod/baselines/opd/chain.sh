#!/usr/bin/env bash
# OPDFormer on SF3D: convert (owner pod only) -> train (upstream recipe) ->
# evaluate on MotionNet_test -> export shared JSONL -> CHAIN_DONE.
#   OWNER=1 bash chain.sh c_rgbd      # also runs the LMDB -> h5 converter
#   bash chain.sh p_rgbd | p_rgb       # waits for the converter's .done_all
# Upstream recipe (configs/opd_{c,p}_real.yaml -> opd_base.yaml): R50 Mask2Former,
# batch 16, AdamW 1e-4, 60k iters, steps (36k,48k), COCO Mask2Former init.
# Fit-to-data overrides only: 8 SF3D affordance classes, per-dataset pixel stats.
set -euo pipefail
V="${1:?variant c_rgbd|p_rgbd|p_rgb}"
B=/workspace/datasets/baselines; DATA=$B/data/opd_sf3d; RUNS=$B/runs/opd_$V; LOGS=$B/logs
REPO="${OPD_REPO:-OPDMulti}"; R=$B/repos/$REPO/opdformer
export TMPDIR=/workspace/tmp; mkdir -p $RUNS $LOGS /workspace/tmp
case $V in
  c_rgbd) CFG=configs/opd_c_real.yaml; FMT=RGBD;;
  p_rgbd) CFG=configs/opd_p_real.yaml; FMT=RGBD;;
  p_rgb)  CFG=configs/opd_p_real.yaml; FMT=RGB;;
  *) echo "bad variant $V"; exit 1;;
esac
echo "== $(date -u) chain $V on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [ "${OWNER:-0}" = "1" ]; then
  if [ ! -f $DATA/.done_all ]; then
    cd /workspace/SegAffordance
    python tools/baselines_sf3d/run.py tools/baselines_sf3d/sf3d_to_opd.py --out $DATA --workers "${CONVERT_WORKERS:-$(nproc)}" > $LOGS/convert_opd.log 2>&1
    python - <<'PY'
import json
for s in ["train","valid","test"]:
    d = json.load(open(f"/workspace/datasets/baselines/data/opd_sf3d/MotionDataset_h5/annotations/MotionNet_{s}.json"))
    print(s, "images", len(d["images"]), "annotations", len(d["annotations"]))
PY
    touch $DATA/.done_all
  fi
else
  while [ ! -f $DATA/.done_all ]; do echo "$(date -u +%H:%M) waiting for converter"; sleep 120; done
fi
read -r MEAN STD <<<"$(python - <<PY
import json
s = json.load(open("$DATA/stats.json")); n = 4 if "$FMT" == "RGBD" else 3
f = lambda xs: "[" + ",".join(f"{x:.6f}" for x in xs[:n]) + "]"
print(f(s["pixel_mean"]), f(s["pixel_std"]))
PY
)"
echo "pixel stats $FMT: $MEAN $STD"
cd $R
COMMON_OPTS="MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN $MEAN MODEL.PIXEL_STD $STD DATALOADER.NUM_WORKERS 8 INPUT.MASK_FORMAT bitmask"
if [ ! -f $RUNS/model_final.pth ]; then
  python train.py --config-file $CFG --output-dir $RUNS --data-path $DATA/MotionDataset_h5 --input-format $FMT \
    --model_attr_path $DATA/obj_info.json --opts $COMMON_OPTS DATASETS.TEST "('MotionNet_valid',)" ${EXTRA_OPTS:-} \
    > $LOGS/train_opd_$V.log 2>&1
fi
echo "== $(date -u) train done"; ls $RUNS | tail -5
python evaluate_on_log.py --config-file $CFG --output-dir $RUNS/test --data-path $DATA/MotionDataset_h5 --input-format $FMT \
  --model_attr_path $DATA/obj_info.json --opts MODEL.WEIGHTS $RUNS/model_final.pth $COMMON_OPTS DATASETS.TEST "('MotionNet_test',)" \
  > $LOGS/test_opd_$V.log 2>&1
echo "== $(date -u) test done"; grep -E "AP|motion" $LOGS/test_opd_$V.log | tail -20
cd /workspace/SegAffordance
python tools/baselines_sf3d/run.py tools/baselines_sf3d/opd_preds_to_jsonl.py --pred $RUNS/test/inference/instances_predictions.pth --data-dir $DATA --out $RUNS/preds.jsonl
wc -l $RUNS/preds.jsonl; cp $RUNS/config.yaml $RUNS/config_resolved.yaml 2>/dev/null || true
touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $V"
