#!/usr/bin/env bash
# MOPD on SF3D: wait for OPDFormer-P RGB -> compose full init ckpt (OPDFormer-P
# RGB + EfficientSAM ViT-S + geffnet B5) -> their 1000-iteration fine-tune
# (configs/opd_p_real.yaml with MOPD's opd_base.yaml: lr 5e-6, batch 16,
# BMOC_V1) -> evaluate on MotionNet_test -> shared JSONL -> CHAIN_DONE.
#   SCHEDULE=opdformer OPD_DATA=$B/data/opd_sf3d_512 RUN_NAME=mopd512_rgb IMG_SIZE=512 INIT_RUN=$B/runs/opd_p_rgb bash chain.sh
#     -> MOPD architecture TRAINED with the OPDFormer recipe (lr 1e-4, 60k iters, steps 36k/48k)
#        at 512x384 from the composed init, instead of their 1k-iteration fine-tune.
set -euo pipefail
B=/workspace/datasets/baselines; DATA="${OPD_DATA:-$B/data/opd_sf3d}"; NAME="${RUN_NAME:-mopd_rgb}"; RUNS=$B/runs/$NAME; LOGS=$B/logs; R=$B/repos/MOPD/opdformer
INIT_RUN="${INIT_RUN:-$B/runs/opd_p_rgb}"
export TMPDIR=/workspace/tmp; mkdir -p $RUNS $LOGS /workspace/tmp
while [ ! -f $INIT_RUN/model_final.pth ]; do echo "$(date -u +%H:%M) waiting for opd_p_rgb model_final.pth"; sleep 300; done
echo "== $(date -u) MOPD chain ($NAME, data $DATA, init $INIT_RUN, schedule ${SCHEDULE:-mopd}) on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /workspace/SegAffordance
[ -f $RUNS/init.pth ] || python tools/baselines_sf3d/run.py tools/baselines_sf3d/mopd_compose_ckpt.py --opd $INIT_RUN/model_final.pth --esam $B/ckpt/efficient_sam_vits.pt --mopd-repo $B/repos/MOPD --out $RUNS/init.pth
read -r MEAN STD <<<"$(python - <<PY
import json
s = json.load(open("$DATA/stats.json")); n = 3
f = lambda xs: "[" + ",".join(f"{x:.6f}" for x in xs[:n]) + "]"
print(f(s["pixel_mean"]), f(s["pixel_std"]))
PY
)"
cd $R
COMMON_OPTS="MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN $MEAN MODEL.PIXEL_STD $STD DATALOADER.NUM_WORKERS 8 INPUT.MASK_FORMAT bitmask"
if [ -n "${IMG_SIZE:-}" ]; then  # see opd/chain.sh: frames are used at h5 size; INPUT.*SIZE* lifted to match
  COMMON_OPTS="$COMMON_OPTS INPUT.IMAGE_SIZE $IMG_SIZE INPUT.MIN_SIZE_TRAIN ($IMG_SIZE,) INPUT.MAX_SIZE_TRAIN $IMG_SIZE INPUT.MIN_SIZE_TEST $IMG_SIZE INPUT.MAX_SIZE_TEST $IMG_SIZE"
fi
# MOPD's opd_base.yaml differs from OPDMulti's ONLY in BASE_LR 5e-6 / MAX_ITER 1000 / CHECKPOINT_PERIOD 200 /
# EVAL_PERIOD 50 (diffed 2026-09-14). SCHEDULE=opdformer restores OPDFormer's values, i.e. trains the
# MOPD architecture for the full detector schedule instead of fine-tuning it.
if [ "${SCHEDULE:-mopd}" = "opdformer" ]; then
  COMMON_OPTS="$COMMON_OPTS SOLVER.BASE_LR 0.0001 SOLVER.MAX_ITER 60000 SOLVER.STEPS (36000,48000) SOLVER.CHECKPOINT_PERIOD 10000 TEST.EVAL_PERIOD 10000"
fi
if [ ! -f $RUNS/model_final.pth ]; then
  python train.py --config-file configs/opd_p_real.yaml --output-dir $RUNS --data-path $DATA/MotionDataset_h5 --input-format RGB \
    --model_attr_path $DATA/obj_info.json --opts MODEL.WEIGHTS $RUNS/init.pth $COMMON_OPTS DATASETS.TEST "('MotionNet_valid',)" ${EXTRA_OPTS:-} \
    > $LOGS/train_$NAME.log 2>&1
fi
echo "== $(date -u) train done"
python evaluate_on_log.py --config-file configs/opd_p_real.yaml --output-dir $RUNS/test --data-path $DATA/MotionDataset_h5 --input-format RGB \
  --model_attr_path $DATA/obj_info.json --opts MODEL.WEIGHTS $RUNS/model_final.pth $COMMON_OPTS DATASETS.TEST "('MotionNet_test',)" \
  > $LOGS/test_$NAME.log 2>&1
echo "== $(date -u) test done"; grep -E "AP|motion" $LOGS/test_$NAME.log | tail -20
cd /workspace/SegAffordance
python tools/baselines_sf3d/run.py tools/baselines_sf3d/opd_preds_to_jsonl.py --pred $RUNS/test/inference/instances_predictions.pth --data-dir $DATA --out $RUNS/preds.jsonl
wc -l $RUNS/preds.jsonl; cp $RUNS/config.yaml $RUNS/config_resolved.yaml 2>/dev/null || true
touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $NAME"
