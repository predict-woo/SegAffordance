#!/usr/bin/env bash
# Prove the data-parallel resume path before moving a live MOPD run: copy the latest checkpoint of
# $SRC into a scratch run dir, resume it for $EXTRA iterations on all GPUs (the end-of-training
# validation runs too), print the iteration lines and the validation AP (to compare with the
# source run's validation at the same checkpoint).
#   bash runpod/baselines/mopd/ddp_smoke.sh [src run name] [extra iters]
set -euo pipefail
SRC="${1:-mopd512_rgb}"; EXTRA="${2:-40}"
B=/workspace/datasets/baselines; DATA=$B/data/opd_sf3d_512; R=$B/repos/MOPD/opdformer; LOGS=$B/logs
S=$B/runs/$SRC; D=$B/runs/${SRC}_ddp_smoke; NPROC=$(nvidia-smi -L | wc -l)
CK=$(cat $S/last_checkpoint); IT=$(echo $CK | grep -oE '[0-9]+' | sed 's/^0*//')
rm -rf $D; mkdir -p $D; cp $S/$CK $D/; cp $S/last_checkpoint $D/; cp $S/init.pth $D/ 2>/dev/null || true
echo "== $(date -u) ddp smoke: $NPROC GPUs, resume $CK (iter $IT) for $EXTRA iters"
python /workspace/SegAffordance/runpod/baselines/mopd/patch_ddp.py $R/train.py
read -r MEAN STD <<<"$(python - <<PY
import json
s = json.load(open("$DATA/stats.json")); n = 3
f = lambda xs: "[" + ",".join(f"{x:.6f}" for x in xs[:n]) + "]"
print(f(s["pixel_mean"]), f(s["pixel_std"]))
PY
)"
cd $R
python train.py --config-file configs/opd_p_real.yaml --output-dir $D --data-path $DATA/MotionDataset_h5 --input-format RGB \
  --num-gpus $NPROC --resume --model_attr_path $DATA/obj_info.json \
  --opts MODEL.WEIGHTS $S/init.pth MODEL.SEM_SEG_HEAD.NUM_CLASSES 8 MODEL.PIXEL_MEAN $MEAN MODEL.PIXEL_STD $STD \
  DATALOADER.NUM_WORKERS 8 INPUT.MASK_FORMAT bitmask INPUT.IMAGE_SIZE 512 "INPUT.MIN_SIZE_TRAIN" "(512,)" INPUT.MAX_SIZE_TRAIN 512 \
  INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TEST 512 SOLVER.BASE_LR 0.0001 SOLVER.STEPS "(36000,48000)" SOLVER.CHECKPOINT_PERIOD 10000 \
  TEST.EVAL_PERIOD 10000 SOLVER.MAX_ITER $((IT + 1 + EXTRA)) DATASETS.TEST "('MotionNet_valid',)" > $LOGS/ddp_smoke_$SRC.log 2>&1 || true
grep -E 'Resuming|Starting training from iteration|iter: [0-9]+ ' $LOGS/ddp_smoke_$SRC.log | head -3 | cut -c1-140
grep -oE 'iter: [0-9]+ .*time: [0-9.]+' $LOGS/ddp_smoke_$SRC.log | tail -2 | grep -oE 'iter: [0-9]+|time: [0-9.]+' | tr '\n' ' '; echo
grep -E 'copypaste: [0-9]' $LOGS/ddp_smoke_$SRC.log | cut -c40-140
grep -m3 -E 'Traceback|Error' $LOGS/ddp_smoke_$SRC.log | cut -c1-160 || true
echo "== $(date -u) ddp smoke done"
