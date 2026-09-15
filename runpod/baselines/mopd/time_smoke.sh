#!/usr/bin/env bash
# Time MOPD training on this pod: resume the run's latest checkpoint in a scratch dir for $EXTRA
# iterations on $NPROC GPUs against $DATA (the full set or the 320-image opd512_timing subset,
# identical h5 layout) and print s/iter. Same config as chain.sh SCHEDULE=opdformer IMG_SIZE=512.
#   DATA=$B/data/opd512_timing bash runpod/baselines/mopd/time_smoke.sh [src run] [extra iters]
set -euo pipefail
SRC="${1:-mopd512_rgb}"; EXTRA="${2:-100}"
B=/workspace/datasets/baselines; DATA="${DATA:-$B/data/opd_sf3d_512}"; R=$B/repos/MOPD/opdformer; LOGS=$B/logs
S=$B/runs/$SRC; D=$B/runs/${SRC}_time_smoke; NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"; mkdir -p $LOGS /workspace/tmp; export TMPDIR=/workspace/tmp
CK=$(cat $S/last_checkpoint); IT=$(echo $CK | grep -oE '[0-9]+' | sed 's/^0*//')
rm -rf $D; mkdir -p $D; cp $S/$CK $D/; cp $S/last_checkpoint $D/
echo "== $(date -u) time smoke: $NPROC x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(lscpu | grep 'Model name' | cut -d: -f2 | xargs), resume $CK for $EXTRA iters, data $DATA"
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
  INPUT.MIN_SIZE_TEST 512 INPUT.MAX_SIZE_TEST 512 SOLVER.BASE_LR 0.0001 SOLVER.STEPS "(36000,48000)" SOLVER.CHECKPOINT_PERIOD 100000 \
  TEST.EVAL_PERIOD 100000 SOLVER.MAX_ITER $((IT + 1 + EXTRA)) DATASETS.TEST "('MotionNet_valid',)" ${EXTRA_OPTS:-} > $LOGS/time_smoke_$SRC.log 2>&1 || true
grep -E 'Starting training from iteration' $LOGS/time_smoke_$SRC.log | head -1 | cut -c1-120
grep -E 'iter: [0-9]+ ' $LOGS/time_smoke_$SRC.log | tail -3 | grep -oE 'iter: [0-9]+|  time: [0-9.]+|data_time: [0-9.]+|max_mem: [0-9]+M' | paste -sd' ' | sed 's/iter:/\niter:/g' | tail -3
grep -m3 -E 'Traceback|Error' $LOGS/time_smoke_$SRC.log | cut -c1-160 || true
echo "== $(date -u) time smoke done"
