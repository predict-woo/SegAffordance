#!/usr/bin/env bash
# bash run.sh smoke|train|eval <arm|tag> [ckpt_dir]
set -euo pipefail
W=/workspace; R=$W/repo/particulate; cd $R
LAMBDA=${LAMBDA:-1.0}; STEPS=${STEPS:-4000}; LR=${LR:-2e-5}
mode=$1; arm=$2; shift 2
arm_args() { if [ "$1" = ours ]; then echo "loss_weight_part_motion_axis_revolute=0 loss_weight_part_motion_axis_prismatic=0 loss_weight_screw_h1=$LAMBDA loss_weight_screw_anchor=$(python -c "print(0.5*$LAMBDA)") loss_weight_screw_prismatic=$LAMBDA"; else echo ""; fi; }
# split: PartNet test = SINGAPO split test ids (7 cats); train = everything else in cached/
python - <<'PY'
import json, os, glob
ids=[os.path.basename(p)[:-4] for p in glob.glob('/workspace/data/cached/*.npz')]
test=set(json.load(open('/workspace/ours/pm_test_ids.json'))) if os.path.exists('/workspace/ours/pm_test_ids.json') else set()
json.dump({"train":[i for i in ids if i not in test],"test":[i for i in ids if i in test]}, open('/workspace/data/split.json','w'))
print("split train/test:", len(ids)-len(test&set(ids)), len(test&set(ids)))
PY
COMMON="dataset_args.datasets=[{root:/workspace/data/cached,data_split_file:/workspace/data/split.json,sampling_rate:1.0}] output_dir=$W/runs init_weights=$W/ckpt_init/model.pt logger_type=none learning_rate=$LR scale_lr=false lr_warmup_steps=100 num_workers=8"
case $mode in
  smoke) python train.py --config configs/train-particulate-B.yaml $COMMON exp_name=smoke_$arm max_train_steps=30 log_interval=1 ckpt_interval=100000 $(arm_args $arm) "$@" 2>&1 | grep -E "init_weights|loss|Error|Traceback" | tail -8 ;;
  train) python train.py --config configs/train-particulate-B.yaml $COMMON exp_name=ft_$arm max_train_steps=$STEPS ckpt_interval=1000 $(arm_args $arm) "$@" > $W/logs/train_$arm.log 2>&1; ls $W/runs | grep ft_$arm ;;
  eval) echo "eval: see eval.sh (infer.py --eval on test split + evaluate.py + eval_axis.py)" ;;
esac
