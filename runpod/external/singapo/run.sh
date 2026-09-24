#!/usr/bin/env bash
# bash run.sh smoke|train|eval <arm|tag> [extra omegaconf overrides...]
#   smoke  theirs|ours     : 30 steps, prints loss magnitudes
#   train  theirs|ours     : 20-epoch matched fine-tune (LAMBDA env for ours)
#   eval   <tag> <ckpt>    : test on PM (5 samples) + their metrics + axis metrics
set -euo pipefail
W=/workspace; R=$W/repo/singapo; cd $R
export PYOPENGL_PLATFORM=egl WANDB_MODE=offline
CKPT_INIT=${CKPT_INIT:-$(grep -v cage $W/ckpt_init/ckpt_list.txt | head -1)}
LAMBDA=${LAMBDA:-1.0}
mode=$1; arm=$2; shift 2
arm_args() { if [[ "$1" == ours* ]]; then echo "system.name=sys_singapo_screw system.screw_weight=$LAMBDA"; else echo "system.name=sys_singapo"; fi; }
case $mode in
  smoke)
    python train_ft.py --config configs/finetune.yaml --init_weights "$CKPT_INIT" name=smoke_$arm $(arm_args $arm) \
      trainer.max_epochs=1 trainer.limit_train_batches=30 trainer.limit_val_batches=0 checkpoint.every_n_epochs=100 trainer.log_every_n_steps=1 "$@" 2>&1 | tail -5
    python - "$arm" <<'PY'
import pandas as pd,glob,sys
f=sorted(glob.glob(f'exps/smoke_{sys.argv[1]}/v1/logs/csv/version_*/metrics.csv'))[-1]
d=pd.read_csv(f); cols=[c for c in d.columns if c.startswith('train/')]
print(d[cols].dropna(how='all').tail(10).mean().to_string())
PY
    ;;
  train)
    python train_ft.py --config configs/finetune.yaml --init_weights "$CKPT_INIT" name=ft_$arm $(arm_args $arm) "$@" \
      > $W/logs/train_$arm.log 2>&1
    echo "train $arm done: $(ls exps/ft_$arm/v1/ckpts)" ;;
  eval)
    tag=$arm; ckpt=$1; shift
    sysname=sys_singapo; [[ "$tag" == *ours* ]] && sysname=sys_singapo_screw
    python test_ft.py --config configs/finetune.yaml --ckpt "$ckpt" name=eval_$tag system.name=$sysname "$@" > $W/logs/eval_$tag.log 2>&1
    tdir=$(ls -d exps/eval_$tag/v1/output/test/epoch_* | head -1)
    python eval_axis.py --test_dir "$tdir" --gt_root ../../data --out "$tdir/metrics_axis.json"
    mkdir -p $W/results/$tag && cp "$tdir"/metrics*.json $W/results/$tag/ 2>/dev/null; ls $W/results/$tag ;;
esac
