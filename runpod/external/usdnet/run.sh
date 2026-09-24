#!/usr/bin/env bash
# bash run.sh smoke|train|eval <arm|tag> [ckpt] [hydra overrides...]
set -euo pipefail
W=/workspace; R=$W/repo/USDNet; cd $R
export OMP_NUM_THREADS=3 WANDB_MODE=offline
CKPT_INIT=${CKPT_INIT:-$W/ckpt_init/mov_trainval.ckpt}
LAMBDA=${LAMBDA:-1.0}; EPOCHS=${EPOCHS:-20}; LR=${LR:-2e-5}
mode=$1; arm=$2; shift 2
COMMON="data/datasets=articulate3d_challenge_mov general.num_targets=4 general.eval_on_segments=false general.train_on_segments=false \
general.eval_articulation=true general.eval_hierarchy_inter=false data.num_labels=3 data.batch_size=1 data.voxel_size=0.02 \
data.load_articulation=true data.use_hierarchy=false data.cropping=true data.crop_length=5.5 data.crop_min_size=75000 \
data.use_coarse_to_fine=true data.c2f_rad=0.1 data.c2f_decay=0.4 data.c2f_alpha=100 model.num_queries=100 \
model.predict_articulation_mode=2 model.predict_hierarchy_interaction=false model.predict_articulation=true \
loss.regular_arti_loss=false loss.losses=[labels,masks,articulations] logging=minimal general.project_name=ext_ab +general.experiment_id=ab +general.version=0"
arm_args() { if [[ "$1" == ours* ]]; then echo "loss.screw_weight=$LAMBDA"; else echo "loss.screw_weight=0"; fi; }
case $mode in
  smoke)
    rm -rf $W/runs/smoke_$arm
    python main_instance_segmentation_articulation.py $COMMON general.experiment_name=smoke_$arm general.save_dir=$W/runs/smoke_$arm \
      data.train_mode=train general.checkpoint=$CKPT_INIT trainer.max_epochs=1 +trainer.limit_train_batches=${SMOKE_STEPS:-40} +trainer.limit_val_batches=2 \
      trainer.check_val_every_n_epoch=1 trainer.num_sanity_val_steps=0 optimizer.lr=$LR $(arm_args $arm) "$@" > $W/logs/smoke_$arm.log 2>&1 || true; grep -E "SCREW|loss_origin|loss_axis|Error|error|ap50|Traceback" $W/logs/smoke_$arm.log | tail -25 ;;
  train)
    python main_instance_segmentation_articulation.py $COMMON general.experiment_name=ft_$arm general.save_dir=$W/runs/ft_$arm \
      data.train_mode=train general.checkpoint=$CKPT_INIT trainer.max_epochs=$EPOCHS trainer.check_val_every_n_epoch=${VAL_EVERY:-10} \
      trainer.num_sanity_val_steps=0 optimizer.lr=$LR $(arm_args $arm) "$@" > $W/logs/train_$arm.log 2>&1
    echo "train $arm done: $(ls $W/runs/ft_$arm | tr '\n' ' ')" ;;
  eval)
    tag=$arm; ckpt=$1; shift
    rm -rf $W/runs/eval_$tag
    python main_instance_segmentation_articulation.py $COMMON general.experiment_name=eval_$tag general.save_dir=$W/runs/eval_$tag \
      general.train_mode=false data.test_mode=validation data.train_mode=train general.checkpoint=$ckpt data.cropping=false \
      loss.screw_weight=0 "$@" > $W/logs/eval_$tag.log 2>&1 || true
    grep -E "MA|MO|MAO|ap_50|all_ap" $W/logs/eval_$tag.log | tail -40 > $W/results/eval_$tag.txt; tail -12 $W/results/eval_$tag.txt ;;
esac
