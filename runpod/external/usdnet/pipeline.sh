#!/usr/bin/env bash
# Full A/B: train both arms sequentially, then evaluate init/theirs/ours on validation.
set -uo pipefail
W=/workspace; export LAMBDA=${LAMBDA:-0.08} EPOCHS=${EPOCHS:-20} LR=${LR:-2e-5}
echo "[pipeline] start $(date) LAMBDA=$LAMBDA EPOCHS=$EPOCHS LR=$LR"
for arm in theirs ours; do
  [ -f $W/runs/ft_$arm/DONE ] && { echo "[pipeline] $arm already trained"; continue; }
  echo "[pipeline] train $arm $(date)"; bash $W/ours/run.sh train $arm && touch $W/runs/ft_$arm/DONE
done
echo "[pipeline] eval $(date)"
bash $W/ours/run.sh eval init $W/ckpt_init/mov_trainval.ckpt
for arm in theirs ours; do bash $W/ours/run.sh eval ft_$arm $W/runs/ft_$arm/last-epoch.ckpt; done
echo "[pipeline] done $(date)"
