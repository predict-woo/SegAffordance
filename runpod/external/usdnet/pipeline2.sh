#!/usr/bin/env bash
# (a) init-ckpt validation numbers via a zero-LR 1-step "train"; (b) v2 arm: add-mode capped H1.
set -uo pipefail
W=/workspace; R=$W/repo/USDNet; cd $R
python /workspace/ours/patch_cap.py
echo "[p2] init-val $(date)"
rm -rf $W/runs/initval
LAMBDA=0 EPOCHS=1 LR=0 bash $W/ours/run.sh train initval +trainer.limit_train_batches=1 trainer.check_val_every_n_epoch=1
echo "[p2] train ours_v2 $(date)"
LAMBDA=${LAMBDA:-0.05} EPOCHS=20 bash $W/ours/run.sh train ours_v2 loss.screw_mode=add loss.screw_cap=4
echo "[p2] done $(date)"
