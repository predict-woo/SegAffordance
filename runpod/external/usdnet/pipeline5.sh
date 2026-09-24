#!/usr/bin/env bash
# Best-epoch protocol: rerun the seed-1 pair with validation every 2 epochs, all ckpts kept.
set -uo pipefail
W=/workspace; cd $W/repo/USDNet
until grep -q '^\[p4\] done' $W/logs/pipeline4.log 2>/dev/null; do sleep 60; done
echo "[p5] start $(date)"
[ -f $W/runs/ft_theirs_s1_v2e/DONE ] || { LAMBDA=0 EPOCHS=20 VAL_EVERY=2 bash $W/ours/run.sh train theirs_s1_v2e general.seed=1 && touch $W/runs/ft_theirs_s1_v2e/DONE; }
echo "[p5] train ours_both_s1_v2e $(date)"
[ -f $W/runs/ft_ours_both_s1_v2e/DONE ] || { LAMBDA=0.05 EPOCHS=20 VAL_EVERY=2 bash $W/ours/run.sh train ours_both_s1_v2e general.seed=1 loss.screw_mode=add loss.screw_term=both loss.screw_cap=4 && touch $W/runs/ft_ours_both_s1_v2e/DONE; }
echo "[p5] done $(date)"
