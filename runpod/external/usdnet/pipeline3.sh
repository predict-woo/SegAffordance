#!/usr/bin/env bash
# Seed-matched pair (general.seed=1): theirs_s1 then ours_v2_s1. Waits for pipeline2.
set -uo pipefail
W=/workspace; cd $W/repo/USDNet
until grep -q '^\[p2\] done' $W/logs/pipeline2.log 2>/dev/null; do sleep 60; done
echo "[p3] start $(date)"
[ -f $W/runs/ft_theirs_s1/DONE ] || { LAMBDA=0 EPOCHS=20 bash $W/ours/run.sh train theirs_s1 general.seed=1 && touch $W/runs/ft_theirs_s1/DONE; }
echo "[p3] train ours_v2_s1 $(date)"
[ -f $W/runs/ft_ours_v2_s1/DONE ] || { LAMBDA=0.05 EPOCHS=20 bash $W/ours/run.sh train ours_v2_s1 general.seed=1 loss.screw_mode=add loss.screw_cap=4 && touch $W/runs/ft_ours_v2_s1/DONE; }
echo "[p3] done $(date)"
