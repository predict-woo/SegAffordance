#!/usr/bin/env bash
# Add-mode arms with L2+H1 (screw_term=both), seed 1, matched against ft_theirs_s1.
set -uo pipefail
W=/workspace; cd $W/repo/USDNet
python /workspace/ours/patch_both.py
echo "[p4] start $(date)"
[ -f $W/runs/ft_ours_both_s1/DONE ] || { LAMBDA=0.05 EPOCHS=20 bash $W/ours/run.sh train ours_both_s1 general.seed=1 loss.screw_mode=add loss.screw_term=both loss.screw_cap=4 && touch $W/runs/ft_ours_both_s1/DONE; }
echo "[p4] train ours_both_nocap_s1 $(date)"
[ -f $W/runs/ft_ours_both_nocap_s1/DONE ] || { LAMBDA=0.05 EPOCHS=20 bash $W/ours/run.sh train ours_both_nocap_s1 general.seed=1 loss.screw_mode=add loss.screw_term=both loss.screw_cap=0 && touch $W/runs/ft_ours_both_nocap_s1/DONE; }
echo "[p4] done $(date)"
