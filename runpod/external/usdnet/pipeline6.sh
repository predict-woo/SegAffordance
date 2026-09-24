#!/usr/bin/env bash
# Add-mode, uncapped L2 (position quadratic) only, seed 1, vs ft_theirs_s1.
set -uo pipefail
W=/workspace; cd $W/repo/USDNet
python /workspace/ours/patch_both.py
echo "[p6] start $(date)"
[ -f $W/runs/ft_ours_pos_nocap_s1/DONE ] || { LAMBDA=0.05 EPOCHS=20 bash $W/ours/run.sh train ours_pos_nocap_s1 general.seed=1 loss.screw_mode=add loss.screw_term=pos loss.screw_cap=0 && touch $W/runs/ft_ours_pos_nocap_s1/DONE; }
echo "[p6] done $(date)"
