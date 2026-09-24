#!/usr/bin/env bash
# SINGAPO: their loss + λ·L2 (position quadratic) only; H1 and anchor off. vs existing ft_theirs.
set -uo pipefail
W=/workspace; export CKPT_INIT=/workspace/ckpt_init/final/ckpts/last.ckpt LAMBDA=${LAMBDA:-0.5} PYOPENGL_PLATFORM=egl
cd $W/repo/singapo && cp $W/ours/run.sh $W/ours/finetune.yaml . 2>/dev/null; cp $W/ours/finetune.yaml configs/finetune.yaml
echo "[pl2] start $(date) LAMBDA=$LAMBDA"
[ -f exps/ft_ours_l2/v1/DONE ] || { bash $W/ours/run.sh train ours_l2 system.screw_w_h1=0.0 system.screw_w_pos=1.0 system.screw_w_anchor=0.0 && touch exps/ft_ours_l2/v1/DONE; }
echo "[pl2] eval $(date)"
bash $W/ours/run.sh eval ft_ours_l2 exps/ft_ours_l2/v1/ckpts/last.ckpt
echo "[pl2] done $(date)"
