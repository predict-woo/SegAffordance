#!/usr/bin/env bash
set -uo pipefail
W=/workspace; export CKPT_INIT=/workspace/ckpt_init/final/ckpts/last.ckpt LAMBDA=${LAMBDA:-0.05}
cd $W/repo/singapo
echo "[pipeline] start $(date) LAMBDA=$LAMBDA"
for arm in theirs ours; do
  [ -f exps/ft_$arm/v1/DONE ] && continue
  echo "[pipeline] train $arm $(date)"; bash $W/ours/run.sh train $arm && touch exps/ft_$arm/v1/DONE
done
echo "[pipeline] eval $(date)"
bash $W/ours/run.sh eval init $CKPT_INIT
for arm in theirs ours; do bash $W/ours/run.sh eval ft_$arm exps/ft_$arm/v1/ckpts/last.ckpt; done
echo "[pipeline] done $(date)"
