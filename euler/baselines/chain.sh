#!/usr/bin/env bash
# Euler-side chain for both baselines. Same recipes as the RunPod chains
# (runpod/baselines/{a3vlm,threedoi}/chain.sh); only the paths and the venv differ.
#   bash chain.sh a3vlm  {smoke|train|eval}
#   bash chain.sh 3doi   {smoke|train|export}
set -euo pipefail
WHICH="${1:?a3vlm|3doi}"; MODE="${2:-train}"
BASE="${EULER_BASE:-$SCRATCH/baselines}"
SEG="${SEG:-$HOME/SegAffordance}"
LOGS=$BASE/logs; mkdir -p $LOGS $BASE/runs
NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"; mkdir -p "$TMPDIR"
echo "== $(date -u) euler chain $WHICH/$MODE on $NPROC GPU(s)"

case $WHICH in
a3vlm)
  R=$BASE/repos/LLaMA2-Accessory/accessory; CK=$BASE/ckpt/sphinx1k; DATA=$BASE/data/a3vlm
  V=$BASE/venv_a3vlm; MP=2; DP=$((NPROC / MP)); ACCUM="${ACCUM:-$((128 / (2 * DP)))}"
  NAME="${NAME:-a3vlm}"; [ "$MODE" = "smoke" ] && NAME="${NAME}_smoke"
  RUNS=$BASE/runs/$NAME; mkdir -p $RUNS
  # sdp shards the optimizer over DATA-parallel ranks only: DP=1 needs ~75 GB/GPU and OOMs.
  [ "$MODE" = "eval" ] || [ "$DP" -ge 2 ] || { echo "ERROR: need >= 4 GPUs (DP=$DP); got $NPROC" >&2; exit 2; }
  export PYTHONPATH=$BASE/repos/LLaMA2-Accessory TORCH_HOME=$BASE/ckpt/torch HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 NCCL_LL_THRESHOLD=0
  # the data YAML carries absolute image paths: rewrite the RunPod paths to this scratch tree
  YAML=$RUNS/data.yaml
  python3 - "$SEG/config/baselines/a3vlm_sf3d.yaml" "$DATA" "$YAML" <<'PY'
import sys, re
src, data, dst = sys.argv[1:4]
open(dst, "w").write(re.sub(r"/workspace/bl/data/a3vlm", data, open(src).read()))
print("data yaml ->", dst)
PY
  # image paths inside the JSONs too (written for the RunPod pod)
  for f in $DATA/*_train.json $DATA/*_test.json $DATA/*_bvalid.json; do
    grep -q '/workspace/bl/data/a3vlm/images' "$f" 2>/dev/null && \
      python3 -c "
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
p.write_text(s.replace('/workspace/bl/data/a3vlm/images', sys.argv[2] + '/images'))
print('repathed', p.name)" "$f" "$DATA"
  done
  cd $R
  PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
  last=$(ls -d $RUNS/epoch* 2>/dev/null | sort -V | tail -1); resume=()
  [ -n "$last" ] && resume=(--resume "$last") && echo "resuming from $last"
  $V/bin/torchrun --nproc_per_node $NPROC --master_port $PORT main_finetune.py \
    --output_dir $RUNS --epochs "${EPOCHS:-3}" --warmup_epochs 0.03 \
    --batch_size 2 --accum_iter $ACCUM --num_workers 4 --max_words 2048 \
    --lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
    --data_parallel sdp --model_parallel_size $MP --checkpointing \
    --llama_type llama_ens5 --llama_config $CK/config.json --tokenizer_path $CK/tokenizer.model \
    --pretrained_path $CK --pretrained_type consolidated \
    --data_config $YAML --dialog --image_transform padded_resize --precision bf16 \
    --save_interval 1 --save_iteration_interval "${SAVE_IT:-500}" --cache_ann_on_disk "${resume[@]}" \
    2>&1 | tee -a $LOGS/train_$NAME.log
  ;;
3doi)
  R=$BASE/repos/3DOI/monoarti; D=$BASE/data/3doi; V=$BASE/venv_3doi
  NAME="${NAME:-3doi}"; [ "$MODE" = "smoke" ] && NAME="${NAME}_smoke"
  RUNS=$BASE/runs/$NAME
  export WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
  export SF3D_LIMIT_VAL_ITERS="${SF3D_LIMIT_VAL_ITERS:-200}"
# stop once the validation loss stops falling; export the best-val checkpoint, not the last
export SF3D_EARLY_STOP_PATIENCE="${SF3D_EARLY_STOP_PATIENCE:-4}"
  cd $R
  resume=(); [ -f $RUNS/checkpoints/checkpoint.pth ] && resume=(checkpoint_path=$RUNS/checkpoints/checkpoint.pth) && echo "resuming"
  if [ "$MODE" = "export" ]; then
    $V/bin/python $SEG/tools/baselines_sf3d/run.py $SEG/tools/baselines_sf3d/threedoi_export.py \
      --repo $R --ckpt "${CKPT:-$([ -f $RUNS/checkpoints/checkpoint_best.pth ] && echo $RUNS/checkpoints/checkpoint_best.pth || echo $RUNS/checkpoints/checkpoint.pth)}" --data $D --out $RUNS/preds.jsonl --batch 2 --workers 4 \
      2>&1 | tee -a $LOGS/export_$NAME.log
  else
    $V/bin/accelerate launch --num_processes $NPROC --mixed_precision fp16 \
      --main_process_port $(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 )) \
      train.py --config-name sam_sf3d hydra.run.dir=$RUNS output_dir=$RUNS \
      optimizer.max_epochs="${EPOCHS:-62}" validation_epoch_interval="${VAL_EVERY:-2}" "${resume[@]}" 2>&1 | tee -a $LOGS/train_$NAME.log
  fi
  ;;
*) echo "unknown: $WHICH" >&2; exit 1 ;;
esac
echo "== $(date -u) euler chain $WHICH/$MODE done"
