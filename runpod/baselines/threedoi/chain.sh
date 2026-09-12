#!/usr/bin/env bash
# 3DOI (monoarti, SAM ViT-B) on SF3D with the released recipe (configs/sam.yaml -> our
# config/baselines/3doi_sam_sf3d.yaml): accelerate on all GPUs, batch 2 per GPU, AdamW 1e-4 /
# 1e-5 backbone, fp16, 200 epochs, then the shared JSONL export on the test split.
#   MODE=smoke bash chain.sh   # 30 iterations on 1 epoch, checkpoint, resume 10 more, export 30 frames
#   MODE=train bash chain.sh   # full run; resumes from $RUNS/checkpoints/checkpoint.pth when present
#   MODE=export bash chain.sh  # export only (CKPT=... to pick a checkpoint)
set -euo pipefail
B=/workspace/datasets/baselines; R=$B/repos/3DOI/monoarti; D=$B/stage/3doi; LOGS=$B/logs; VENV=/opt/venv_3doi
MODE="${MODE:-train}"; NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"; EPOCHS="${EPOCHS:-200}"
NAME="${NAME:-3doi}"; [ "$MODE" = "smoke" ] && NAME="${NAME}_smoke"
RUNS=$B/runs/$NAME; mkdir -p $LOGS /workspace/tmp
export TMPDIR=/workspace/tmp WANDB_MODE=offline OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
SEG=/workspace/SegAffordance
echo "== $(date -u) $MODE $NAME on $NPROC x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
cd $R

train() {  # $1 epochs, rest = hydra overrides
  local epochs=$1; shift
  local resume=()
  [ -f $RUNS/checkpoints/checkpoint.pth ] && resume=(checkpoint_path=$RUNS/checkpoints/checkpoint.pth) && echo "resuming from $RUNS/checkpoints/checkpoint.pth"
  $VENV/bin/accelerate launch --num_processes $NPROC --mixed_precision fp16 --main_process_port $(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 )) \
    train.py --config-name sam_sf3d hydra.run.dir=$RUNS output_dir=$RUNS optimizer.max_epochs=$epochs "${resume[@]}" "$@"
}

export_preds() {  # $1 ckpt, $2 out jsonl, rest = extra args
  local ck=$1 out=$2; shift 2
  $VENV/bin/python $SEG/tools/baselines_sf3d/run.py $SEG/tools/baselines_sf3d/threedoi_export.py \
    --repo $R --ckpt $ck --data $D --out $out --batch 2 --workers 4 "$@"
}

case $MODE in
  smoke)
    rm -rf $RUNS
    SF3D_LIMIT_ITERS=30 train 1 validation_epoch_interval=1000 > $LOGS/train_$NAME.log 2>&1 || { tail -40 $LOGS/train_$NAME.log; exit 1; }
    ls -la $RUNS/checkpoints; grep -E "loss|sec/it" $LOGS/train_$NAME.log | tail -3
    SF3D_LIMIT_ITERS=10 train 2 validation_epoch_interval=1000 > $LOGS/train_${NAME}_resume.log 2>&1 || { tail -40 $LOGS/train_${NAME}_resume.log; exit 1; }
    grep -E "Resuming|resuming" $LOGS/train_${NAME}_resume.log | tail -2; ls $RUNS/checkpoints
    export_preds $RUNS/checkpoints/checkpoint.pth $RUNS/preds_smoke.jsonl --limit 30 > $LOGS/export_$NAME.log 2>&1 || { tail -30 $LOGS/export_$NAME.log; exit 1; }
    tail -2 $LOGS/export_$NAME.log; wc -l $RUNS/preds_smoke.jsonl
    ;;
  train)
    if [ ! -f $RUNS/.train_done ]; then
      train $EPOCHS > $LOGS/train_$NAME.log 2>&1
      touch $RUNS/.train_done
    fi
    echo "== $(date -u) train done"; ls $RUNS/checkpoints | tail -3
    export_preds "${CKPT:-$RUNS/checkpoints/checkpoint.pth}" $RUNS/preds.jsonl > $LOGS/export_$NAME.log 2>&1
    tail -1 $LOGS/export_$NAME.log; wc -l $RUNS/preds.jsonl
    touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $NAME"
    ;;
  export)
    export_preds "${CKPT:-$RUNS/checkpoints/checkpoint.pth}" $RUNS/preds.jsonl > $LOGS/export_$NAME.log 2>&1
    tail -1 $LOGS/export_$NAME.log; touch $RUNS/CHAIN_DONE
    ;;
esac
