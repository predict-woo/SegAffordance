#!/usr/bin/env bash
# A3VLM on SF3D: the released recipe (A3VLM/model/accessory/scripts/a3vlm_train.sh: SPHINX-1k
# 13B, llama_ens5, MP 2, sdp, bf16, 3 epochs, lr 2e-5, warmup 0.03, batch 2 x accum 8 x 8 GPUs
# = 128 samples/step) on our VQA JSONs, then their eval_affordance_v2.py on the test questions
# and the shared JSONL export.
#   MODE=smoke  bash chain.sh   # 200-sample subset, 1 epoch, save every 5 it, resume, eval 20 q's
#   MODE=train  bash chain.sh   # full training (resumes from the newest epoch*/ dir if present)
#   MODE=eval   bash chain.sh   # eval + export from $EVAL_CKPT (default: newest epoch dir)
set -euo pipefail
B=/workspace/bl; R=$B/repos/LLaMA2-Accessory/accessory; CK=$B/ckpt/sphinx1k; DATA=$B/data/a3vlm; LOGS=$B/logs
MODE="${MODE:-train}"; NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"; MP=2
NAME="${NAME:-a3vlm}"; [ "$MODE" = "smoke" ] && NAME="${NAME}_smoke"
RUNS=$B/runs/$NAME; mkdir -p $RUNS $LOGS /workspace/tmp
export TMPDIR=/workspace/tmp TORCH_HOME=$B/ckpt/torch HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONPATH=$B/repos/LLaMA2-Accessory
export NCCL_LL_THRESHOLD=0
SEG=/workspace/SegAffordance
# 8 GPUs / MP 2 -> 4 data-parallel ranks; their 2 nodes x 8 GPUs / MP 2 = 8 ranks x batch 2 x accum 8 = 128.
DP=$((NPROC / MP)); ACCUM="${ACCUM:-$((128 / (2 * DP)))}"
# sdp shards the optimizer across DATA-parallel ranks only. With DP=1 each GPU holds the full
# AdamW state for its half of the 13B model (~75 GB) and OOMs on an 80 GB card -- measured on
# 2 x H100 2026-09-12. DP >= 2 (i.e. >= 4 GPUs) is the floor; the released recipe uses DP=8.
if [ "$MODE" != "eval" ] && [ "$DP" -lt 2 ]; then
  echo "ERROR: need at least $((MP * 2)) GPUs (DP=$DP shards nothing -> OOM); got $NPROC" >&2; exit 2
fi
echo "== $(date -u) $MODE $NAME on $NPROC x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); dp=$DP accum=$ACCUM"
python $SEG/runpod/baselines/a3vlm/patch_eval.py $R/eval_affordance_v2.py
# Pick a base port with a free range [PORT, PORT+NPROC/MP] -- the eval shards use PORT+1+i.
# A bare random pick collided with Docker's embedded DNS (127.0.0.11:44453) on 2026-09-13 and
# torchrun died with "failed to bind". Their own scripts loop on nc -z for the same reason.
port_free() { ! (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -qE "[:.]$1 "; }
PORT=0
for _try in $(seq 1 50); do
  cand=$(( ((RANDOM<<15)|RANDOM) % 20000 + 20000 ))
  ok=1
  for off in $(seq 0 $(( NPROC / MP ))); do port_free $(( cand + off )) || ok=0; done
  [ "$ok" = "1" ] && { PORT=$cand; break; }
done
[ "$PORT" != "0" ] || { echo "no free port range found" >&2; exit 4; }
echo "using master port $PORT"
cd $R

# `|| true` is load-bearing: with `set -o pipefail` the ls fails when no checkpoint exists yet,
# so the pipeline returns non-zero, so `last=$(newest_epoch)` is a failing assignment, which
# `set -e` treats as fatal -- killing the run silently before torchrun, leaving a 0-byte log.
newest_epoch() { ls -d $RUNS/epoch* 2>/dev/null | sort -V | tail -1 || true; }

train() {  # $1 = data yaml, $2 = epochs, $3 = save_iteration_interval, rest = extra args
  local yaml=$1 epochs=$2 save_it=$3; shift 3
  local resume=(); local last; last=$(newest_epoch)
  # NB: `[ cond ] && a && b` as a bare statement returns 1 when cond is false, which under
  # `set -e` kills the script -- silently, before torchrun ever starts. Use an if.
  if [ -n "$last" ]; then resume=(--resume "$last"); echo "resuming from $last"; fi
  torchrun --nproc_per_node $NPROC --master_port $PORT main_finetune.py \
    --output_dir $RUNS --epochs $epochs --warmup_epochs 0.03 \
    --batch_size 2 --accum_iter $ACCUM --num_workers 4 --max_words 2048 \
    --lr 0.00002 --min_lr 0 --clip_grad 8 --weight_decay 0 \
    --data_parallel sdp --model_parallel_size $MP --checkpointing \
    --llama_type llama_ens5 --llama_config $CK/config.json --tokenizer_path $CK/tokenizer.model \
    --pretrained_path $CK --pretrained_type consolidated \
    --data_config $yaml --dialog --image_transform padded_resize --precision bf16 \
    --save_interval 1 --save_iteration_interval $save_it "${resume[@]}" "$@"
}

# --max_gen_len: their default is 2048 new tokens; our longest answer (an 8-vertex box) is ~130 tokens.
# Without a cap the fine-tuned model ran every batch to 2048 tokens (92 s/batch, ~24 h for the test
# set, measured 2026-09-13). Truncation can only affect answers that would not have parsed anyway.
# eval_affordance_v2.py generates on ONE model-parallel group (rank 0 drives, the rest follow),
# so a single call uses only $MP GPUs. Shard the question file into NPROC/MP pieces and run one
# group per GPU pair concurrently, then concatenate. Same model, same prompts, same generation
# settings -- only the work split differs.
evaluate() {  # $1 = question json, $2 = ckpt dir, $3 = flag ; results -> $R/vqa_logs/$3/<name>.json
  local q=$1 ck=$2 flag=$3 base; base=$(basename ${q%.json})
  local shards=$((NPROC / MP)) sd=$RUNS/eval/shards/$base
  rm -rf $sd $R/vqa_logs/$flag/$base.json; mkdir -p $sd $R/vqa_logs/$flag
  python - "$q" "$sd" "$shards" <<'PYSPLIT'
import json, sys
q, sd, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(q))
for i in range(n):
    json.dump(d[i::n], open(f"{sd}/shard{i}.json", "w"))
print(f"{len(d)} questions -> {n} shards", flush=True)
PYSPLIT
  local pids=()
  for i in $(seq 0 $((shards - 1))); do
    CUDA_VISIBLE_DEVICES=$(seq -s, $((i * MP)) $((i * MP + MP - 1))) \
    torchrun --nproc-per-node=$MP --master_port $((PORT + 1 + i)) eval_affordance_v2.py \
      --llama_type llama_ens5 --llama_config $CK/config.json --tokenizer_path $CK/tokenizer.model \
      --pretrained_path $ck --dataset $sd/shard$i.json --batch_size "${EVAL_BS:-8}" --input_size 448 \
      --model_parallel_size $MP --addition_flag ${flag}_s$i --sampled_num 1000000 --remove_space \
      --max_gen_len "${MAX_GEN:-192}" \
      > $LOGS/eval_${base}_s${i}_$NAME.log 2>&1 &
    pids+=($!)
  done
  local rc=0; for pid in "${pids[@]}"; do wait $pid || rc=1; done
  [ $rc -eq 0 ] || { echo "eval shard failed for $base"; tail -20 $LOGS/eval_${base}_s0_$NAME.log; return 1; }
  python - "$R/vqa_logs" "$flag" "$base" "$shards" <<'PYMERGE'
import json, sys
root, flag, base, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out = []
for i in range(n):
    out += json.load(open(f"{root}/{flag}_s{i}/shard{i}.json"))
json.dump(out, open(f"{root}/{flag}/{base}.json", "w"))
print(f"merged {len(out)} answers -> {flag}/{base}.json", flush=True)
PYMERGE
}

export_all() {  # $1 = ckpt dir, $2 = flag, $3 = rec/joint question dir
  local ck=$1 flag=$2 qd=$3 out=$RUNS/eval; mkdir -p $out
  evaluate $qd/rec_test.json $ck $flag
  evaluate $qd/joint_test.json $ck $flag
  python $SEG/tools/baselines_sf3d/run.py $SEG/tools/baselines_sf3d/a3vlm_preds_to_jsonl.py make-joint \
    --rec-results $R/vqa_logs/$flag/rec_test.json --rec-questions $qd/rec_test.json --out $out/joint_pred_test.json
  evaluate $out/joint_pred_test.json $ck $flag
  python $SEG/tools/baselines_sf3d/run.py $SEG/tools/baselines_sf3d/a3vlm_preds_to_jsonl.py export \
    --joint-results $R/vqa_logs/$flag/joint_pred_test.json --joint-questions $out/joint_pred_test.json \
    --meta $DATA/meta_test.json --out $out/preds_chain.jsonl
  python $SEG/tools/baselines_sf3d/run.py $SEG/tools/baselines_sf3d/a3vlm_preds_to_jsonl.py export \
    --joint-results $R/vqa_logs/$flag/joint_test.json --joint-questions $qd/joint_test.json \
    --meta $DATA/meta_test.json --gt-box --out $out/preds_gtbox.jsonl
  cp -r $R/vqa_logs/$flag $out/vqa_logs; wc -l $out/preds_*.jsonl
}

case $MODE in
  smoke)
    rm -rf $RUNS; mkdir -p $RUNS/data
    python - <<EOF
import json, random
random.seed(0)
for t in ("det", "rec", "joint"):
    d = json.load(open("$DATA/%s_train.json" % t)); random.shuffle(d); json.dump(d[:70], open("$RUNS/data/%s.json" % t, "w"))
for t in ("rec", "joint"):
    d = json.load(open("$DATA/%s_test.json" % t)); json.dump(d[:20], open("$RUNS/data/%s_test.json" % t, "w"))
open("$RUNS/data/smoke.yaml", "w").write("META:\n" + "".join("  -\n    path: '$RUNS/data/%s.json'\n    type: 'image_text'\n    ratio: 1\n" % t for t in ("det", "rec", "joint")))
EOF
    # each save is ~103 GB / ~3 min, and the smoke only needs to prove one save + one resume
    train $RUNS/data/smoke.yaml 1 "${SMOKE_SAVE_IT:-25}" > $LOGS/train_$NAME.log 2>&1 || { tail -40 $LOGS/train_$NAME.log; exit 1; }
    ls $RUNS; grep -E "effective batch|Epoch: \[0\]|loss" $LOGS/train_$NAME.log | tail -5
    # resume proof: a second epoch from the saved epoch0
    train $RUNS/data/smoke.yaml 2 "${SMOKE_SAVE_IT:-25}" > $LOGS/train_${NAME}_resume.log 2>&1 || { tail -40 $LOGS/train_${NAME}_resume.log; exit 1; }
    grep -E "resum|Epoch: \[1\]" $LOGS/train_${NAME}_resume.log | tail -3; ls $RUNS
    export_all "$(newest_epoch)" ${NAME}_flag $RUNS/data
    ;;
  train)
    if [ ! -f $RUNS/.train_done ]; then
      # EPOCHS was hardcoded to 3 here while the smoke path used variables; the 2-epoch restart on
      # 2026-09-13 therefore ran a third epoch for ~1.4 h ($26) before being caught, and its LR
      # schedule was the 3-epoch cosine throughout.
      train $SEG/config/baselines/a3vlm_sf3d.yaml "${EPOCHS:-3}" "${SAVE_IT:-500}" --cache_ann_on_disk > $LOGS/train_$NAME.log 2>&1
      touch $RUNS/.train_done
    fi
    echo "== $(date -u) train done"; ls $RUNS
    export_all "${EVAL_CKPT:-$(newest_epoch)}" ${NAME}_test $DATA
    touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $NAME"
    ;;
  eval)
    export_all "${EVAL_CKPT:-$(newest_epoch)}" ${NAME}_test $DATA
    touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $NAME (eval)"
    ;;
esac
