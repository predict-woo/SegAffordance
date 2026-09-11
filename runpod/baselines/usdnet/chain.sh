#!/usr/bin/env bash
# USDNet on SF3D scenes: wait for scans -> convert -> train (upstream
# scripts/train_mov.sh recipe, from the Mask3D scannet200 backbone) -> export
# test predictions (general.debug=true -> preds.pkl) -> per-frame shared JSONL.
#   bash chain.sh            # full run
#   SMOKE=1 bash chain.sh    # 40-step smoke into runs/usdnet_smoke
set -euo pipefail
B=/workspace/datasets/baselines; R=$B/repos/USDNet; D=$B/data/usdnet_sf3d; SCANS=$B/data/sf3d_scans; LOGS=$B/logs; CKPT=$B/ckpt
EPOCHS="${EPOCHS:-200}"; NAME="${NAME:-usdnet}"; [ "${SMOKE:-0}" = "1" ] && NAME=usdnet_smoke
RUNS=$B/runs/$NAME; export TMPDIR=/workspace/tmp OMP_NUM_THREADS=3 WANDB_MODE=offline; mkdir -p $RUNS $LOGS /workspace/tmp
while [ ! -f $SCANS/.done_download ]; do echo "$(date -u +%H:%M) waiting for scans"; sleep 120; done
echo "== $(date -u) chain $NAME on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
cd /workspace/SegAffordance
if [ ! -f $D/.done_convert ]; then
  python tools/baselines_sf3d/sf3d_to_usdnet.py --scans $SCANS --out $D --splits experiments/baselines_sf3d/splits.json --workers "${CONVERT_WORKERS:-16}" > $LOGS/convert_usdnet.log 2>&1
  tail -5 $LOGS/convert_usdnet.log; touch $D/.done_convert
fi
# USDNet's dataset configs hard-code data/processed/articulate3d_challenge_mov; point that path at our data.
mkdir -p $R/data/processed && ln -sfn $D $R/data/processed/articulate3d_challenge_mov
cd $R
COMMON="data/datasets=articulate3d_challenge_mov general.num_targets=4 general.eval_on_segments=false general.train_on_segments=false \
general.eval_articulation=true general.eval_hierarchy_inter=false data.num_labels=3 data.batch_size=1 data.voxel_size=0.02 \
data.load_articulation=true data.use_hierarchy=false data.cropping=true data.crop_length=5.5 data.crop_min_size=75000 \
data.use_coarse_to_fine=true data.c2f_rad=0.1 data.c2f_decay=0.4 data.c2f_alpha=100 model.num_queries=100 \
model.predict_articulation_mode=2 model.predict_hierarchy_interaction=false model.predict_articulation=true \
loss.regular_arti_loss=false loss.losses=[labels,masks,articulations] logging=minimal general.project_name=sf3d_baselines"
if [ "${SMOKE:-0}" = "1" ]; then
  python main_instance_segmentation_articulation.py $COMMON general.experiment_name=$NAME general.save_dir=$RUNS data.train_mode=train \
    general.checkpoint=$CKPT/scannet200_benchmark.ckpt trainer.max_epochs=1 +trainer.limit_train_batches=40 +trainer.limit_val_batches=2 \
    trainer.check_val_every_n_epoch=1 trainer.num_sanity_val_steps=0 optimizer.lr=0.0001 > $LOGS/train_$NAME.log 2>&1 || true
  grep -E "Error|error|Traceback|ap_50|loss" $LOGS/train_$NAME.log | tail -15; exit 0
fi
if [ ! -f $RUNS/last.ckpt ] || [ ! -f $RUNS/.train_done ]; then
  python main_instance_segmentation_articulation.py $COMMON general.experiment_name=$NAME general.save_dir=$RUNS data.train_mode=train \
    general.checkpoint=$CKPT/scannet200_benchmark.ckpt trainer.max_epochs=$EPOCHS trainer.check_val_every_n_epoch=20 \
    trainer.num_sanity_val_steps=0 optimizer.lr=0.0001 > $LOGS/train_$NAME.log 2>&1
  touch $RUNS/.train_done
fi
echo "== $(date -u) train done"; ls $RUNS | tail
BEST=$(ls -t $RUNS/epoch=*val_mean_ap_50*.ckpt 2>/dev/null | python3 -c "
import sys,re; fs=[l.strip() for l in sys.stdin if l.strip()]
best=max(fs, key=lambda f: float(re.search(r'val_mean_ap_50=([0-9.]+)', f).group(1))) if fs else ''
print(best)")
[ -n "$BEST" ] || BEST=$RUNS/last.ckpt; echo "eval ckpt: $BEST"; echo "$BEST" > $RUNS/eval_ckpt.txt
python main_instance_segmentation_articulation.py $COMMON general.experiment_name=${NAME}_test general.save_dir=$RUNS/test \
  general.train_mode=false general.debug=true general.checkpoint="$BEST" data.train_mode=train data.validation_mode=test data.cropping=false \
  > $LOGS/test_$NAME.log 2>&1 || true
grep -E "MA|MO|MAO|ap_50|Error|Traceback" $LOGS/test_$NAME.log | tail -20
ls -la $RUNS/test/debug/val_preds/preds.pkl
cd /workspace/SegAffordance
python tools/baselines_sf3d/usdnet_preds_to_jsonl.py --preds $RUNS/test/debug/val_preds/preds.pkl --usd-dir $D --out $RUNS/preds.jsonl
wc -l $RUNS/preds.jsonl; touch $RUNS/CHAIN_DONE; echo "== $(date -u) CHAIN_DONE $NAME"
