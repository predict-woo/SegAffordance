#!/bin/bash
# Cross-eval on the HOI4D v2 held-out split (110 objects) with the HOI4D
# teacher_forcing config (data + GT-anchored proj metric), for SF3D-trained
# checkpoints. Dev pod (24 GB): small val batch.
cd /workspace/SegAffordance
OUT=experiments/20260907_xeval_sf3d_on_hoi4d/logs; mkdir -p $OUT
CFG=config/hoi4d_v2_teacher_forcing.yaml
for e in 20260821_sf3d_g19_dct 20260907_sf3d_g19_dct_ft_hoi4d_tf 20260907_sf3d_g19_dct_ft_hoi4d 20260907_hoi4d_2d_v2_teacher_forcing; do
  ck=$(ls experiments/$e/checkpoints/best-*.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "=== NO CKPT for $e"; continue; }
  echo "=== TEST $e $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $CFG --ckpt_path $ck \
    --trainer.logger=false --data.batch_size_val 16 --data.num_workers_val 8 > $OUT/test_$e.log 2>&1
  echo "=== TESTDONE $e exit=$?"
done
echo XEVAL_DONE
