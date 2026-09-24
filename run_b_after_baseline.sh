#!/bin/bash
# Pod B: let the running `baseline` python finish, prune its ckpts, then wait
# for teacher_forcing (pod A writes sweep_queue_a2.log with TF_DONE_A) — or run
# it here if pod A has not started it within 15 min after baseline ends —
# then test passes for all three arms and the SF3D post-training.
cd /workspace/SegAffordance
T() { exp=experiments/$1; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1); echo "=== TEST $1 $ck $(date)"
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 > $exp/logs/test.log 2>&1; echo "=== TESTDONE $1 exit=$?"; }
prune() { d=experiments/$1/checkpoints; best=$(ls $d | grep best- | sed 's/.*valloss\([0-9.]*\)\.ckpt/\1 &/' | sort -g | head -1 | cut -d' ' -f2); for f in $d/*.ckpt; do [ "$(basename $f)" = "$best" ] || rm -f "$f"; done; echo "$1 -> $best"; }
while pgrep -f "_better.py fi[t]" >/dev/null; do sleep 30; done
echo "=== baseline finished $(date)"; prune 20260907_hoi4d_2d_v2_baseline
t0=$(date +%s)
while ! grep -q TF_DONE_A sweep_queue_a2.log 2>/dev/null; do
  if [ ! -f experiments/20260907_hoi4d_2d_v2_teacher_forcing/checkpoints/last.ckpt ] && [ $(( $(date +%s) - t0 )) -gt 900 ]; then
    echo "=== pod A has not started teacher_forcing after 15 min: running it HERE $(date)"
    bash runpod/sweep_queue.sh 20260907_hoi4d_2d_v2_teacher_forcing:config/hoi4d_v2_teacher_forcing.yaml
    T 20260907_hoi4d_2d_v2_teacher_forcing; echo TF_DONE_A >> sweep_queue_a2.log; break
  fi
  sleep 60
done
T 20260907_hoi4d_2d_v2_dct_baseline; T 20260907_hoi4d_2d_v2_baseline
[ -f experiments/20260907_hoi4d_2d_v2_teacher_forcing/logs/test.log ] || T 20260907_hoi4d_2d_v2_teacher_forcing
echo HOI4D_ARMS_DONE
