#!/bin/bash
# After the HOI4D depth-complete arms are scored, pick the init checkpoint
# (lr3e5_depth unless tf beats it on held-out mIoU+PDet), patch the SF3D
# post-training config, run it, test it.
cd /workspace/SegAffordance
until grep -q FINAL_DONE sweep_queue_c.log; do sleep 60; done
# tf was SKIPPED by the c-queue (stale last.ckpt from a fast_dev_run smoke): run it now + its test pass
rm -f experiments/20260907_hoi4d_2d_v2_tf/checkpoints/*.ckpt; rm -rf experiments/20260907_hoi4d_2d_v2_tf/logs
bash runpod/sweep_queue.sh 20260907_hoi4d_2d_v2_tf:config/hoi4d_v2_tf.yaml
exp=experiments/20260907_hoi4d_2d_v2_tf; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
echo "=== TEST tf $ck"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb --data.num_workers_val 8 > $exp/logs/test.log 2>&1
echo "=== TESTDONE tf exit=$?"; echo HOI4D_TF_DONE
pick=$(/opt/venv/bin/python - <<'PY'
import re
def score(a):
    try: t=open(f"experiments/20260907_hoi4d_2d_v2_{a}/logs/test.log").read().replace("\r","\n")
    except FileNotFoundError: return None
    g=lambda k: float(re.search(rf"test/{k}\s+([0-9.]+)", t).group(1))
    return g("mean_iou")*100 + g("p_det")
s={a:score(a) for a in ("lr3e5_depth","tf")}
print("tf" if (s["tf"] or -1) > (s["lr3e5_depth"] or -1) else "lr3e5_depth")
PY
)
exp=experiments/20260907_hoi4d_2d_v2_$pick; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
echo "=== INIT from $pick: $ck"
sed -i "s#finetune_from_path: \".*\"#finetune_from_path: \"/workspace/SegAffordance/$ck\"#" config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml experiments/20260907_sf3d_g19_dct_ft_hoi4d/config.yaml
grep finetune_from_path config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml
bash runpod/sweep_queue.sh 20260907_sf3d_g19_dct_ft_hoi4d:config/sf3d_train_runpod_g19_dct_ft_hoi4d.yaml
exp=experiments/20260907_sf3d_g19_dct_ft_hoi4d; ck=$(ls $exp/checkpoints/best-*.ckpt | head -1)
echo "=== TEST sf3d_ft $ck"
HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_SF3D_better.py test --config $exp/config.yaml --ckpt_path $ck --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb > $exp/logs/test.log 2>&1
echo "=== TESTDONE sf3d_ft exit=$?"; echo SF3D_FT_DONE
