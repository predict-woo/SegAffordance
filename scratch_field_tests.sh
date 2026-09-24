#!/bin/bash
# field model test passes on the DEV pod (the chain's tests failed on the datamodule binding; pod already deleted)
cd /workspace/SegAffordance
export TORCHINDUCTOR_CACHE_DIR=/root/inductor_cache TRITON_CACHE_DIR=/root/triton_cache
E=20260913_field_joint4_l2anchor; ck=$(ls experiments/$E/checkpoints/best-*.ckpt | head -1)
MP="--model.model_params.articulation_readout dense --model.model_params.dense_hidden 256 --model.model_params.compile_model false"
T() { logn=$1; shift; HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_field_better.py test --config config/sf3d_test_decoder_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false $MP --data.batch_size_val 8 --data.num_workers_val 6 "$@" > experiments/$E/logs/$logn 2>&1; echo "=== TESTDONE $logn exit=$?"; }
T test_sf3d.log
T test_sf3d_writerlen.log --model.model_params.trajectory_decoder_length writer
DEC="$MP --model.model_params.use_trajectory_head false --model.model_params.trajectory_decoder analytic --model.model_params.trajectory_dct_coeffs 0"
for c in hoi4d_v2 epic_v1 arctic_v1; do
  HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 /opt/venv/bin/python train_field_better.py test --config config/${c}_rgb_scalefree.yaml --ckpt_path $ck --trainer.logger=false $DEC --data.batch_size_val 8 --data.num_workers_val 6 > experiments/$E/logs/test_${c}.log 2>&1
  echo "=== TESTDONE $c exit=$?"
done
echo FIELD_TESTS_DONE
