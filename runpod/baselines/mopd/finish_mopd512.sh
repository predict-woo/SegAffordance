#!/usr/bin/env bash
# Finish mopd512_rgb on the MAIN volume after the H200 run was copied back (2026-09-15): the chain's
# export step needs the SF3D GT cache (/workspace/cache) that only the main volume has. Waits for the
# copied files, exports preds.jsonl, fills results/, scores. Run detached on the dev pod.
set -uo pipefail
B=/workspace/datasets/baselines; R=$B/runs/mopd512_rgb; cd /workspace/SegAffordance
until [ "$(stat -c %s $R/test/inference/instances_predictions.pth 2>/dev/null)" = 129307104 ] && \
      [ "$(stat -c %s $R/model_final.pth 2>/dev/null)" = 769543526 ] && [ -f $B/logs/test_mopd512_rgb.log ]; do sleep 30; done
sleep 60; echo "== $(date -u) files present"
python tools/baselines_sf3d/run.py tools/baselines_sf3d/opd_preds_to_jsonl.py --pred $R/test/inference/instances_predictions.pth --data-dir $B/data/opd_sf3d_512 --out $R/preds.jsonl 2>&1 | tail -2
wc -l $R/preds.jsonl
mkdir -p $B/results/mopd512_rgb; cp $R/preds.jsonl $B/results/mopd512_rgb/; cp $R/config.yaml $B/results/mopd512_rgb/config_resolved.yaml
cp $B/logs/train_mopd512_rgb.log $B/logs/test_mopd512_rgb.log $B/results/mopd512_rgb/ 2>/dev/null; touch $R/CHAIN_DONE
bash runpod/baselines/score_run.sh mopd512_rgb 20260914_mopd_rgb_512_full mopd512_rgb 2>&1 | tail -3
echo "== $(date -u) FINISH_DONE"
