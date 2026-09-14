#!/usr/bin/env bash
# Re-export + score the A3VLM chained protocol from the regenerated answers, then regenerate every
# per-sample CSV (6 decimals). Runs on the dev pod: bash runpod/baselines/a3vlm/export_chain_fix.sh (2026-09-14, after the x100-box incident)
set -euo pipefail
cd /workspace/SegAffordance
R=/workspace/datasets/baselines/results/a3vlm; D=/workspace/datasets/baselines/stage/a3vlm
E=experiments/baselines_sf3d; RES=/workspace/datasets/baselines/results
python tools/baselines_sf3d/a3vlm_preds_to_jsonl.py export --joint-results $R/vqa_logs/joint_pred_test.json \
  --joint-questions $R/joint_pred_test.json --meta $D/meta_test.json --out $R/preds_chain.jsonl
python - <<PY
import json
rows=[json.loads(l) for l in open("$R/preds_chain.jsonl")]
print("chain rows", len(rows), "matched", sum(r["matched"] for r in rows), "with mask", sum(r["mask_rle"] is not None for r in rows), "revolute", sum(r["type"]==1 for r in rows))
PY
python tools/baselines_sf3d/score_predictions.py --preds $R/preds_chain.jsonl --out $R/metrics_chain.json > /dev/null
echo SCORED_CHAIN
python tools/baselines_sf3d/per_sample_csv.py \
  --model opdformer_c_rgbd $RES/opd_c_rgbd/preds.jsonl $E/20260912_opdformer_c_rgbd/per_sample_metrics.csv \
  --model opdformer_p_rgbd $RES/opd_p_rgbd/preds.jsonl $E/20260912_opdformer_p_rgbd/per_sample_metrics.csv \
  --model opdformer_p_rgb $RES/opd_p_rgb/preds.jsonl $E/20260912_opdformer_p_rgb/per_sample_metrics.csv \
  --model mopd_rgb $RES/mopd_rgb/preds.jsonl $E/20260912_mopd_rgb/per_sample_metrics.csv \
  --model usdnet $RES/usdnet/preds.jsonl $E/20260912_usdnet/per_sample_metrics.csv \
  --model a3vlm_gtbox $R/preds_gtbox.jsonl $E/20260913_a3vlm/per_sample_metrics_gtbox.csv \
  --model a3vlm_chain $R/preds_chain.jsonl $E/20260913_a3vlm/per_sample_metrics_chain.csv
echo CSV_ALL_DONE
