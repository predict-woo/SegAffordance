#!/usr/bin/env bash
# Score one finished baseline run on the dev pod: our metrics.json, the per-sample CSV, and the
# confidence-thresholded PDet/mIoU the paper reports next to the oracle numbers.
#   bash runpod/baselines/score_run.sh <run> <experiment id> <model name> [preds.jsonl]
#   e.g. bash runpod/baselines/score_run.sh opd512_p_rgb 20260914_opdformer_p_rgb_512 opdformer_p_rgb_512
# Writes experiments/baselines_sf3d/<id>/{metrics.json,per_sample_metrics.csv,thresholded.json}.
set -euo pipefail
RUN="$1"; EXP="$2"; MODEL="$3"
cd /workspace/SegAffordance
B=/workspace/datasets/baselines; E=experiments/baselines_sf3d/$EXP; P="${4:-$B/results/$RUN/preds.jsonl}"
mkdir -p $E
python tools/baselines_sf3d/score_predictions.py --preds $P --out $E/metrics.json > /dev/null
echo SCORED
python tools/baselines_sf3d/per_sample_csv.py --model $MODEL $P $E/per_sample_metrics.csv
python - "$E" <<'PY'
import csv, json, sys
import numpy as np
E = sys.argv[1]
rows = list(csv.DictReader(open(f"{E}/per_sample_metrics.csv")))
iou = np.array([float(r["mask_iou"]) for r in rows]); conf = np.array([float(r["confidence"]) for r in rows])
conf = np.nan_to_num(conf, nan=-1.0)
out = {"n": len(rows), "oracle": {"p_det": 100 * float(np.mean(iou >= 0.5)), "mean_iou": float(iou.mean())}}
for t in (0.3, 0.5, 0.7):
    keep = conf > t
    out[f"conf>{t}"] = {"n_kept": int(keep.sum()), "p_det": 100 * float(np.mean((iou >= 0.5) & keep)),
                        "mean_iou": float(np.mean(iou * keep))}
json.dump(out, open(f"{E}/thresholded.json", "w"), indent=1)
print(json.dumps(out))
PY
echo SCORE_RUN_DONE
