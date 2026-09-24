#!/usr/bin/env bash
# bash eval.sh <tag> <ckpt model.pt>  — infer on the PM test split, their metrics + axis metrics
set -euo pipefail
W=/workspace; R=$W/repo/particulate; cd $R; tag=$1; ckpt=$2
mkdir -p $W/data/eval_gt $W/runs/eval_$tag/infer $W/results/$tag
python - <<'PY'
import json,os,subprocess
ids=json.load(open('/workspace/ours/pm_test_ids.json'))
for i in ids:
    d=f'/workspace/data/preprocessed/{i}'
    if os.path.isdir(d) and not os.path.exists(f'/workspace/data/eval_gt/{i}.npz'):
        subprocess.run(['python','-m','particulate.data.cache_points','--root',d,'--output_path',f'/workspace/data/eval_gt/{i}','--num_points','100000','--ratio_sharp','0','--format','eval'],check=False,capture_output=True)
print('eval gt cached:', len(os.listdir('/workspace/data/eval_gt')))
PY
python infer.py --input_mesh "$W/data/preprocessed/*/original.obj" --eval --output_dir $W/runs/eval_$tag/infer --up_dir Z --ckpt_path "$ckpt" > $W/logs/infer_$tag.log 2>&1 || tail -5 $W/logs/infer_$tag.log
python evaluate.py --gt_dir $W/data/eval_gt --result_dir $W/runs/eval_$tag/infer --output_dir $W/runs/eval_$tag/metrics > $W/logs/eval_$tag.log 2>&1 || tail -5 $W/logs/eval_$tag.log
python $W/ours/eval_axis_particulate.py --gt_dir $W/data/eval_gt --result_dir $W/runs/eval_$tag/infer --out $W/results/$tag/metrics_axis.json
cp -r $W/runs/eval_$tag/metrics $W/results/$tag/ 2>/dev/null || true; ls $W/results/$tag
