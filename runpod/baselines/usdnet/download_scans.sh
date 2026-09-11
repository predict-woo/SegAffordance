#!/usr/bin/env bash
# Download the SceneFun3D assets USDNet needs (laser_scan_5mm, annotations,
# motions, one video transform per visit) for every scene in our key list.
# Plain curl over the toolkit's URL scheme (data_downloader/download_utils/
# download_data.py: {SceneFun3D_url}/{train|test}/{visit}/{file}); the
# toolkit's own downloader drags in pandas + moviepy and fights the USDNet env.
# ~100 MB/scene, 224 scenes. Idempotent: existing complete files are skipped.
set -uo pipefail
B=/workspace/datasets/baselines; R=$B/repos/scenefun3d; D=$B/data/sf3d_scans; mkdir -p $D
URL=https://cvg-data.inf.ethz.ch/scenefun3d/v1
[ -d $R/.git ] || git clone -q https://github.com/SceneFun3D/scenefun3d $R
python3 - <<'PY'
import pickle, csv, os
D = '/workspace/datasets/baselines/data/sf3d_scans'
cache = pickle.loads(open('/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl','rb').read())
keys = [k.decode() if isinstance(k, bytes) else k for k in cache['keys']]
first = {}
for k in keys:
    v, vid = k.split('/')[:2]; first.setdefault(v, vid)
trainval = set()
with open('/workspace/datasets/baselines/repos/scenefun3d/benchmark_file_lists/train_val_set.csv') as f:
    for row in csv.DictReader(f): trainval.add(str(row['visit_id']))
with open(f'{D}/list.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['visit_id', 'video_id', 'split'])
    for v in sorted(first): w.writerow([v, first[v], 'train' if v in trainval else 'test'])
print('visits', len(first), 'not in train_val_set:', sum(1 for v in first if v not in trainval))
PY
fetch() {  # fetch <url> <dst>
  local url=$1 dst=$2
  [ -s "$dst" ] && return 0
  mkdir -p "$(dirname "$dst")"
  curl -sS --fail --retry 3 --retry-delay 5 -o "$dst.tmp" "$url" && mv "$dst.tmp" "$dst" || { echo "FAIL $url"; rm -f "$dst.tmp"; return 1; }
}
export -f fetch; export URL D
tail -n +2 $D/list.csv | while IFS=, read -r v vid split; do
  echo "$URL/$split/$v/${v}_laser_scan.ply $D/$v/${v}_laser_scan.ply"
  echo "$URL/$split/$v/${v}_annotations.json $D/$v/${v}_annotations.json"
  echo "$URL/$split/$v/${v}_motions.json $D/$v/${v}_motions.json"
  echo "$URL/$split/$v/$vid/${vid}_refined_transform.npy $D/$v/$vid/${vid}_transform.npy"
done > $D/jobs.txt
wc -l $D/jobs.txt
xargs -P "${DL_PAR:-8}" -n 2 bash -c 'fetch "$0" "$1"' < $D/jobs.txt
echo "== verify"; n=0; miss=0
for v in $(tail -n +2 $D/list.csv | cut -d, -f1); do
  if [ -s $D/$v/${v}_laser_scan.ply ] && [ -s $D/$v/${v}_annotations.json ] && [ -s $D/$v/${v}_motions.json ] && ls $D/$v/*/*_transform.npy >/dev/null 2>&1; then n=$((n+1)); else miss=$((miss+1)); echo "MISSING $v"; fi
done
echo "complete visits: $n, missing: $miss"; du -sh $D; touch $D/.done_download
