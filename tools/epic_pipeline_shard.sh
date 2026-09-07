#!/bin/bash
# EPIC/VISOR production shard: for videos[i::n] (all VISOR-covered videos in
# coverage.json, sorted): chunked fetch -> SAM2 propagation work items ->
# delete the video (unless KEEP_VIDEOS=1). Resumable: finished videos are
# skipped via <out>/_video_done/<vid>. Every step is under `timeout` so a
# stall cannot hang the shard. Log markers: "=== VIDEO", FETCH_FAIL,
# PROP_FAIL, SHARD_DONE.
#   bash tools/epic_pipeline_shard.sh <i> <n> [out_dir]
set -u
i=$1; n=$2; OUT=${3:-/workspace/datasets/epic_processed_2d/work}
cd /workspace/SegAffordance
PY=/opt/venv/bin/python; VIDS=/workspace/datasets/epic_videos
mkdir -p "$OUT/_video_done" "$VIDS"
$PY - "$i" "$n" <<'PY' > /tmp/shard_videos.txt
import json, sys
i, n = int(sys.argv[1]), int(sys.argv[2])
cov = json.load(open("/workspace/datasets/visor/coverage.json"))
vids = sorted({o["video_id"] for o in cov if o["hits"]})
for v in vids[i::n]: print(v)
PY
echo "=== SHARD $i/$n: $(wc -l < /tmp/shard_videos.txt) videos, $(date)"
while read -r vid; do
  [ -f "$OUT/_video_done/$vid" ] && { echo "=== SKIP $vid (done)"; continue; }
  echo "=== VIDEO $vid start $(date +%H:%M:%S)"
  ok=0
  for attempt in 1 2; do
    if timeout 2700 $PY tools/epic_fetch_video.py "$vid" "$VIDS" --chunks 16; then ok=1; break; fi
    echo "fetch attempt $attempt failed for $vid"; sleep 30
  done
  [ $ok = 1 ] || { echo "FETCH_FAIL $vid $(date +%H:%M:%S)"; rm -f "$VIDS/$vid.MP4"*; continue; }
  if timeout 5400 $PY tools/epic_visor_propagate_batch.py --video "$vid" --out "$OUT" --max-d 60; then
    echo "=== VIDEO $vid done $(date +%H:%M:%S)"
  else
    echo "PROP_FAIL $vid exit=$? $(date +%H:%M:%S)"
  fi
  [ "${KEEP_VIDEOS:-0}" = "1" ] || rm -f "$VIDS/$vid.MP4"
done < /tmp/shard_videos.txt
echo "SHARD_DONE $i/$n $(date)"
