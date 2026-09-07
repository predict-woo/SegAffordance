#!/bin/bash
# Fetch VISOR dense-interpolation zips for every VISOR video that overlaps
# our EPIC interactions (from coverage.json), 12 parallel streams, with
# retries + zip validation. -> /workspace/datasets/visor/interpolations/<split>/
B=https://data.bris.ac.uk/datasets/2v6cgv1x04ol22qp9rm9x2j6a7/Interpolations-DenseAnnotations
V=/workspace/datasets/visor; mkdir -p $V/interpolations/train $V/interpolations/val
cd $V
python3 - <<'PY' > lists/interp_targets.txt
import json
cov = json.load(open("/workspace/datasets/visor/coverage.json"))
seen = {}
for o in cov: seen[o["video_id"]] = o["split"]
for v, s in sorted(seen.items()): print(s, v)
PY
echo "targets: $(wc -l < lists/interp_targets.txt)"
fetch() { split=$1; vid=$2; z=$V/interpolations/$split/${vid}_interpolations.zip
  for a in 1 2 3 4; do
    python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$z') else 1)" 2>/dev/null && { echo "ok $vid"; return; }
    curl -s -m 1800 --retry 3 -o "$z" "$B/$split/${vid}_interpolations.zip" || rm -f "$z"
  done
  python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$z') else 1)" 2>/dev/null && echo "ok $vid" || echo "FAILED $vid"
}
export -f fetch; export V B
xargs -a lists/interp_targets.txt -P 12 -L 1 bash -c 'fetch $0 $1' 2>&1 | tee lists/interp_fetch.log | grep -c "^ok"
echo "failed: $(grep -c FAILED lists/interp_fetch.log)"; du -sh $V/interpolations
echo INTERP_DONE
