#!/bin/bash
# (Re)start the EPIC axis annotator server on the dev pod, detached. Kills whatever holds port 8080 first.
# Run ON THE POD:  bash runpod/epic_annot_serve.sh    (then on the Mac: ssh -N -L 8080:localhost:8080 segaff-dev)
# Never `pkill -f` the server's script name from an ssh command that also contains it: the shell matches itself.
cd /workspace/SegAffordance
fuser -k 8080/tcp > /dev/null 2>&1 || true
sleep 1
nohup /opt/venv/bin/python tools/epic_axis_annotator.py serve \
  --clouds /workspace/datasets/epic_gt_annot/clouds \
  --out /workspace/datasets/epic_gt_annot/annotations --port 8080 > /workspace/epic_annot.log 2>&1 < /dev/null &
sleep 3
echo "procs $(pgrep -f 'epic_axis_annotator.p[y]' | wc -l)  page $(curl -s -m 10 localhost:8080/ | grep -c faceNormalCam)  records $(curl -s -m 10 localhost:8080/api/records | /opt/venv/bin/python -c 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null)"
