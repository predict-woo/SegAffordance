#!/usr/bin/env bash
# One training pod per experiment, run in parallel, deleted when done.
#
#   bash runpod/train_pod.sh create  <name> [gpu_count]
#   bash runpod/train_pod.sh launch  <name> <exp_id> <config> [train_script] [pythonpath]
#   bash runpod/train_pod.sh status  <name> <exp_id>
#   bash runpod/train_pod.sh delete  <name>
#
# Why one pod per experiment: N pods in parallel cost the same as one pod
# running N experiments sequentially (you pay GPU-hours either way), but finish
# in 1/N the wall time and keep the dev pod free for smoke tests. Training on
# the dev pod squats the GPU and blocks exactly the smoke tests you need in
# order to launch the next run.
#
# Training pods need NO manual bootstrap: code, datasets and cached model
# weights live on the shared network volume, and `launch` builds the python
# env on the pod's local NVMe from the committed requirements.lock (~50s,
# runpod/ensure_env.sh) — imports from the FUSE volume are painfully slow.
set -u

VOLUME_ID=bckt1t9uuf
IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# RTX PRO 6000 Blackwell (96GB) in EU-RO-1. Two SKUs exist and stock differs
# between them — try both before giving up. No A100/H100 in this datacenter.
GPUS=(
  "NVIDIA RTX PRO 6000 Blackwell Server Edition"
  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
  "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
  # Fallback: the gen-5 workhorse (~490 samples/s SF3D). PRO 6000 stock in
  # EU-RO-1 is unreliable ("Low" listings routinely fail actual creates).
  # NOTE 4500-class hosts cap /dev/shm at 29G — launch stages LMDBs on
  # container NVMe (/root/lmdb) and runs 24 workers (see launch).
  "NVIDIA RTX PRO 4500 Blackwell"
)

pod_id() {
  runpodctl pod list 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); pods=d if isinstance(d,list) else d.get('pods',[])
print(next((p['id'] for p in pods if p.get('name')=='$1'), ''))"
}

ssh_info() {
  runpodctl pod get "$1" 2>/dev/null | python3 -c "
import json,sys
p=json.load(sys.stdin); p=p[0] if isinstance(p,list) else p
s=p.get('ssh') or {}
print(f\"{s.get('ip','')} {s.get('port','')} {p.get('costPerHr','')}\")"
}

case "${1:-}" in
  create)
    name="$2"; pod="segaffordance-${name}"; count="${3:-1}"
    for gpu in "${GPUS[@]}"; do
      echo "trying: $gpu x$count"
      out=$(runpodctl pod create --name "$pod" --cloud-type SECURE \
        --gpu-id "$gpu" --gpu-count "$count" --image "$IMAGE" \
        --network-volume-id "$VOLUME_ID" \
        --container-disk-in-gb 40 --ports "22/tcp" 2>&1)
      if ! grep -q '"error"' <<<"$out"; then echo "created with $gpu x$count"; break; fi
      echo "  unavailable, trying next SKU"
    done
    id=$(pod_id "$pod")
    [ -z "$id" ] && { echo "pod create failed for every SKU" >&2; exit 1; }
    # SSH details live under the top-level `ssh` key. `runtime.ports` stays
    # null on these pods — polling it waits forever.
    until [ -n "$(ssh_info "$id" | awk '{print $2}')" ]; do sleep 10; done
    read -r ip port cost <<<"$(ssh_info "$id")"
    python3 - "$name" "$ip" "$port" <<'PY'
import pathlib, re, sys
host, ip, port = f"segaff-{sys.argv[1]}", sys.argv[2], sys.argv[3]
cfg = pathlib.Path.home()/".ssh"/"config"
s = cfg.read_text()
s = re.sub(rf"Host {host}\n(?:  .*\n)*", "", s)
s = s.rstrip("\n") + f"\n\nHost {host}\n  HostName {ip}\n  User root\n  Port {port}\n  IdentityFile ~/.runpod/ssh/runpodctl-ssh-key\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n"
cfg.write_text(s)
print(f"ssh alias: {host}")
PY
    echo "$pod: id=$id ${ip}:${port} \$${cost}/hr"
    ;;

  launch)
    name="$2"; exp="$3"; cfg="$4"; script="${5:-train_OPDReal_better.py}"; pp="${6:-}"
    host="segaff-${name}"
    # Warm RN50 into the page cache (a cold read stalls model construction
    # for minutes), and stage the SF3D LMDBs into /dev/shm: even warm, FUSE
    # mmap access costs ~3.5 ms per LMDB get (profiled 2026-08-03, ~7 ms of
    # the 12.7 ms/sample) — from tmpfs it's a plain memcpy. Sequential cp
    # runs at ~155 MB/s, ~2.5 min for 26 GB, once per pod.
    extra=""
    case "$cfg" in *sf3d*)
      extra="--data.lmdb_path /dev/shm/data.lmdb --data.frame_cache_path /dev/shm/frames.lmdb"
    ;; esac
    ssh -o BatchMode=yes "$host" "bash /workspace/SegAffordance/runpod/ensure_env.sh
      cat /workspace/models/RN50.pt > /dev/null 2>&1 || true
      case '${cfg}' in *dinov3*)
        cat /workspace/cache/dinov3/*.pth > /dev/null 2>&1 || true
      ;; esac
      case '${cfg}' in *sf3d*)
        mkdir -p /dev/shm/data.lmdb /dev/shm/frames.lmdb
        [ -f /dev/shm/data.lmdb/data.mdb ] || time cp /workspace/datasets/sf3d_processed_v2/data.lmdb/data.mdb /dev/shm/data.lmdb/
        [ -f /dev/shm/frames.lmdb/data.mdb ] || time cp /workspace/datasets/sf3d_processed_v2/frames.lmdb/data.mdb /dev/shm/frames.lmdb/
      ;; esac"
    # Detached: nohup inside a subshell, in its OWN ssh invocation. Plain
    # `& disown` in the same ssh call can hang the session.
    # ulimit: pods default to 1024 open files; >=96 dataloader workers dies
    # with Errno 24. Raised unconditionally — cheap insurance at any count.
    ssh -o BatchMode=yes "$host" "( ulimit -n 65536; cd /workspace/SegAffordance && mkdir -p experiments/${exp}/logs experiments/${exp}/checkpoints && \
      HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 ${pp:+PYTHONPATH=$pp} \
      nohup /opt/venv/bin/python ${script} fit --config ${cfg} ${extra} \
      > experiments/${exp}/logs/train.log 2>&1 < /dev/null & ) ; sleep 2; echo launched"
    echo "$name -> $exp ($script)"
    ;;

  status)
    name="$2"; exp="$3"
    ssh -o BatchMode=yes "segaff-${name}" "
      pgrep -f '_better\.py fi[t]' >/dev/null && echo 'proc: RUNNING' || echo 'proc: ENDED'
      nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
      ls /workspace/SegAffordance/experiments/${exp}/checkpoints/ 2>/dev/null | grep best- \
        | sed 's/.*valloss\\([0-9.]*\\)\\.ckpt/\\1 &/' | sort -g | head -3 | cut -d' ' -f2-"
    ;;

  delete)
    pod="segaffordance-${2}"
    id=$(pod_id "$pod")
    [ -z "$id" ] && { echo "no pod named $pod"; exit 0; }
    runpodctl pod delete "$id" && echo "deleted $pod ($id) — billing stopped, /workspace persists"
    ;;

  *) sed -n '2,6p' "$0"; exit 2 ;;
esac
