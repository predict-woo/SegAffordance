#!/usr/bin/env bash
# One training pod per experiment, run in parallel, deleted when done.
#
#   bash runpod/train_pod.sh create  <name>
#   bash runpod/train_pod.sh launch  <name> <exp_id> <config> [pythonpath]
#   bash runpod/train_pod.sh status  <name> <exp_id>
#   bash runpod/train_pod.sh delete  <name>
#
# Why one pod per experiment: N pods in parallel cost the same as one pod
# running N experiments sequentially (you pay GPU-hours either way), but finish
# in 1/N the wall time and keep the dev pod free for smoke tests. Training on
# the dev pod squats the GPU and blocks exactly the smoke tests you need in
# order to launch the next run.
#
# Training pods need NO bootstrap: /workspace/venv, the code, the datasets and
# any cached model weights all live on the shared network volume. A pip install
# done once on any pod (into /workspace/venv) is visible to every later pod.
set -u

VOLUME_ID=bckt1t9uuf
IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
# RTX PRO 6000 Blackwell (96GB) in EU-RO-1. Two SKUs exist and stock differs
# between them — try both before giving up. No A100/H100 in this datacenter.
GPUS=(
  "NVIDIA RTX PRO 6000 Blackwell Server Edition"
  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
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
    name="$2"; pod="segaffordance-${name}"
    for gpu in "${GPUS[@]}"; do
      echo "trying: $gpu"
      out=$(runpodctl pod create --name "$pod" --cloud-type SECURE \
        --gpu-id "$gpu" --image "$IMAGE" --network-volume-id "$VOLUME_ID" \
        --container-disk-in-gb 20 --ports "22/tcp" 2>&1)
      if ! grep -q '"error"' <<<"$out"; then echo "created with $gpu"; break; fi
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
    name="$2"; exp="$3"; cfg="$4"; pp="${5:-}"
    host="segaff-${name}"
    # Detached: nohup inside a subshell, in its OWN ssh invocation. Plain
    # `& disown` in the same ssh call can hang the session.
    ssh -o BatchMode=yes "$host" "( cd /workspace/SegAffordance && mkdir -p experiments/${exp}/logs experiments/${exp}/checkpoints && \
      HF_HOME=/root/hfcache HF_HUB_OFFLINE=1 ${pp:+PYTHONPATH=$pp} \
      nohup /workspace/venv/bin/python train_OPDReal_better.py fit --config ${cfg} \
      > experiments/${exp}/logs/train.log 2>&1 < /dev/null & ) ; sleep 2; echo launched"
    echo "$name -> $exp"
    ;;

  status)
    name="$2"; exp="$3"
    ssh -o BatchMode=yes "segaff-${name}" "
      pgrep -f 'train_OPDReal_bette[r]' >/dev/null && echo 'proc: RUNNING' || echo 'proc: ENDED'
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
