#!/usr/bin/env bash
# Helpers for the isolated external-baseline experiments (spec
# docs/specs/2026-08-29_external_screw_loss_ab.md). One network volume per
# experiment in EU-FR-1 (H100-capable). Volumes are NEVER deleted here.
#
#   bash pods.sh volume <name> <size_gb>          # create (idempotent by name)
#   bash pods.sh gpu <pod-name> <volume-id> [gpu] # create H100 pod on volume
#   bash pods.sh cpu <pod-name> <volume-id>       # create CPU pod on volume
#   bash pods.sh ssh-info <pod-name>              # "ip port"
#   bash pods.sh run <pod-name> "<cmd>"           # run via ssh (bash -lc)
#   bash pods.sh delete <pod-name>                # delete pod, reconcile list
#   bash pods.sh list                             # pods + volumes
set -euo pipefail
DC="${EXT_DC:-EU-RO-1}"   # EU-FR-1 (H100) was empty on 2026-08-29; EU-RO-1 provisions reliably
IMAGE="${EXT_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o LogLevel=ERROR"

pod_id() { runpodctl pod list 2>/dev/null | python3 -c "
import json,sys; ps=json.load(sys.stdin); ps=ps if isinstance(ps,list) else ps.get('pods',[])
print(next((p['id'] for p in ps if p.get('name')=='$1'),''))"; }
ssh_info() { runpodctl ssh info "$1" 2>/dev/null | python3 -c "
import json,sys; d=json.load(sys.stdin); print(d.get('ip',''), d.get('port',''))"; }

case "${1:-}" in
  volume)
    name="$2"; size="$3"
    existing=$(runpodctl network-volume list 2>/dev/null | python3 -c "
import json,sys; vs=json.load(sys.stdin); vs=vs if isinstance(vs,list) else vs.get('networkVolumes',[])
print(next((v['id'] for v in vs if v.get('name')=='$name'),''))")
    if [ -n "$existing" ]; then echo "$existing"; exit 0; fi
    out=$(runpodctl network-volume create --name "$name" --size "$size" --data-center-id "$DC" 2>&1) || true
    # 500s can still create — re-list before trusting the error
    sleep 3
    id=$(runpodctl network-volume list 2>/dev/null | python3 -c "
import json,sys; vs=json.load(sys.stdin); vs=vs if isinstance(vs,list) else vs.get('networkVolumes',[])
print(next((v['id'] for v in vs if v.get('name')=='$name'),''))")
    [ -n "$id" ] || { echo "volume create failed: $out" >&2; exit 1; }
    echo "$id" ;;
  gpu)
    pod="$2"; vol="$3"
    id=$(pod_id "$pod"); [ -n "$id" ] && { echo "exists $id"; exit 0; }
    if [ -n "${4:-}" ]; then GPUS=("$4"); else GPUS=(
      "NVIDIA RTX PRO 6000 Blackwell Server Edition"
      "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
      "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" "NVIDIA H200"); fi
    for gpu in "${GPUS[@]}"; do
      echo "trying: $gpu"
      runpodctl pod create --name "$pod" --cloud-type SECURE --gpu-id "$gpu" --gpu-count "${EXT_GPU_COUNT:-1}" \
        --data-center-ids "$DC" --image "$IMAGE" --network-volume-id "$vol" \
        --container-disk-in-gb "${EXT_DISK_GB:-60}" --ports "22/tcp" >/dev/null 2>&1 || true
      id=$(pod_id "$pod"); [ -n "$id" ] && break
    done
    [ -n "$id" ] || { echo "pod create failed for every SKU" >&2; exit 1; }
    until [ -n "$(ssh_info "$id" | awk '{print $2}')" ]; do sleep 10; done
    echo "created $pod $id: $(ssh_info "$id")" ;;
  cpu)
    pod="$2"; vol="$3"
    id=$(pod_id "$pod"); [ -n "$id" ] && { echo "exists $id"; exit 0; }
    runpodctl pod create --name "$pod" --compute-type cpu --cloud-type SECURE \
      --data-center-ids "$DC" --image "$IMAGE" --network-volume-id "$vol" \
      --container-disk-in-gb 20 --ports "22/tcp" >/dev/null
    id=$(pod_id "$pod"); [ -n "$id" ] || { echo "pod create failed" >&2; exit 1; }
    until [ -n "$(ssh_info "$id" | awk '{print $2}')" ]; do sleep 10; done
    echo "created $pod $id: $(ssh_info "$id")" ;;
  ssh-info) ssh_info "$(pod_id "$2")" ;;
  run)
    read -r ip port <<<"$(ssh_info "$(pod_id "$2")")"
    ssh $SSH_OPTS -p "$port" "root@$ip" "bash -lc $(printf '%q' "$3")" ;;
  scp-to)   # scp-to <pod> <local> <remote>
    read -r ip port <<<"$(ssh_info "$(pod_id "$2")")"
    scp $SSH_OPTS -P "$port" -r "$3" "root@$ip:$4" ;;
  scp-from) # scp-from <pod> <remote> <local>
    read -r ip port <<<"$(ssh_info "$(pod_id "$2")")"
    scp $SSH_OPTS -P "$port" -r "root@$ip:$3" "$4" ;;
  delete)
    id=$(pod_id "$2"); [ -n "$id" ] && runpodctl pod delete "$id" >/dev/null && echo "deleted $2 ($id)"
    sleep 3; echo "remaining pods:"; runpodctl pod list 2>/dev/null | python3 -c "
import json,sys; ps=json.load(sys.stdin); ps=ps if isinstance(ps,list) else ps.get('pods',[])
[print(' ',p['id'],p.get('name'),p.get('desiredStatus'),p.get('costPerHr')) for p in ps] or print('  (none)')" ;;
  list)
    echo "pods:"; runpodctl pod list 2>/dev/null | python3 -c "
import json,sys; ps=json.load(sys.stdin); ps=ps if isinstance(ps,list) else ps.get('pods',[])
[print(' ',p['id'],p.get('name'),p.get('desiredStatus'),p.get('costPerHr')) for p in ps] or print('  (none)')"
    echo "volumes:"; runpodctl network-volume list 2>/dev/null | python3 -c "
import json,sys; vs=json.load(sys.stdin); vs=vs if isinstance(vs,list) else vs.get('networkVolumes',[])
[print(' ',v['id'],v.get('name'),v.get('size'),'GB',v.get('dataCenterId')) for v in vs]" ;;
  *) echo "usage: see header"; exit 1 ;;
esac
