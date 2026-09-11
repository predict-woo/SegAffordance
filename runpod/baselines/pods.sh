#!/usr/bin/env bash
# Pod helpers for the external SF3D baselines: A100-class pods on the MAIN
# volume (bckt1t9uuf, EU-RO-1) with the torch 2.1.1 / cu121 devel image
# (torch 2.1 has no Blackwell kernels; detectron2 + MinkowskiEngine build on
# A100/H100). Delegates to runpod/external/common/pods.sh.
#
#   bash runpod/baselines/pods.sh gpu <pod-name>        # create, wait for ssh
#   bash runpod/baselines/pods.sh run <pod-name> "<cmd>"
#   bash runpod/baselines/pods.sh scp-to <pod> <local> <remote>
#   bash runpod/baselines/pods.sh scp-from <pod> <remote> <local>
#   bash runpod/baselines/pods.sh delete <pod-name>
#   bash runpod/baselines/pods.sh list
set -euo pipefail
export EXT_DC="${EXT_DC:-EU-RO-1}"
export EXT_IMAGE="${EXT_IMAGE:-runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04}"
VOL="${BL_VOLUME:-bckt1t9uuf}"
COMMON="$(cd "$(dirname "$0")/../external/common" && pwd)/pods.sh"
case "${1:-}" in
  gpu)
    name="$2"
    for gpu in "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA H100 80GB HBM3" "NVIDIA H100 PCIe"; do
      out=$(bash "$COMMON" gpu "$name" "$VOL" "$gpu" 2>&1) && { echo "$out"; exit 0; }
      echo "no stock: $gpu"
    done
    echo "pod create failed for every SKU" >&2; exit 1 ;;
  *) bash "$COMMON" "$@" ;;
esac
