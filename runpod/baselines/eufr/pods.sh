#!/usr/bin/env bash
# Pod helpers for the A3VLM / 3DOI baselines on the EU-FR-1 volume `bl-eufr`
# (5pw4vigftc, 150 GB). No volume-capable datacenter has A100s, so these run
# on H100 SXM (2x for smokes, 8x for the A3VLM run). Delegates to
# runpod/external/common/pods.sh.
#
#   bash runpod/baselines/eufr/pods.sh gpu <pod-name> [n_gpus]   # create, wait for ssh
#   bash runpod/baselines/eufr/pods.sh run <pod-name> "<cmd>"
#   bash runpod/baselines/eufr/pods.sh scp-to <pod> <local> <remote>
#   bash runpod/baselines/eufr/pods.sh delete <pod-name>
#   bash runpod/baselines/eufr/pods.sh list
set -euo pipefail
export EXT_DC="${EXT_DC:-EU-FR-1}"
# A3VLM's LLaMA2-Accessory stack pins torch 2.0.1 (cu117/cu118); this image ships exactly that.
export EXT_IMAGE="${EXT_IMAGE:-runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04}"
export EXT_DISK_GB="${EXT_DISK_GB:-80}"
VOL="${BL_VOLUME:-5pw4vigftc}"
COMMON="$(cd "$(dirname "$0")/../../external/common" && pwd)/pods.sh"
case "${1:-}" in
  gpu)
    name="$2"; export EXT_GPU_COUNT="${3:-1}"
    for gpu in "NVIDIA H100 80GB HBM3" "NVIDIA H200"; do
      out=$(bash "$COMMON" gpu "$name" "$VOL" "$gpu" 2>&1) && { echo "$out"; exit 0; }
      echo "no stock: $gpu x$EXT_GPU_COUNT"
    done
    echo "pod create failed for every SKU" >&2; exit 1 ;;
  *) bash "$COMMON" "$@" ;;
esac
