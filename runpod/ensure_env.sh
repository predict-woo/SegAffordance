#!/usr/bin/env bash
# Build/refresh the python env on pod-local NVMe (runs ON a pod).
#
#   bash runpod/ensure_env.sh            # make /opt/venv match requirements.lock
#   bash runpod/ensure_env.sh --relock   # re-resolve requirements.txt -> requirements.lock
#
# The env is a plain (non system-site) venv built entirely from the committed
# lockfile, so every pod boots the exact same package set — the image's
# system python is never mixed in (that mixing is how cu12/cu13 cudnn drift
# happened). It lives on the container disk, which is wiped on every pod
# stop; rebuilding cold takes ~50s (measured 2026-08-03: uv prepared all 70
# packages, incl. the 3.5GB cu128 torch, in 40s — the datacenter pipe to
# PyPI is far faster than any volume-staging scheme). When the env already
# matches the lock this is a sub-second no-op, so it's safe to call from
# every entry point (setup.sh, dev.sh run, train_pod.sh launch, ~/.bashrc).
#
# To change the env: edit requirements.txt, run `ensure_env.sh --relock` on
# the dev pod, commit both files. Every pod (and the next boot of this one)
# picks it up automatically.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/opt/venv
UV=/workspace/bin/uv
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# uv itself: a single binary kept on the volume (sequential read — fast).
# Bootstrap it from the network once per volume.
if [ ! -x "$UV" ]; then
  mkdir -p /workspace/bin
  curl -LsSf https://astral.sh/uv/install.sh \
    | env UV_INSTALL_DIR=/workspace/bin UV_NO_MODIFY_PATH=1 sh >/dev/null
fi

if [ "${1:-}" = "--relock" ]; then
  "$UV" pip compile "$REPO/requirements.txt" -o "$REPO/requirements.lock" \
    --emit-index-url --extra-index-url "$TORCH_INDEX" \
    --python-version 3.12 --quiet
  echo "ensure_env: wrote $REPO/requirements.lock — commit it (from the Mac side)"
fi

[ -x "$VENV/bin/python" ] || "$UV" venv "$VENV" >/dev/null 2>&1

# sync = install missing + remove extraneous: the venv IS the lockfile.
# --no-cache: the uv cache would die with the container disk anyway, and
# skipping it halves peak disk usage during a cold build.
"$UV" pip sync --python "$VENV/bin/python" --no-cache --quiet "$REPO/requirements.lock"
