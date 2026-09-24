#!/usr/bin/env bash
# USDNet data staging (runs on a CPU or GPU pod attached to the ext-usdnet volume).
# Idempotent via markers on /workspace.
set -euo pipefail
W=/workspace; mkdir -p $W/repo $W/data $W/ckpt_init $W/results $W/logs
python -m pip install -q --break-system-packages gdown 2>/dev/null || python -m pip install -q gdown
echo "== repo"; [ -d $W/repo/USDNet/.git ] || git clone -q --recursive https://github.com/insait-institute/USDNet $W/repo/USDNet
cd $W/repo/USDNet && git checkout -q 0ba303d90375b86f61fb79007f6ff9757f5f2c15 && git log -1 --format="%h %cd"
echo "== processed data (5.8 GB, Google Drive)"; cd $W/data
if [ ! -f .done_processed ]; then
  gdown --continue "https://drive.google.com/uc?id=1QS_D5CBoF5AssleA3kdMMirwFEWPMaMW" -O processed.zip
  ls -la processed.zip; unzip -l processed.zip | sed -n 4,12p
  unzip -q -o processed.zip && rm -f processed.zip && ls && touch .done_processed
fi
du -sh $W/data/* | head
echo "== checkpoints"; cd $W/ckpt_init
if [ ! -f .done ]; then
  gdown --continue "https://drive.google.com/uc?id=1YtSnYhJqvnOePKxOq5FIgxHwjoUKpmWm" || true
  gdown --continue "https://drive.google.com/uc?id=1i6ohSVXBGEPkXkWJMyD-YhAmo7vAjoR0" || true
  curl -sL -o scannet200_benchmark.ckpt https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt
  ls -la; touch .done
fi
echo "== staging done"
