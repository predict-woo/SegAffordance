#!/usr/bin/env bash
# A3VLM env (LLaMA2-Accessory fork, SPHINX-1k 13B) on the EU-FR-1 pods created by
# runpod/baselines/eufr/pods.sh (image runpod/pytorch:2.0.1-py3.10-cuda11.8.0 = the
# torch 2.0.1 pin of LLaMA2-Accessory's requirements.txt). Idempotent.
set -euo pipefail
B=/workspace/bl; R=$B/repos; CK=$B/ckpt; mkdir -p $R $CK $B/logs $B/runs $B/data /workspace/tmp
export PIP_BREAK_SYSTEM_PACKAGES=1 TMPDIR=/workspace/tmp
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'gpus')"
# LLaMA2-Accessory requirements.txt minus the torch lines (already in the image) + our converter deps.
# Pins: unpinned open_clip_torch/timm/bitsandbytes drag torch to 2.14+cu130 (seen 2026-09-12); these are the
# late-2023 releases contemporary with LLaMA2-Accessory. bitsandbytes (quant only) is left out.
[ -f /root/.done_pip_a3vlm ] || { pip install -q fairscale sentencepiece "transformers==4.34.1" "open_clip_torch==2.23.0" "timm==0.9.12" fire packaging pyyaml Ninja einops regex h5py pandas pyarrow tensorboard "gradio==3.50.2" "httpx[socks]" tqdm "huggingface_hub[cli]" dacite pycocotools opencv-python-headless lmdb && touch /root/.done_pip_a3vlm; }
python -c "import torch; assert torch.__version__.startswith('2.0.1'), torch.__version__" || { pip uninstall -y -q torch torchvision torchaudio triton; pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118; }
pip check || true
[ -d $R/A3VLM/.git ] || git clone -q https://github.com/changhaonan/A3VLM $R/A3VLM
cd $R/A3VLM && git checkout -q 436e715 && echo "A3VLM @ $(git rev-parse --short HEAD)"
# The fork's model/accessory is a partial tree (no llama.py, global_configs.py, ...): it is meant to be
# overlaid on upstream LLaMA2-Accessory ("follow the original repository to set up"). Use the last
# upstream commit before the fork's commit date (2024-10-07) and copy the fork's files over it.
[ -d $R/LLaMA2-Accessory/.git ] || git clone -q https://github.com/Alpha-VLLM/LLaMA2-Accessory $R/LLaMA2-Accessory
cd $R/LLaMA2-Accessory && git checkout -q $(git rev-list -1 --before=2024-10-08 main) && echo "LLaMA2-Accessory @ $(git rev-parse --short HEAD) ($(git log -1 --format=%ci))"
cp -r $R/A3VLM/model/accessory/. $R/LLaMA2-Accessory/accessory/
# flash-attn is optional in LLaMA2-Accessory (torch sdpa fallback). Opt-in only (FLASH=1): a source build against
# torch 2.0.1 takes >30 min and the sdpa path is what the released recipe falls back to anyway.
[ "${FLASH:-0}" = "1" ] && { python -c "import flash_attn" 2>/dev/null || pip install -q flash-attn --no-build-isolation > $B/logs/flash_attn_pip.log 2>&1 || echo "flash-attn not installed (sdpa fallback)"; }
# dinov2: torch.hub.load(..., pretrained=False) still needs the hub repo checked out.
export TORCH_HOME=$CK/torch; mkdir -p $TORCH_HOME/hub
[ -d $TORCH_HOME/hub/facebookresearch_dinov2_main ] || git clone -q https://github.com/facebookresearch/dinov2 $TORCH_HOME/hub/facebookresearch_dinov2_main
# SPHINX-1k weights (2 x 19.9 GB) + tokenizer, resumable.
mkdir -p $CK/sphinx1k; cd $CK/sphinx1k
HF=https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/resolve/main/finetune/mm/SPHINX/SPHINX-1k
for f in config.json meta.json tokenizer.model; do [ -s $f ] || curl -sL -o $f "$HF/$f"; done
for f in consolidated.00-of-02.model.pth consolidated.01-of-02.model.pth; do
  [ -f .done_$f ] || { curl -sL -C - -o $f "$HF/$f" && [ "$(stat -c %s $f)" = 19909875004 ] && touch .done_$f; } &
done
wait; ls -la $CK/sphinx1k; cat $CK/sphinx1k/config.json; echo
cd $R/LLaMA2-Accessory/accessory && python -c "import sys; sys.path.insert(0, '..'); from accessory.model.LLM import llama_ens5; from accessory.model.meta import MetaModel; print('a3vlm import ok')"
touch $B/.env_ready   # artefact-based readiness flag (ENV_READY), see scripts/a3vlm_auto.sh
echo "== a3vlm env done"
