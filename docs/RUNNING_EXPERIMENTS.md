# Running experiments in SegAffordance

A practical guide for getting the code training on your own machine.
(Repo-state context lives in `STATE.md`; our RunPod infra specifics in
`runpod/README.md` — you don't need either to train.)

## 1. What this is

Language-conditioned affordance segmentation + articulation prediction
(PyTorch Lightning). One model predicts, from an RGB-D frame and a text
instruction: the part mask, an interaction point, articulation type
(rotation/translation), axis, origin, and a 20-point 3D motion
trajectory. Entry points are `train_SF3D_better.py` (main dataset),
`train_OPDReal_better.py`, `train_OPDMulti_better.py` — each a
`LightningCLI` app driven by a YAML config from `config/`.

## 2. Environment

Python **3.12**, CUDA GPU. The installable artifact is
`requirements.lock` (pinned to cu128 torch 2.9.1 builds):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock
```

(`requirements.txt` is intent only — always install the lock. To change
deps, edit `requirements.txt` and regenerate the lock.)

VRAM: the current configs run the dinov3 backbone at 512 px with
`batch_size_train: 64` → ~28 GB. On a smaller GPU, lower
`batch_size_train`/`batch_size_val` in your config; nothing else needs
to change.

## 3. Data and weights you need

Ask us for access (we share via a RunPod S3 bucket) — none of this is in
git:

| artifact | size | config field |
|---|---|---|
| `sf3d_processed_v3/` (LMDB: records + annotations) | 13 GB | `data.train_data_dir` |
| `sf3d_frames_512.lmdb` (pre-resized frame cache) | 39 GB | `data.frame_cache_path` |
| dinov3 ViT-L backbone weights (`dinov3_vitl16_pretrain_lvd1689m-*.pth`) | 1.2 GB | `model.model_params.dinov3_backbone_weights` |
| dinov3-txt text encoder (`..._dinotxt_vision_head_and_text_encoder-*.pth`) | 2.1 GB | `model.model_params.dinotxt_weights` |
| a clone of the `facebookresearch/dinov3` repo | small | `model.model_params.dinov3_repo_dir` |

The key cache (`data.key_cache_path`) is built automatically on the
first run (~5 min) and memoised; point it anywhere writable.

## 4. Configs

Every experiment is one YAML in `config/`. Start from the current best
recipe and edit paths:

- `sf3d_train_runpod_g17_splitax.yaml` — best full-3D recipe
- `sf3d_train_runpod_g17_2d_dct.yaml` — best 2D-only recipe

To make your own: copy one, then change (a) the two experiment paths
(`ModelCheckpoint.dirpath`, `CSVLogger.save_dir`), (b) the data paths
from §3, (c) whatever you're experimenting with. Configs are heavily
commented — the comments are the documentation of why each knob is set.
Keep one config file == one experiment; never reuse an old experiment's
checkpoint/log dirs.

## 5. Train

```bash
python train_SF3D_better.py fit --config config/<your>.yaml
```

- Progress: stdout progress bar; metrics stream to
  `<save_dir>/csv/version_0/metrics.csv`.
- Checkpoints: top-3 by `val/loss_total` + `last.ckpt` in the configured
  `dirpath`.
- Any config field can be overridden on the CLI, e.g.
  `--data.batch_size_train 32 --trainer.max_epochs 10`.

Quick sanity check before a long run: `SMOKE_ONLY=g17
python tools/smoke_dinov3_stack.py` runs a tiny forward/backward of the
current stack.

## 6. Evaluate

```bash
python train_SF3D_better.py test --config config/<your>.yaml \
  --ckpt_path <checkpoints>/best-epochXX-....ckpt --trainer.logger=false
```

Prints the full metric table on the standard test split (~5,088
samples): mIoU/PDet (mask + grounding), point errors, axis errors
(sign-aware variants included), origin/radius, trajectory direction,
2D-reprojection (`traj_proj2d_*`), and smoothness (`traj_rough_*`).
Never pass `--trainer.enable_checkpointing` (it crashes the CLI).
Caveat: metrics are only comparable between runs with the same data
filters (`min_revolute_radius` etc.) — don't mix eras; see
`experiments/INDEX.md` for which runs are comparable.

## 7. Record what you ran (our conventions)

Per experiment, a directory `experiments/YYYYMMDD_<dataset>_<tag>/`
containing the exact `config.yaml`, the `metrics.csv`, and a short
`notes.md` (goal / setup / result table / decision) — plus one summary
row in `experiments/INDEX.md`. Checkpoints stay out of git. The habit
that matters: write the notes the moment the test pass finishes, and
record negative results too.

## 8. Visualize predictions

```bash
python tools/sf3d_vis_predictions.py \
  --model <name> config/<your>.yaml <ckpt> \
  --data-root <sf3d_processed_v3> --frame-cache-path <frames_512.lmdb> \
  --input-size 512 --key-cache <key cache pkl> \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/YYYYMMDD_<subject>_panels
```

Repeat `--model` for side-by-side columns. Seed 42421 is our standard
sample set (comparable across all our batches). Batches live in dated
dirs under `viz/` with a README — see `viz/INDEX.md` for examples.
