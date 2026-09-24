# External A/B — DIPO (NeurIPS 2025) — status: ON HOLD (release incomplete)

Findings 2026-08-29 (repo github.com/RQ-Wu/DIPO @ 78efeff):
- README data/checkpoint links are empty `[here]()`; checkpoint found in the HF Space
  `HorizonRobotics/DIPO` (`ckpts/dipo.ckpt`, 24.6 MB); PM-X train data
  `HorizonRobotics/DIPO-Dataset/train_data.zip` (6.9 GB); retrieval assets
  `wuruiqi0722/DIPO_data/data/data.tar` (0.95 GB).
- Training code is a debug snapshot: `data_module.py` trains on `train_ids[:10]`,
  val on `val_ids[:50]`; `json_root`/`split_file` are hardcoded to the authors'
  home paths (SINGAPO-format jsons); `on_test_end` (retrieval + metrics) is
  commented out; features expect their own dual-state Blender renders
  (`features/*.npy` with first/last-frame DINOv2 features).
- Decision: do not spend budget until the zip contents are checked; if it lacks
  the PM dual-state features, DIPO is not reproducible as released. Zip listing
  in progress on the SINGAPO pod's container disk (`/tmp/dipo/listing.txt`).

## Verdict (2026-08-29 22:20 CEST): DROPPED
`train_data.zip` (6.9 GB) listing: 115,100 entries — `object.json` (5,002 objects: PM
categories + augmented_data(_complex) + gpt_blender + aug_joint_* variants) and
`imgs/*.mp4` animations (100,040 files). **0 feature `.npy`, no `data_split.json`,
no `test/`.** The released dataloader requires precomputed per-view DINOv2
features (first/last frame) that are not shipped, and the training entry is a
debug snapshot. Not reproducible as released; not worth budget. Retrieval assets
(`wuruiqi0722/DIPO_data`) and the Space checkpoint would only serve inference.
