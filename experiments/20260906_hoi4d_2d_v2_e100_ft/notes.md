# 20260906_hoi4d_2d_v2_e100_ft — HOI4D v2 sweep arm `e100_ft`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 1e-05, milestones [80, 92], batch 64, initialized from the SF3D 2D-DCT best checkpoint (strict=False).

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3638** (epoch 74); test pass on the 110 held-out objects (= val split): mIoU **0.690**, PDet **85.2**, point err 0.0179, traj proj-2D shape 0.0330, traj_dir acc 47.7 (chance — as in v1, direction does not emerge from hand tracks). SF3D 2D-DCT init helps at lr 1e-5 (0.690 vs 0.626) but less than raising LR; user: fine-tune line NOT wanted; combination arm ft+lr3e5 was cancelled at ep 1. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
