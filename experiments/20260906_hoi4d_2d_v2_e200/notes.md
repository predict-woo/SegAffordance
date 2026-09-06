# 20260906_hoi4d_2d_v2_e200 — HOI4D v2 sweep arm `e200`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 200, lr 1e-05, milestones [160, 185], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3678** (epoch 100); test pass on the 110 held-out objects (= val split): mIoU **0.656**, PDet **82.2**, point err 0.0167, traj proj-2D shape 0.0331, traj_dir acc 47.7 (chance — as in v1, direction does not emerge from hand tracks). confirms the horizon: best at ep 100/200, behind the higher-LR arms. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
