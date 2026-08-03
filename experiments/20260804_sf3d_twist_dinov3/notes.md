# 20260804_sf3d_twist_dinov3

**Goal:** same arm as 20260804_sf3d_twist_clip with a FROZEN DINOv3
ViT-L/16 + dino.txt tower in place of frozen CLIP RN50 — backbone is the
single variable (regime mirrors 20260726_opdreal_dinov3l).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist_dinov3.yaml @ launch).

**Result:** trained 16/16 epochs @ ~497 samples/s. Val plateaued
immediately (best 1.3457 @ epoch 2; final 1.370) while train fell to 0.61.
Hint-free eval on 43,870 val samples, best-ckpt vs last-ckpt:
type acc (head) 95.4/95.8%, MA pass 20.1/20.1%, twist axis err
50.9/49.7 deg, twist type-from-|omega| 59.9/... , twist_dir_acc
57.5/60.4%, traj_dir_cos 0.20/0.23, traj_dir_acc 62.8/64.0%, mIoU
0.004/0.007. Weak on every twist metric vs the old CLIP arms (axis
~35-39 deg) — frozen DINOv3 + this head stack did not transfer well at
these settings; direction barely above chance (runs predate the 1-cos
consistency change). Eval logs: logs/eval_best.log, logs/eval_last.log.

**Decision:** CLIP remains the backbone for the twist work. Direction
supervision needs the 1-cos generation (this run trained with sin^2
consistency; only the twist L2 pushed direction).
