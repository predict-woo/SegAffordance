# 20260804_sf3d_twist_dinov3_g3

**Goal:** gen-3 twist arm on DINOv3 ViT-L + dino.txt, backbone UNFROZEN
(the frozen gen-2 run was weak everywhere — this tests whether the
features or the freezing was the problem). Same gen-3 stack as
20260804_sf3d_twist_g3 (1-cos, no axis/type heads, pitch-free twists),
same lr 1e-5 as the CLIP arm which also fine-tunes its backbone —
backbone is the only variable vs the g3 CLIP run.

**Setup:** config.yaml (= config/sf3d_train_runpod_twist_dinov3.yaml @ launch).

**Result:** 16/16 epochs @ ~190 samples/s (~9h; two early crashes — the
volume-quota wall and a ModelCheckpoint resume-state KeyError — both
resolved, ~1h lost). Val improved essentially to the end: best 0.8789 @
ep13 (last.ckpt deduplicated to it), BETTER than clip_g3's 0.9042 on
identical loss composition. Hint-free eval vs clip_g3 (backbone the
only variable):
- WINS every articulation metric, best of ALL runs to date: twist axis
  36.95 deg (clip_g3 42.8, gen-2 39.6), twist_dir_acc 80.2% (71.5),
  traj_dir_acc 85.7% / cos 0.597 (79.7 / 0.508), type-from-|omega|
  71.0% (60.2; even beats gen-2's emergent 67), pass_ma 21.9% (14.0),
  line_dist 2.14 m (3.08), point err 0.112 (0.148).
- LOSES masks badly: mIoU 0.013 (clip_g3 0.103), PDet 0.6% (4.0) — the
  dynamic-kernel mask projector appears to depend on CLIP-style
  text-visual alignment that the dino.txt space doesn't give it.
Freezing, not the features, was gen-2's DINOv3 problem.
Eval log: logs/evaldg3.log.

**Decision:** DINOv3-unfrozen is the articulation-metrics champion;
CLIP the mask champion. Paths forward: (a) fix the mask projector for
dino.txt alignment, (b) dual-backbone or distillation, (c) accept the
split per use-case. Also pending: the type-commitment ablation shared
with clip_g3 (gen-3 stacked changes).

vis: viz/20260805_sf3d_g3_clip_vs_dinov3 (head-to-head at best ckpts)
