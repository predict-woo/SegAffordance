# 20260804_sf3d_twist_g3

**Goal:** gen-3 CLIP twist arm — the twist as the SOLE motion
representation, plus direction-everywhere. vs gen-2
(20260804_sf3d_twist_clip): (1) 1-cos consistency (L_screw_gt/self are
now direction-sensitive — tests the hypothesis that sin^2 licensed the
fold-back zigzag and the under-extension), (2) axis head removed,
(3) type head removed (type emergent from |omega|; NOTE eval M-Pass/type
metrics now MEASURE the emergent type — expect lower than gen-2's 95%
CE-head number by construction, gen-2's emergent figure was 67%),
(4) structurally pitch-free twists (TwistMLP output map projects v
against omega — no helices representable).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).

**Result:** 16/16 epochs @ ~685 samples/s (first run at full unthrottled
speed; one quota crash at ep5, resumed — volume expanded 500->600GB).
Best val 0.9042 @ ep8, mild drift after (0.961 @ 15). NOTE last.ckpt on
disk deduplicated to the ep8 file (Lightning save_last link quirk across
the resume), so ep8 is the evaluated checkpoint. Hint-free eval, 43,870
samples, vs gen-2 (20260804_sf3d_twist_clip):
- DIRECTION (the 1-cos target): twist_dir_acc 71.5% (was 64.8%),
  traj_dir_acc 79.7% (was 75.2%), traj_dir_cos 0.508 (was 0.43) — the
  direction-sensitive consistency clearly worked.
- REGRESSIONS: type-from-|omega| 60.2% (was 67.1% emergent),
  twist_pass_rate_ma 14.0% (was 23.0%), twist axis err 42.8 deg (was
  39.6). The pass-rate drop tracks the type drop.
- mIoU 0.103 (best of any arm), PDet 4.0%, point err 0.148.
Leading suspects for the type/axis regression: (a) the removed type-CE
gradient was shaping shared condition features that helped |omega|
commit; (b) the pitch-free projection changes the omega/v coupling under
sign-sensitive MSE. Disentangling needs a one-variable arm (e.g. gen-3
minus pitch-free, or type head back at low weight).
Eval log: logs/evalg3_best.log.

**Decision:** keep 1-cos (direction goal achieved). The
type-head-removal + pitch-free combination needs an ablation before
adopting both permanently — direction went up while type/axis
commitment went down, and the three stacked changes cannot be
attributed individually from this run.
