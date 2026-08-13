# 20260813_sf3d_split_g6 — split articulation heads, classical losses

**Goal:** first run of the 2026-08-13 split-heads spec — retire the se(3)
twist as the predicted representation: type CE + axis direction (1-cos,
sign-sensitive) + absolute-3D origin (canonical q* target) + direct
absolute-3D interaction point (replaces the 2D heatmap entirely, user
Option B) + K=1 trajectory; all-predicted consistency L_pp (soft type
gate, radial+drift circle residual); NO teacher forcing anywhere
(predicted-mask pooling, detached); CLIP UNFROZEN; radius-filtered data.

**Setup:** config.yaml (= config/sf3d_train_runpod_split.yaml @ launch).
RTX PRO 4500 ($0.72/hr; all 3 PRO 6000 SKUs out of stock), LMDBs on
container NVMe (29G shm cap), 24 workers, batch 128, 16 epochs ~4.3h
(~$4 incl. eval). Teardown hang after final val killed manually
(worker-shutdown hang; everything already on disk).

**Best: epoch 10, val 1.0214** (traj 0.0141, origin 0.8651, point3d
0.2138, motion 0.2974, type 0.2493, mask 0.4003, L_pp 0.0029). Val
plateaued from ~ep4; post-LR-drop epochs overfit (train 0.246, val
~1.065 at ep13-15). last.ckpt md5-identical to ep10 (known quirk) —
single evaluated checkpoint. 36k filtered test samples.

**Result — the split cleanly separates what works from what doesn't:**

- **Articulation semantics: best ever, big margins.** Axis direction
  err 22.1 deg all / 15.1 matched (g5 twist 29.4, DINOv3 36.95, g4
  48.4); type pass_rate_m 97.5% (g5 89.4 via |omega|) — the CE head
  makes type commitment a non-problem, as designed; pass_rate_ma 42.2
  (g5 22.0). traj_dir 92.0% / cos 0.742 ~= g5's 92.6/0.750 WITH K=1:
  **no zero-motion collapse** (val traj 0.0141 vs 0.018 baseline;
  gen-3's K=1 sat AT the baseline). The K=4 WTA machinery was not
  needed for the trajectory once the type hedge left the
  representation.
- **Metric localization: weak, exactly as risked.** point_err_3d
  0.582 m, origin_err 0.688 m (line 0.630 m), radius_err 0.368 m;
  mIoU 0.098 (g4 0.118, g3 0.103, g5 0.090), PDet 3.5. Diagnosis
  (mid-train viz + panels): both 3D points are pooled-vector MLP
  regressions — no pixel grid, no structural mask tie, no dense
  localization supervision on the trunk (heatmap BCE + coord L1 were
  deleted with the 2D machinery); grounding failures visible
  (washer-for-oven). Unfreezing CLIP did NOT rescue masks (0.098 vs
  frozen g5's 0.090) — the lost point-heatmap co-training is the
  stronger suspect.
- Legacy 2D metrics (mean_point_error, mean_origin_error_m) log 0.0
  by design — no coords_hat on this arm.

**Decision:** split parameterization + classical losses VALIDATED for
type/direction/trajectory — this is the semantics baseline going
forward. The localization pipeline (Option B direct-3D points) is the
regression; gen-7 (proposed, docs/slides/2026-08-14_gen7_structure.html)
replaces both points with heatmap+depth lifts: 3-channel projector
(mask + point + origin heatmaps, Gaussian BCE), scalar depth heads,
composed 3D unprojection losses, origin channel supervised at projected
q* (in-frame 99.6-100% on v2 data — tools/diag_origin_inframe.py),
frozen CLIP, origin_uv + point_uv in the condition vector. Open:
pooling policy, local feature sample for depth heads.

vis: viz/20260814_sf3d_g6_mid_e5_panels, viz/20260814_sf3d_g6_mid_e10_panels
(both MID-TRAIN; e10 = the best checkpoint's weights)

Eval log: logs/test_best.log. ckpt md5s: ckpt_md5.txt. Spec:
docs/superpowers/specs/2026-08-13-split-heads-gen6-design.md (amended
x2 during implementation: L_pp drift term, q* canonicalization).
