# 20260911_joint4_decoder_rgb_scalefree — joint 2D+3D with the analytic decoder (no trajectory head)

**Design (user, 2026-09-11).** The trajectory is no longer a learned head. On hand-video batches (HOI4D
2,821 labelled windows, EPIC 359 VLM-labelled, ARCTIC 2,559) the analytic decoder renders it from the
GT-routed type, the split axis heads, the lifted origin/point and a learned ARC LENGTH (rot: angle =
L / radius, radius detached, clamped at π; trans: L · direction), in the scale-free frame, and the
unit-anchor projection loss trains it; type CE 0.5 on every source. On SF3D batches the closed-form
position quadratic at sweep 2π is the only trajectory-related loss (no direct axis loss, no rendered
curve). No residual, no consistency term, no derivative terms. Everything else = the joint4 recipe
(source-homogeneous batches, hand ×10, lr 2e-5, 20 ep, monitor `val/sf3d/loss_total`, from scratch).
At test the decoder uses the predicted type; SF3D is tested with the learned length and with the
writer's constants (π/2, 0.7 m) for comparability with the head-based arms.

**Comparison rows.** joint4 (MA 31.41 / signed 30.15, mIoU 0.2738, PDet 22.7, traj_dir 96.4, HOI4D
0.676 / 85.2), joint v2 (32.94 / 32.08, 0.250 / 19.1), DCT chain (32.80 / 32.43), cf_l2_noaxis at π/2
(MA 23.80, rot flips 20.1; the L2-only loss this recipe uses, on the arm-B config) and the 2π arm in
flight (`20260911_sf3d_cf_l2_noaxis_2pi`).

**Result (2026-09-11 06:20 local, pod jdec, 4 h 20 min train + 6 min tests, pod deleted — verified).**
Best `val/sf3d/loss_total` **1.0089 at epoch 16** of 20 (still improving; val L_cf_position 0.80 → 0.34,
val L_traj_proj on the decoded curve plateaus at 0.50–0.65 from epoch 4 — the arc/line is only as good
as the axis/origin it is rendered from; EPIC val rises from epoch 12 as in every ×10 run).

SF3D test (predicted type; "writer" length pass identical on every metric — none of them reads the length):

| metric | joint4 (head) | joint v2 (head) | **decoder** |
|---|---|---|---|
| MA / signed | 31.41 / 30.15 | **32.94 / 32.08** | 31.29 / 30.86 |
| type pass | 91.2 | 91.6 | **91.8** |
| matched axis / all / signed-all | 20.1 / 26.1 / 34.4 | **16.4** / 25.2 / 32.0 | 18.4 / 25.3 / 32.1 |
| flips all / rot | 11.9 / **13.6** | 10.8 / 15.2 | **10.7** / 13.9 |
| origin / line | 0.327 / 0.297 | 0.355 / 0.320 | **0.273 / 0.247** |
| radius / point 3D | 0.163 / 0.285 | 0.161 / 0.279 | **0.124** / 0.289 |
| mIoU / PDet / point 2D | **0.274 / 22.7 / 0.104** | 0.250 / 19.1 / 0.113 | 0.266 / 22.5 / 0.108 |
| traj_dir acc / cos | **96.4 / 0.830** | 96.0 / 0.828 | 91.1 / 0.745 |
| roughness | 0.0090 | 0.0096 | **0.0000** |

Per-source (single-source configs + decoder overrides; type pass now = accuracy against the new labels):

| source | joint4 mIoU / PDet / point / shape / dir / rough | **decoder** |
|---|---|---|
| HOI4D | **0.676 / 85.2** / 0.0144 / 0.0343 / 52.5 / 0.0051 | 0.615 / 74.4 / 0.0164 / **0.0340** / **62.3** / 0.0006, type 100 |
| EPIC | **0.305** / 15.1 / 0.048 / **0.091** / 79.2 / 0.0026 | 0.267 / 15.1 / 0.054 / 0.100 / **86.8** / 0.0003, type 100 |
| ARCTIC | **0.618** / 68.7 / 0.062 / **0.083** / 73.9 / 0.0058 | 0.607 / **69.3** / 0.068 / 0.086 / **81.5** / 0.0005, type 100 |

Panels `viz/20260912_joint_decoder_panels` (SF3D 12 val samples decoder | joint4; 8 per hand source).

**Reading.** (1) **It works.** Every rendered trajectory is a clean arc on the decoded orbit or a line on
the direction ray (roughness 0 by construction); closet door: hinge line on the door edge, 6° axis error,
arc along the orbit where joint4's head draws a jittery loop; drum pedal: the decoder localises the pedal
where joint4 lands on the drum. (2) **Origin and radius are the big win**: 0.273 / 0.124 vs 0.327 / 0.163
— the 2π position quadratic's 3:1 radial weighting (cf_l2_noaxis_2pi showed the same on the arm-B
config) plus the projection loss now reaching the origin head through the arc. (3) MA 31.29 = joint4
within noise, below joint v2's 32.94; matched axis 18.4 between them; type pass and all-flips best of the
three. Masks 0.266 / 22.5 ≈ joint4 (the v2 mask regression did not recur). (4) **Trajectory direction
accuracy drops 96 → 91 on SF3D**: the decoded curve inherits every axis-sign error (rot flips 13.9 %,
all-axis 25°), whereas the free head learned direction on its own (96 %) — i.e. the free head knew the
sign better than the axis head does. The decoder makes the trajectory exactly as good as the articulation,
no better; the SF3D side has no direct axis loss in this arm, so the l2anchor variant is the direct test.
(5) HOI4D masks 0.676 → 0.615 / PDet 85 → 74: the projection loss now drives the trunk through the
axis/origin/point heads instead of a private trajectory head — the same trunk-pull seen with derivative
losses, milder than v2 (0.592). On the hand sources the decoded trajectory's DIRECTION accuracy improves
on all three (+10 / +8 / +8) and shape holds. (6) Type accuracy 100 % on all three hand sources: the new
labels are category-determined and the type head learns them; HOI4D type is no longer a held-out number.

**Verdict.** The decoder is a sound replacement for the trajectory head: same MA, far better
origin/radius, structurally smooth, masks kept, direction inherited from the axis. Variants launched
(autonomous night): `l2anchor` (L2 2π + direct axis loss — the sign fix), `h1anchor` (the August MA
recipe), `cfframe` (the shape-designed loss).


**Arc-length probe (2026-09-11 07:15, 400 SF3D val samples, scratch script length_probe.py).** The
length head (trained only by the hand-video projection loss) predicts a metric arc length L·z_p of
median 0.213 m for slides and 0.212 m for hinges — about 0.30× the writer's constants (0.70 m / ~0.62 m
arc) and uncorrelated with them (corr 0.03 / −0.31; p10–p90 0.10–0.37 m). Expected by design: SF3D never
supervises the length, so the head reports "how far a hand moves in a clip", and the rendered SF3D sweeps
are ~30 % of the orbit (the short arcs in the panels). No test metric reads the length (direction, shape,
MA are extent-free), so nothing else is affected; for full-orbit renders use `trajectory_decoder_length
writer`. A GT-extent metric would need real extents, which SF3D does not have. Type accuracy on the same
400: 93.3 %.
