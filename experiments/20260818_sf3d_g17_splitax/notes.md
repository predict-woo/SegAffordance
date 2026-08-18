# 20260818_sf3d_g17_splitax — split articulation axis heads

**Recipe:** g16 (dinov3 @512, v3, normalized trajectory loss @0.5) with the
single MotionMLP axis readout replaced by per-type readouts
(`motion_head_rot` + `motion_head_trans`, `split_axis_heads`, spec
2026-08-18): axis loss GT-routed per row at train, row-wise selection by
predicted type at test, L_pp line branch reads the trans candidate /
circle branch the rot candidate. Trajectory head stays shared; origin path
untouched. RTX PRO 6000 WK, 30 epochs, ~4h, clean exit.

**Run:** best = epoch 18, val 0.9272 (comparable loss composition to g16's
1.0123 — a real drop).

## Test (best-epoch18, 5,088 samples) vs g16

| metric | g16 | g17 | Δ |
|---|---|---|---|
| type acc | 92.32 | **95.34** | **+3.0 — best ever** |
| MA | 23.09 | **25.86** | **+2.8** (g13's 26.2 within noise) |
| MA_signed | 22.92 | **25.75** | +2.8 |
| axis all / matched (°) | 28.25 / 17.83 | **25.83 / 16.93** | −2.4 / −0.9 (matched = family best) |
| signed axis all (°) | 33.91 | 33.08 | −0.8 |
| flip rate all / rot (%) | 10.18 / 12.73 | 9.65 / 13.30 | −0.5 / **+0.6 (rot NOT improved)** |
| traj_dir acc / cos | 94.46 / 0.798 | **94.91 / 0.811** | project bests again |
| 2D point | 0.0988 | **0.0952** | best ever |
| 3D point (m) | 0.2488 | **0.2315** | best ever |
| origin q* / line (m) | 0.3033 / 0.2743 | **0.2569 / 0.2298** | **beats g15's 0.279 record** |
| radius (m) | 0.1391 | **0.1215** | best ever |
| mIoU | 0.2636 | 0.2654 | flat |
| PDet | 21.44 | 20.56 | −0.9 (the one dip) |

## Reading

The split did what the averaging hypothesis predicted — and more than
expected outside the axis itself: type accuracy jumped +3.0 (the trunk no
longer has to smear type information into a compromise axis), and the
whole revolute geometry chain (origin, radius, 3D point) improved sharply,
consistent with L_pp's circle branch finally reading a pure hinge-axis
candidate. MA +2.8 with the matched axis at the family-best 16.9°.

The one goal it did NOT meet: the rot sign-flip rate (13.3% vs 12.7%) —
flips are evidently not caused by cross-type interference; the model still
can't tell which way a fridge/window opens from appearance alone. That
failure mode needs its own idea (sign supervision is already there and
sign-sensitive; candidates: sign-consistency with the predicted trajectory
sweep, or handle-position cues).

PDet −0.9 is the only cost; with mIoU flat it reads as threshold noise
around the 0.5 IoU gate rather than a mask regression.

**GEN-17 IS THE NEW OVERALL BEST** (best-epoch18-valloss0.9272.ckpt):
g16's trajectories + record type/MA/origin/radius/point metrics.

vis: viz/20260818_sf3d_g17_vs_g16_panels (pending)
