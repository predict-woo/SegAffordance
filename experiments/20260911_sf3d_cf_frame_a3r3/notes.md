# 20260911_sf3d_cf_frame_a3r3 — cf_frame 3:1 with radius weight 0.30

**Question.** a3 fixed the sign (flips 13.0, MA 31.90) but cost origin (0.262) and masks (0.249). Does doubling the log-radius weight buy the origin back without losing the sign?

**Recipe.** cf_frame_a3 with `closed_form_frame_radius 0.30`.

**Comparison rows.** cf_frame_a3: 31.90 / 16.2 / 13.0 / origin 0.262 / mIoU 0.249. cf_frame: 31.03 / 17.0 / 17.8 / 0.245 / 0.270.

**Status.** launcher started 2026-09-11 ~09:00 (autonomous night).
