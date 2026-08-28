# 20260828_sf3d_cf_h1only — the closed-form fdiff: NEW ALL-TIME MA RECORD (30.64), no trajectory head

**Question:** the fdiff family's continuous limit is Sobolev H1 — the
velocity term closes to the derivative quadratic (angle/length have no
closed form). How much of the closed form's gain is the derivative term
alone?

**Answer: more than the pair.** H1-only at the fdiff velocity weight
(1.0) + the axis anchor beats BOTH the full closed form and the all-time
record that needed head+fdiff+L_pp.

**Recipe:** exactly 20260828_sf3d_closedform with
`closed_form_trajectory_weight 0.5 -> 0.0`,
`closed_form_velocity_weight 0.5 -> 1.0`. Axis loss ON (vae 0.5). No
trajectory head, no L_pp, 30 epochs, seed 42. Best = epoch 29
(val 1.1303; final epoch — still improving at cutoff).

## Test (5,088) — the closed-form family + the old record

| metric | closedform (pos+der) | cf_noaxis (pos+der, no axis) | **cf_h1only (der only)** | g19_fdiff (old MA record) |
|---|---|---|---|---|
| MA / MA_signed | 29.19 / 28.89 | 27.71 / 27.26 | **30.64 / 30.11 (records)** | 29.91 / 29.56 |
| axis matched / all | 22.3° / 26.8° | 17.6° / 26.4° | **16.6° / 24.5°** | 17.3° / 25.7° |
| flips all / rot | 10.7 / **13.7** | 10.5 / 15.9 | **9.8** / 15.4 | 12.2 / 15.2 |
| origin (m) | **0.250** | 0.253 | 0.254 | 0.296 |
| radius (m) | 0.128 | **0.123** | 0.130 | 0.156 |
| 3D point | 0.238 | **0.231** | 0.234 | 0.227 |
| mIoU / PDet | 0.258 / 20.1 | 0.263 / 20.8 | **0.266 / 21.8** | 0.250 / 18.0 |
| pass_rate_m | 93.3 | 93.8 | **95.5** | — |

## Reading

1. **The position quadratic was HURTING MA.** Dropping it (+ doubling
   the derivative) is worth +1.45 MA over the closed-form pair and +0.7
   over g19_fdiff — with zero trajectory machinery. The distillation now
   BEATS its teacher: one interpretable H1 quadratic + the classical
   per-branch losses is the best articulation recipe we have.
2. **No threshold-vs-precision trade this time — both improve.** Matched
   16.6° (best outside fdiff_dir's 14.6°), all-axis 24.5° (best ever),
   flips-all 9.8 (best ever), pass_rate_m 95.5. The H1 term supervises
   shape-of-motion (where the discriminative signal lives) while the
   axis 1-cos anchors sign; the position term apparently mostly re-blurs
   what those two already handle, at the cost of MA.
3. **Origin barely misses the record** (0.254 vs closedform's 0.250) —
   the position quadratic IS worth ~4mm of origin: it is the only
   cf term anchoring absolute lever placement. Small position weight
   (0.1-0.2) is the obvious sweep point if origin matters.
4. **Masks: best of every trajectory-supervised arm** (0.266/21.8,
   approaching arm B's no-pressure 0.268/22.6) — the H1-only gradient is
   the gentlest trunk-side trajectory pressure measured so far.
5. Triangulating with cf_noaxis: axis anchor + H1 compose (sign from
   1-cos, sharpness from H1); cf_noaxis showed what H1+position lose
   without the anchor; this arm shows position was the deadweight.

**Follow-ups (recorded, not commissioned):** small-position sweep
(0.1/0.2) for the origin record + MA record in one run; H1 weight sweep;
distilled gen-22 should now be H1-only + DCT head; seed replicate to
confirm the record isn't seed luck.

test pass: logs/test.log (ckpt best-epoch29-valloss1.1303). Pod deleted.

vis: viz/20260829_sf3d_cf_h1only_val_panels (GT | g19_fdiff | cf_h1only, 16 val, seed 42421)
