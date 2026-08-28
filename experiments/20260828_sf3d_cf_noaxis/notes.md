# 20260828_sf3d_cf_noaxis — closed form minus the direct axis loss: works, but the anchor earns its 1.5 MA

**Question:** the identifiability proof (2026-08-28 discussion) says the
closed-form Gram quadratics + 3D origin/point losses have their unique
global zero at the correct articulation WITHOUT the 1-cos axis term
(trans rows: the cf trans term IS 2(1-cos); rot rows: the parallel-and-
perpendicular argument, needs r* != 0 — guaranteed by
min_revolute_radius). Does the landscape cooperate?

**Recipe:** exactly 20260828_sf3d_closedform with `vae_weight: 0.5 -> 0.0`.
No trajectory head, no L_pp, cf position 0.5 + derivative 0.5, 30 epochs,
seed 42. Best = epoch 29 (val 1.0720 — NOT comparable to closedform's
1.1792: the total no longer contains 0.5*L_vae). Best being the FINAL
epoch means the run was still improving at cutoff.

## Test (5,088) vs closedform (identical but axis loss on)

| metric | closedform (axis 0.5) | **cf_noaxis (axis 0)** | arm B (nothing) |
|---|---|---|---|
| MA / MA_signed | **29.19 / 28.89** | 27.71 / 27.26 | 20.40 / — |
| axis matched / all | 22.3° / 26.8° | **17.6° / 26.4°** | 23.3° / 29.2° |
| flips all / rot | 10.7 / **13.7** | **10.5** / 15.9 | — / 21.8 |
| origin (m) | **0.250** | 0.253 | 0.290 |
| radius (m) | 0.128 | **0.123** | 0.136 |
| 3D point | 0.238 | **0.231** | 0.243 |
| mIoU / PDet | 0.258 / 20.1 | **0.263 / 20.8** | 0.268 / 22.6 |

## Reading

1. **Theory confirmed: training works.** MA 27.7 with ZERO direct axis
   supervision — the closed-form quadratics alone carry the axis 7.3 MA
   above arm B. The unique-zero argument holds in practice, not just on
   paper.
2. **But the 1-cos anchor earns its keep: −1.5 MA, all revolute sign.**
   Rot flips 13.7 -> 15.9 (+2.2) — exactly the predicted failure mode:
   the antipodal saddle with no scale-free second term to kick off it,
   and lever-scaled sign signal on short-lever parts. Overall flips are
   flat (10.5) because trans rows lost nothing (cf IS their axis loss).
3. **Surprise: matched axis SHARPENS 22.3° -> 17.6°** — recovering the
   sampled-fdiff arm's precision (17.9°) with the exact loss. The direct
   1-cos was apparently blurring matched precision (pulling all rows
   toward moderate alignment) while buying pass-rate sign-robustness —
   the threshold-vs-precision trade, now visible WITHIN the closed-form
   family. Radius, 3D point, and masks also improve slightly.
4. **Verdict:** keep the axis loss for MA-optimizing runs (its job is
   flips, not precision); drop it when matched-axis precision matters.
   A small vae_weight (0.1-0.25) or a rot-flip-only hinge is the obvious
   interpolation if one run must do both. The distilled gen-22 candidate
   (cf + DCT head) should keep the axis loss on.

test pass: logs/test.log on the volume (ckpt best-epoch29-valloss1.0720).
Launch hiccup for the record: the first launch died on a missing config —
the mutagen mirror routes through the (stopped) dev pod, so new files
must be scp'd to the volume when training with the dev pod off. Pod
deleted after the test pass.
