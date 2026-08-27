# 20260828_sf3d_closedform — the continuous loss, exactly: near-record MA, no trajectory anything

**Recipe:** arm B's config (NO trajectory head, no L_pp/dir) + the
CLOSED-FORM continuous trajectory loss (theory note
2026-08-28_continuous_trajectory_loss.html): position L2 + derivative H1
Gram quadratics on the radial/tangential residuals, per-row normalized,
GT-type routed, weights 0.5/0.5 (`closed_form_screw_loss`, commit
914ed32). No sampled points, no decoded curve tensor, zero added
parameters. 30 epochs, best = epoch 22 (val 1.1792), full-speed host.

## Test (5,088) vs the family

| metric | arm B (no traj) | sampled decode+fdiff | **closed form** | joint (fdnolpp) | g19_fdiff (head+L_pp, MA champion) |
|---|---|---|---|---|---|
| MA / MA_signed | 20.40 / — | 26.73 / 26.43 | **29.19 / 28.89** | 27.54 / 26.93 | 29.91 / 29.56 |
| axis matched / all | 23.3° / 29.2° | **17.9°** / 26.2° | 22.3° / 26.8° | 15.1° / 27.4° | 17.3° / 25.7° |
| flips all / rot | — / 21.8 | 10.1 / **13.0** | 10.7 / 13.7 | 10.4 / 13.5 | 12.2 / 15.2 |
| origin (m) | 0.290 | 0.259 | **0.250 (record)** | 0.266 | 0.296 |
| radius (m) | 0.136 | 0.135 | 0.128 | **0.113** | 0.156 |
| 3D point | 0.243 | 0.235 | 0.238 | **0.231** | 0.227 |
| mIoU / PDet | **0.268 / 22.6** | 0.257 / 19.4 | 0.258 / 20.1 | 0.246 / 19.2 | 0.250 / 18.0 |

## Reading

1. **The distillation essentially completes.** Two interpretable
   quadratic loss terms on the articulation heads — no trajectory head,
   no curve tensor — reach MA 29.19, within 0.7 of the all-time record
   that required the full trajectory apparatus (head + fdiff + L_pp),
   and set a NEW ORIGIN RECORD (0.250, prev best 0.257 = g17). The
   entire "redundant label" benefit is now two named residuals with
   fixed Gram weights.
2. **Exact beats sampled by +2.5 MA** — NOT a quadrature effect (that's
   ~3e-4). The arms differ in the derivative-side norm: the sampled arm
   used the fdiff trio (non-squared velocity/angle/length norms), the
   closed form uses the H1 quadratic. The principled quadratic weighting
   outperforms the empirically-tuned trio in this no-head setting.
3. **The threshold-vs-precision trade again, now favorably:** the closed
   form wins pass-rate MA and origin but concedes matched-axis
   sharpness (22.3° vs the sampled arm's 17.9°) — the H1 term's equal
   radial/tangential weighting sharpens where it counts for the pass
   threshold rather than for the matched mean.
4. Masks: 0.258/20.1 — best of the decode arms and above the joint arm;
   arm B's 0.268 remains the no-trajectory-pressure reference.

**Follow-ups (recorded, not commissioned):** weight/Θ sweep of the two
Gram terms (the theory note's principled knob); closed form + DCT
trajectory head (masks) as the distilled gen-22; matched-axis gap
diagnosis (is it the missing angle-term's non-quadratic norm?).

test pass: logs/test.log (ckpt best-epoch22-valloss1.1792). Pod deleted.
