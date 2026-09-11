# 20260912_multi3_decoder_rgb_scalefree — the multi-source 2D arm with the analytic decoder (chain stage 1)

**Goal.** multi3 recipe, decoder instead of the DCT head, projection loss on the decoded curve + type CE 0.5, 12 ep. Stage 1 of the decoder chain; stage 2 = `20260912_sf3d_decoder_l2anchor_ft_multi3dec`.

**Comparison rows.** multi3 DCT arm (HOI4D 0.753 / 89.8 / shape 0.0337, EPIC 0.520 / 58.5, ARCTIC 0.694 / 79.6); joint decoder per-source (HOI4D 0.615 / 74.4).

**Result (2026-09-11 10:40 local, pod mdec, 62 min).** Best val 0.5584 at epoch 9 of 12 (val projection
term 0.535 — vs the DCT arm's 0.516 at its best; type CE 0.20).

| held-out split | n | DCT arm (ep 11) mIoU / PDet / point / shape / dir | **decoder arm (ep 9)** |
|---|---|---|---|
| HOI4D | 459 | **0.753 / 89.8 / 0.0137 / 0.0337** / — | 0.646 / 80.1 / 0.0192 / 0.0355 / 53.0, type 100 |
| EPIC | 53 | **0.520 / 58.5** / 0.053 / 0.090 / — | 0.433 / 37.7 / 0.062 / 0.107 / 77.4 |
| ARCTIC | 329 | **0.694 / 79.6** / 0.062 / 0.085 / — | 0.631 / 69.0 / 0.070 / 0.092 / 71.7 |
| union | 841 | **0.715 / 83.8** / 0.035 / 0.057 / — | 0.626 / 72.9 / 0.042 / 0.063 / 62.1 |

**Reading.** With ONLY 2D data the decoder arm is clearly worse than the DCT-head arm on masks and
detection (union −0.09 mIoU / −11 PDet) and slightly worse on the 2D track fit (shape 0.063 vs 0.057).
Same direction as the joint run's HOI4D regression, larger in magnitude: with no 3D data to anchor the
axis/origin/point heads, the projection loss has to invent the articulation from the hand track and
drives the trunk through those heads. The trajectory is structurally smooth (roughness 0.0002).
Stage 2 (`20260912_sf3d_decoder_l2anchor_ft_multi3dec`) tells whether this init still transfers.

