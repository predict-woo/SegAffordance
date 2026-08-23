# 20260823_sf3d_g21_dct_dir — g21: DCT head + screw-direction term (3D)

**Recipe:** exactly `20260821_sf3d_g19_dct` (g17 split-axis + DCT
trajectory head) + `pred_pred_art_dir_weight: 0.1` — the midpoint
screw-direction sign-consistency term in L_pp (spec
2026-08-23-label-efficiency-v2-dir-term-design.md). Two jobs: THE
dir-term experiment on 3D (one change vs a measured baseline), and arm
C' (100% upper bound) of label-efficiency v2. 30 epochs, best = epoch 21
(val 0.9991; ≈ g19_dct's 0.9652 + the term's own ~0.03 contribution).
CSV logger is version_1 (a dead PRO 4500 launch claimed version_0).

## Test (5,088) vs g19_dct (identical minus the term) and g17

| metric | g17 | g19_dct (term OFF) | **g21 (term ON)** |
|---|---|---|---|
| axis_flip_rate all / rot | 13.3 (rot) | 9.98 / 13.30 | 11.56 / **15.45** |
| err_adir all / matched / signed-all | 25.8 / 16.9 / — | 25.29 / 18.21 / 32.99 | 27.04 / 20.76 / 35.19 |
| MA / MA_signed | 25.9 / — | 25.98 / 25.83 | **26.61 / 26.40** |
| mIoU / PDet | 0.265 / 20.6 | 0.2685 / 21.72 | **0.2712 / 23.21 (records)** |
| origin / radius (m) | 0.257 / 0.122 | 0.276 / 0.141 | 0.278 / 0.140 |
| 3D point / 2D point | 0.232 / 0.095 | 0.259 / 0.105 | **0.244 / 0.102** |
| traj_dir acc / cos | 94.9 / 0.811 | 94.52 / 0.800 | **88.78 / 0.724** |
| roughness (m) | 0.0915 | 0.00903 | **0.00849** |

## Reading — the flip-rate hypothesis FAILED on 3D

The term was built to cut the ~13% rot sign-flip tail. It did not: rot
flip rate went 13.30 → 15.45 and every axis-error column worsened by
~2°. The likely mechanism is visible in traj_dir (94.5 → 88.8, the
biggest regression): the term's gradient flows into the TRAJECTORY as
well as the axes, so where the predicted axis is wrong (exactly the flip
tail), the term drags the trajectory toward the wrong axis's velocity
field instead of only pushing the axis toward the (GT-anchored)
trajectory. In 3D, where GT already teaches axis sign directly, the
term's coupling is net harmful to articulation.

Not all bad: mIoU 0.2712 / PDet 23.21 are new overall records, 3D/2D
point improve, MA nudges up (+0.6, and the MA↔MA_signed gap is ~0.2 —
matched-axis flips at threshold are essentially gone), roughness best
ever. And in the 2D pretrain (B'1, where no GT sign exists) the term's
val value fell 0.64→0.07 — its real home may be the 2D path only.

**Candidate follow-up (parked):** detach the trajectory in the dir term
(axis-only gradients) for 3D use — one-line change, would keep the sign
teaching without the trajectory drag. Cross-read pending: the other
session's 20260824_sf3d_supabl2_nolpp (arm D) gives L_pp-off-entirely on
this recipe.

test pass: logs/test.log (ckpt best-epoch21-valloss0.9991).
Role in label-efficiency v2: arm C' upper bound; also reused as "arm A
(joint)" by the supervision-ablation-v2 session.
