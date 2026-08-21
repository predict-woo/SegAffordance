# 20260821_sf3d_g19_dct — ARM 1: truncated-DCT trajectory head

**Recipe:** g17 + `trajectory_dct_coeffs: 6` — the head emits 6 DCT
coefficients per axis, a fixed IDCT buffer decodes to 20 points; jitter
above the 6th frequency is unrepresentable (spec 2026-08-21; survey
knowledge/trajectory-parameterization-survey.md). Losses unchanged.
30 epochs, best = epoch 20 (val 0.9652).

## Test (5,088) vs g17 (rough rerun in logs/test_rough.log of g17's dir)

| metric | g17 | g19-dct |
|---|---|---|
| traj ROUGHNESS (2nd-diff m; GT floor 0.0032) | 0.0915 | **0.0090 (10.2×)** |
| mIoU / PDet | 0.2654 / 20.56 | **0.2685 / 21.72 — both records** |
| MA / matched axis | 25.86 / 16.93° | 25.98 / 18.21° |
| traj_dir acc / cos | 94.91 / 0.811 | 94.52 / 0.800 |
| type | 95.34 | 95.15 |
| 2D pt / 3D pt | 0.0952 / 0.2315 | 0.1048 / 0.2592 |
| origin q* / radius | 0.2569 / 0.1215 | 0.2764 / 0.1410 |

## Reading

The DCT basis delivered exactly its promise: roughness fell 10× to 2.8×
the GT floor, and the freed capacity even helped the shared trunk (mask +
PDet records). Costs are small geometry dips (point +1/+2.8cm, origin
+2cm, matched axis +1.3°) — the 6-coeff bottleneck slightly stiffens the
curve's fine placement. traj_dir essentially flat.

Note (2026-08-22): fdiff-arm comparison and combined recommendation in
20260821_sf3d_g19_fdiff/notes.md.

vis: viz/20260822_sf3d_g19_smoothness_panels
