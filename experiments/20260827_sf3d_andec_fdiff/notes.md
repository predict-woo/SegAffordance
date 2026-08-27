# 20260827_sf3d_andec_fdiff — the mechanism verdict transfers: ~89% loss geometry

**Recipe:** the analytic_decode arm (arm-B config, NO trajectory head,
writer-mirror decode at 0.5) + the gen-19 fdiff losses applied to the
DECODED curve (fdiff-on-decode trainer block, commit 13ffad8). Zero
parameters vs arm B. The fdiff-family version of the "why does a
redundant label help" discriminator. 30 epochs, best = epoch 26
(val 1.0632), full-speed host.

## Test (5,088) between the fdiff-family anchors

| metric | arm B (no traj) | **decode+fdiff** | fdnolpp (joint, head) |
|---|---|---|---|
| MA / MA_signed | 20.40 / — | **26.73 / 26.43** | 27.54 / 26.93 |
| axis matched / all | 23.29° / 29.20° | **17.91° / 26.19°** | 15.09° / 27.44° |
| flips all / rot | 21.8 (rot) | **10.12 / 13.01** | 10.36 / 13.48 |
| origin / radius (m) | 0.290 / 0.136 | **0.259** / 0.135 | 0.266 / **0.113** |
| 3D point | 0.243 | 0.235 | 0.231 |
| mIoU / PDet | **0.268 / 22.60** | 0.257 / 19.38 | 0.246 / 19.24 |

## Reading

- **The transfer replicates and strengthens: (26.73−20.40)/(27.54−20.40)
  = ~89% of the fdiff-family articulation gap is recovered with ZERO new
  parameters** (DCT family: ~75%). The mechanism conclusion is
  family-robust: the trajectory's value to articulation is the loss
  geometry, not the head.
- The decode+fdiff arm even BEATS the joint arm on flip rates, origin,
  and masks — the head buys only matched-axis sharpness (15.1° vs 17.9°)
  and radius (0.113 vs 0.135) here.
- vs the no-fdiff decode (20260825: MA 26.5, matched 19.0, rot flips
  15.4): adding fdiff ON THE DECODE improves matched −1.1°, rot flips
  −2.4, origin −0.9cm — first-difference geometry composes with the
  decode, still parameter-free.
- Masks: same directional finding as DCT (decode < arm B: 0.257 vs
  0.268) but milder, and NOT below the joint arm this time — on the
  plain-head family the head is not a mask-saver either (fdnolpp 0.246).
  The "head protects the trunk" effect looks DCT-specific.

test pass: logs/test.log (ckpt best-epoch26-valloss1.0632). Pod deleted.
Companion arm C_f (trajectory-only + fdiff) still training at wrap time.
