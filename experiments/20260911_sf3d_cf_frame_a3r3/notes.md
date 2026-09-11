# 20260911_sf3d_cf_frame_a3r3 — cf_frame 3:1 with radius weight 0.30

**Question.** a3 fixed the sign (flips 13.0, MA 31.90) but cost origin (0.262) and masks (0.249). Does doubling the log-radius weight buy the origin back without losing the sign?

**Recipe.** cf_frame_a3 with `closed_form_frame_radius 0.30`.

**Comparison rows.** cf_frame_a3: 31.90 / 16.2 / 13.0 / origin 0.262 / mIoU 0.249. cf_frame: 31.03 / 17.0 / 17.8 / 0.245 / 0.270.

**Result (2026-09-11 13:15 local, pod cfframea3r3, 3 h 40 min, pod deleted — verified).** Best val 1.4023 at epoch 22.

| metric | cf_frame (2:1, r 0.15) | cf_frame_a3 (3:1, r 0.15) | **a3r3 (3:1, r 0.30)** |
|---|---|---|---|
| MA / signed | 31.03 / 30.27 | 31.90 / 31.33 | **32.47 / 31.56** |
| type pass | 95.1 | 95.0 | 92.1 |
| axis matched / all / signed-all | 17.0 / 24.8 / 33.5 | **16.2 / 25.1 / 32.3** | 21.0 / 27.6 / 34.5 |
| flips all / rot | 9.9 / 17.8 | **9.2 / 13.0** | 11.1 / 13.5 |
| origin / line | **0.245 / 0.215** | 0.262 / 0.234 | 0.270 / 0.242 |
| radius / point 3D | **0.123 / 0.229** | 0.132 / 0.229 | 0.131 / 0.238 |
| mIoU / PDet / point 2D | **0.270 / 21.9 / 0.097** | 0.249 / 19.0 / 0.103 | 0.251 / 19.1 / 0.104 |

**Reading.** The prediction failed: doubling the log-radius weight did NOT buy the origin back (0.262 →
0.270, radius unchanged), and it blurred the axis (matched 16.2 → 21.0°, all 25.1 → 27.6°, all-flips 9.2
→ 11.1) and cost 3 points of type pass. MA nevertheless reads 32.47 (+0.6): a more peaked core under 10°
with a fatter tail — at one seed this is inside the noise, and every other axis metric points down.
So the 2 : 1 arm's origin advantage came from the axis-to-phase balance, not from the radius term; the
radius term at 0.30 is deadweight or worse (it competes with the phase term for the lever's direction).

**Verdict.** Keep 0.15. For the origin/mask side of the trade, the knob is the axis : phase ratio (2 : 1
best origin/masks, 3 : 1 best sign/MA), not the radius weight. On the joint recipe the 2 : 1 form already
holds the MA record (36.36); its 3 : 1 variant is running.

