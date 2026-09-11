# 20260911_sf3d_cf_l2_noaxis_2pi — position L2 alone at sweep 2π

**Question (user, 2026-09-11).** The L2 term's phase leak is ρ = 0.953 at π/2 (a flipped axis buyable down to
0.09) and 0.500 at 2π (floor 0.75); no sweep closes it. How much of the L2-only arm's 20.1 % hinge flips
does the 2π sweep remove?

**Recipe.** `config/sf3d_train_runpod_cf_l2_noaxis_2pi.yaml` = `20260829_sf3d_cf_l2_noaxis` (L2 1.0, no H1,
no direct axis loss, arm-B config, 30 ep, seed 42) with `closed_form_sweep 2π`.

**Comparison rows.** cf_l2_noaxis (π/2): MA 23.80 / 23.35, matched 18.6°, all 28.4°, flips all/rot 10.6 / 20.1,
origin 0.277, mIoU 0.261 / PDet 20.4. cf_noaxis_2pi (L2+H1 at 2π): MA 27.04, rot flips 13.0, matched 20.2°.
cf_frame (this day's other arm): p = 0, ρ = 0.

**Result (2026-09-11 03:35 local, pod cfl22pi, 3 h 55 min, pod deleted — verified).** Best val 1.0597 at
epoch 24 (last 29: 1.0639, flat — unlike the π/2 L2 arm, which peaked at 24 and rose).

| metric | cf_l2_noaxis (π/2) | **cf_l2_noaxis_2pi** | cf_noaxis_2pi (L2+H1, 2π) | cf_h1only (H1 + axis) |
|---|---|---|---|---|
| MA / signed | 23.80 / 23.35 | **26.49 / 25.94** | 27.04 / 26.73 | 30.64 / 30.11 |
| type pass | 94.2 | 92.2 | 94.1 | 95.5 |
| axis matched / all / signed-all | 18.6 / 28.4 / 36.4 | 19.9 / 27.3 / 36.1 | 20.2 / 26.0 / 33.7 | 16.6 / 24.5 / 31.7 |
| flips all / rot | 10.6 / 20.1 | 12.1 / **18.0** | 10.1 / 13.0 | 9.8 / 15.4 |
| origin / line | 0.277 / 0.251 | **0.251 / 0.228** | 0.259 / 0.234 | 0.254 / 0.230 |
| radius / point 3D | 0.149 / 0.240 | 0.135 / 0.229 | 0.125 / 0.230 | 0.130 / 0.234 |
| mIoU / PDet | 0.261 / 20.4 | 0.264 / 21.1 | 0.255 / 21.1 | 0.266 / 21.8 |

**Reading against the master formula (ρ 0.953 → 0.500, compensated floor 0.09 → 0.75, position basis
3:1 radial at 2π).** (1) MA +2.7 and the all-axis error −1.1°: the leak halving is worth about the same
as the H1 term's addition was (cf_noaxis_2pi 27.04 has both). (2) Hinge flips 20.1 → 18.0: down, but
far from the anchored arms (15.4) or the L2+H1 2π arm (13.0). Half the flip penalty is still buyable
and the p = 1 scaling (axis gradient ∝ predicted radius, collapse plateau) is untouched, so L2 alone
cannot fix sign at any sweep — as derived. (3) Origin 0.277 → 0.251: the best origin of the whole
no-anchor family and equal to the anchored record region (0.250–0.254). The 2π position basis weights
the radial residual 3:1, i.e. it is an origin/radius regulariser first. The registered prediction
"origin worse" was wrong; the earlier cf_noaxis_2pi note's "radial-heavy → origin improves" was the
right intuition and this arm confirms it without H1 in the way. (4) Matched axis blurs 18.6 → 19.9°
(predicted): the 2π metric rewards the average orbit, not the tangent. (5) Type pass −2 (noise-level;
single seed).

**Verdict.** For the joint decoder recipe, whose SF3D side is exactly this loss (L2 at 2π, no axis
loss), expect: good origin, mediocre hinge sign. The obvious next SF3D-side variant is L2 2π + the
direct axis loss (`joint4_decoder_l2anchor`) — the never-run {L2 + anchor} corner — which the formula
predicts closes the flip gap while keeping the 2π origin gain.

