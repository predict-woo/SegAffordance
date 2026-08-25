# 20260825_sf3d_fdiff_nolpp — L_pp isolation on the fdiff family

**Recipe:** exactly `20260821_sf3d_g19_fdiff` (g17 split-axis + PLAIN
trajectory head + fdiff losses) with `pred_pred_art_weight: 0` — module
kept so the consistency VALUE logs passively (supabl2 arm-D pattern).
30 fixed epochs, best = epoch 24 (val 1.1830 ≈ g19_fdiff's 1.1780 minus
its L_pp contribution). Ran on a power-capped host (~0.67 it/s — 600W
cap, EU-RO-1-wide that night).

## Test (5,088) vs g19_fdiff (identical, L_pp 0.1)

| metric | g19_fdiff (L_pp ON) | **fdnolpp (L_pp OFF)** |
|---|---|---|
| MA / MA_signed | **29.91 / 29.56** | 27.54 / 26.93 |
| axis matched / all | 17.25° / **25.74°** | **15.09° (record)** / 27.44° |
| flips all / rot | 12.21 / 15.17 | **10.36 / 13.48** |
| origin / radius (m) | 0.296 / 0.156 | **0.266 / 0.113** |
| 3D point | **0.227** | 0.231 |
| mIoU / PDet | 0.250 / 18.00 | 0.246 / **19.24** |
| traj_dir acc / cos | **96.11 / 0.819** | 95.77 / 0.809 |
| roughness (m) | 0.0509 | 0.0499 |

Passive L_pp value: 0.51 → 0.37 over training (consistency partially
emerges for free on this family — unlike DCT-family arm D where it
stayed flat) but never reaches the trained 0.23.

## Reading — the L_pp trade is FAMILY-DEPENDENT

On the DCT family (supabl2 arm D), dropping L_pp bought MA +2.2. Here it
COSTS MA −2.4 while improving nearly every precision column: matched
axis 15.09° is the best ever recorded (previous best g17's 16.9°), flip
rates drop below every prior arm, origin −3cm, radius −4.3cm (a
near-record). The two facts reconcile as a bulk-vs-threshold split:
without L_pp the axis/origin estimates get more precise where they were
already decent, but slightly fewer samples clear the joint MA pass bar.
So "no L_pp is strictly better" (the DCT-family conclusion) does NOT
generalize — with fdiff losses in the mix, L_pp still buys pass-rate MA.
g19_fdiff keeps the MA crown; this arm takes matched-axis/flip/radius.

Consequence for the gen-22 candidate: the earlier "…+ no L_pp" clause is
now uncertain — on the fdiff side L_pp earns its keep at threshold. The
honest formulation: head + analytic decode + fdiff, with L_pp a ±0.1
coin-flip to be measured, not assumed.

test pass: test.log (ckpt best-epoch24-valloss1.1830). Pod deleted.
