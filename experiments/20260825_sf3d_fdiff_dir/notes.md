# 20260825_sf3d_fdiff_dir — the dir term on the PLAIN head: redeemed

**Recipe:** exactly `20260821_sf3d_g19_fdiff` (g17 split-axis + plain
trajectory head + fdiff losses + L_pp 0.1) + `pred_pred_art_dir_weight:
0.1`. First dir-term run WITHOUT the DCT head — separates "the term is
inherently harmful" (the g21 conclusion) from "the term interacts badly
with the DCT basis". 30 fixed epochs, best = epoch 24 (val 1.2487).
Power-capped host (~0.7 it/s).

## Test (5,088) vs its two family siblings

| metric | g19_fdiff (L_pp, no dir) | fdnolpp (neither) | **fddir (L_pp + dir)** |
|---|---|---|---|
| MA / MA_signed | **29.91 / 29.56** | 27.54 / 26.93 | 27.81 / 27.56 |
| axis matched | 17.25° | 15.09° | **14.62° (record)** |
| flips all / rot | 12.21 / 15.17 | 10.36 / 13.48 | 11.87 / **11.24 (record)** |
| origin / radius | 0.296 / 0.156 | 0.266 / **0.113** | 0.288 / 0.134 |
| 3D point | **0.227** | 0.231 | 0.237 |
| mIoU / PDet | 0.250 / 18.00 | 0.246 / **19.24** | 0.241 / 17.30 |
| traj_dir | **96.11 / 0.819** | 95.77 / 0.809 | 95.46 / 0.811 |
| roughness | 0.0509 | 0.0499 | 0.0486 |

## Reading — the g21 "dir term failed" verdict was HALF WRONG

On the plain head the term does exactly what it was designed to do:
**rot flip rate 11.24 — the lowest ever recorded** (the stubborn ~13.3%
tail that split heads, DCT, and fdiff all failed to move) — plus a
matched-axis record (14.62°), at a cost of only −0.7 traj_dir
(vs the −5.7 CRASH on the DCT head) and −2.1 threshold-MA.

So the damage attribution: the g21 failure was substantially the
**dir × DCT interaction** — the DCT basis projects the term's trajectory
drag onto coherent whole-curve rotation (exactly the net-direction mode
traj_dir measures), while the plain head absorbs it as local bending.
The term's design goal (kill the rot sign-flip tail) is achieved here.
The trajectory-detach variant remains the theory-clean follow-up and
might keep the flip win while recovering the MA/traj_dir cost — now
worth its run.

Family grid summary: g19_fdiff keeps threshold-MA; the two interventions
(remove L_pp / add dir) both trade threshold-MA for precision, in
different columns. No arm dominates.

test pass: test.log (ckpt best-epoch24-valloss1.2487). Pod deleted —
no training pods remain.
