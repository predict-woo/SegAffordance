# 20260827_sf3d_supabl3_traj_fdiff — arm C_f: trajectory-only on the fdiff family

**Recipe:** supabl2_traj's head-removal gates verbatim (no axis/type
heads, no origin heatmap or z_q, 2-ch projector) with the DCT head
swapped for the PLAIN head + the gen-19 fdiff losses (1.0/0.5/0.5).
Completes the fdiff-family joint-vs-either grid. 30 epochs, best =
epoch 24 (val 0.7399), full-speed host.

## Test (5,088) — the fdiff-family triangle

| metric | arm B (art-only) | **C_f (traj-only)** | fdnolpp (joint) |
|---|---|---|---|
| mIoU / PDet | **0.268 / 22.60** | 0.261 / 20.22 | 0.246 / 19.24 |
| 2D point | 0.103* | **0.101** | 0.106 |
| 3D point | 0.243 | 0.248 | **0.231** |
| traj_dir acc / cos | — | 94.40 / 0.790 | **95.77 / 0.809** |
| roughness (m) | — | 0.0497 | 0.0499 |
| MA / matched axis | 20.4 / 23.3° | — | **27.5 / 15.1°** |

\* arm B's mean_point_error from its own wrap.

## Reading — joint wins on the fdiff family, on BOTH sides

- **Trajectory→articulation** (joint vs B): MA +7.1, matched −8.2° —
  the huge one, as on DCT.
- **Articulation→trajectory** (joint vs C_f): traj_dir +1.4 acc /
  +0.019 cos — small but REAL, where the DCT family measured ~nil
  (94.26 vs 94.36). With fdiff's direction-sensitive losses in play,
  articulation supervision gives the trajectory a measurable assist.
  So "the coupling is asymmetric" survives, but the fdiff family's weak
  side is not exactly zero.
- **The mask ordering does NOT transfer.** DCT family: C > B > D
  (trajectory-only best). fdiff family: B > C_f > joint (0.268 > 0.261 >
  0.246). The "trajectory supervision is trunk-friendly" story was
  DCT-specific: the plain head + fdiff losses send harsher gradients
  through the pooled features than the smooth 6-coefficient DCT basis
  does. Consistent with every fdiff arm's masks sitting ~0.24-0.25 vs
  DCT arms' 0.265+, and with andec_fdiff's milder mask dip.
- Smoothness is unchanged joint-vs-traj-only (0.0497 vs 0.0499): it
  comes from the fdiff losses, not from articulation supervision.

test pass: logs/test.log (ckpt best-epoch24-valloss0.7399). Pod deleted
— campaign complete, no training pods remain.
