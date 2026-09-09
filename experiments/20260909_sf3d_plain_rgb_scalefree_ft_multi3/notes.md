# 20260909_sf3d_plain_rgb_scalefree_ft_multi3 — SF3D post-training, PLAIN head, from the multi-source 2D arm

**Question:** post-train the multi-source 2D arm (`20260909_multi3_rgb_scalefree` epoch-6: HOI4D + EPIC + ARCTIC, RGB-only scale-free plain-TF recipe, augmentation x8) on SF3D WITHOUT swapping the trajectory head: `trajectory_dct_coeffs 0`, so every weight of the 2D checkpoint — trunk, decoder, mask/point/origin heads, z_p/z_q, axis and type heads, and the full trajectory MLP including its readout — loads 1:1 (loader log: "All model weights loaded successfully"; user 2026-09-09 after learning that the DCT post-training re-initialised the plain readout's last layer). First SF3D post-training with a plain head and the first from a multi-source init. Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/sf3d_train_runpod_plain_rgb_scalefree_ft_multi3.yaml` = `sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml` with `trajectory_dct_coeffs 0` and the multi3 init. 30 ep, lr 1e-5, milestones [24, 28]; losses: mask/point/origin heatmaps, point-3D, origin-3D, axis, type, normalized trajectory (metres = GT z0 · Δ̃), consistency 0.1; fdiff / closed-form / analytic / twist / projection / tether all 0. Pod C (RTX PRO 6000 Workstation), 3 h 47 min right after the multi3 arm. Tested with pred_z_p (logs/test.log) and gt_z0 (logs/test_gt_z0.log) — identical on every reported metric, as for every scale-free arm (no test metric reads the trajectory scale).

**Result:** best val/loss_total **1.0953** (ep 19; val L_trajectory best 0.490 at ep 6, 0.554 at the end — the same early plateau/creep as the other RGB arms, slightly worse; val L_mask 0.331, val L_point_3d 0.037). Test (5,088): MA **31.01** / signed 30.17, PDet **22.72**, mIoU 0.2625, point err 0.1036, point3d 0.270 m, origin 0.317 m, axis all 27.3° / matched 20.8°, rot flips 14.2, traj_dir 88.9, roughness 0.068.

| init → SF3D head | depth | MA | PDet | mIoU | axis all / matched | origin | rough |
|---|---|---|---|---|---|---|---|
| HOI4D tf (DCT) → DCT, the record | yes | 31.13 | 23.27 | 0.266 | 28.0 / 19.7 | — | 0.0079 |
| HOI4D tf_plain → DCT | yes | 30.62 | 20.03 | 0.2555 | 28.0 / 20.0 | 0.286 | 0.0079 |
| HOI4D rgb_scalefree (plain) → DCT | no | 30.07 | 22.35 | 0.2660 | 26.9 / 22.0 | 0.293 | 0.0089 |
| **multi3 rgb_scalefree (plain) → PLAIN (this)** | **no** | **31.01** | 22.72 | 0.2625 | 27.3 / **20.8** | 0.317 | 0.068 |
| scratch rgb_scalefree → DCT | no | 24.88 | 17.39 | 0.2425 | 29.6 / 22.0 | 0.324 | 0.0109 |

**Reading:** (1) Best RGB-only MA so far and within 0.1 of the all-time (depth) record, with no depth anywhere and nothing re-initialised at the hand-over. (2) Against the RGB DCT arm from the HOI4D-only init: +0.9 MA, +0.4 PDet, −0.004 mIoU, better matched axis (20.8° vs 22.0°), 2 cm worse origin. Two things changed at once (plain head kept vs DCT re-init; multi3 init vs HOI4D-only), so the credit is not separable from one seed; a plain post-training from the HOI4D-only arm is the control. (3) The plain head's known cost shows: roughness 0.068 vs 0.009 (jittery sweeps) and traj_dir 88.9 vs 92.9 — the DCT basis is the smoothness fix; a plain→DCT hand-over that PROJECTS the plain readout onto the DCT basis (instead of re-initialising it) would keep both. (4) The scale-free trajectory val loss creeps after epoch ~6 in every RGB arm; still the thing to watch.

**Decision:** this is the new RGB-only reference checkpoint (MA). Follow-ups: the HOI4D-only plain control; the DCT-projection hand-over; per-source pretraining budget ~10 epochs at x8. Single seed. Ckpt best-epoch19-valloss1.0953.ckpt on the volume. vis: viz/20260909_sf3d_plain_ft_multi3_panels.
