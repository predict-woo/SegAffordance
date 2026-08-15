# Visualization index

One row per batch, newest first. Every batch is a dated directory
`YYYYMMDD_<subject>_<what>/` with a tracked `README.md` (and, for
tool-generated batches, an auto-written `manifest.yaml`); the images
themselves are gitignored but live on both sides of the mutagen mirror.
Conventions: `../CLAUDE.md` § Visualization organization.

| Batch | What | Source |
|---|---|---|
| [20260816_sf3d_g11_vs_g10_panels](20260816_sf3d_g11_vs_g10_panels/README.md) | gen-11 vs gen-10 on v3 GT — g11 trans sweeps at 0.7m scale, g10 still 0.1m stubs | `tools/sf3d_vis_predictions.py` |
| [20260816_sf3d_g10_vs_g9_panels](20260816_sf3d_g10_vs_g9_panels/README.md) | normalized-L_pp before/after — g10 trajectories track their own orbits (val1684 fixed) | `tools/sf3d_vis_predictions.py` |
| [20260816_sf3d_g10_lpp_val93_3d](20260816_sf3d_g10_lpp_val93_3d/README.md) | val93 3D re-plot with gen-10 — radial RMS 0.12->0.05m, sweep shortened | `tools/diag_lpp_samples.py` + local matplotlib |
| [20260815_sf3d_g9_lpp_val93_3d](20260815_sf3d_g9_lpp_val93_3d/README.md) | 3D pred-trajectory vs pred-orbit for val93 — radius shrinks 0.29->0.12m inside the predicted circle (L_pp p97) | `tools/diag_lpp_samples.py` + local matplotlib |
| [20260815_sf3d_g9_ablation_panels_b2](20260815_sf3d_g9_ablation_panels_b2/README.md) | b2 re-render: no GT-trajectory overlay on trajectory-less arm B panels | `tools/sf3d_vis_predictions.py` |
| [20260815_sf3d_g9_ablation_panels](20260815_sf3d_g9_ablation_panels/README.md) | supervision ablation 3-arm panels (joint vs art-only vs traj-only) — head guards render correctly, armC trajectories wander | `tools/sf3d_vis_predictions.py` |
| [20260805_sf3d_g3_clip_vs_dinov3](20260805_sf3d_g3_clip_vs_dinov3/README.md) | gen-3 head-to-head — dinov3 wins geometry, clip wins masks | `tools/sf3d_vis_predictions.py` |
| [20260804_sf3d_clip_g3_panels](20260804_sf3d_clip_g3_panels/README.md) | clip gen-3 full panels — smooth trajectories, planar pitch-free orbits, visible masks | `tools/sf3d_vis_predictions.py` |
| [20260804_sf3d_clip_g3_traj_points](20260804_sf3d_clip_g3_traj_points/README.md) | clip gen-3 trajectory points — 1-cos killed the fold-back zigzag AND the under-extension | `tools/sf3d_vis_predictions.py --traj-only` |
| [20260804_sf3d_gen2_traj_points](20260804_sf3d_gen2_traj_points/README.md) | gen-2 arms, trajectory-only points — delta-cumsum turned clouds into ordered sweeps (clip) | `tools/sf3d_vis_predictions.py --traj-only` |
| [20260804_sf3d_gen2_panels](20260804_sf3d_gen2_panels/README.md) | gen-2 full panels: clip vs dinov3 vs 2donly at last.ckpt | `tools/sf3d_vis_predictions.py` |
| [20260804_sf3d_gt_twist_check](20260804_sf3d_gt_twist_check/README.md) | GT twist orbit (both signs) vs GT trajectory, 12 samples — 12/12 stored-sign OK | `tools/sf3d_vis_gt_twist.py` |
| [20260803_sf3d_twist_traj_points](20260803_sf3d_twist_traj_points/README.md) | twist arm, trajectories only, drawn as points — exposes jittery/unordered predicted sequences | `tools/sf3d_vis_predictions.py --traj-only` |
| [20260803_sf3d_twist_vs_2d_twist_panels](20260803_sf3d_twist_vs_2d_twist_panels/README.md) | GT vs twist vs 2d_twist prediction panels, 16 stratified val samples | `tools/sf3d_vis_predictions.py` |
| [20260727_sf3d_reproc_debug](20260727_sf3d_reproc_debug/README.md) | One-off debug renders from the sf3d_processed_v2 rebuild (splat-mask A/B, arc-fix checks, per-frame inspections) | ad-hoc scripts, July 27–28 |
| [20260727_sf3d_v1_dataset_audit](20260727_sf3d_v1_dataset_audit/README.md) | 100-sample audit of the v1 SF3D LMDB (frame/zoom/depth panels, contact sheet, geometry TSV) | `tools/sf3d_vis_samples.py` |
| [20260720_opd_label_audits](20260720_opd_label_audits/README.md) | OPD sample renders + description-regeneration label audits | `tools/show_opd_samples.py`, `tools/label_render.py` |
| 20260815_sf3d_mask_cutoff_curve | data-vs-cutoff curve for min_mask_area_frac (full-DB scan): 0.25%->19.3k, 0.10%->59.6k, 0.05%->120k, 0.01%->293k of 356.7k; knee sits in the useful range |
| 20260815_sf3d_g8_closeup_panels_b2 | 16 more g8 close-up (seed 18827): relational grounding nailed ('top drawer next to bathtub', ax=21deg); the drawer/door annotation-vs-language conflict recurs (trans pull vs GT rot) |
| 20260814_sf3d_g8_closeup_panels | gen-8 close-up best (ep59): masks finally visible blobs (mIoU 0.178), relative-direct trajectories SMOOTH (no zigzag); top-loader lid-hinge axis miss illustrates the 32.8deg tail |
| 20260814_sf3d_g7_e14_panels | gen-7 FINAL best (ep14): heatmap origin lands ON the hinge (rot 6deg highlight); absolute-trajectory ZIGZAG risk materialized; relational grounding still misses |
| 20260814_sf3d_g6_mid_e10_panels | gen-6 MID-TRAIN #2 (epoch 10/16), RANDOM draw seed 9030 — direction strong; failure modes: grounding (washer-for-oven) and drawer/door type calls; origin still coarse |
| 20260814_sf3d_g6_mid_e5_panels | gen-6 split-heads MID-TRAIN (epoch 5/16), seed 3 filtered — first split-arm render: type+direction already strong (rot 19deg / trans 5deg), origin placement + point anchor still coarse |
| 20260813_sf3d_g4_vs_g5_panels | g4 vs g5 best, seed 3, FILTERED val split (no knob-class): omega commitment visible — |w| headers up, first door-scale rot orbits |
| 20260813_sf3d_g4_vs_g5_panels_b2 | 30 more g4-vs-g5 (seed 7, filtered split); highlight 01_rot 'Close the door': g5 rot @ |w|=0.58, axis 27deg vs g4's 95deg; GT/pred axes overlaid |
| 20260812_sf3d_g3_vs_g4_panels_b2 | 30 more g3-vs-g4 panels (seed 7, disjoint samples) — same reading guide as the seed-3 batch |
| 20260812_sf3d_g3_vs_g4_panels | g3 vs g4 best ckpts, 16 val samples: g4 trajectories real + localization better; twist orbits still flat (omega hedge) |
| 20260812_sf3d_g4_traj_points | g4 best, trajectory points only: ordered direction-correct sweeps vs g3's zero-motion stubs |
