# Visualization index

One row per batch, newest first. Every batch is a dated directory
`YYYYMMDD_<subject>_<what>/` with a tracked `README.md` (and, for
tool-generated batches, an auto-written `manifest.yaml`); the images
themselves are gitignored but live on both sides of the mutagen mirror.
Conventions: `../CLAUDE.md` § Visualization organization.

| Batch | What | Source |
|---|---|---|
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
| 20260814_sf3d_g6_mid_e5_panels | gen-6 split-heads MID-TRAIN (epoch 5/16), seed 3 filtered — first split-arm render: type+direction already strong (rot 19deg / trans 5deg), origin placement + point anchor still coarse |
| 20260813_sf3d_g4_vs_g5_panels | g4 vs g5 best, seed 3, FILTERED val split (no knob-class): omega commitment visible — |w| headers up, first door-scale rot orbits |
| 20260813_sf3d_g4_vs_g5_panels_b2 | 30 more g4-vs-g5 (seed 7, filtered split); highlight 01_rot 'Close the door': g5 rot @ |w|=0.58, axis 27deg vs g4's 95deg; GT/pred axes overlaid |
| 20260812_sf3d_g3_vs_g4_panels_b2 | 30 more g3-vs-g4 panels (seed 7, disjoint samples) — same reading guide as the seed-3 batch |
| 20260812_sf3d_g3_vs_g4_panels | g3 vs g4 best ckpts, 16 val samples: g4 trajectories real + localization better; twist orbits still flat (omega hedge) |
| 20260812_sf3d_g4_traj_points | g4 best, trajectory points only: ordered direction-correct sweeps vs g3's zero-motion stubs |
