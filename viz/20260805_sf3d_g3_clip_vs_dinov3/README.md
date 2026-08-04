# gen-3 head-to-head: clip_g3 vs dinov3_g3 (unfrozen)

Full panels ([GT | clip_g3 | dinov3_g3], 12 standard samples, each arm's
best checkpoint, forward-sweep orbit rendering). The metrics split made
visual: dinov3_g3's yellow sweeps and magenta trajectories hug the GT
tracks noticeably tighter (twist axis 36.95 vs 42.8 deg, dir 80 vs 72%),
while its red mask overlays are absent/speckle (mIoU 0.013) where
clip_g3 shows real handle masks (0.103).

- clip_g3: experiments/20260804_sf3d_twist_g3/best-epoch08
- dinov3_g3: experiments/20260804_sf3d_twist_dinov3_g3/best-epoch13
Tool: tools/sf3d_vis_predictions.py (manifest.yaml). Rendered 2026-08-05.
