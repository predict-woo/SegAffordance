# gen-2 panels: clip vs dinov3 vs 2donly

Full prediction panels ([GT | clip | dinov3 | 2donly], 12 stratified val
samples, same seed/samples as every prior batch) for the three 20260804
arms, each at last.ckpt, hint-free inference.

- clip: experiments/20260804_sf3d_twist_clip (twist orbit now decodes ROT
  on true rotations — e.g. counter door |omega|=0.74 -> rot where the old
  generation sat at 0.57 -> trans; trajectory arcs along the orbit)
- dinov3: experiments/20260804_sf3d_twist_dinov3 (trajectories still
  scribbly/jittery, points mislocalised — matches its weak metrics)
- 2donly: experiments/20260804_sf3d_2donly (|omega| = 0.00 everywhere —
  the omega-shrink collapse the eval quantified; orbits degenerate to
  straight lines; cls=n/a: no type head)

Tool: tools/sf3d_vis_predictions.py (manifest.yaml). Rendered 2026-08-04.
