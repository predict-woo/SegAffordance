# twist arm: trajectory-only panels, rendered as points

Diagnostic follow-up to [[20260803_sf3d_twist_vs_2d_twist_panels]]: same 16
stratified val samples, twist model only, `--traj-only` mode — GT track
(cyan) and predicted 3D trajectory (magenta) drawn as DISCRETE POINTS with
the sequence start ringed white; no mask, no twist orbit, no smoothing by
line rendering.

- **Experiment:** `experiments/20260728_sf3d_twist`
  (best-epoch04-valloss0.9891.ckpt), hint-free CVAE-prior inference
- **Tool:** `tools/sf3d_vis_predictions.py --traj-only` (manifest.yaml has
  the exact command)
- **Rendered:** 2026-08-03 on the dev pod

## What the points expose (that line rendering hid)

GT trajectories are clean, monotone, evenly-spaced sweeps. The predicted
20-point sequences are NOT paths in that sense:

1. **Jittery, unordered sequences** — points scatter around a direction
   instead of progressing along it; consecutive indices jump back and
   forth.
2. **Bidirectional hedging** — the cluster typically straddles the anchor
   on both sides (visible in the rot samples), as if the head averages the
   two possible sweep directions rather than committing to one.
3. **Index 0 sits mid-cluster, not at an end** — the head's "relative to
   first point" convention is honoured numerically but the sequence has no
   temporal ordering along the sweep.
4. **Extent under-predicted** relative to GT (the known under-sweep).

Line rendering had made these look like short-but-plausible smooth paths;
the polyline connecting a jittery sequence traces out something
arc-like even when the underlying point set is disordered.
