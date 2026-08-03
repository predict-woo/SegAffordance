# SF3D prediction panels: twist vs 2d_twist

Side-by-side `[GT | twist | 2d_twist]` panels for 16 deterministic,
rot/trans-stratified validation samples (same scene split as training:
seed 42, ratio 0.1). Inference is deployment-condition: CVAE prior
sampling, NO type hint (motion_type_input → NULL).

- **Experiments:** `experiments/20260728_sf3d_twist` (best-epoch04-valloss0.9891.ckpt),
  `experiments/20260728_sf3d_2d_twist` (best-epoch15-valloss1.0906.ckpt)
- **Tool:** `tools/sf3d_vis_predictions.py` @ 79b423e (draws GT track,
  predicted 3D trajectory, predicted 2D track, decoded twist orbit —
  legend in the tool docstring)
- **Rendered:** 2026-08-03 on the dev pod

Regenerate:

```bash
bash runpod/dev.sh run "python tools/sf3d_vis_predictions.py \
  --model twist experiments/20260728_sf3d_twist/config.yaml \
          experiments/20260728_sf3d_twist/checkpoints/best-epoch04-valloss0.9891.ckpt \
  --model 2d_twist experiments/20260728_sf3d_2d_twist/config.yaml \
          experiments/20260728_sf3d_2d_twist/checkpoints/best-epoch15-valloss1.0906.ckpt \
  --out viz/20260803_sf3d_twist_vs_2d_twist_panels --num 16"
```

## Reading the panels

Consistent with the eval metrics: interaction points land on the right
element; predicted 3D trajectories (magenta) head in plausible directions
but are under-swept; the twist head under-commits |omega| on true
rotations (arcs decode as trans at |omega|~0.18 even where the magenta
path clearly arcs); predicted masks are near-invisible at the 0.5
threshold (mIoU ~0.09). 2d_twist's orbits and tracks are visibly more
coherent than twist's, matching its win on every twist metric.
