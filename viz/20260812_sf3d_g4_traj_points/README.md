# 20260812_sf3d_g4_traj_points

Trajectory-focused panels (points only, start ringed white; no mask, no
twist orbit) for the gen-4 best checkpoint — the direct visual check that
the trajectory head now predicts ordered, connected, direction-correct
motion instead of the gen-3 zero-motion hedge. Same 16 samples/seed as
the g3-vs-g4 batch for cross-reference.

- ckpt: `20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt`

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --out viz/20260812_sf3d_g4_traj_points --num 16 --seed 3 --traj-only
```
