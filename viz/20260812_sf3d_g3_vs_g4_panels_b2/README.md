# 20260812_sf3d_g3_vs_g4_panels_b2

Second, larger sampling of the g3-vs-g4 head-to-head (30 stratified val
samples, seed 7 — disjoint draw from the seed-3 batch). Same checkpoints
and reading guide as `20260812_sf3d_g3_vs_g4_panels/` (see its README:
magenta trajectory = the gen-4 win; yellow twist orbit = still hedged,
known open item).

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g3 experiments/20260804_sf3d_twist_g3/config.yaml experiments/20260804_sf3d_twist_g3/checkpoints/best-epoch08-valloss0.9042.ckpt \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --out viz/20260812_sf3d_g3_vs_g4_panels_b2 --num 30 --seed 7
```
