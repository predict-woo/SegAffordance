# 20260813_sf3d_g4_vs_g5_panels_b2

Larger draw of the g4-vs-g5 head-to-head: 30 stratified val samples,
**seed 7 — the same samples as `20260812_sf3d_g3_vs_g4_panels_b2/`**.
Same checkpoints and reading guide as `20260813_sf3d_g4_vs_g5_panels/`.

Highlight: `05_rot_val11758.jpg` — "open the leftmost cabinet door":
gen-5 decodes rot at |w| = 0.71 with a door-scale curved orbit at the
hinge side; gen-4 called the same sample trans at 0.30.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --model g5 experiments/20260812_sf3d_twist_g5/config.yaml experiments/20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt \
  --out viz/20260813_sf3d_g4_vs_g5_panels_b2 --num 30 --seed 7
```
