# 20260813_sf3d_g4_vs_g5_panels_b2

Larger draw of the g4-vs-g5 head-to-head: 30 stratified val samples,
seed 7, from the FILTERED val split (no knob-class rotations) — same
checkpoints and reading guide as `20260813_sf3d_g4_vs_g5_panels/`.

Highlight: `01_rot_val16602.jpg` — "Close the door": gen-5 decodes rot
at |w| = 0.58 with a wide door-scale orbit and a trajectory tracking the
GT sweep; gen-4 called it trans at 0.22.

Axis overlays (added same day): GREEN = GT articulation axis (revolute:
the hinge line through the annotated origin; prismatic: a direction ray
from the element), RED = the axis decoded from the predicted twist, both
with a dot at the +sign end; the header's `ax=` is the signed angular
error of the decoded direction. The thin green copy on model panels is
the GT reference for direct comparison.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --model g5 experiments/20260812_sf3d_twist_g5/config.yaml experiments/20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl --min-revolute-radius 0.10 \
  --out viz/20260813_sf3d_g4_vs_g5_panels_b2 --num 30 --seed 7
```
