# 20260814_sf3d_g7_e14_panels

Gen-7 (heatmap + depth lifts) FINAL best checkpoint —
best-epoch14-valloss1.3171.ckpt — 16 random val samples (seed 30662),
FILTERED split (min_revolute_radius 0.10). Rendered while the test eval
was still running; metrics in experiments/20260814_sf3d_g7/notes.md.

Gen-7 drawing semantics: point marker = projected lifted p_hat; RED
circle = origin_uv (the origin heatmap's readout); red axis = lifted
q_hat + direction; trajectory = the ABSOLUTE 20-point readout projected
directly (no anchoring).

Reading, checked against two panels:

- **Origin/axis: the gen-7 win.** `15_rot_val34448` "Close the bedroom
  door": rot, ax=6 deg, red axis parallel AND adjacent to the GT hinge
  line, origin marker on the actual hinge edge — the heatmap origin
  fixes the gen-6 floating-origin failure on grounded samples.
- **Trajectory: the accepted zigzag risk MATERIALIZED.** The absolute
  direct 20-point readout renders as disordered mid-air scribbles
  (panel 15's magenta tangle) — the exact failure mode delta-cumsum
  was introduced to fix (viz/20260803_sf3d_twist_traj_points).
  Recorded fallback: trajectory_delta_cumsum + relative frame.
- **Grounding still shaky on relational referring expressions.**
  `01_trans_val3826` "second drawer next to the washing machine"
  grounds the OVEN drawer across the room (type still trans, but
  ax=98 deg because it describes a different drawer's geometry).

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g7e14 config/sf3d_train_runpod_g7.yaml experiments/20260814_sf3d_g7/checkpoints/best-epoch14-valloss1.3171.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl --min-revolute-radius 0.10 \
  --out viz/20260814_sf3d_g7_e14_panels --num 16 --seed 30662
```
