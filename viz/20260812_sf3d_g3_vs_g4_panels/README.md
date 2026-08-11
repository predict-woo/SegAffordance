# 20260812_sf3d_g3_vs_g4_panels

Head-to-head panels: gen-3 CLIP twist arm vs gen-4 (body-metric loss +
K=4 WTA articulation bundles), 16 stratified val samples (seed 3), both
best checkpoints, hint-free.

- g3: `20260804_sf3d_twist_g3/checkpoints/best-epoch08-valloss0.9042.ckpt`
- g4: `20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt`

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g3 experiments/20260804_sf3d_twist_g3/config.yaml experiments/20260804_sf3d_twist_g3/checkpoints/best-epoch08-valloss0.9042.ckpt \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --out viz/20260812_sf3d_g3_vs_g4_panels --num 16 --seed 3
```

Interpretation (matches the recorded eval: experiments/20260811_sf3d_twist_g4/notes.md):

- **Trajectories (magenta) are the gen-4 win.** g3's are stubs (the head
  had converged to the zero-motion baseline); g4's have real extent and
  track the GT sweep direction (e.g. 15_rot: follows the door arc;
  03_trans: a small flick at a light switch — scale-appropriate).
- **Localization is visibly better in g4** (e.g. 03_trans: g4 puts the
  point ON the switch where g3 drifts to the wrong counter; matches
  mIoU 0.118 vs 0.103, PDet 5.6 vs 4.0).
- **The yellow twist sweeps are still wrong in g4** — near-prismatic
  (|w| ~ 0.1-0.3 in the headers, "-> trans" on rot samples): the
  omega-hedge persists inside the WTA cells. This is the documented open
  item (rho/K knobs), NOT fixed by this run; read articulation from the
  magenta trajectory, not the yellow orbit.
