# 20260813_sf3d_g4_vs_g5_panels

Head-to-head: gen-4 vs gen-5 (radius filter + rho 0.75 + frozen CLIP),
16 stratified val samples, **seed 3 — the same samples as
`20260812_sf3d_g3_vs_g4_panels/`**, so the three generations can be
compared panel-by-panel across the two batches.

- g4: `20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt`
- g5: `20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt`

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --model g5 experiments/20260812_sf3d_twist_g5/config.yaml experiments/20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt \
  --out viz/20260813_sf3d_g4_vs_g5_panels --num 16 --seed 3
```

NOTE these panels draw from the UNfiltered val split (the viz tool keeps
the legacy key cache) — knob-class samples still appear here even though
gen-5 neither trained nor evaluated on them; expect hedged omega on those.

Interpretation (numbers: experiments/20260812_sf3d_twist_g5/notes.md):

- The header `|w|` values tell the commitment story: gen-5's revolute
  samples sit visibly higher (0.30-0.71 here vs gen-4's 0.19-0.30), and
  panels above the 0.5 threshold decode "-> rot" with a door-scale
  yellow orbit for the first time (see b2/05_rot: |w| 0.71, curved
  sweep at the cabinet hinge; gen-4 says trans at 0.30).
- Trajectories (magenta) remain strong in both — that was the gen-4 win
  and gen-5 keeps it (traj_dir 92.6%).
- Not every rot sample commits (|omega| median 0.59, still short of 1);
  the below-threshold ones still render flat-ish orbits.
