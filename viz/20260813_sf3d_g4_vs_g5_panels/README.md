# 20260813_sf3d_g4_vs_g5_panels

Head-to-head: gen-4 vs gen-5 (radius filter + rho 0.75 + frozen CLIP),
16 stratified val samples, seed 3. **Sampled from the FILTERED val split**
(min_revolute_radius 0.10 — the split gen-5 trains and evaluates on), so
no knob/dial-class degenerate rotations appear. NOTE this makes the
sample set different from the 20260812 g3-vs-g4 batches (unfiltered) —
regenerated 2026-08-13 at user request, replacing an unfiltered draw
from earlier the same day.

- g4: `20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt`
- g5: `20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt`

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g4 experiments/20260811_sf3d_twist_g4/config.yaml experiments/20260811_sf3d_twist_g4/checkpoints/best-epoch13-valloss0.7992.ckpt \
  --model g5 experiments/20260812_sf3d_twist_g5/config.yaml experiments/20260812_sf3d_twist_g5/checkpoints/best-epoch11-valloss0.7316.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl --min-revolute-radius 0.10 \
  --out viz/20260813_sf3d_g4_vs_g5_panels --num 16 --seed 3
```

Reading guide (numbers: experiments/20260812_sf3d_twist_g5/notes.md):
the header |w| is the commitment readout — gen-5 revolute samples sit at
0.3-0.7 (gen-4: 0.1-0.3), and above-0.5 panels decode "-> rot" with
door-scale yellow orbits for the first time. Magenta trajectories strong
in both generations.
