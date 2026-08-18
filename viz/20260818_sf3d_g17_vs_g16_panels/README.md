# 20260818_sf3d_g17_vs_g16_panels — split axis heads, before/after

16 val samples (seed 42421 — the same picks as the g16-vs-g13 batch, so
three generations are directly comparable), rows: GT | g16trajnorm
(best-epoch21) | g17splitax (best-epoch18), 512-px inputs, v3 GT.

Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g16trajnorm config/sf3d_train_runpod_g16_trajnorm.yaml \
    experiments/20260817_sf3d_g16_trajnorm/checkpoints/best-epoch21-valloss1.0123.ckpt \
  --model g17splitax config/sf3d_train_runpod_g17_splitax.yaml \
    experiments/20260818_sf3d_g17_splitax/checkpoints/best-epoch18-valloss0.9272.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260818_sf3d_g17_vs_g16_panels
```

Interpretation:

- **13_rot_val164 "Open the fridge" — the marquee panel.** g16's axis is
  the textbook sign flip (170°, red axis upside-down against green GT);
  g17 lands at 20° with the right sign. The famous failure case from the
  g13/g16 batches is fixed here — even though the AGGREGATE rot flip rate
  didn't move (13.3% vs 12.7%), the flips redistributed rather than
  reduced.
- **08_rot_val1684 (door by the nightstand):** both arms sweep the full
  door arc along the cyan GT; g17's yellow orbit is wider and its magenta
  points ride the GT track a touch closer (ax 11° vs 7° — sample-level
  axis noise runs both ways; the test set says g17 wins on average).
- **05_rot_val93 (top oven door):** both full-extent arcs on the handle;
  g16 happens to nail the axis (1° vs 14°) — same story in reverse.
- **Trans panels (00/01/02/…):** rays essentially unchanged, as designed —
  the trajectory head wasn't touched. The known grounding tail persists
  unchanged (02_trans_val3962: both arms pick the wrong drawer for
  "second drawer of the wooden cabinet" while getting a clean 6–7° ray).

Matches the metrics: type 92.3→95.3, MA 23.1→25.9, matched axis
17.8→16.9°, origin 0.303→0.257 m, traj_dir 94.5→94.9/cos 0.811.
