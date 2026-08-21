# 20260822_sf3d_g19_smoothness_panels — the smoothness fix, three ways

16 val samples (seed 42421, family-standard picks), rows: GT | g17
(best-epoch18) | g19-dct (best-epoch20) | g19-fdiff (best-epoch29),
512-px inputs, v3 GT.

Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g17 config/sf3d_train_runpod_g17_splitax.yaml \
    experiments/20260818_sf3d_g17_splitax/checkpoints/best-epoch18-valloss0.9272.ckpt \
  --model g19-dct config/sf3d_train_runpod_g19_dct.yaml \
    experiments/20260821_sf3d_g19_dct/checkpoints/best-epoch20-valloss0.9652.ckpt \
  --model g19-fdiff config/sf3d_train_runpod_g19_fdiff.yaml \
    experiments/20260821_sf3d_g19_fdiff/checkpoints/best-epoch29-valloss1.1780.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260822_sf3d_g19_smoothness_panels
```

Interpretation (matches the metrics — roughness 0.0915 / 0.0090 / 0.0509,
GT floor 0.0032):

- **08_rot_val1684 (door):** g17's magenta points scatter around the arc
  (the jitter that motivated gen-19); g19-dct's points form a visibly
  ordered smooth curve; g19-fdiff's are also clean, riding the GT track
  and orbit tightly (its 96.1% direction record is visible as better
  track adherence).
- The DCT arm's curves are the smoothest everywhere — by construction —
  while fdiff's are smoother than g17 but retain some waviness; fdiff's
  arcs align with the GT direction best.
- Axes/masks: consistent with the tables (dct panels show the mask/PDet
  gains; fdiff shows tighter direction, slightly weaker masks).
