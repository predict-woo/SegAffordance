# 20260819_sf3d_g17_2d_vs_g17_panels — what 2D-only supervision buys

16 val samples (seed 42421 — the family-standard picks, comparable with the
g16/g17 batches), rows: GT | g17splitax (best-epoch18, full 3D supervision)
| g17-2donly (best-epoch27, ZERO 3D GT), 512-px inputs, v3 GT.

Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g17splitax config/sf3d_train_runpod_g17_splitax.yaml \
    experiments/20260818_sf3d_g17_splitax/checkpoints/best-epoch18-valloss0.9272.ckpt \
  --model g17-2donly config/sf3d_train_runpod_g17_2donly.yaml \
    experiments/20260818_sf3d_g17_2donly/checkpoints/best-epoch27-valloss2.3453.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260819_sf3d_g17_2d_vs_g17_panels
```

Interpretation (matches the metrics: traj_dir 81.0 vs chance 50 · type
75.2 ≈ majority baseline · axis 58.7° ≈ random · mIoU 0.057):

- **What survived with zero 3D labels:** interaction-point placement and
  language grounding are still there — 08_rot_val1684 puts the point ON
  the door handle; 02_trans_val3962 picks a knob (the same wrong "second
  drawer" as g17 — the grounding tail is supervision-independent). The
  magenta trajectory clusters head roughly along the cyan GT track's
  initial direction — the 81% direction signal, visible but compressed
  and noisy.
- **What was lost:** masks are near-invisible blobs or absent (the mIoU
  0.265→0.057 collapse — identical mask supervision, shared features
  wrecked by the noisy projection/L_pp gradients); axes point ~randomly
  (08: door called trans at 71°); no full-extent sweeps.
- Use this batch as the picture of the gen-17-2D verdict: the projection
  data term teaches WHERE and WHICH WAY, but L_pp alone cannot teach the
  articulation frame, and the 2D-only gradient mix actively damages the
  segmentation pathway.
