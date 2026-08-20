# 20260821_sf3d_g17_2d_detach_panels — arm A alone, GT vs model

16 val samples (seed 42421, family-standard picks), rows: GT |
g17-2d-detach (best-epoch24) — the anchor-detach fix arm, trained with
ZERO 3D GT. No comparison column by request.

Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g17-2d-detach config/sf3d_train_runpod_g17_2d_detach.yaml \
    experiments/20260820_sf3d_g17_2d_detach/checkpoints/best-epoch24-valloss1.5182.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260821_sf3d_g17_2d_detach_panels
```

Interpretation (matches the metrics: proj2d shape 0.0997 = 3D-supervised
level, mIoU 0.242 recovered, axis/type still ~unlearned):

- **08_rot_val1684 (door): the money panel** — the magenta trajectory
  rides the cyan GT closing arc point-for-point, learned purely from 2D
  tracks. Point on the handle.
- **05_rot_val93 (oven): masks are BACK** — a real blob on the oven door
  (the broken g17-2d had nothing), full-extent downward sweep along the
  GT direction, drifting right near the end.
- **00_trans_val452 (toilet flush):** downward push direction correct with
  a rightward drift; mask blob on the button.
- **Still missing everywhere:** axes are wrong (76–148° on these panels)
  and types default trans — the articulation frame remains unlearned
  (L_pp-is-too-weak, unchanged by this fix). The magenta shapes are the
  fix's contribution; the red axes are the open problem.
