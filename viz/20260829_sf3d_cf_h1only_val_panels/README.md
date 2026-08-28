# 20260829_sf3d_cf_h1only_val_panels — GT | g19_fdiff | cf_h1only

16 val samples (seed 42421, family-standard picks), columns: GT |
g19_fdiff (best-epoch29, old MA record 29.91, full trajectory apparatus)
| cf_h1only (best-epoch29, NEW MA record 30.64, H1 quadratic + axis
anchor, NO trajectory head).

Note on the cf_h1only column: the model has no trajectory head, so there
is no magenta curve; the yellow sweep is decoded analytically from its
predicted articulation (type-routed forward sweep, 90 deg / 0.7 m) —
i.e. exactly the curve its training loss compared to GT.

Rendered on the dev pod. Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g19-fdiff config/sf3d_train_runpod_g19_fdiff.yaml \
    experiments/20260821_sf3d_g19_fdiff/checkpoints/best-epoch29-valloss1.1780.ckpt \
  --model cf-h1only config/sf3d_train_runpod_cf_h1only.yaml \
    experiments/20260828_sf3d_cf_h1only/checkpoints/best-epoch29-valloss1.1303.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260829_sf3d_cf_h1only_val_panels
```

Interpretation (consistent with the metrics: MA 30.64 vs 29.91, matched
16.6 vs 17.3 deg, origin 0.254 vs 0.296):

- **Axes track GT closely on the standard cases in BOTH columns**, cf
  usually within a few degrees of g19 (oven door 05: 14 vs 10 deg; door
  08: 13 vs 10; toilet flush 00: 7 vs 8 — cf slightly tighter). The red
  predicted axis hugs the green GT axis, and cf's red origin dot sits
  visibly closer to the GT hinge line in the door/oven panels (the
  origin story: 0.254 vs g19's 0.296).
- **cf_h1only's analytic sweep (yellow) is a clean, plausible motion arc**
  — door 08's sweep follows the leaf's closing arc, the oven door's
  drops along the GT track. Trajectory realism survives the head's
  removal because the curve IS the articulation now.
- **Masks are crisp and comparable on handles/doors** (cf's are the
  better of the two by the numbers, 0.266 vs 0.250, and it shows
  slightly in the oven-handle blob).
- **Shared failure case: 13_rot_val164** (fridge, extreme close
  viewpoint, element mostly out of frame): g19 flips outright
  (163 deg), cf is wrong but less catastrophically (104 deg). Consistent
  with flips being the residual failure mode for both.

checkpoints: 20260821_sf3d_g19_fdiff/best-epoch29-valloss1.1780,
20260828_sf3d_cf_h1only/best-epoch29-valloss1.1303. vis pointer added to
both experiments' notes.md.
