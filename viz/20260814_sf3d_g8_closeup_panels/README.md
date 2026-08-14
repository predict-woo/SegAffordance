# 20260814_sf3d_g8_closeup_panels

Gen-8 close-up experiment, FINAL best checkpoint
(best-epoch59-valloss1.0173.ckpt), 16 random samples (seed 14406) from
the CLOSE-UP split (min_revolute_radius 0.10 + min_mask_area_frac
0.0025 — 19,296 records; metrics comparable to nothing before it).

Reading, checked on `15_rot_val1345` "Open the washing machine door":

- **Masks are back.** The close-up split gives the mask head real
  targets (median element no longer ~5x5 px at 256^2): predicted mask
  is a visible, correctly-placed blob on the handle. Test mIoU 0.178 —
  ~1.8x the best any full-split arm ever reached (0.118) — with PDet
  9.4 (vs 3-5.6). Easier data, but the first proof the mask head
  works when the target is resolvable.
- **The relative direct trajectory is SMOOTH** — connected magenta
  curves, no gen-7 scribbles, no delta-cumsum needed: the Euler-era
  formulation (option 1) visually and numerically validated
  (traj_dir 94.8%, cos 0.783, best ever).
- **Axis orientation is the remaining error mode** on unusual
  articulations: the top-loader's lid hinge (horizontal) is predicted
  as a side-swing (vertical, ax=86 deg) — the err_adir_all 32.8 deg
  mean carries a tail of such cases while matched samples sit at
  17.2 deg.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g8e59 config/sf3d_train_runpod_g8_closeup.yaml experiments/20260814_sf3d_g8_closeup/checkpoints/best-epoch59-valloss1.0173.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac00025.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.0025 \
  --out viz/20260814_sf3d_g8_closeup_panels --num 16 --seed 14406
```
