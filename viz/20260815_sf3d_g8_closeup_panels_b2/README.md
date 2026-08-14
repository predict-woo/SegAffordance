# 20260815_sf3d_g8_closeup_panels_b2

Second random draw (seed 18827) from the gen-8 close-up best checkpoint
(best-epoch59-valloss1.0173.ckpt), 16 samples, close-up split
(minrad010 + maskfrac00025 cache). Same reading guide as
`20260814_sf3d_g8_closeup_panels`, plus (re-rendered same day): trajectories
are now DOTS-only (start ringed white — no line smoothing), and the GT track
is overlaid on the model panels as small cyan dots (`cyn=GTtraj`).

Checked highlights:

- `00_trans_val1097` "top drawer of the cabinet next to the bathtub":
  relational grounding WORKS here — mask+point on the correct top knob
  among three identical ones, trans at ax=21 deg, smooth pull-out
  trajectory along the ray. The good case of this generation in one
  panel.
- `04_rot_val295` "left drawer of the wooden drawer table": the
  drawer/door annotation-vs-language conflict again (gen-6's antique-
  cabinet case) — text says "drawer", model grounds the right handle
  and predicts a physically-plausible trans pull at 90 deg to the GT,
  but GT annotates rot with a hinge at the cabinet edge. Counts
  against type/axis metrics; arguably the model follows the language.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g8e59 config/sf3d_train_runpod_g8_closeup.yaml experiments/20260814_sf3d_g8_closeup/checkpoints/best-epoch59-valloss1.0173.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac00025.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.0025 \
  --out viz/20260815_sf3d_g8_closeup_panels_b2 --num 16 --seed 18827
```
