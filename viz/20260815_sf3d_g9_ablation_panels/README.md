# 20260815_sf3d_g9_ablation_panels — supervision-ablation 3-arm comparison

16 random val samples (seed 42421) from the gen-9 split (mask >0.1%, 5%
edge margin, radius ≥0.10 m), each panel row: GT | arm A (g9 joint) |
arm B (articulation-only) | arm C (trajectory-only).

Checkpoints:
- g9joint: `20260815_sf3d_g9_closeup010` / `best-epoch23-valloss0.9191.ckpt`
- armB: `20260815_sf3d_g9abl_artonly` / `best-epoch21-valloss0.9491.ckpt`
- armC: `20260815_sf3d_g9abl_trajonly` / `best-epoch22-valloss0.4090.ckpt`

Regen (dev pod, /workspace/SegAffordance):

```
/opt/venv/bin/python tools/sf3d_vis_predictions.py \
  --model g9joint config/sf3d_train_runpod_g9_closeup010.yaml experiments/20260815_sf3d_g9_closeup010/checkpoints/best-epoch23-valloss0.9191.ckpt \
  --model armB config/sf3d_train_runpod_g9abl_artonly.yaml experiments/20260815_sf3d_g9abl_artonly/checkpoints/best-epoch21-valloss0.9491.ckpt \
  --model armC config/sf3d_train_runpod_g9abl_trajonly.yaml experiments/20260815_sf3d_g9abl_trajonly/checkpoints/best-epoch22-valloss0.4090.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260815_sf3d_g9_ablation_panels
```

Interpretation: the absent-head guards render correctly — arm B panels
have the red axis/yellow orbit but no magenta trajectory dots; arm C
panels have trajectory dots but `cls=n/a` and no axis/orbit/origin ring.
Qualitatively matches the metrics: arm C trajectories wander off the GT
sweep direction more often than the joint model's (e.g. `13_rot_val164`,
fridge — armC dots head down-right while GT sweeps right), consistent
with its traj_dir 87.6% vs joint 93.1%.
