# 20260814_sf3d_g6_mid_e10_panels

**MID-TRAIN snapshot #2** of the gen-6 split-heads arm — epoch 10 of 16
(best-epoch10-valloss1.0214.ckpt), rendered on the dev pod while the
training pod ran epoch ~13. RANDOM sample draw (seed 9030, user request
— not the recurring seed-3/7 sets), 16 stratified val samples, FILTERED
split (min_revolute_radius 0.10).

Same split-arm drawing semantics as `20260814_sf3d_g6_mid_e5_panels`:
marker = projected predicted 3D point, trajectory anchored there, RED
axis = predicted origin + direction (no twist decode), GREEN = GT.

Reading (val @ ep10: total 1.0214, traj ~0.013, origin ~0.87):
direction stays the strong suit; the visible failure modes at ep10 are
(a) language grounding — `14_rot_val3747` "Open the top oven door"
localizes the WASHING MACHINE next to the oven (point+mask on the
wrong appliance, hence cls=trans), and (b) type on ambiguous
furniture — `05_rot_val29821` "right drawer of the antique cabinet"
predicts a physically plausible horizontal drawer-pull (trans, ax=88)
where the GT annotates a revolute joint with a vertical axis at the
cabinet edge; grounding there is correct. Origin placement remains
coarse on true rotations. Masks still patchy.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g6e10 config/sf3d_train_runpod_split.yaml experiments/20260813_sf3d_split_g6/checkpoints/best-epoch10-valloss1.0214.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl --min-revolute-radius 0.10 \
  --out viz/20260814_sf3d_g6_mid_e10_panels --num 16 --seed 9030
```
