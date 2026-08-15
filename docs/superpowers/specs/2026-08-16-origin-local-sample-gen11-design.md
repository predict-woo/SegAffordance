# Gen-11: Origin Local Sample + Meaningful Prismatic Sweep Length

**Date:** 2026-08-16
**Status:** DRAFT — awaiting user review
**Goal:** Two user-decided changes on top of the gen-10 recipe:
(A) the origin depth head gets the same grid-sampled local feature the
point depth head has (path symmetry), and (B) the synthetic prismatic
trajectory length changes from the arbitrary 0.1 m to **0.7 m** — the
measured median 90° revolute arc length — via a reader-side rescale.

## A · Origin local sample (`use_origin_local_feature`)

Diagrams: `docs/slides/2026-08-16_origin_local_sample.html`.

**Rationale (reverses a gen-7 assumption):** gen-7 kept ẑ_q condition-only
on the belief that the pixel under the origin is occluded/meaningless.
Evidence since: q* is the perpendicular foot from the interaction point —
for door-class objects the visible door/frame seam at handle height — and
the gen-7 viz batch showed the origin heatmap localizing exactly those
seams (99.6–100% of q* projections are in-frame). The hinge pixel's
appearance IS depth evidence, the same way the handle pixel is for ẑ_p.
Loop-free: location-in → depth-out, no feedback.

**Change (`model/segmenter.py`):** new ModelParams flag
`use_origin_local_feature: bool = False`. When true (requires
`use_origin_heatmap`; ValueError otherwise):

- `origin_depth_head_g7` input dim: `vae_condition_dim` →
  `vae_condition_dim + fpn_out[1]`.
- Forward adds one grid_sample, the mirror of the point path's:

```python
        if self.origin_depth_head_g7 is not None:
            zq_in = vae_condition
            if self.use_origin_local_feature:
                ogrid = origin_uv.view(-1, 1, 1, 2) * 2.0 - 1.0
                olocal = F.grid_sample(
                    fq, ogrid, align_corners=False
                ).flatten(1)                                  # (B, fpn_out[1])
                zq_in = torch.cat([vae_condition, olocal], dim=1)
            z_q = self.origin_depth_head_g7(zq_in)
```

No loss changes: ẑ_q keeps training only through the composed 3D origin
loss (revolute rows; prismatic fully masked, as today). The remaining
point/origin asymmetry is supervision-side only.

## B · Prismatic sweep length 0.7 m (`trans_traj_length_m`)

**Measured basis (2026-08-16, gen-9 split, 13,297 rot rows):** 90° arc
length mean 0.731 m, **median 0.699 m**, p10–p90 0.392–1.115 m. The current
trans sweep (0.1 m) is ~7× shorter, so with 20 points the per-point
supervision energy on trans rows is ~50× smaller than on rot rows even
though 77.5% of rows are trans. User decision: set trans length to
**0.7 m** ≈ the revolute median — per-point spacing then matches the median
arc exactly.

**Mechanism — reader-side rescale, NO LMDB rewrite.** The stored trans
trajectory is a straight ray from the annotated motion origin
(`traj[0] == origin` by construction, `tools/sf3d_process.py:585`). New
`SF3DDataset` param `trans_traj_length_m: float = 0.0` (0 = off, exact
current behavior). When > 0, for trans rows only, applied after the
20-point subsample:

1. `L_cur` = polyline length of the subsampled trajectory.
2. **Only rescale the standard synthesis:** if `0.05 < L_cur < 0.15`
   (the 0.1 m rays; degenerate 0.01 m fallback segments and any legacy
   oddities are left untouched), set `s = trans_traj_length_m / L_cur`
   and `traj = traj[0] + s · (traj − traj[0])`.
3. `traj[0]` is the pivot → the 3D interaction-point target and the 2D
   point target (`trajectory_2d[0]`, its projection) are unchanged.
4. **Reproject the 2D track:** the stored `trajectory_2d_image_coords` is
   the projection of the OLD ray; when the record has
   `camera_intrinsics` + `image_dimensions_wh`, recompute the 20-point 2D
   track and its valid flags from the rescaled 3D points with the same
   convention as the preprocessor's `project_trajectory_to_2d` (z > eps,
   in-bounds check). Records without intrinsics keep the stored track
   (2D-head training is off in all current arms; the track is viz/aux).
5. Key-cache neutral: the option filters nothing, so cache files and
   their validation dicts are untouched. The edge filter reads
   `trajectory_2d[0]`, which the pivot preserves.

Passthrough: `SF3DDataModule`, and `--trans-traj-length` on
`tools/sf3d_vis_predictions.py` and `tools/diag_lpp_samples.py` (viz and
probe must see the same GT the trainer sees).

**Metric consequences (accepted):** trajectory-MSE and per-point-error
values on trans rows change scale — gen-11 trajectory losses/metrics are a
FRESH baseline vs gens 8–10 on trans rows. Direction metrics (traj_dir)
remain comparable. L_pp is unaffected in kind (its line branch is
normalized by trajectory energy since gen-10; the energy rises, which is
the point — trans rows stop being near-degenerate).

## Gen-11 run

`config/sf3d_train_runpod_g11_closeup010.yaml` = gen-10 config plus ONLY:

```yaml
model_params:
  use_origin_local_feature: true
data:
  trans_traj_length_m: 0.7
```

and experiment paths → `experiments/20260816_sf3d_g11_closeup010`. Same
split, 30 epochs, milestones [24, 28], seed 42, frozen CLIP, RTX PRO 4500,
launched with `train_pod.sh launch train-g11 … train_SF3D_better.py`.

## Success criteria & evaluation

1. Standard test pass (same 5,088 samples): primary watch =
   `origin_err_m` / `origin_line_err_m` (the local sample's target;
   gen-10: 0.369 / 0.328) and trans-row trajectory quality via traj_dir
   (gen-10: 91.5%). Type/axis/point/mask must hold (±2 pts band).
2. Consistency probe (ref-512): normalized L_pp should stay in gen-10's
   band (~0.054) — the longer trans sweeps make the line branch's
   denominator larger, not a free pass.
3. Viz batch (seed 42421) vs gen-10; check prismatic magenta sweeps are
   ~0.7 m scale and origin rings/axes on hinge seams.

## Code changes required

1. `model/segmenter.py`: flag + dim change + gated grid_sample (above),
   plus the `use_origin_heatmap` requirement check in `__init__`.
2. `config/opd_train.py`: `use_origin_local_feature: bool = False`.
3. `datasets/scenefun3d.py`: `trans_traj_length_m` param, rescale +
   reproject in `__getitem__` after the subsample; passthrough in
   `datasets/scenefun3d_datamodule.py`.
4. `tools/sf3d_vis_predictions.py`, `tools/diag_lpp_samples.py`:
   `--trans-traj-length` flag → dataset kwarg.
5. Config as above.
6. Tests: arm construction (origin head input dim, ValueError without
   heatmap); forward parity (flag off → ẑ_q identical to gen-10 path);
   rescale unit tests (length becomes 0.7, traj[0] fixed, degenerate
   0.01 m segment untouched, rot rows untouched, 2D reprojection matches
   manual projection); config test (only the two new knobs + paths differ
   from gen-10).

## Out of scope

- Any preprocessor/LMDB change (the rescale is reader-side).
- Trajectory-length prediction or per-object real travel ranges.
- The removed "point refinement" concept — never part of any plan.
