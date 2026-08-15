# Gen-11: Origin Local Sample + Meaningful Prismatic Sweep Length

**Date:** 2026-08-16
**Status:** DRAFT — awaiting user review
**Goal:** One model change on top of the gen-10 recipe — the origin depth
head gets the same grid-sampled local feature the point depth head has
(path symmetry) — trained on **sf3d_processed_v3** (whose prismatic sweeps
are 0.7 m; the length change is a DATASET version, built and verified
2026-08-16, not part of this spec).

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

## B · (moved) Prismatic sweep length is dataset v3

Originally specced here as a reader-side rescale; user decision 2026-08-16:
it is a dataset version instead. `sf3d_processed_v3` was derived from v2 by
`tools/sf3d_build_v3.py` (trans trajectories recomputed as 0.7 m rays from
the stored per-frame origin/direction, 2D tracks reprojected, rot rows
byte-identical, frames/images/depth shared by symlink) and verified by
`tools/diag_verify_v3.py`: 458,265/458,265 entries, 200/200 rot
byte-identity and 200/200 trans checks sampled, reader smoke with the gen-9
key cache passing (caches remain valid — no filter input changed). 0.7 m =
the measured median 90° revolute arc length (mean 0.731, median 0.699 over
the gen-9 split, `tools/diag_arc_length.py`). No `trans_traj_length_m`
reader option exists or is planned.

Gen-11 (and any future SF3D run) points `data.train_data_dir` at
`/workspace/datasets/sf3d_processed_v3`. Metric consequences unchanged from
the original analysis: trajectory-magnitude losses/metrics on trans rows
are a FRESH baseline vs gens 8–10; direction metrics stay comparable.

## Gen-11 run

`config/sf3d_train_runpod_g11_closeup010.yaml` = gen-10 config plus ONLY:

```yaml
model_params:
  use_origin_local_feature: true
data:
  train_data_dir: "/workspace/datasets/sf3d_processed_v3"
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
3. Config as above (v3 data root; viz/probe runs pass the v3 root via
   their existing --data-root flags).
4. Tests: arm construction (origin head input dim, ValueError without
   heatmap); forward parity (flag off → ẑ_q identical to gen-10 path);
   config test (only the flag + data root + paths differ from gen-10).

## Out of scope

- Any preprocessor/LMDB change (the rescale is reader-side).
- Trajectory-length prediction or per-object real travel ranges.
- The removed "point refinement" concept — never part of any plan.
