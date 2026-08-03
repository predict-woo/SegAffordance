# SF3D reprocessing debug renders (historical archive)

One-off renders from debugging the `sf3d_processed_v2` rebuild
(2026-07-27/28, ad-hoc scripts — kept as evidence, not regenerable):

- `mask_ab/`, `mask_ab_final/` — A/B of the point-projection masks vs the
  splat masks that shipped in v2
- `edge_overlay/` — mask/edge alignment checks on individual frames
- `gate_check/` — sensor-occlusion gating spot-checks
- `inspect_007/`, `inspect_015/`, `inspect_015_vid84/` — per-frame
  inspections of specific visits while chasing the revolute-arc bug
- `verify100/` — 100-sample verification pass over the finished v2 LMDB

Outcomes are recorded in `knowledge/` and `runpod/README.md` (v2 rebuild
notes); these images are the raw material behind those conclusions.
