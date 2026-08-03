# Visualization index

One row per batch, newest first. Every batch is a dated directory
`YYYYMMDD_<subject>_<what>/` with a tracked `README.md` (and, for
tool-generated batches, an auto-written `manifest.yaml`); the images
themselves are gitignored but live on both sides of the mutagen mirror.
Conventions: `../CLAUDE.md` § Visualization organization.

| Batch | What | Source |
|---|---|---|
| [20260803_sf3d_twist_vs_2d_twist_panels](20260803_sf3d_twist_vs_2d_twist_panels/README.md) | GT vs twist vs 2d_twist prediction panels, 16 stratified val samples | `tools/sf3d_vis_predictions.py` |
| [20260727_sf3d_reproc_debug](20260727_sf3d_reproc_debug/README.md) | One-off debug renders from the sf3d_processed_v2 rebuild (splat-mask A/B, arc-fix checks, per-frame inspections) | ad-hoc scripts, July 27–28 |
| [20260727_sf3d_v1_dataset_audit](20260727_sf3d_v1_dataset_audit/README.md) | 100-sample audit of the v1 SF3D LMDB (frame/zoom/depth panels, contact sheet, geometry TSV) | `tools/sf3d_vis_samples.py` |
| [20260720_opd_label_audits](20260720_opd_label_audits/README.md) | OPD sample renders + description-regeneration label audits | `tools/show_opd_samples.py`, `tools/label_render.py` |
