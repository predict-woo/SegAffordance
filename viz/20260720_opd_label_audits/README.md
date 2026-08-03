# OPD sample renders + description-regeneration audits (historical archive)

From the OPD description-regeneration work (2026-07-19..21):

- `sample_viz/` — OPDMulti sample renders from `tools/show_opd_samples.py`
  (image + annotation JSON pairs)
- `label_audit/` — before/after pairs auditing the Codex-regenerated
  `description` texts against the rendered annotations
- `label_viz/`, `label_viz2/`, `label_viz3/` — iterations of
  `tools/label_render.py` output while tuning the audit rendering

The regeneration itself is documented in `runpod/README.md` (style rules,
pipeline, `description_source` tag).
