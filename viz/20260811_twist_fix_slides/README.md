# 20260811_twist_fix_slides

Two supervisor-facing 16:9 slides summarizing the gen-3 twist/trajectory
diagnosis and the gen-4 fixes. All numbers are measured: clip_g3 checkpoint
diagnostics (`tools/diag_twist_radius.py`, 800 stratified val samples), the
closed-form mixture study, the loss-pricing measurements in the 2026-08-11
specs, and the gen-4 smoke run.

- `slide1_problem.png` — MSE mode-averaging: radius-vs-P(rot) mixture curve
  + the three measured symptoms (|omega| 0.295, radius 10x one-sided,
  trajectory at the zero-motion baseline).
- `slide2_fixes.png` — the body-frame kinetic-energy metric (pricing bars)
  and the K=4 winner-takes-all bundle head.
- `twist_fix_slides.pdf` — both slides, one file for sharing.

Regenerate: session scratchpad `make_slides.py` (numbers hardcoded from the
measurements above; specs in `docs/superpowers/specs/2026-08-11-*.md` are
the source of truth).
