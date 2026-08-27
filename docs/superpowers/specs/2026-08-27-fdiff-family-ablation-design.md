# fdiff-family ablation + mechanism transfer — design

**Date:** 2026-08-27. **Commissioned:** user ("see if the ablation
experiments we ran for the dct heads transfer over to the fdiff versions
… first see if for the fdiff family, jointly training is better than
either path alone, and … the 'Why does a redundant label help?' answer
for the fdiff family").

## Reused corners (no retraining)

- **Articulation-only** = supabl2 arm B (MA 20.4, mIoU 0.268) — shared
  across families by construction: fdiff losses die with the trajectory
  head, exactly like L_pp did.
- **Joint, fdiff, no consistency confounds** = `20260825_sf3d_fdiff_nolpp`
  (MA 27.5, matched 15.1°, mIoU 0.246).
- Reference points: `analytic_decode` (decode without fdiff, MA 26.5) and
  the DCT-family grid (arm C masks 0.2689, arm D MA 28.2).

## New arms (both zero-code-risk: one trainer extension, two configs)

1. **C_f — trajectory-only + fdiff** (`20260827_sf3d_supabl3_traj_fdiff`):
   supabl2_traj's head-removal gates verbatim, DCT→plain head, + gen-19
   fdiff weights (1.0/0.5/0.5). Answers joint-vs-either and whether the
   DCT family's mask ordering (C > B > joint) and the
   articulation→trajectory null transfer to fdiff.
2. **decode+fdiff** (`20260827_sf3d_andec_fdiff`): the analytic_decode
   arm + the fdiff losses applied to the DECODED curve (new
   fdiff-on-decode block in the trainer's analytic section — same knobs
   and conventions as the head fdiff block; mutually exclusive by
   construction since one requires trajectory_pred None, the other
   present). Zero parameters vs arm B. Answers the mechanism split for
   the fdiff family: fraction of the B→fdnolpp gap (7.1 MA) recovered by
   loss geometry alone, now including the fdiff geometry.

## Readouts

- Joint-vs-either (MA and the articulation columns): fdnolpp vs B vs C_f.
- Mechanism split: (andec_fdiff − B) / (fdnolpp − B) on MA, matched,
  flips; masks expected BELOW B again if the DCT-family finding holds.
- Mask ordering C_f vs B vs fdnolpp; traj metrics C_f vs fdnolpp
  (articulation→trajectory coupling on fdiff).

## Execution

Standard: smoke fast_dev_run on dev pod (new trainer block) → two PRO
6000 pods (pollers, verify GPU + clocks) → 30 fixed epochs → test pass →
delete pods → wraps. Cost ≈ $16–25 depending on host quality.
