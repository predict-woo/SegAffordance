# Gen-17: Split Articulation Axis Heads (rot / trans readouts)

**Date:** 2026-08-18
**Status:** APPROVED (user, 2026-08-18) — implementation next
**Companion:** docs/slides/2026-08-18_gen17_split_axis_heads.html
**Baseline:** g16 (20260817_sf3d_g16_trajnorm, best-epoch21) — including its
new sign-aware axis columns: signed err 33.91°, flip rate 10.18% all /
12.73% rot, MA_signed 22.92.

## Motivation

One `MotionMLP.motion_head` 3-vector currently serves as BOTH the revolute
hinge axis (⊥ to the motion) and the prismatic slide direction (= the
motion), with no type conditioning (the GT hint was removed in gen-7).
Under type ambiguity (documented drawer/door conflicts) the cos-optimal
single output is a blend of the two type-conditional answers — wrong for
both — and even on unambiguous rows the two types' gradient statistics
(vertical hinge axes vs horizontal slide directions) fight over the same
771 readout weights. Investigated alternative (sigmoid octant constraint)
was REFUTED 2026-08-18 (segmenter rescales to (−1,1)); the sigmoid+rescale
form is kept unchanged.

## Change: `split_axis_heads` (ModelParams, default False)

Flag-gated so every existing checkpoint keeps loading byte-identically.

1. **`model/layers.py` — MotionMLP** gains `split_axis_heads: bool = False`.
   - False: exact legacy module (`motion_head`, same state-dict keys).
   - True: `motion_head` is replaced by `motion_head_rot` and
     `motion_head_trans` (each `Sequential(Linear(hidden, 3), Sigmoid())`,
     the legacy form). Forward returns
     `(motion_rot, motion_trans, type_logits)` in split mode and
     `(motion_pred, type_logits)` otherwise (call sites gate on the flag).
   - `with_motion_head=False` + split → ValueError (no axis to split).

2. **`model/segmenter.py`** — in the `motion_mlp` branch, split mode:
   - rescale BOTH candidates with the existing `(x − 0.5) * 2.0`;
   - row-wise select `motion_pred` by **predicted** type argmax:
     `stack([trans, rot], dim=1)[arange(B), type_logits.argmax(-1)]`
     (index 1 = rot, matching `motion_type_gt == 1`); done model-side so
     every downstream consumer (metrics, viz, probes, radius metric) works
     unchanged;
   - `ModelOutputs` gains `motion_pred_rot=None` / `motion_pred_trans=None`
     (`model/outputs.py`), populated only in split mode.
   - Guards (ValueError at `__init__`): `split_axis_heads` requires
     `use_motion_head` AND `use_motion_type_head` (selection needs type
     logits); incompatible with `use_cvae` and `use_twist_head`.

3. **`train_OPDReal_better.py` — axis loss GT routing.** When
   `outputs.motion_pred_rot is not None`:
   ```python
   pred_stack = torch.stack(
       [outputs.motion_pred_trans, outputs.motion_pred_rot], dim=1)  # (B,2,3)
   routed = pred_stack[torch.arange(B, device=...), motion_type_gt.long()]
   L_motion = axis_direction_loss(routed, motion_gt, sign_agnostic=...)
   ```
   Gather → each row's gradient reaches ONLY its GT-type head (the trunk
   still learns from every row). No data-dependent control flow
   (compile-safe). Legacy path untouched when the fields are None.

4. **`model/losses/geometric.py` — L_pp branch pairing.** In
   `PredPredArticulationLoss`, when `motion_pred_rot`/`motion_pred_trans`
   are present: the LINE branch normalizes `motion_pred_trans` for its
   axis, the CIRCLE branch `motion_pred_rot`. p_rev weighting, energy
   masks, normalization — all unchanged. Legacy: both branches keep
   reading `motion_pred`.

5. **`config/opd_train.py`**: `split_axis_heads: bool = False`.

## What does NOT change

- Trajectory head: stays SHARED (explicit user decision).
- Origin path: already revolute-only by loss masking — gen-17 makes the
  axis follow the same pattern (GT-routed at train, predicted-type-selected
  at test), it does not touch the origin modules.
- Type head, point path, masks, backbone (dinov3 stays frozen), sigmoid +
  rescale form, all loss weights.

## Gen-17 run

`config/sf3d_train_runpod_g17_splitax.yaml` = **g16** config plus ONLY
`split_axis_heads: true` and paths → `experiments/20260818_sf3d_g17_splitax`.
From scratch, same seed/epochs/data (v3, 512, frames_512.lmdb).

## Success criteria

1. **Flip rate down** (the headline): rot flip rate < 12.73%; signed axis
   err (all) < 33.91°.
2. **Matched axis / MA improve or hold:** matched ≤ 17.8°, MA ≥ 23.1
   (blend tax removed on correctly-typed rows).
3. **Type accuracy flat** (~92.3): the trunk gradient mix changed slightly;
   a drop > ~1 pt means the split hurt the selector.
4. **Everything else flat:** mIoU ~0.264, PDet ~21.4, 2D pt ~0.099,
   traj_dir ~94.5/0.798 — nothing upstream changed.
5. Standard wrap: test pass (now includes sign-aware columns), probe, viz
   batch, notes/INDEX.

## Tests

- Flag off: MotionMLP state-dict keys and forward outputs byte-identical
  to legacy (regression).
- Split: state dict has `motion_head_rot.*`/`motion_head_trans.*`, no
  `motion_head.*`.
- GT routing: a batch with pure-rot rows backwards gradient ONLY into
  `motion_head_rot` (trans head grads exactly zero), and vice versa.
- Selection: `outputs.motion_pred[i]` equals the rot candidate exactly
  where `type_logits.argmax == 1`, trans candidate elsewhere.
- L_pp pairing: with artificially distinct candidates, the line residual
  matches a manual compute with the trans axis and the circle residual
  with the rot axis; legacy outputs (fields None) reproduce old values.
- Guards: ValueError for split without motion head, without type head,
  with use_cvae, with use_twist_head.
- Config chain: g17 == g16 + `split_axis_heads` + paths.
