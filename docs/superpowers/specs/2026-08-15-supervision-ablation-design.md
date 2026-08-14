# Supervision Ablation: Joint vs Trajectory-only vs Articulation-only (gen-9 arms)

**Date:** 2026-08-15
**Status:** IMPLEMENTED (commits cab5b18, 81e9ba7; user approved 2026-08-15)
**Question:** Does jointly training on trajectory and articulation supervision
improve performance over either supervision alone?

## Experiment design

Three arms, identical in everything except which supervision (and the heads
that carry it) is present. Arm A is the gen-9 run currently training — it is
reused, not re-run.

| | A: joint (gen-9) | B: articulation-only | C: trajectory-only |
|---|---|---|---|
| Mask (DiceBCE) | ✓ | ✓ | ✓ |
| Interaction point: heatmap + coord + z_p lift + 3D loss | ✓ | ✓ | ✓ |
| Origin: heatmap + z_q lift + canonical-q* loss + map BCE | ✓ | ✓ | — |
| Type CE | ✓ | ✓ | — |
| Axis direction (1−cos) | ✓ | ✓ | — |
| Trajectory head + per-point loss (relative direct readout) | ✓ | — | ✓ |
| L_pp consistency (pred-pred) | ✓ | — | — |

Decisions locked with the user (2026-08-15):

1. **The interaction point is static grounding, not "trajectory data".**
   All arms keep the full point pipeline (heatmap, coord, z_p, 3D loss vs
   GT `trajectory[:, 0]`). Only the 20-point motion trajectory counts as
   trajectory supervision.
2. **Arm C removes articulation from the architecture entirely** — no
   type head, no axis head, no origin heatmap channel, no z_q head, and
   `origin_uv` leaves the condition vector (condition = [features,
   point_uv]). Not "heads present but unsupervised".
3. **Evaluation: one standard test pass per arm** on the identical test
   split; the comparison table is assembled from the three metric CSVs.
   Metrics for absent heads are simply absent.

### What each comparison answers

- **A vs B** (on axis angle, type accuracy, origin errors): does trajectory
  supervision improve articulation estimation?
- **A vs C** (on traj_dir, per-point trajectory error): does articulation
  supervision improve trajectory prediction?
- **A vs B vs C** (on mIoU, 2D/3D point error): do either or both help the
  shared grounding heads?

**Attribution caveat (accepted):** arm A differs from B and C by *two*
mechanisms each — the extra supervision signal AND the L_pp consistency
coupling. If joint wins, these arms alone cannot separate co-training from
the consistency loss. A joint-minus-L_pp arm is a possible follow-up, not
part of this experiment.

## Constants across arms

Everything from `config/sf3d_train_runpod_g9_closeup010.yaml` that is not a
head/loss gate, verbatim:

- Data: sf3d_processed_v2, key cache
  `sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl` (59,174 records:
  sensor 0.5 + min_revolute_radius 0.10 + mask ≥0.1% + edge margin 5%),
  val_split_ratio 0.1, point_source "element", batch 128.
- Schedule: 30 epochs, lr 1e-5, milestones [24, 28], gamma 0.1, seed 42.
- Backbone: CLIP frozen, channels_last + compile.
- GPU class: RTX PRO 4500 ($0.72/hr); arms B and C run **sequentially on
  the existing g9 pod after gen-9 finishes** (no new pod creates). ~2.2 h /
  ~$1.6 each.

## Arm B: articulation-only

Config `config/sf3d_train_runpod_g9abl_artonly.yaml`, experiment dir
`experiments/20260815_sf3d_g9abl_artonly`. Deltas vs the g9 config:

```yaml
model_params:
  use_trajectory_head: false     # TrajectoryMLP not constructed
loss_params:
  trajectory_weight: 0.0         # belt-and-suspenders; loss already gates on None
  geometric_loss: "none"         # L_pp needs the predicted trajectory
  pred_pred_art_weight: 0.0
```

Model consequence: `trajectory_pred` is None end-to-end. The existing
trainer gate (`outputs.trajectory_pred is not None`) skips L_trajectory;
`geometric_loss: "none"` removes L_pp. GT trajectories are still loaded by
the dataset (they define the 3D point target `traj[:, 0]`) — the point is
supervised, the motion is not.

## Arm C: trajectory-only

Config `config/sf3d_train_runpod_g9abl_trajonly.yaml`, experiment dir
`experiments/20260815_sf3d_g9abl_trajonly`. Deltas vs the g9 config:

```yaml
model_params:
  use_motion_head: false         # no axis-direction head (MotionMLP skipped —
  use_motion_type_head: false    #   both sub-heads off ⇒ module is None)
  use_origin_heatmap: false      # projector back to 2 channels; no origin_uv,
                                 #   no z_q head, no origin lift
loss_params:
  vae_weight: 0.0                # axis loss (inert once motion_pred is None)
  motion_type_weight: 0.0
  origin_weight: 0.0
  origin_map_weight: 0.0
  geometric_loss: "none"         # L_pp needs type probs + axis + origin
  pred_pred_art_weight: 0.0
```

Model consequences: `motion_pred`, `motion_type_logits`, `origin_uv`,
`origin_logits`, `origin_pred` all None; condition vector = [features,
point_uv] (dims self-adjust; TrajectoryMLP and z_p input dims follow
`vae_condition_dim`). All four loss gates already exist in the trainer.

## Code changes required (beyond the two configs)

The flags all exist; the work is an **audit of unguarded consumers** so a
None head cannot crash train/val/test:

1. `train_OPDReal_better.py` validation-visualization path indexes
   `trajectory_pred[i]` unconditionally (~line 734) — guard it.
2. Sweep both trainers for indexing/`.to()` on `motion_pred`,
   `motion_type_logits`, `origin_uv`, `origin_logits`, `origin_pred`,
   `trajectory_pred` outside existing None-gates (test_step metric blocks in
   `train_SF3D_better.py` are mostly guarded already — verify each split
   metric: type acc, axis angle, origin_err/origin_line/radius, traj_dir,
   traj point errors).
3. `PredPredArticulationLoss` is never constructed in either arm
   (`geometric_loss: "none"`) — no internal changes.
4. `tools/sf3d_vis_predictions.py`: minimal guards so the split-arm branch
   skips absent overlays (axis/orbit/origin ring in arm C; trajectory dots
   in arm B) instead of crashing.
5. Tests: one forward-shape + loss-gating test per arm in the existing
   suite style (`_StubBackbone` monkeypatch): arm-B model returns
   `trajectory_pred is None` and training step runs; arm-C model returns
   None articulation fields, 2-channel projector output, and training step
   runs. Assert `loss_total` is finite in both.

Metrics logged as zeros for gated-off losses are acceptable (existing
zero-log convention); absent *metrics* (not zero) are required in the test
CSVs so the comparison table can't mistake "head missing" for "error = 0".
Where the current code would log a 0.0 metric for a missing head, skip the
log instead.

## Execution order

1. Gen-9 (arm A) finishes; run its normal wrap-up (eval, records).
   **Keep the pod.**
2. Launch arm B on the same pod (same launch pattern: LMDBs on /root/lmdb,
   24/16 workers, log under the arm's experiment dir). Monitor as usual.
3. Wrap up arm B (kill teardown hang, md5 last-vs-best, test eval, records).
4. Launch arm C, same procedure.
5. Wrap up arm C, then **delete the pod**.
6. Write the three-way comparison into both experiment notes and the INDEX:
   per-metric table (shared metrics across all arms; articulation metrics
   A vs B; trajectory metrics A vs C), with the L_pp attribution caveat
   stated.

## Implementation notes (2026-08-15, post-review)

- Spec item 4 (viz-tool guards) required NO code change: every consumer of
  the six optional head fields in `tools/sf3d_vis_predictions.py` was
  already None-guarded (verified by the final review, not assumed).
- `test/mean_origin_error_m` is the LEGACY pseudo-origin metric (point_uv +
  depth-patch unprojection) — it is computed identically in all arms and
  still appears in arm C's CSV. It is NOT an origin-head metric; the
  comparison table must not present it as arm C "origin" performance.
- Absent CSV column now means "no data", not only "head absent":
  `test/point_traj0_gap_m` is absent on relative-trajectory arms and
  `test/mean_origin_error_m` is absent when the test split has zero
  rotational samples.
- `val/loss_total` sums different term sets per arm — checkpoint selection
  is within-arm only; never compare loss values across arms.

## Out of scope

- Joint-minus-L_pp arm (only if the joint result is interesting).
- Any dataset change: all arms read the same records and the same key cache.
- 2D-pretraining interactions; full-split (non-close-up) replication.
