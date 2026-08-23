# 20260824_sf3d_supabl2_art — supervision ablation v2, arm B: ARTICULATION ONLY

**Recipe:** the g21 recipe minus the trajectory path —
`use_trajectory_head: false`, `trajectory_weight: 0`, and L_pp + the dir
term die with it (`PredPredArticulationLoss.forward` requires
`outputs.trajectory_pred`, so this is forced, not chosen). The interaction
point (heatmap + coord + z_p lift + 3D loss vs `trajectory[:, 0]`) is
static grounding and stays, exactly as in the 2026-08-15 version of this
arm. Spec: `docs/superpowers/specs/2026-08-24-supervision-ablation-v2-design.md`.
30 epochs, best = **epoch 27 (val 0.7889)**. Val losses are NOT comparable
across arms — different term sets.

Trajectory metrics are correctly ABSENT (head removed), never zeroed.

## The comparison that matters: B vs D (test, 5,088)

Arm D (`20260824_sf3d_supabl2_nolpp`) is the joint arm with L_pp and the
dir term OFF. Neither B nor D has any consistency coupling, so B vs D
isolates **co-training alone** — the question the 2026-08-15 ablation
could not answer, because there the joint arm also carried L_pp.

| metric | B (articulation only) | D (joint, no L_pp) | effect of adding trajectory GT |
|---|---|---|---|
| MA | 20.40 | **28.22** | **+7.8** |
| axis matched (°) | 23.29 | **17.07** | **−6.2°** |
| axis all (°) | 29.20 | **25.30** | **−3.9°** |
| axis signed all (°) | 38.54 | **33.61** | −4.9° |
| axis flip rate ROT | 21.82 | **14.79** | **−7.0** |
| type acc | **93.85** | 93.63 | flat (−0.2) |
| origin q* (m) | 0.2900 | **0.2726** | −1.7 cm |
| radius err (m) | 0.1361 | **0.1201** | −1.6 cm |
| 3D point (m) | **0.2429** | 0.2459 | flat |
| 2D point | **0.0984** | 0.1008 | flat |
| mIoU | **0.2680** | 0.2650 | −0.003 (B better) |
| PDet | **22.60** | 21.95 | −0.65 (B better) |

## Reading

**Trajectory supervision helps articulation, and much more than in v1 —
but it moves a different metric.** Adding the trajectory head and its loss
is worth **+7.8 MA, −6.2° matched axis, and −7.0 points of rot sign-flip
rate**. The 2026-08-15 run found the effect in *type accuracy* (−3.0 when
trajectory was removed); here type is FLAT (93.85 vs 93.63) and the whole
effect lands on the **axis** instead. That is consistent with the stack
having changed underneath: gen-9 had one shared axis head and a weak type
head at 95.0, while g17's split rot/trans heads already read type at ~94–95
without help, leaving the axis as the thing a swept curve can still teach.

**The sign result is the striking one.** Removing the trajectory nearly
DOUBLES the rot-axis flip rate (14.79 → 21.82). The trajectory's time
ordering is the only signal in the whole loss set that distinguishes an
axis from its antipode — L_pp's locus residuals are sign-blind by
construction, and the GT axis loss is applied with
`axis_sign_agnostic: false` but evidently does not carry it alone. This
is direct evidence for why the sign problem exists, and it was obtained
with the dir term nowhere in the comparison.

**Masks prefer fewer tasks, replicating v1.** B beats D on mIoU (0.2680 vs
0.2650) and PDet (22.60 vs 21.95) — the same mild task-competition on the
shared decoder that the gen-9 ablation saw (its trajectory-only arm had
the best masks of the three). Small, consistent, and in the same direction
two generations apart.

## Run incident

Restarted once, deliberately: the volume quota was exhausted mid-run
(project-wide event, see STATE.md), so this arm was stopped at epoch 1
BEFORE its next checkpoint write could truncate, then resumed from its
intact epoch-0 `last.ckpt` with checkpoints redirected to pod-local
`/dev/shm/ckpt_art`. Zero volume checkpoint writes thereafter, zero
`Disk quota` errors in the log, and the run reached
`max_epochs=30` normally.

- `metrics.csv` is MERGED from `logs/csv/version_0` (epoch 0) and
  `logs/csv/version_1` (epochs 1–29).
- **The weights no longer exist** — pod-local checkpoints died with
  `segaffordance-abl-art` when it was deleted after this test pass. The
  metrics here and in `logs/test.log` are the durable artifact; re-running
  costs ~4 h / ~$8.
- Checkpoint sizes here are 4,337,038,877 B vs the other arms'
  4,348,918,xxx — that is NOT truncation, it is the genuinely absent
  TrajectoryMLP (~12 MB of params + Adam state).

test pass: `logs/test.log` on the volume (ckpt best-epoch27-valloss0.7889);
pod deleted immediately after.
