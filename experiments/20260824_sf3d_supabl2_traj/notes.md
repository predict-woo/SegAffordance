# 20260824_sf3d_supabl2_traj — supervision ablation v2, arm C: TRAJECTORY ONLY

**Recipe:** the g21 recipe minus every articulation path — no type head,
no axis heads (`MotionMLP` skipped entirely), no origin heatmap channel
and no z_q lift (projector back to 2 channels, condition = `[features,
point_uv]`). Mask + interaction point + the 20-point DCT trajectory
remain. Two gates that did NOT exist in the gen-9 version of this arm had
to go off as well — `split_axis_heads: false` and
`use_origin_local_feature: false` — because both raise `ValueError` in
`CRIS.__init__` without their dependencies. Spec:
`docs/superpowers/specs/2026-08-24-supervision-ablation-v2-design.md`.
30 epochs, best = **epoch 20 (val 0.4926)**. Val losses are NOT comparable
across arms.

Articulation metrics are correctly ABSENT (heads removed), never zeroed.
`test/mean_origin_error_m` is present but is the LEGACY pseudo-origin
(point_uv + depth patch), not an origin-head output — same caveat as v1.

## The comparison that matters: C vs D (test, 5,088)

Arm D (`20260824_sf3d_supabl2_nolpp`) is the joint arm with L_pp and the
dir term OFF, so C vs D isolates **co-training alone** in the reverse
direction: does articulation supervision improve the trajectory?

| metric | C (trajectory only) | D (joint, no L_pp) | effect of adding articulation GT |
|---|---|---|---|
| traj_dir acc | 94.36 | 94.26 | **flat (−0.1)** |
| traj_dir cos | 0.7799 | **0.8003** | +0.020 |
| roughness (m) | 0.00898 | **0.00852** | flat |
| 3D point (m) | **0.2456** | 0.2459 | flat |
| 2D point | **0.0978** | 0.1008 | flat |
| mIoU | **0.2689** | 0.2650 | −0.004 (C better) |
| PDet | **22.82** | 21.95 | −0.87 (C better) |

## Reading — this is the non-replication

**v1's single largest effect is GONE.** The 2026-08-15 ablation found that
removing articulation supervision cost the trajectory **−5.5 points of
traj_dir accuracy** (93.10 → 87.58) and **−0.105 cosine** (0.736 → 0.631),
and called it "the largest single effect in the ablation." On the g21
stack the accuracy effect is **zero** (94.36 vs 94.26) and only a small
cosine gap survives (+0.020).

The most likely reason is that the trajectory head no longer needs the
help. Between gen-9 and now it gained the **normalized trajectory loss**
(g16, the rot-collapse fix) and the **truncated-DCT parameterization**
(g19, which makes jitter unrepresentable and drove roughness 10× down).
Both directly supply what articulation supervision used to contribute
indirectly — a well-scaled, smooth, correctly-oriented curve. Arm C hits
traj_dir 94.36 and roughness 0.0090 with NO knowledge of joint type, axis
or origin at all, which is at or above every joint arm before g19.

So the coupling is now **asymmetric**: the trajectory teaches articulation
a great deal (arm B: MA +7.8, matched axis −6.2°, rot flips −7.0), while
articulation teaches the trajectory almost nothing. In v1 it looked
mutual.

**Masks: best of all arms, replicating v1's task competition.** C leads on
mIoU (0.2689) and PDet (22.82), ahead of B (0.2680 / 22.60) and D (0.2650
/ 21.95). The gen-9 ablation found exactly the same ordering — its
trajectory-only arm had the best masks of the three. Two generations,
different backbone, different resolution, same effect: each extra head
costs the shared decoder a little mask quality.

## Run notes

Launched once and ran to `max_epochs=30` with no incident — the only arm
that needed no restart, because it started after the quota event and used
pod-local `/dev/shm/ckpt_traj` from the beginning. Single
`logs/csv/version_0`, so `metrics.csv` is unmerged.

- **The weights no longer exist** — pod-local checkpoints died with
  `segaffordance-abl-traj` when it was deleted after this test pass.
  Metrics here and in `logs/test.log` are the durable artifact;
  re-running costs ~4 h / ~$8.
- Checkpoint size 4,267,770,992 B, smaller than the other arms' — that is
  the genuinely absent MotionMLP, origin heads and third projector
  channel, not truncation.

test pass: `logs/test.log` on the volume (ckpt best-epoch20-valloss0.4926);
pod deleted immediately after — no pods remain.
