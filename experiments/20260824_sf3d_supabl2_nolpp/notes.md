# 20260824_sf3d_supabl2_nolpp — supervision ablation v2, arm D: JOINT MINUS L_pp

**Recipe:** the g21 recipe (`sf3d_train_runpod_g21_dct_dir.yaml`) with
IDENTICAL supervision and the consistency coupling switched off —
`pred_pred_art_weight: 0.0`, `pred_pred_art_dir_weight: 0.0`.
`geometric_loss` deliberately STAYS `"pred_pred_art"` so the module is
still built and `L_geo_pred_pred_art` is still logged: the arm **measures**
head consistency every step while training on none of it. Spec:
`docs/superpowers/specs/2026-08-24-supervision-ablation-v2-design.md`.
30 epochs, best = **epoch 26 (val 0.9244)**.

This is the deconfounding arm the 2026-08-15 ablation named as "a possible
follow-up, not part of this experiment" and never ran.

## The headline: consistency does NOT emerge for free

`val/L_geo_pred_pred_art`, the same quantity in both runs — one trains on
it, one only watches it:

| arm | L_pp weight | epoch 0 | final |
|---|---|---|---|
| A₀ `20260821_sf3d_g19_dct` | 0.1 | 0.2337 | **0.1240** |
| **D** (this run) | **0** | 0.3820 | **0.3704** |

Arm D's value is FLAT across all 30 epochs (0.382 → 0.370, never below
0.37) while its losses fall normally (train 2.41 → 0.299). So the
trajectory, axis and origin heads do **not** drift into agreement on
their own just because each is supervised toward the same ground truth —
they agree only when L_pp makes them. That is the affirmative case for the
term's existence, and it could not be read off any previous run.

## A₀ vs D — the clean L_pp isolation (test, 5,088)

Both are g17 + DCT head, same split/schedule/seed, dir term absent from
BOTH. The ONLY difference is `pred_pred_art_weight` 0.1 vs 0. (This is a
better comparison than A-vs-D as originally specced: arm A =
`20260823_sf3d_g21_dct_dir` also carries the dir term, which the
concurrent session showed is net-negative — see the spec amendment.)

| metric | A₀ (L_pp on) | D (L_pp off) | L_pp's effect |
|---|---|---|---|
| MA | 25.98 | **28.22** | **−2.2** |
| axis matched (°) | 18.21 | **17.07** | −1.1 worse |
| axis all (°) | **25.29** | 25.30 | flat |
| type acc | **95.15** | 93.63 | **+1.5** |
| axis flip rate ROT | **13.30** | 14.79 | **+1.5** |
| 3D point (m) | 0.2592 | **0.2459** | −1.3 cm worse |
| origin q* (m) | 0.2764 | **0.2726** | flat |
| radius err (m) | 0.1410 | **0.1201** | **−2.1 cm worse** |
| mIoU | **0.2685** | 0.2650 | +0.004 |
| PDet | 21.72 | **21.95** | flat |
| traj_dir acc / cos | **94.52** / 0.7999 | 94.26 / **0.8003** | flat |
| roughness (m) | 0.0090 | **0.0085** | flat |

**Reading — L_pp is a trade, not a free win.** Turning it OFF *improves*
MA by 2.2 points, matched axis by 1.1°, 3D point by 1.3 cm and radius by
2.1 cm. What it buys is **type accuracy (+1.5) and sign stability (rot
flip rate 13.3 vs 14.8)** — the two places where making the heads agree
with each other actually pays. Masks, detection, trajectory direction and
smoothness are all indifferent to it.

That reframes the 2026-08-15 result: arm A there beat arms B and C while
carrying L_pp, and the natural reading was that consistency helped. On
this stack the consistency term is NOT what drives the joint arm's
articulation quality — if anything it costs MA — so whatever advantage
joint supervision has must come from co-training itself. Arms B and C
test exactly that (their comparisons run against D, not just A₀).

## Run incident — quota crash and resume

Died silently at **epoch 9** with no traceback: the volume quota was
exhausted (`OSError: [Errno 122] Disk quota exceeded`, surfaced on the
resume attempt's hparams write). Resumed from the intact `last.ckpt`
and completed epochs 9→29 with checkpoints redirected to POD-LOCAL
`/dev/shm/ckpt_nolpp`, so training made zero volume checkpoint writes.
Consequences to know when reading this run:

- `metrics.csv` here is MERGED from `logs/csv/version_0` (epochs 0–8,
  pre-crash) and `logs/csv/version_4` (epochs 9–29, the resume).
  Versions 1–3 are empty dirs from the failed resume attempts.
- Two superseded checkpoints (epoch06 val 1.0753, epoch07 val 1.0284)
  were deleted to reclaim quota; both were strictly worse than the
  epoch08 kept at the time, and `save_top_k: 3` would have pruned them.
- **The weights no longer exist.** Pod-local checkpoints died with
  `segaffordance-abl-nolpp` when it was deleted after this test pass.
  The metrics here and in `logs/test.log` are the durable artifact.
  Re-running the arm costs ~4 h / ~$8 if the weights are ever needed.
- No viz batch for this arm: a cross-arm panel would need all three
  arms' checkpoints on ONE pod, and the quota freeze made consolidating
  them on the volume impossible.

test pass: `logs/test.log` on the volume (ckpt best-epoch26-valloss0.9244);
pod deleted immediately after.
