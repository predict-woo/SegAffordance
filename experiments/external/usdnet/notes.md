# External A/B — USDNet / Articulate3D (ICCV 2025) + closed-form screw loss

Spec: `docs/specs/2026-08-29_external_screw_loss_ab.md`. Scripts: `runpod/external/usdnet/`.
Upstream: github.com/insait-institute/USDNet @ 0ba303d (Mask3D + MinkowskiEngine,
torch 2.1.1/cu121 per their Dockerfile → needs a pre-Blackwell GPU: A100/H100).

## Infra
- Volume: `yde8krkk9k` (`ext-usdnet`, 120 GB, EU-RO-1).
- CPU staging pod `ext-usdnet-cpu` (`rzmcijxk9166lo`) running `stage_data.sh`:
  repo clone, `processed.zip` (5.8 GB, gdown), trained ckpts (Drive folder
  1qv2hTF8_U7nM1tAz1hltO1EGggstVaeR: `mov_trainval.ckpt`, `inter_trainval.ckpt`),
  Mask3D scannet200 backbone ckpt.
- GPU pod: `ext-usdnet-gpu` (`cnpgq5bxgzl1pr`), **A100 80GB PCIe, $1.39/hr**, image
  `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04` — landed 22:50 CEST after
  ~70 min of retries. CPU staging pod deleted after staging finished.
- Staged: `processed/articulate3d_challenge_mov` (train 390 / validation 84 / test 42
  files, `instance_gt`, `expand_dict`), `mov_trainval.ckpt`, `inter_trainval.ckpt`,
  `scannet200_benchmark.ckpt`.

## Log
- 2026-08-29 21:45: staging launched. 22:50: staging done; env build (`setup_env.sh`,
  MinkowskiEngine compile) launched on the A100 pod.
- Caveat to carry into results: `mov_trainval.ckpt` was trained on train+validation;
  the test split ships without articulation GT (`load_articulation` is forced off in
  test mode), so the A/B fine-tunes on `train` and evaluates on `validation` — both
  arms share the same contaminated init; report deltas, not absolutes.

## Calibration (smoke, 150 steps, 'ours' code path, λ=1)
- SCREW_STATS over 7,000 matched rotation targets (all decoder layers):
  mean H1 term 12.45 (dimensionless, normalised by |r*|²) vs mean of their
  symmetric line-distance term 0.99 m → **λ = 0.08** equalises the two at step 0.
- Env: torch 2.1.1+cu121, MinkowskiEngine 0.5.4 built with the Dockerfile's thrust
  patches (arch 8.0;9.0), pytorch-lightning 1.7.2, hydra 1.0.5, numpy 1.26.4
  (volumentations pulled numpy 2 once — repinned). Loss tests 20/20 on pod.
- Throughput: ~1.05 it/s at batch 1 (cropped 5.5 m) → ~6 min/epoch on train (390).

## A/B launched 2026-08-29 23:55 CEST
`pipeline.sh`: theirs (loss.screw_weight=0) → ours (screw_weight=0.08, mode
replace, term h1, min_radius 0.1 m; their 1−cos axis term kept in both) → eval
init / theirs / ours on validation (84 files) with their MA/MO/MAO AP50.
Both arms: init `mov_trainval.ckpt`, data.train_mode=train, 20 epochs, AdamW
lr 2e-5 with OneCycle (their default scheduler), batch 1, seed default, val every 10.

## Results (validation split, 84 files; AP₅₀ from their evaluator)

| arm | epoch | M | MA (+axis 15°) | MO (+origin 0.25 m) | MAO (+both) | MAO_ST | rot MA/MO/MAO | trans MA |
|---|---|---|---|---|---|---|---|---|
| **init** (`mov_trainval.ckpt`, no training) | 0 | 0.588 | 0.465 | 0.472 | 0.070 | 0.345 | rot AP50 0.702 | trans 0.474 |
| theirs | 10 | 0.568 | 0.458 | 0.463 | 0.117 | 0.350 | 0.643 / 0.526 / 0.158 | 0.272 |
| theirs | 20 (final) | 0.573 | 0.453 | 0.468 | 0.093 | 0.345 | — | — |
| ours v1 (replace, λ=0.08, H1 uncapped) | 10 | **0.507** | **0.361** | **0.309** | **0.075** | 0.212 | rot AP50 0.685 | trans 0.329 |
| ours v1 | 20 (final) | 0.548 | 0.383 | 0.329 | 0.066 | 0.221 | — | — |
| ours v2 (add, cap 4, λ=0.05) | 10 | 0.508 | 0.396 | 0.399 | 0.073 | 0.285 | rot AP50 0.699 | trans 0.316 |
| ours v2 | 20 (final) | 0.530 | 0.419 | 0.405 | 0.081 | 0.302 | — | — |
| **theirs, seed=1** | 10 | 0.513 | 0.422 | 0.393 | 0.062 | 0.299 | — | — |
| **theirs, seed=1** | 20 (final) | 0.552 | 0.439 | 0.446 | 0.094 | 0.334 | — | — |
| **ours v2, seed=1** | 10 | 0.537 | 0.413 | 0.405 | 0.084 | 0.283 | — | — |
| **ours v2, seed=1** | 20 (final) | 0.532 | 0.416 | 0.423 | 0.100 | 0.310 | — | — |

(epoch = 195 iters ≈ 3 min; both arms 20 epochs; init eval + final evals pending)
- 'theirs' arm: 62 min wall-clock. Note MAO drifts 0.117 → 0.093 over the last 10 epochs
  while M/MO tick up — continued fine-tuning with their loss does not improve the joint
  metric on validation (init was trained on train+val, so this is near a fixed point).
- 'ours' arm running (launched 22:47 UTC); SCREW_STATS at epoch 2: mean H1 15.2 vs
  line-dist 1.07 (n=26k) — ratio ~14, λ=0.08 keeps the contribution comparable.
- **v1 verdict at epoch 10: negative, and not only on joints** — M (pure segmentation)
  fell 0.568 → 0.507, so the term is perturbing the shared trunk. Diagnosis: the H1
  quadratic is a *relative* error normalised by |r*|²; on real scenes lever arms are
  0.1–1 m and early-decoder-layer origins are metres off → per-instance values of
  10²–10³ (mean 15, flip costs only 2.0) dominate gradients. In SegAffordance the
  scene/object frame is bounded and the head is separate; here it isn't.
- v1 final (epoch 20): recovers some segmentation (M 0.548) but joints stay far below
  theirs (MA −0.07, MO −0.14, MAO −0.03). SCREW_STATS end: mean H1 12.4 vs line-dist
  1.13 over 180k instances. **v1 = negative result** for the naive replace-mode slot-in.
- The standalone `eval` runs (`results/eval_*.txt`) hit the 42-file test split which has
  no articulation GT (their test_mode override is ignored) — ignore them; the numbers
  above come from the identical evaluator run on validation during training.
- Launched pipeline2 (00:25 CEST 2026-08-30): init-ckpt validation via a zero-LR
  1-step run, then v2.
- v2 (running, launched 00:02 UTC): mode=add, cap=4, λ=0.05. SCREW_STATS at epoch 2:
  capped H1 mean 1.28 vs line-dist 1.02 (n=22k) — the cap removes the tail as intended.
- **Confound found**: `general.seed: null` in their config → every arm ran with a
  different random seed (data order, crops, dropout). The M drop in both 'ours' arms
  could be partly seed noise on an 84-file validation set. Queued `pipeline3.sh`: a
  seed-matched pair (`general.seed=1`) theirs_s1 vs ours_v2_s1, 20 epochs each.
- v2 final: SCREW_STATS end mean capped-H1 1.32 vs line-dist 1.07 (n=178k). v2 > v1 on
  every column but still < theirs by 4–6 AP points on M/MA/MO and ~1 on MAO. Whether
  that gap is the loss or the seed is what pipeline3 answers.
- **Seed variance is large**: theirs@seed1 ep10 = 0.513/0.422/0.393/0.062 vs the
  unseeded theirs ep10 = 0.568/0.458/0.463/0.117 — a 5–7 AP swing from the seed alone,
  the same size as the 'ours vs theirs' gaps above. Those gaps are therefore NOT
  attributable to the loss; only the seed-matched pair (theirs_s1 vs ours_v2_s1) is a
  valid comparison, and even that is one seed on 84 scenes.
## Follow-up v3 (user-requested, 2026-08-30 15:00 CEST): add-mode with L2 + H1
All of their losses kept; added λ·(position quadratic + H1 quadratic), λ=0.05, seed 1
(matched against `ft_theirs_s1`). Two arms: per-term cap 4 (`ours_both_s1`) and uncapped
(`ours_both_nocap_s1`). New pod `ext-usdnet-gpu` p8httahvk60ce5 (A100 PCIe, $1.39/hr);
env rebuilt from `setup_env.sh` (container was lost with the previous pod).
Step-0 magnitudes: capped (L2+H1) mean 2.85 vs their line-dist 1.20 (n=21k) → the added
term is ≈12% of their origin term at λ=0.05.

| arm (seed 1) | epoch | M | MA | MO | MAO | MAO_ST |
|---|---|---|---|---|---|---|
| theirs_s1 | 10 | 0.513 | 0.422 | 0.393 | 0.062 | 0.299 |
| theirs_s1 | 20 | 0.552 | 0.439 | 0.446 | 0.094 | 0.334 |
| ours_v2_s1 (add, H1, cap 4) | 10 | 0.537 | 0.413 | 0.405 | 0.084 | 0.283 |
| ours_v2_s1 | 20 | 0.532 | 0.416 | 0.423 | 0.100 | 0.310 |
| **ours_both_s1 (add, L2+H1, cap 4)** | 10 | 0.540 | 0.436 | 0.412 | 0.090 | 0.310 |
| **ours_both_s1** | 20 | 0.551 | 0.437 | 0.431 | 0.095 | 0.321 |
| ours_both_nocap_s1 (add, L2+H1, uncapped) | 10 | 0.551 | 0.402 | 0.422 | **0.145** | 0.300 |
| ours_both_nocap_s1 | 20 | 0.545 | **0.350** | 0.424 | **0.131** | 0.274 |

v3 final vs theirs_s1: M −0.001, MA −0.002, MO −0.015, MAO +0.001 (rot-only AP50 0.724 vs
0.693, trans-only 0.378 vs 0.410). The epoch-10 lead (+0.014…+0.028) was a transient —
theirs climbs in the second half — so v3 is **null** too. End SCREW_STATS: capped L2+H1
mean 2.65 vs line-dist 1.08 (n=180k). Uncapped variant final: **mixed** — +both (MAO) 0.131 vs theirs 0.094 (+0.037, the largest
positive delta of the whole campaign) and MAO 0.145 at epoch 10, but +axis (MA) 0.350 vs
0.439 (−0.089, the largest negative), MO ≈ −0.02, M −0.007; rot-only AP50 0.678 vs 0.693,
trans-only 0.413 vs 0.410. Uncapped term mean 24.4 vs line-dist 1.05 (n=180k) → at λ=0.05
it contributes ≈1.2 per instance, i.e. it DOMINATES their origin term (unlike the capped
arm at ≈0.14). Reading: the strong, heavy-tailed quadratic pulls origins toward the GT
lever geometry hard enough to move the joint gate, at the cost of axis-only accuracy —
an axis/origin trade, not a clean win, on one seed. Worth a second seed only if the
MAO gain is the metric that matters to the user.

Protocol caveat (user, 2026-08-30 17:40 CEST): all numbers above are FINAL-epoch, with
validation only at epochs 10 and 20 — no best-epoch selection was possible. Queued
`pipeline5.sh`: rerun of the seed-1 pair (theirs vs add L2+H1 cap 4) with validation
every 2 epochs and all checkpoints kept; report final AND best-epoch (`best_epoch.py`).
Limitation: their evaluator logs only aggregate AP per pass, so selection and reporting
share the validation set (optimistic for both arms equally); a cross-column proxy
(select on M, report MAO and vice versa) is included.
**Cancelled by the user at 18:05 CEST** (the selection-on-reporting-set issue makes it
uninformative); only the uncapped L2+H1 arm remains to finish.

## Follow-up v4 (user-requested, 2026-08-30 19:20 CEST): add-mode, uncapped L2 only
All their losses kept; added 0.05 × position quadratic (uncapped), seed 1, vs `ft_theirs_s1`.
Arm `ours_pos_nocap_s1`. New pod `ext-usdnet-gpu` o4mobkzarix9dc (A100), env rebuilt.
Step-0: uncapped L2 mean 16.9 vs line-dist 1.17 → ≈0.85/instance at λ=0.05.

| arm (seed 1) | epoch | M | MA | MO | MAO | MAO_ST |
|---|---|---|---|---|---|---|
| theirs_s1 | 10 / 20 | 0.513 / 0.552 | 0.422 / 0.439 | 0.393 / 0.446 | 0.062 / 0.094 | 0.299 / 0.334 |
| ours_both_nocap_s1 | 10 / 20 | 0.551 / 0.545 | 0.402 / 0.350 | 0.422 / 0.424 | 0.145 / 0.131 | 0.300 / 0.274 |
| **ours_pos_nocap_s1** | 10 / 20 | 0.533 / 0.547 | 0.432 / 0.432 | 0.405 / 0.419 | **0.159 / 0.144** | 0.316 / 0.321 |

v4 final vs theirs_s1: M −0.005, MA −0.007, MO −0.027, **MAO +0.050** (rot-only AP50 0.704 vs
0.693, trans-only 0.389 vs 0.410). End SCREW_STATS: uncapped L2 mean 13.0 vs line-dist 1.04
(n=180k) → ≈0.65/instance at λ=0.05. This is the strongest +both result of the campaign
(0.144; init 0.070, theirs 0.094, L2+H1 uncapped 0.131) and, unlike L2+H1 uncapped, it
keeps the axis gate essentially intact (−0.007 vs −0.089). Pattern across the uncapped
arms: the position quadratic drives the combined gate; the H1 term is what costs axis
accuracy on this model. One seed, final epoch, validation = reporting set — needs a
second seed before it is a claim.

Interpretation (agreed with the sibling session, 2026-08-30 21:30 CEST): the H1-vs-L2
inversion relative to SF3D is coherent with the internal closed-form 2×2, which showed the
terms are context-dependent stabilizers. USDNet already carries a strong 1−cos axis loss
(its native supervision IS our anchor) and a sign-blind metric, so the only headroom is on
the origin — where L2's radial Gram weight acts — while H1's axis pressure can only fight
their existing axis term (MA −0.089 on the uncapped L2+H1 arm). One line: the right term to
export depends on which constraint the host model is missing.

## Verdict (2026-08-30 03:30 CEST): NULL / slightly negative

Seed-matched pair (seed 1, 20 epochs, identical everything but the loss):
theirs 0.552 / 0.439 / 0.446 / 0.094 vs ours-v2 0.532 / 0.416 / 0.423 / 0.100
(M / MA / MO / MAO AP₅₀). MAO +0.006, M/MA/MO −0.02 — all inside the ~5-point
seed swing observed between the two 'theirs' seeds. No evidence that adding a
capped H1 screw term (λ=0.05) to USDNet's articulation loss improves its
articulation-gated AP; the naive replace-mode slot-in (v1) is clearly harmful.

Why it does not transfer (hypotheses, not tested):
1. USDNet's origin is a per-point *vote* (mean of point + offset over the mask) and
   its axis a mean of point-wise + query predictions; the lever-arm quadratic
   couples these through the GT part centroid, but the gradient lands on
   per-point offsets whose mean already gets a line-distance gradient — little new
   signal, more variance.
2. Real-scene scale: lever arms 0.1–1 m with early-decoder origins metres off →
   relative errors 10²–10³ (mean 12–15) — the loss only behaves after capping,
   at which point it is close to a constant for most rows.
3. The gate metric (15°, 0.25 m) is coarse; sign errors (what the H1 term uniquely
   penalises) are invisible to it, and their eval uses |cos|.
4. One seed, 84 validation scenes, contaminated init (trainval) — low power.

Cost: A100 PCIe $1.39/hr × ~7 h ≈ $10 + CPU staging. Pod deleted; volume
`yde8krkk9k` kept (120 GB: data, all checkpoints, runs).
Bundle: `bundle/` (metrics_*.csv per arm, patch.diff, pipeline logs, screw stats).

- Reading so far (superseded by the seed finding): fine-tuning with THEIR loss moves MAO 0.070 → 0.117 (ep10) → 0.093
  (ep20) with M 0.588 → 0.573; v1 ours is below init on every column. mode=add (their line-distance term kept), H1 capped
  per instance at 4.0 (2× the flip value — preserves sign/lever information, removes the
  tail), λ=0.05.


## Volumes deleted (2026-08-31)
All campaign volumes removed at the user's direction; checkpoints/data existed only there. Results survive in bundle/.
