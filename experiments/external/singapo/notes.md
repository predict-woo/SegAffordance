# External A/B — SINGAPO (ICLR 2025) + closed-form screw loss

Spec: `docs/specs/2026-08-29_external_screw_loss_ab.md`. Scripts: `runpod/external/singapo/`.
Upstream: github.com/3dlg-hcvc/singapo @ de4b466. Loss slot-in: x̂₀-space auxiliary
term on node token 2 (axis dir/origin) with p = AABB centre (detached), min-SNR
weighting γ=5, rows = valid non-fixed nodes with |r*| ≥ 0.05.
Headline "ours": H1 (1.0) + 1−cos anchor (0.5), position quadratic off; λ set to
match ε-MSE magnitude at step 0 (smoke).

## Infra
- Volume: `pye36wkz0m` (`ext-singapo-ro`, 250 GB, EU-RO-1). Pod: `ext-singapo-gpu`
  `u6zprqqu9uimzc`, RTX PRO 6000 Blackwell 96 GB, $2.09/hr (healthy: 2430 MHz / 600 W).
- EU-FR-1 had no H100/H200/CPU stock at 21:00 CEST 2026-08-29; empty leftover volume
  `86sqhsh4c5` (`ext-singapo`, 250 GB, EU-FR-1) — **user to delete**.

## Log
- 2026-08-29: files written; loss port passes 20/20 tests locally.
- 2026-08-29 21:10: pod up, `setup.sh` launched (repo @ de4b466, pip env, pm.zip 13 GB +
  augmented_train.zip 76 GB, ckpts). Log: `/workspace/logs/setup.log` on the pod.

- 2026-08-30 02:35 CEST: setup done — pm (13 GB) + augmented (76 GB) unpacked
  (125 GB on volume after zip removal; the MooseFS unzip of ~100k small files took
  ~4 h — the dominant cost of this experiment's setup), all 3,157 split ids present,
  released ckpt `ckpt_init/final/ckpts/last.ckpt` + CAGE init, loss tests 20/20 on pod.
- 02:40: smoke arms launched (30 steps each) for λ calibration.

- Smoke (30 steps, both arms OK; needed torchvision + `weights_only=False` for their
  OmegaConf-bearing ckpt). Step-0 magnitudes: ε-MSE 0.0013, fg 0.010 (w=0.01),
  screw@λ=1 0.065 (H1 0.060, anchor 0.010, ~813 valid rows/step). **λ = 0.05** chosen
  (screw ≈ 0.003: above MSE, below fg).
- 03:20 CEST 2026-08-30: pipeline launched — theirs → ours → eval init/theirs/ours
  (20 epochs each, constant LR 5e-5 adapter / 1e-5 base, seed 42 both arms).

- 04:40: eval path dry-run on the init ckpt — denoising 154 inputs × 5 samples = 12.6 min;
  outputs `exps/dryrun/.../{idx}@{Cat}@{id}/{0..4}/object.json`; retrieval + their metrics
  rerun without timeout (in progress). Their `limit_test_batches` isn't honoured, so every
  eval is the full test set.

## Results (PM test split: 77 objects × 2 views = 154 inputs, 5 samples each)

Axis metrics (`eval_axis.py`; rows = GT non-fixed joints, index-aligned; angles in deg,
origin = point-to-GT-axis line distance in the normalized [-1,1]³ object frame;
GT direction sign canonicalised the SINGAPO way; "avg" = mean over the 5 samples,
"best" = sample with lowest unsigned angle per input):

| ckpt | angle unsigned (avg / best) | angle signed | flip rate | origin line dist (avg / best) | type acc |
|---|---|---|---|---|---|
| init (released) | 12.38 / 4.37 | 12.38 | 6.7% / 2.9% | 0.312 / 0.241 | 0.854 / 0.939 |
| theirs, 20 ep | 12.94 / 7.27 | 12.94 | 4.7% / 3.7% | 0.311 / 0.276 | 0.849 / 0.907 |
| **ours, 20 ep** (H1 1.0 + anchor 0.5, λ=0.05) | 13.27 / 5.98 | 13.28 | 8.1% / 3.8% | 0.306 / 0.230 | 0.850 / 0.926 |

Their metrics (`scripts/eval_metrics.py` after mesh retrieval; lower is better for all;
"avg" over 5 samples / "best" of 5; paper's released-model row for reference):

| ckpt | RS-IoU | AS-IoU | RS-cDist | AS-cDist | RS-CD | AS-CD | AOR |
|---|---|---|---|---|---|---|---|
| paper (Table quant_pm) | 0.484 | 0.497 | 0.045 | 0.096 | 0.017 | 0.091 | 0.0043 |
| init (released), avg | 0.436 | 0.452 | 0.032 | 0.075 | 0.011 | 0.085 | 0.0030 |
| init, best-of-5 | 0.378 | 0.397 | 0.026 | 0.069 | 0.008 | 0.079 | 0.0007 |
| theirs 20 ep, avg | 0.449 | 0.464 | 0.033 | 0.077 | 0.011 | 0.085 | 0.0024 |
| theirs 20 ep, best-of-5 | 0.389 | 0.407 | 0.027 | 0.070 | 0.009 | 0.080 | 0.0005 |
| **ours 20 ep, avg** | 0.458 | 0.473 | 0.031 | 0.076 | 0.0095 | 0.085 | 0.0021 |
| ours 20 ep, best-of-5 | 0.393 | 0.413 | 0.026 | 0.070 | 0.0071 | 0.079 | 0.0005 |

- 'theirs' arm (20 ep, seed 42, constant LR) evaluated 10:00 CEST: essentially a no-op vs
  init with mild drift (avg axis +0.6°, best-of-5 axis 4.4° → 7.3°, IoU metrics +0.01
  worse, flips −2 pts, AOR −0.0006). The released model is at a fixed point of its own loss.

- 'ours' arm at epoch 9 (11:20 CEST): train/loss_screw 0.0605 (H1 0.0555, anchor 0.00995,
  ~916 valid rows/step) vs 0.065 at step 0 — slow decrease; ε-MSE 0.00127, fg 0.00999.

## Follow-up (user-requested, 2026-08-30 23:10 CEST): their loss + L2 only
Same protocol (init = released ckpt, seed 42, 20 ep) but **λ=0.5** (10× the earlier arm: the
added L2 term ≈0.03 at step 0, ~25× the ε-MSE — the 'dominant term' regime where the USDNet
result appeared; user's choice) with `screw_w_pos=1.0`,
`screw_w_h1=0`, `screw_w_anchor=0` — the configuration that gave USDNet its best +both.
Arm `ft_ours_l2`, compared against the existing `ft_theirs`. Waiting for GPU stock in
EU-RO-1 (all SKUs out at launch time; retry loop landed an RTX PRO 6000 Workstation
`jqvrkdnj89kb80` $1.89/hr after ~1 h). Training started 20:55 CEST.
- Step-0 (first 5 logged steps): pos 0.090 → weighted 0.045 vs ε-MSE 0.0011, fg 0.010 —
  the term is ~4× fg and ~40× MSE, i.e. it dominates the objective (by design).
- ~1.4k steps in: pos 0.074 (falling), but the unweighted anchor diagnostic rose
  0.013 → 0.022 — direction drifting while the position term pulls (same L2-vs-axis
  tension as USDNet's uncapped L2+H1 arm).
- 21:45 CEST: pod `jqvrkdnj89kb80` is a power-capped lemon (SM 532 MHz @ 100% util, 602 W vs
  600 W cap, max 3090 MHz; 1.36 it/s vs 5.5 on the previous Server-edition pod). Deleted
  after 2 epochs (no ckpt yet); re-provisioning and restarting from scratch.
- 22:20 CEST: second pod `v1eenk8081buhv` (also Workstation edition) throttles identically
  (532 MHz @ 100%, 1.37 it/s). Server edition / A100 / H100 all out of stock in EU-RO-1,
  so continuing on it: ~30 min/epoch → ~10 h for 20 epochs (~$19). Probing for a healthy
  SKU in parallel; will swap if one appears early (ckpts every 5 epochs).
- 05:30 CEST 2026-08-31: Server edition flashed 'Low' in stock; deleted the throttled pod
  (epoch 0, no ckpt) but the Server create failed (stock gone). Now retrying Server/A100/H100
  every 5 min for ≤3 h; relaunch is automatic on success.
- 06:15 CEST: A100 80GB PCIe `0f94jcx2g3eggo` ($1.39/hr, 1410 MHz/300 W) landed; env rebuild
  + L2 arm relaunched. `setup.sh` fixed to auto-detect the CUDA arch for the pytorch3d build
  (was hard-coded to Blackwell 12.0). A100 runs this at 2.0 it/s (~22 min/epoch, ~7.5 h).
- Epoch 5 loss trajectory: pos 0.076 → 0.045–0.058 (falling, noisy), ε-MSE flat 0.0012–0.0016,
  anchor diagnostic 0.011 → 0.014 (mild direction drift).
- Full trajectory (epochs 0→14): pos 0.065 → 0.044 (−30%), ε-MSE flat 0.0014–0.0016, anchor
  diagnostic 0.014 → 0.019 → 0.0155. Training finished 06:25 CEST; eval's `_save_html_end`
  crashed on a missing metrics.json (their eval_metrics subprocess failed silently);
  retrieval + metrics rerun manually.

Axis metrics, L2 arm (λ=0.5, 20 ep):

| ckpt | angle unsigned (avg / best) | flip | origin (avg / best) | type acc |
|---|---|---|---|---|
| init | 12.38 / 4.37 | 6.7% | 0.312 / 0.241 | 0.854 |
| theirs 20 ep | 12.94 / 7.27 | 4.7% | 0.311 / 0.276 | 0.849 |
| ours H1+anchor λ=0.05 | 13.27 / 5.98 | 8.1% | 0.306 / 0.230 | 0.850 |
| **ours L2 λ=0.5** | **12.03 / 6.81** | 5.5% | 0.315 / 0.229 | **0.869** |

vs theirs: avg axis −0.9° (best fine-tuned arm, slightly better than init), flips +0.8 pts,
origin +0.004, type accuracy +2 pts. Their metrics: pending rerun.

Per-input paired read (`bundle/analyze_samples_l2.py`): ours_l2 − theirs mean −0.90° but
**median +0.05°, ours better on only 27% of inputs**; within-input std 5.07 (lowest of all
arms; theirs 5.19); best-of-5 mean −0.46°, median +0.03°. Tail: frac(mean > 45°) 0.12 vs
0.15 (theirs) / 0.14 (init) — i.e. ~4–5 fewer wrong-axis objects out of 154 — while the
near-perfect majority got marginally worse (median 0.16° vs 0.10°). So the −0.9° mean is
a tail-count effect (a handful of objects recovered), not a distribution shift; same
structure as the earlier best-of-5 ordering, opposite sign. One seed.
Their metrics, L2 arm (lower is better):

| ckpt | RS-IoU | AS-IoU | RS-cDist | AS-cDist | RS-CD | AS-CD | AOR |
|---|---|---|---|---|---|---|---|
| init | 0.436 | 0.452 | 0.032 | 0.075 | 0.0106 | 0.0846 | 0.0030 |
| theirs 20 ep | 0.449 | 0.464 | 0.033 | 0.077 | 0.0114 | 0.0853 | 0.0024 |
| ours H1+anchor λ=0.05 | 0.458 | 0.473 | 0.031 | 0.076 | 0.0095 | 0.0849 | 0.0021 |
| **ours L2 λ=0.5** | **0.502** | **0.514** | 0.037 | 0.082 | 0.0137 | 0.0904 | **0.0089** |

## Verdict for the L2 arm (2026-08-31 07:50 CEST): NEGATIVE on their metrics, tiny tail gain on axis
At λ=0.5 the position quadratic dominates the objective and the model pays for it in
layout quality: RS/AS-IoU +0.05 worse than theirs (+0.07 vs init), cDist/CD worse across
the board, and AOR (part inter-penetration) 3.7× higher (0.0089 vs 0.0024). The only
positive is a −0.9° mean axis error that the per-input read attributes to ~4–5 recovered
wrong-axis objects, with the median slightly worse. Net: the "dominant L2" regime that
helped USDNet's origin gate hurts SINGAPO, whose outputs are bounding boxes + joints in
one tensor — a strong term on the joint channels perturbs the box channels through the
shared denoiser. Cost: ~$14 (two throttled Workstation pods + the A100 run + evals).
Bundle: `bundle/{their,axis}_metrics_eval_ft_ours_l2.json`, `train_ours_l2.csv`,
`analyze_samples_l2.py`, `pipeline_l2.sh`. Pod deleted; volume `pye36wkz0m` kept.

## Verdict (2026-08-30 13:00 CEST): NULL

Matched arms (seed 42, same data order, same constant LR, 20 epochs = 52k steps each).
Ours vs theirs: avg axis error +0.33° (worse), best-of-5 axis −1.3° (better), flip rate
+3.4 pts (worse — the opposite of the sign-anchor story), origin line distance −0.005
(better), RS/AS-IoU +0.009 (worse), RS/AS-CD −0.002/−0.0004 (better), AOR −0.0003.
Every delta is inside the init→theirs drift band (which is itself a no-op fine-tune of a
converged model), and inside the 5-sample diffusion noise. No evidence that an
x̂₀-space H1+anchor term at λ=0.05 changes SINGAPO's articulation quality either way.

Why it likely cannot show here: (1) the joint channels are 6 of 30 in an ε-prediction
diffusion model; x̂₀ reconstructed at random t is dominated by noise for most of the
timestep range, and min-SNR weighting leaves only the low-noise tail carrying signal;
(2) the released model is at a fixed point of its own objective — the 'theirs' arm
does not move either; (3) λ=0.05 was chosen conservatively (term ≈ 2.5× ε-MSE); a larger
λ risks the USDNet v1 failure mode. A stronger test would train from the CAGE init
(their real recipe, 200 epochs, ~$50) rather than fine-tune a converged model.

Cost: RTX PRO 6000 $2.09/hr × ~19 h (4 h of it the MooseFS unzip, 1.6 h lost to the
validation/checkpoint incident) ≈ $40. Pod deleted; volume `pye36wkz0m` kept (125 GB:
data, both arms' checkpoints, all eval outputs). Bundle: `bundle/` (their metrics +
axis metrics per checkpoint, train CSVs, the five added files, pipeline log).

## Where the samples moved (per-input analysis, `bundle/analyze_samples.py`, 13:40 CEST)

The axis-error distribution is extremely bimodal: the **median per-input mean error is
≈0.1°** for all three checkpoints (SINGAPO's axes snap to canonical directions), and all of
the mean is carried by a tail of ~21% of inputs with mean error > 20° (14–16% > 45°, i.e.
the wrong axis). Within-input std over the 5 samples: init 6.3°, theirs 5.2°, ours 5.7° —
both fine-tunes narrowed the sample spread a little, ours less than theirs.
Paired per input, ours vs theirs: mean-angle Δ median **+0.02°** (ours better on 36% of
inputs), best-of-5 Δ median **+0.01°** (ours better on 44%), std Δ median +0.01°, flip Δ
median 0 (ours better on 5%). The best-of-5 ordering init 4.37 < ours 5.98 < theirs 7.27
is therefore not a sharpening/mode effect — it is the count of tail inputs that keep at
least one good sample: frac(best < 3°) = 0.94 / 0.92 / 0.90, i.e. 3–6 objects out of 154.
Same conclusion as the means: no distributional shift attributable to the loss.

## Incident 05:05 CEST: first pipeline attempt lost (no checkpoints)
`limit_val_batches: 0.1` × their 4-batch val loader → Lightning MisconfigurationException at
the first validation (after epoch 10) — and because ModelCheckpoint fires after validation,
`ckpts/` was EMPTY for 'theirs' (1.6 h of GPU lost). Killed the 'ours' arm (same fate),
fixed `finetune.yaml` (`limit_val_batches: 0`, `check_val_every_n_epoch: 1000`,
`checkpoint.save_on_train_epoch_end: true`), relaunched both arms from scratch at 05:15 CEST
(20 epochs each, ~2.7 h per arm + evals → results ≈ 11:30 CEST).


## Volumes deleted (2026-08-31)
All campaign volumes removed at the user's direction; checkpoints/data existed only there. Results survive in bundle/.
