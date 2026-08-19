# 20260818_sf3d_g17_2donly — 2D-only training on the g17 stack

**Recipe:** the g17 model VERBATIM trained with zero 3D GT (spec
2026-08-18): data term = normalized TrajectoryProjectionLoss @0.5
(energy floor 0.0025), consistency = normalized L_pp @0.1 (sole teacher
of axis heads / origin lift / p_rev), z_p depth tether @0.5, p_rev
batch-balance prior @0.5 (target 0.225), mask/point/coord @0.5. 3D
trajectory / axis / type CE / origin losses all zero. RTX PRO 6000 WK,
30 epochs, clean exit; best = epoch 27 (val 2.3453, still improving).

**Launch 1 (killed at ep1, ~$1):** the label-free gate collapsed to
trans (p_rev_mean 0.44→0.0015 — line residual bounded ≤1, circle
unbounded) and near-degenerate tracks spiked the projection ratio
30–50×. Fixed by the balance prior + energy floor (commit b29147c);
launch 2's p_rev_mean held 0.15–0.17 throughout.

## Test (best-epoch27, 5,088 samples) vs g17 [3D-supervised]

| metric | g17 | g17-2d 2D-only | read |
|---|---|---|---|
| traj_dir acc / cos | 94.9 / 0.811 | **80.96 / 0.379** | REAL signal — direction far above the 50% chance floor, from ordered 2D matching alone |
| type acc | 95.3 | 75.2 | ≈ the 77.5% majority-trans baseline — gate stayed ALIVE (16% rot calls) but L_pp did not sort samples; weak/no separation |
| axis (sign-agnostic, all) | 25.8° | 58.7° | ≈ the 57.3° random-direction expectation — axis did NOT emerge from L_pp alone |
| axis flip rate (rot) | 13.3% | 54.9% | ≈ random sign, exactly as predicted (sign unobservable) |
| mIoU / PDet | 0.265 / 20.6 | **0.057 / 1.1** | THE SURPRISE — masks collapsed despite identical 2D mask supervision |
| 2D point | 0.095 | 0.307 | dragged down with the shared features |
| origin q* / radius | 0.257 / 0.122 | 1.41 / 0.26 | dark, as expected (z_q trains only through L_pp) |

## Reading

Three findings, one per tier of the design:

1. **The projection data term works.** 81% trajectory-direction accuracy
   with zero 3D labels is the positive result — ordered 2D matching
   teaches signed 3D motion direction (twist-era 2donly managed 60%).
2. **L_pp alone is too weak a teacher for articulation.** Axis at the
   random-direction expectation and type at the majority baseline: with
   nothing pinning the trajectory's SHAPE tightly (projection fixes 2 of
   3 DOF, loosely early), the consistency loss has too little signal to
   back-propagate into axis/origin/type. The balance prior kept the gate
   alive but alive ≠ informative.
3. **The unexpected cost: mask/point collapse (mIoU 0.265→0.057).**
   Identical 2D supervision to g17, so the mask heads were NOT starved —
   the noisy, initially-huge projection+L_pp gradients flowing through
   the shared FPN/decoder/projector evidently wrecked the shared features
   the mask channel needs. g17's 3D losses coexisted fine; the 2D-only
   composition does not. Candidate fixes if pursued: projection-term
   warmup (0 for N epochs), gradient detach into the trunk from the
   trajectory path, or a lower projection weight.

**Verdict:** mechanism proof PARTIAL — better than the twist-era proof on
direction and gate survival, but not a usable pretraining checkpoint as-is
(the mask collapse would poison a finetune start). The
finetune-from-2D-vs-scratch payoff experiment is NOT worth running on this
checkpoint; fix the interference first if 2D-only stays a priority.

vis: viz/20260819_sf3d_g17_2d_vs_g17_panels
