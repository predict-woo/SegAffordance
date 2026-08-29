# 20260829_sf3d_cf_h1_noaxis — H1 alone: additivity breaks; the anchor and L2 are substitutes

**Question:** the 2x2's fourth corner — does H1 carry articulation
entirely by itself?

**Recipe:** cf_h1only with `vae_weight 0`. Only trajectory-derived
supervision: the H1 quadratic (1.0). 30 epochs, seed 42. Best =
epoch 19 (val 1.0770; peaked before cutoff, like L2-only, unlike the
anchored arms). Run note: first host was a LEMON (600W-capped, 562 MHz
under load, 0.51 it/s, passed the idle clock check) — killed at ep14,
resumed from last.ckpt on a healthy pod (runbook updated, 056bcfb);
metrics.csv is version_0+version_1 merged.

**Registered prediction:** MA 28.5-29.5 (additive ~29.2 "minus a
bite"), rot flips 16+, matched <16.6, origin ~0.255, masks >= cf_h1only.
**Got the side effects right, the headline wrong: the bite ate 3.3 MA.**

## Test (5,088), best-epoch19-valloss1.0770 — THE COMPLETED 2x2

| MA | + anchor | no anchor |
|---|---|---|
| **L2 + H1** | 29.19 (closedform) | 27.71 (cf_noaxis) |
| **H1 only** | **30.64 (cf_h1only, RECORD)** | 25.92 (this arm) |
| **L2 only** | (unrun) | 23.80 (cf_l2_noaxis) |

| metric | cf_noaxis (L2+H1) | **cf_h1_noaxis (H1)** | cf_h1only (H1+anchor) |
|---|---|---|---|
| MA / signed | **27.71 / 27.26** | 25.92 / 25.20 | 30.64 / 30.11 |
| axis matched / all | **17.6° / 26.4°** | 19.3° / 26.7° | 16.6° / 24.5° |
| flips all / rot | 10.5 / **15.9** | **9.4** / 16.5 | 9.8 / 15.4 |
| origin (m) | **0.253** | 0.254 | 0.254 |
| 3D point | 0.231 | **0.228** | 0.234 |
| mIoU / PDet | **0.263 / 20.8** | 0.250 / 18.0 | 0.266 / 21.8 |

## Reading

1. **Additivity is badly violated — interaction term −3.2 MA.** From
   closedform: drop anchor = −1.5, drop L2 = +1.45, drop both = −3.3
   (additive prediction: −0.03). The anchor and the position quadratic
   are SUBSTITUTES: H1 needs some complementary stabilizer — either the
   scale-free axis anchor (best: 30.64) or, failing that, the position
   quadratic (27.71) — and with the anchor present, L2's contribution
   flips negative (redundant + blurring).
2. **H1 truly alone is under-constrained, and it shows beyond axis
   numbers:** matched blurs to 19.3°, val peaked at ep19, and the TRUNK
   degrades — masks 0.250/18.0, the family's worst PDet. The pure
   derivative gradient, unanchored, is apparently a noisier signal for
   the shared features, not just for the axis readouts.
3. **flips-all 9.4 is technically the best ever** but rot flips 16.5 are
   near-worst — the all-rows number rides on trans rows (where H1 IS
   the axis loss); revolute sign needs the anchor, third confirmation.
4. Ranking, no-anchor family: L2+H1 27.71 > H1 25.92 > L2 23.80 —
   the terms genuinely complement each other without the anchor.
   With the anchor: H1 30.64 > L2+H1 29.19 — the anchor makes L2
   deadweight. Loss design lesson in one line: **one scale-free
   direct constraint + one geometry-coupled derivative term is the
   sweet spot; more terms are not more signal.**

**Follow-ups (recorded, not commissioned):** the {L2+anchor} corner
would complete the picture but is low-value now (both its neighbors are
measured and the story is consistent); Gram weight/Theta sweep and the
H1(pi/2)+small-pos(2pi) anti-flip combo remain the interesting ones.

test pass: logs/test.log (ckpt best-epoch19-valloss1.0770). Pod deleted.
