# 20260829_sf3d_cf_l2_noaxis — position L2 alone: the weak half, decisively

**Question:** the mirror of cf_h1_noaxis — what does the position
quadratic do BY ITSELF (no H1, no axis anchor)?

**Recipe:** cf_noaxis base, position 1.0 / derivative 0.0, vae 0, sweep
π/2, 30 epochs, seed 42. Best = **epoch 24** (val 1.0867) — the ONLY
arm of the family whose val loss peaked and then ROSE (1.0907, 1.0936
at 26/29); every H1-carrying arm was still improving at cutoff.

## Test (5,088), best-epoch24-valloss1.0867

| metric | arm B (none) | **cf_l2_noaxis (L2)** | cf_noaxis (L2+H1) | cf_h1only (H1+anchor) |
|---|---|---|---|---|
| MA / signed | 20.40 / — | 23.80 / 23.35 | 27.71 / 27.26 | **30.64 / 30.11** |
| axis matched / all | 23.3° / 29.2° | 18.6° / 28.4° | 17.6° / 26.4° | **16.6° / 24.5°** |
| flips all / rot | — / 21.8 | 10.6 / 20.1 | 10.5 / 15.9 | **9.8** / 15.4 |
| origin (m) | 0.290 | 0.277 | **0.253** | 0.254 |
| radius (m) | 0.136 | 0.149 | **0.123** | 0.130 |
| 3D point | 0.243 | 0.240 | **0.231** | 0.234 |
| mIoU / PDet | **0.268 / 22.6** | 0.261 / 20.4 | 0.263 / 20.8 | 0.266 / 21.8 |

## Reading

1. **The derivative term is the engine; position alone is a weak
   learner.** L2-only buys +3.4 MA over arm B, vs +7.3 (L2+H1, no
   anchor) and +10.2 (H1+anchor). It also overfits earliest (val peak
   ep24, rising after) — consistent with a blurrier, more-compensable
   loss surface.
2. **Rot flips 20.1 — essentially unmitigated (arm B: 21.8), despite
   position carrying the family's LARGEST flip penalty (2.75).** Second
   independent confirmation of the 2π lesson: flip-penalty magnitude is
   not the sign mechanism. What fixes sign is either the scale-free
   anchor or a clean (decoupled or derivative-side) tangential
   gradient; the π/2 position term's cross-term lets sign errors be
   traded away.
3. **Even the term's "own" metric doesn't survive alone: origin 0.277**,
   worst of the family (predictions said ~0.25). The position term
   improves origin only in COMBINATION with H1 (0.250-0.254 there) —
   its radial signal is too entangled with axis/point error to work
   unaided.
4. Predictions registered before launch: MA 25-27 (got 23.8 — too
   generous), origin competitive (wrong), flips moderate (wrong — near
   arm-B). The consistent theme across this family: mechanisms are
   about GRADIENT GEOMETRY (what can be traded against what), not about
   penalty magnitudes at symmetric points.

**Grid state after this arm:** {L2+H1+anchor} 29.19 · {L2+H1} 27.71 ·
{H1+anchor} 30.64 RECORD · {L2} 23.80 · {H1} in flight · {L2+anchor}
unrun. Ranking so far: H1 > L2+H1 > L2 ≫ nothing; anchor worth +1.5-3.

test pass: logs/test.log (ckpt best-epoch24-valloss1.0867). Pod deleted.
