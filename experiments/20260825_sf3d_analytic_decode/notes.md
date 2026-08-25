# 20260825_sf3d_analytic_decode — MECHANISM VERDICT: it's the loss geometry

**Recipe:** supabl2 arm B's exact config (NO trajectory head, no L_pp, no
dir term) + `analytic_trajectory_weight: 0.5`: the trajectory decoded
DIFFERENTIABLY from the predicted articulation parameters (exact mirror
of the GT writer; tests/test_analytic_decode.py) and supervised with the
gen-16 normalized trajectory loss. **Zero parameters added vs arm B.**
30 epochs, best = epoch 23 (val 0.9780). Spec:
2026-08-25-trajectory-mechanism-design.md.

## Test (5,088) between the two anchors

| metric | arm B (no traj) | **analytic decode** | arm D (traj head) |
|---|---|---|---|
| MA / MA_signed | 20.40 / — | **26.45 / 25.88** | 28.22 / — |
| axis matched / all | 23.29° / 29.20° | **19.02° / 27.06°** | 17.07° / — |
| rot flip rate | 21.82 | **15.45** | ~14.8 |
| origin / radius (m) | 0.290 / 0.136 | **0.268** / 0.136 | ~0.28 / 0.120 |
| 3D point | 0.243 | **0.235** | 0.246 |
| mIoU / PDet | 0.268 / 22.60 | 0.256 / 18.93 | 0.265 / 21.95 |

## Reading — the answer to "why does a redundant label help?"

**~75% of the trajectory→articulation transfer is the LOSS GEOMETRY, not
the head.** A deterministic decode with zero new capacity recovers MA
+6.1 of the +7.8 gap, matched axis −4.3° of −6.2°, and nearly all of
the flip-rate improvement. The trajectory target carries no new
information — but expressing the SAME articulation quantities as a
20-point curve gives a dense, conjunction-coupled, GT-anchored objective
(axis ∧ origin ∧ point must be jointly right to place the curve) that
is better-conditioned than the marginal angular/positional losses.
Notably this transfer does NOT need the trajectory gradients to route
through shared features at all — the decode hits the articulation heads
directly.

The residual ~25% (D's MA 28.2 vs 26.5, radius 0.120 vs 0.136) is what
the head itself adds — but arm D carried the DCT head, so the residual
confounds "generic head feature shaping" with "DCT smoothness prior";
fdnolpp (plain head, landing next) narrows this.

**The masks are the flip side:** mIoU 0.256 / PDet 18.9 — BELOW arm B.
The decode loss adds articulation-gradient pressure on the pooled
features without the dense spatial supervision a real trajectory head
provides, and the trunk pays. So the two mechanisms split cleanly:
articulation gains = loss geometry (the decode gets them alone); mask
gains (arm C/D's best-of masks) = the head's dense feature shaping (the
decode conspicuously fails to get them). Both hypotheses were right —
about different metrics.

Toy-probe corollary recorded in the spec: the saddle story is NOT the
mechanism (both losses saddle at the antipode; toy angular beats toy
point-space head-to-head on a single latent). The conditioning advantage
only materializes with the real system's coupled, imperfect heads —
which is why the naive toy was null.

**Practical candidate (gen-22):** trajectory HEAD (for the trunk/masks)
+ analytic decode (for the articulation heads) + fdiff losses, no L_pp —
each component now has a measured, attributed role.

test pass: logs/test.log (ckpt best-epoch23-valloss0.9780). Pod deleted
after the pass (delete delayed by a RunPod API outage; self-healing
retry armed).
