# Can a revolute trajectory loss contain 1 − cos(n, n*)?  (2026-09-11)

Follows `docs/slides/2026-08-28_continuous_trajectory_loss.html` and the
cf_noaxis / cf_h1only / cf_l2_noaxis arms. Every closed form below is
verified against Simpson quadrature (N = 201) of the sampled curve-space
definition on 100 random articulations (generic, flipped, 1 mm levers,
large along-axis offsets); max abs error ≤ 5e-5 (the L2 forms at π/2,
quadrature-limited) and ≤ 1e-9 otherwise. Script and full output:
`tools/revolute_loss_check.py`
(`check_output.txt` next to it).

Notation: k = n·n*, λ = |r|/|r*| (predicted / GT lever radius), r̂, t̂ the
predicted lever and tangent directions, r̂*, t̂* the GT ones. Two in-plane
phases are defined through the 2×2 frame overlap M_ij = (GT col i)·(pred col j):
(1+k)cos ψ = r̂·r̂* + t̂·t̂*, (1+k)sin ψ = t̂*·r̂ − r̂*·t̂  (ψ = lever phase after
aligning the axes; ψ = ∠(r̂, r̂*) when n = n*), and (1−k)cos χ = r̂·r̂* − t̂·t̂*,
(1−k)sin χ = r̂*·t̂ + t̂*·r̂ (the phase that only matters when the axes differ).

## 0. Verdict

Yes — but only one way, and it is not free. Every point-arc loss in this
family (position or velocity, any sweep, any normaliser) has the exact form

    L = radius(λ) + λ^p (1−k)·[1 + ρ_Θ cos(χ − χ_Θ)] + λ^p (1+k)·(1 − cos ψ)      (★)

so the axis term is a (1 − cos) multiplied by λ^p (lever scaling) and by a
phase-dependent factor 1 ± ρ_Θ (cancellable by the lever). The current loss
has p = 1 (normaliser is GT-only) and ρ_Θ > 0 (ρ = 0.95 for L2 at π/2, 0.64
for H1 at π/2); the flipped axis can be compensated down to 0.09 (L2) / 0.60
(H1) by moving origin/point, which is the mechanism behind the flip rates.
A clean, scale-free, uncancellable unit-weight (1 − k) appears iff p = 0
(normalise by the predicted lever, or compare unit tangents) AND ρ_Θ = 0,
which for the velocity (H1) term happens exactly at Θ ∈ {π, 2π} and for the
position term never. The cleanest instance is the H1 loss at Θ = π on unit
levers: L = (1−k) + (1+k)(1−cos ψ) = ½(|r̂−r̂*|² + |t̂−t̂*|²), whose value at any
flipped axis is exactly 2 regardless of origin/point (same as the anchor).
Its costs: the antipode is a (−1, +1) saddle in n (escape only by rotating
the axis about the lever), not the anchor's (−1, −1) maximum; radius is no
longer in the term; and it is singular when the predicted axis is parallel
to p0 − o (projected lever → 0). The rigid-body view explains the original
asymmetry: the point arc is the *translational* half of the screw motion;
the *orientation* half's body-velocity H1 loss is exactly |n − n*|² — the
anchor is the revolute analogue of "prismatic trajectory loss = axis loss",
not an ad-hoc extra. Recommendation in §3: keep the anchor; if a single
trajectory term must carry sign, use unit-lever H1 at Θ = π (+ the existing
position term at small weight for radius/origin) and test it against
cf_h1only.

## 1. Why the current formulation cannot contain 1 − cos

The closed form is a quadratic Q(dr, dt) in dr = r − r*, dt = n×r − n*×r*,
divided by a GT-only normaliser. Writing it out in the frame variables (all
verified) gives (★) with

| term | p | radius(λ) | ρ_Θ (phase leak) | χ_Θ |
|---|---|---|---|---|
| L2 / |r*|² (code, `rot_pos`) | 1 | (λ−1)² | √((A−C)²+B2²)/(A+C) = 0.953 (π/2), 0.81 (π), 0.50 (2π) | atan2(B2, A−C) |
| H1 / |r*|² (code, `rot_der`) | 1 | (λ−1)² | sin Θ/Θ = 0.637 (π/2), 0 (π, 2π) | Θ |

Three consequences, each numerically confirmed:

1. **Homogeneity (why no scale-free term).** The numerator is degree-2 in the
   predicted lever and the normaliser is GT-only, so every axis-carrying term
   is multiplied by λ (p = 1). The axis gradient at 90° error scales linearly
   with the predicted radius: 1.63λ (L2), 1.19λ (H1) — 0.016 at λ = 0.01. A
   degree-0 term such as 1 − k cannot occur; at λ = 0 the loss is exactly 1
   for every n (an n-blind plateau).
2. **Phase leak (why the sign is compensable).** The coefficient of (1−k) is
   1 + ρ_Θ cos(χ − χ_Θ); the lever phase χ (set by origin/point) can shrink
   it to 1 − ρ_Θ. For the position term ρ_Θ ≥ 0.5 at every sweep because its
   basis {cos θ − 1, sin θ} is never isotropic (A ≠ C, B2 = −(1−cos Θ)² < 0
   for all Θ ∉ 2πZ); for the derivative basis {−sin θ, cos θ}, ρ_Θ = |sin Θ|/Θ
   vanishes at half-turn multiples.
3. **The "antipodal saddle" is really a compensated state.** With exact
   (o, p0) the antipode is a strict local *maximum* in n for both terms
   (Hessian eigenvalues −2.7/−0.8 for L2, −1.6/−0.4 for H1 at π/2) — the
   gradient in n is zero there, as it must be for any smooth loss maximal at
   the antipode, including 1 − cos. The problem is the competing descent
   through (o, p0): minimising over origin/point at n = −n* gives

   | loss | value at exact (o,p0) | floor | at λ, phase | full 8-D Hessian min eig at floor |
   |---|---|---|---|---|
   | L2 π/2 | 2.752 | **0.091** | 0.95, −113° | −0.041 (7 of 8 ≥ 0: a near-trap) |
   | L2 2π | 1.000 | 0.750 | 0.50, 0° | −0.25 |
   | H1 π/2 | 2.000 | **0.595** | 0.64, −90° | −0.23 |
   | H1 π | 2.000 | 1.000 | 0 (collapse) | 0 (flat plateau) |
   | anchor 1−k | 2 | 2 | — | −1 |

   The L2-only arm (20% revolute flips) was training against a loss whose
   flipped-axis floor is 0.09 — 3% of its antipode value — reached by a
   ~R*-sized in-plane displacement of the point; the escape curvature there
   is −0.04 in one direction against +1.6…+4.7 in the others. H1 at π/2 is
   better (floor 0.60, escape −0.23) but still offers the trade. The anchor
   adds a floor of 2 that no other head can lower. (Also: with an along-axis
   offset a = n*·(p0 − o) ≠ 0 the current terms have a spurious n-gradient
   of 1.63a / 1.19a at the antipode that points toward shrinking the
   projected lever, not toward n*; the p = 0 forms have none.)

## 2. Candidates

### 2.1 Unit tangents  u(θ) = −sin θ r̂ + cos θ t̂, (1/Θ)∫|u − u*|²

Closed form: (1/Θ)[C|δr̂|² + C2|δt̂|² − sin²Θ δr̂·δt̂] = (★) with p = 0, no
radius term, ρ_Θ = sin Θ/Θ. Uses (n×r̂)·(n*×r̂*) = k(r̂·r̂*) − (n·r̂*)(n*·r̂):
the (1 − k) survives with r̂ ≠ r̂* but is entangled through ψ, χ.

- Θ = π/2: L = 2 − (1+k)cos ψ − (2/π)(r̂·t̂* + r̂*·t̂); axis coefficient
  1 ± 0.64; antipode floor 0.727 (phase −90°).
- Θ = π or 2π: L = (1 − k) + (1 + k)(1 − cos ψ). Exact. Floor 2 at every
  flipped axis (flat in λ and ψ there); gradient at 90° = 1.0 for all λ.
  Antipode Hessian in n: (−1, +1) — tilting the axis about t̂* (toward r̂*)
  leaves t̂ = −t̂* to first order while tilting the lever out of plane, which
  increases |u − u*|; tilting about r̂* decreases it. Half the escape
  directions of the anchor. Adding the anchor at weight w gives (−1−w, 1−w).
- Full circle *with phase minimised* (unmatched parameterisation):
  min_β L_ut^{2π} = 1 − k exactly (verified, 1e-5 on a 0.5° grid). Two great
  circles of tangent directions compared up to a time shift is the axis
  loss — which shows the axis lives in the unit-tangent circle and the
  lever phase is the only other thing a point arc knows.
- Stops supervising: radius (entirely), along-axis origin (as before).
  Singular at n ∥ (p0 − o) (projected lever → 0; r̂ undefined) — bounded by
  4 with an eps clamp but the gradient is wild there.

### 2.2 Frenet binormal / normalised angular momentum

For a circle b = ẋ×ẍ/|·| = r×ṙ/|·| = n for all θ. A loss on b is 1 − k
exactly. It is a legitimate curve functional (needs the curve to second
order, no explicit axis) but loses radius, origin, phase — it *is* the
axis loss. The full Frenet frame (T, N, B) in L2 is sweep-independent
(verified to 1e-15 between Θ = 0.3 and 5.0): (1/Θ)∫(|δT|²+|δN|²+|δB|²) =
‖Q − Q*‖²_F = 4(1 − cos ω) = 4(1−k) + 2(1+k)(1−cos ψ) — the chordal SO(3)
distance between the frames [r̂ t̂ n] and [r̂* t̂* n*]. Contains 1 − k with
weight 4, floor 8, antipode Hessian (−4, 0).

### 2.3 Arc-length parameterisation

Normalised arc length σ = s/(RΘ) ∈ [0,1] is identical to matched θ (s = Rθ),
so the L2/H1 of normalised-unit-speed curves are the current terms. True
(unnormalised) arc length makes the predicted curve sweep Θ R*/R — a
different angle — and the integrals become beat terms sin((1/R − 1/R*)s)/(…);
the unit-speed H1 is 2.1 at mismatched angles and the H2 (curvature) term
compares 1/R with 1/R* (diverges at small radius). No clean 1 − k. Rejected.

### 2.4 Rigid-body trajectory: SO(3) and SE(3)

Via Rodrigues, tr(R(θn)ᵀR(θn*)) = 3 − 4(1−c) + 2s²k + (1−c)²(k²+1)
(c = cos θ, s = sin θ), hence ‖R(θn) − R(θn*)‖²_F = 8(1−c)(1−k) − 2(1−c)²(1−k)²
and, with I1 = ∫(1−cos) = Θ − sin Θ and A as in the code,

    ∫₀^Θ ‖R − R*‖²_F dθ = 8 I1 (1−k) − 2 A (1−k)²      (verified, 1e-10)

A function of n·n* only — trivially scale-free, no lever, no origin. It is
a *concave* reshaping of 1 − cos: steeper near the truth, flatter near the
antipode (normalised to 1 at the antipode: gradient 0.5 at 90°, Hessian
−0.27 vs the anchor's 1.0 and −1). Monotone in the axis angle only for
Θ ≲ 2.1 rad (g'(2) = 8(I1 − A) ≥ 0); at Θ = π and 2π the loss *peaks at
k = −1/3* and the antipode is a local minimum (a reversed full turn visits
the same rotations). Do not use it at large sweeps.

The body-velocity H1 of the same trajectory, ∫‖RᵀṘ − R*ᵀṘ*‖²_F = ∫‖[n−n*]ₓ‖²
= 2Θ|n − n*|², normalised by 2Θ, is **exactly |n − n*|² = 2(1 − k)** — the
same expression the prismatic rows already produce. That is the answer to
"why does prismatic give 1 − cos automatically": the prismatic rigid motion
is all translation (T = [I, θL d̂]) and its trajectory is the direction; the
revolute rigid motion's direction lives in the orientation part, which the
point-arc loss discards. SE(3) adds the translation (I − R(θn))o — the
current quadratic with the world origin as the "point" (frame-dependent,
metres vs radians needs a scale ρ) — i.e. "SO(3) term + origin-arc term".

### 2.5 Normalisations

Dividing the H1 numerator by Θ|r||r*| instead of Θ|r*|² gives (★) with
p = 0 and radius(λ) = λ + 1/λ − 2 = 4 sinh²(½ log λ):

    H1/(Θ|r||r*|) = (λ + 1/λ − 2) + (1−k)[1 + (sin Θ/Θ)cos(χ − Θ)] + (1+k)(1 − cos ψ)

and at Θ = π: **(λ + 1/λ − 2) + (1 − k) + (1 + k)(1 − cos ψ)** (verified,
2e-13). This is the exact "shape/scale split": unit-tangent loss + a
symmetric log-radius term, with unit weight on 1 − k. The collapse plateau
is gone (λ → 0 costs 1/λ) and the axis gradient is λ-independent (1.0 at
90° for λ = 0.01…10). Costs: gradient into p0/o at small predicted lever is
1/λ² (needs a floor; at n ∥ (p0−o) the projected lever vanishes and the
loss → ∞ — the current loss is bounded by 1 there); antipode Hessian (−1, +1)
as in 2.1. Dividing by |r|² instead gives (λ−1)²/λ² → bounded (= 1) as
λ → ∞: an infinite-lever plateau mirroring the current zero-lever one.
The same treatment of the position term (L2/((A+C)|r||r*|)) keeps
ρ_Θ ≥ 0.5, floor 0.094 at π/2 — normalisation fixes scale, not the leak.

### 2.6 Sweep

Θ only changes ρ_Θ and χ_Θ in (★). For H1, Θ ∈ {π, 2π} zeroes the leak
(cross term 0 and C = C2); the note's "2π decouples" holds already at π.
For L2 no sweep works (ρ_2π = 0.5: floor 0.75, at λ = ½ — below the
note's flip value 1.0, which assumed dr = 0). No sweep separates a pure
1 − k out of the p = 1 quadratic (homogeneity); with p = 0 and H1, π does.

### 2.7 Angular velocity / velocity field

∫|ω − ω*|² with matched rate = 2Θ(1 − k): identical to 2.4's body-velocity
H1 — legitimate as the orientation trajectory's H1 seminorm, and it *is* the
anchor. A velocity-field loss over the part mask, ∫_mask |n×(x−o) − n*×(x−o*)|²,
is the current H1-type quadratic averaged over mask points (each with its
own lever): the lever scaling is replaced by the mask's second moment about
the axis — a mask-shaped, not a scale-free, coefficient. Same structure.

## 3. Recommendation

Gradient/curvature summary (exact o, p0; Riemannian gradient in n):

| loss | |∇| at 90° | |∇| at 170° | antipode Hess (exact) | floor | Hess at floor |
|---|---|---|---|---|---|
| L2 π/2 (code) | 1.63 (∝λ) | 0.40 | −2.7, −0.8 | 0.091 | +0.36, +2.26 |
| H1 π/2 (code) | 1.19 (∝λ) | 0.21 | −1.6, −0.4 | 0.595 | −0.13, +1.14 |
| H1 π /|r||r*| | 1.00 | 0.18 | −1, +1 | 2.000 | −1, +1 |
| unit tangent π | 1.00 | 0.17 | −1, +1 | 2.000 | −1, +1 |
| Frenet frame | 4.00 | 0.00 | −4, 0 | 8 | −4, 0 |
| SO(3) π/2 (norm.) | 0.50 | 0.05 | −0.27, −0.27 | 1 | same |
| anchor 1 − k | 1.00 | 0.17 | −1, −1 | 2 | same |

1. **Keep the separate axis loss, and stop calling it ad hoc.** It is the
   body-velocity H1 of the rigid-body orientation trajectory — the revolute
   twin of the prismatic term. No point-arc functional reproduces its
   (−1, −1) antipodal curvature; the best a point arc can do is (−1, +1),
   because a point arc encodes the axis only through the tangent circle,
   whose phase is a nuisance variable that the origin/point heads control.
2. **If one trajectory term should carry the sign anyway**: use the
   derivative term at Θ = π on unit levers,
   `rot_der_unit = 0.5·(|r̂ − r̂*|² + |t̂ − t̂*|²)` (= (1−k) + (1+k)(1−cos ψ); a
   two-line change in `closed_form_screw_loss`: normalise r_p, r_g, t_p, t_g
   with an eps floor on |r_p| — e.g. 0.1·|r_g| — and use sweep = π, which
   makes the Gram (C, C2, sin²Θ) = (π/2, π/2, 0)). It keeps the flipped-axis
   value at exactly 2 for any origin/point (vs floor 0.60 now), is
   λ-independent, and has no along-offset artefact. It drops radius, so
   compose as: `rot_der_unit` (weight ≈ the current derivative weight) +
   the existing `rot_pos` at 0.1–0.2 (the note's origin regulariser; it is
   the only cf term with (λ−1)²) + the anchor at its current 0.5 (the
   antipode then has (−1.5, +0.5): the +1 direction is still not closed
   without w ≥ 1). The Frenet-frame loss is the same thing with the anchor
   built in at weight 2 relative to the tangent part.
3. **Experiment to run before believing any of this**: cf_h1unit_pi_noaxis
   (rot_der_unit 1.0, pos 0, vae 0) against cf_noaxis (rot flips 15.9%)
   and cf_h1only (15.4%, MA 30.64). The prediction from (★): flips move
   toward the anchor arm's rate without an anchor; MA may drop because the
   term no longer sees radius and because Θ = π reweights the metric that
   set the record at π/2 — which is why the anchor-on version (2.) is the
   safer configuration. Do not try the SO(3) Frobenius term at Θ ≥ 2.1 rad
   (antipode becomes a local minimum) and do not use the 1/λ geometric-mean
   normaliser without a lever floor.
