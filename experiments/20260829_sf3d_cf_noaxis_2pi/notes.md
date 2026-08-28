# 20260829_sf3d_cf_noaxis_2pi — the Θ knob is real, and every prediction was wrong

**Question:** what does the sweep extent do, isolated? Base = cf_noaxis
(pos 0.5 + der 0.5, NO axis anchor); only change `closed_form_sweep:
2*pi` (general-Θ implementation, commit 3abfd9e). At 2π the cross-term
vanishes (residuals decouple), position reweights 3:1 radial, position
flip penalty 2.75 → 1.0, H1's stays 2.0.

**Registered predictions:** origin improves (radial-heavy position);
rot flips worsen (weakest anti-flip pressure yet); matched sharpens.
**All three were wrong in direction.**

## Test (5,088), best-epoch29-valloss1.0635

| metric | cf_noaxis (π/2) | **cf_noaxis_2pi** | closedform (π/2 + anchor) |
|---|---|---|---|
| MA / signed | **27.71 / 27.26** | 27.04 / 26.73 | 29.19 / 28.89 |
| axis matched / all | **17.6°** / 26.4° | 20.2° / **26.0°** | 22.3° / 26.8° |
| flips all / rot | 10.5 / 15.9 | **10.1 / 13.0** | 10.7 / 13.7 |
| origin (m) | **0.253** | 0.259 | 0.250 |
| radius (m) | **0.123** | 0.125 | 0.128 |
| 3D point | 0.231 | **0.230** | 0.238 |
| mIoU / PDet | **0.263** / 20.8 | 0.255 / **21.1** | 0.258 / 20.1 |

## Reading

1. **Rot flips 15.9 → 13.0 — the best sign performance of any
   no-anchor arm, beating even closedform WITH its anchor (13.7).** The
   flip-penalty magnitudes (position 2.75 → 1.0) predicted the opposite;
   what evidently matters is the DECOUPLING: at π/2 the −Δr·Δt cross
   term lets radial and tangential errors trade against each other,
   creating descent directions that reduce loss without fixing sign; at
   2π the tangential (sign-carrying) residual gets its own clean,
   undisturbed gradient in both terms.
2. **Matched blurred 17.6° → 20.2°, the mirror cost.** Less tangential
   weight in the position term (3:1 radial at 2π vs 1:2.2 at π/2) =
   less direction sharpening on matched rows. All-axis still improved
   slightly (26.4 → 26.0) — flips leaving the >90° bucket pull the mean
   down even as matched blurs.
3. **Origin did NOT improve (0.253 → 0.259)** despite the position term
   going radial-heavy — the "purified origin term" story oversimplified;
   the radial residual couples origin, point, AND axis errors, so more
   radial weight is not more origin accuracy.
4. Net MA −0.7: sign gains don't cover the 10°-threshold losses from
   matched blur. The Θ trade in one line: **Θ controls a
   sign-robustness ↔ direction-precision dial** (2π = robust/blunt,
   π/2 = sharp/flippier), not a quality knob.

**Implication for the two-record combo:** the H1(π/2) + small
pos(2π) idea survives with a different rationale — pos(2π) as an
anti-flip decoupled term, not an origin term. But the flip mechanism is
now murky enough that a Θ mid-point (e.g. π) may be the better sweep
candidate.

test pass: logs/test.log (ckpt best-epoch29-valloss1.0635). Pod deleted.
