# 20260829_sf3d_cf_noaxis_2pi — full-circle Gram sweep, unconfounded (IN FLIGHT)

**Question:** what does the sweep extent Theta do, isolated? Base is
cf_noaxis (pos 0.5 + der 0.5, NO axis anchor) so the Gram change is the
only knob turned — no anchor to overlap with (user request 2026-08-29).

**Recipe:** cf_noaxis + `closed_form_sweep: 2*pi` (new general-Theta
implementation in closed_form_screw_loss, locked by 7 new tests, suite
at 25 for this file). At 2*pi: cross-term = 0 (residuals decouple),
position reweights 3:1 radial (purified origin/radius term), position
flip penalty 2.75 -> 1.0, H1 flip penalty stays 2.0 (= 2 + sin(2T)/T,
exactly 2 at quarter-turn multiples). Trans rows sweep-invariant.

**Predictions vs cf_noaxis (MA 27.71, matched 17.6, flips 10.5/15.9,
origin 0.253):** origin improves (radial-heavy position); rot flips
worsen (weakest anti-flip pressure of any arm yet: no anchor + 1.0
position flip penalty); matched may sharpen (less tangential weight).

Result: PENDING
