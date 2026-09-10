# 20260911_sf3d_cf_l2_noaxis_2pi — position L2 alone at sweep 2π

**Question (user, 2026-09-11).** The L2 term's phase leak is ρ = 0.953 at π/2 (a flipped axis buyable down to
0.09) and 0.500 at 2π (floor 0.75); no sweep closes it. How much of the L2-only arm's 20.1 % hinge flips
does the 2π sweep remove?

**Recipe.** `config/sf3d_train_runpod_cf_l2_noaxis_2pi.yaml` = `20260829_sf3d_cf_l2_noaxis` (L2 1.0, no H1,
no direct axis loss, arm-B config, 30 ep, seed 42) with `closed_form_sweep 2π`.

**Comparison rows.** cf_l2_noaxis (π/2): MA 23.80 / 23.35, matched 18.6°, all 28.4°, flips all/rot 10.6 / 20.1,
origin 0.277, mIoU 0.261 / PDet 20.4. cf_noaxis_2pi (L2+H1 at 2π): MA 27.04, rot flips 13.0, matched 20.2°.
cf_frame (this day's other arm): p = 0, ρ = 0.

**Status.** launched 2026-09-11 (pod `segaffordance-cfl22pi`, `run_cf_l2_2pi_chain.sh`, log `cf_l2_2pi_chain.log`).
