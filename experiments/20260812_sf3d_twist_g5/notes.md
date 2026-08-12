# 20260812_sf3d_twist_g5 — radius filter + rho 0.75 + frozen CLIP

**Goal:** fix the omega half of the gen-4 result (trajectories fixed,
omega still hedged at |omega| ~ 0.1). Two changes aimed at it plus one
requested regime change:
- `min_revolute_radius: 0.10` — drop the knob/dial/faucet mode (40% of
  revolute records, radii bimodal: 40% < 3 cm, then doors at 0.3-0.5 m).
  Gen-4's hedge was radius-correlated (pred |omega| 0.06 knob vs 0.25
  door, corr 0.58): knobs supervise omega through pure ambiguity.
  417,765 -> 356,747 records; applies to train AND val/test.
- `twist_metric_rho: 0.25 -> 0.75` — a full omega-hedge now costs ~0.28
  of WTA distortion (was ~0.03), comparable to the trajectory terms, so
  the Voronoi cells must also split along omega.
- `freeze_backbone: true` — frozen CLIP (user-requested regime).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).
RTX PRO 4500 pod, ~490 samples/s, 16 epochs ~3.2 h (~$3). LMDBs on
container-local NVMe (29G shm cap), 24 workers — the g4 recipe.

**IMPORTANT — metrics not comparable to gen-3/4:** the eval split is
filtered too (knob-class removed), which removes the hardest type/axis
samples. Cross-generation comparisons below use the diagnostics' door-
class-only breakdowns where possible.

**Result (16/16 epochs; best ep11 val 0.7316; last.ckpt DEDUPLICATED to
the ep11 file — identical md5, the known Lightning quirk — so ep11 is
the only evaluated checkpoint; 36,406 test samples):**

- **Omega commitment: the fixes worked.** Selected revolute |omega|
  median 0.59 (gen-4: 0.10 overall, 0.25 on door-class — a genuine 2.3x
  on comparable samples), prismatic 0.10 (clean 6x type separation);
  decoded-as-rot 61% of revolute (gen-4: 6%). Radius ratio median 2.13
  (gen-4: 9.6). Commitment grew monotonically all run (0.30 @ ep5,
  0.53 @ ep8, 0.59 final) — not yet saturated at |omega| = 1.
- **Articulation metrics, best of any arm** (filtered-eval caveat):
  type-from-|omega| 89.4% (g4 52.0, g3 60.2, DINOv3-unfrozen 71),
  twist_dir_acc 87.5% (g4 70.7), traj_dir_acc 92.6% / cos 0.750
  (g4 82.7 / 0.546), twist axis err 29.4 deg (g4 48.4; previous best
  DINOv3 36.95), twist_pass_rate_ma 22.0 (g3 14.0), point err 0.115.
- **WTA machinery:** spread 0.885, all 4 bundles alive, selector 0.60;
  selector picks higher-|omega| bundles than the GT-nearest oracle
  (0.59 vs 0.49) — selection is not the bottleneck.
- **Cost: masks regressed.** mIoU 0.090 (g4 0.118, g3 0.103), PDet 3.2
  (g4 5.6). Frozen CLIP is the prime suspect (same pattern as frozen
  DINOv3 in gen-3); the filtered eval set may contribute.

**Decision:** gen-5 recipe (body metric + WTA bundles + fast anneal +
radius filter + rho 0.75) is the new articulation baseline. Open items:
(a) masks — unfreeze CLIP (one-variable vs g5) or revisit the projector;
(b) |omega| at 0.59 not saturated — candidates: longer training (it was
still climbing), rho higher still, or a |omega|-magnitude term in
screw_self; (c) selector 0.60 has headroom (CE weight / calibration).

vis: viz/20260813_sf3d_g4_vs_g5_panels, viz/20260813_sf3d_g4_vs_g5_panels_b2

Eval logs: logs/test_best.log (test_last.log = same ckpt, see
ckpt_md5.txt). Diagnostics: logs/diag_hyps.log, logs/diag_radius.log.
Specs: docs/superpowers/specs/2026-08-11-twist-*.md.
