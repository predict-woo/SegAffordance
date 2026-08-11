# 20260811_sf3d_twist_g4 — body-metric loss + K=4 WTA articulation bundles (CLIP)

**Goal:** first run of the two 2026-08-11 specs (body-frame kinetic-energy
twist metric + annealed winner-takes-all (twist, trajectory) bundles),
targeting the gen-3 diagnosis: posterior-mean hedging had collapsed
revolute `|omega|` to 0.295 (radius 10x inflated) and parked the
trajectory head exactly at the zero-motion baseline (0.0182 vs 0.0181).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).
K=4 bundles, body metric rho=0.25 anchored at GT element point,
trajectory_weight 4.0 (subsumed into the WTA distortion), hint-free eval.
RTX 4090 pod (~430 samples/s, ~$3.2 of the ~$6 total incl. false starts).

**Run history (4 attempts, all findings recorded):**
1. RTX PRO 6000 pods x2: never published SSH endpoints; deleted unused.
2. shm bus error: this 4090 host caps /dev/shm at 29G; the staged LMDBs
   left 4.3G for 64 workers' tensors. Fix: LMDBs on container-local NVMe.
3. Silent container-level OOM kill (host RAM fine): 64 workers too fat
   for the per-GPU container memory cap. Fix: 24 workers.
4. **Hypothesis collapse under the literature anneal schedule** (T0=10
   over 80% of epochs): 8+ epochs of near-uniform softmin pulled all 4
   bundles onto the posterior-mean barycenter (pairwise body-dist 0.005
   at ep9, selected |omega| median 0.04, CE pinned at ln 4) — an
   identical-outputs symmetric fixed point WTA cannot escape at lr 1e-5.
   Aborted at ep9; fix: T0=0.5 (mode-distance scale), hard winner from
   epoch 2. tools/diag_wta_hyps.py is the diagnostic that caught it.

**Result (attempt 4, 16/16 epochs, best ep13 val 0.7992; dual eval —
best and last are distinct ckpts, md5-checked; 43,870 test samples):**

- **Trajectory head: FIXED.** val winner-traj MSE 0.0099 (zero-motion
  baseline 0.018; gen-3 sat exactly AT the baseline). traj_dir_acc 82.7%
  (gen-3 79.7), traj_dir_cos 0.546 (0.508). The bundled WTA + reweighting
  did what it was designed to do on the trajectory side.
- **WTA machinery works:** hypothesis spread 0.452 (vs 0.005 collapsed),
  all 4 bundles alive (win shares 0.21-0.28), selector picks the oracle
  bundle 56% (random 25%), selected-vs-oracle median gap small
  (0.086 vs 0.072).
- **Twist |omega| commitment: NOT achieved.** Even the ORACLE bundle's
  |omega| on revolute samples is ~0.08 (selected 0.10); radius ratio
  median 9.6x (98% >1), decoded-rot rate 6%. Type-from-|omega| 52.0%
  (gen-3 60.2), twist axis err 48.4 deg (42.8), twist_dir_acc 70.7
  (71.5). Type hint again changes nothing.
- Segmentation/localization improved: mIoU 0.118 (best CLIP arm yet;
  gen-3 0.103), PDet 5.6 (4.0), point err 0.135 (0.148).
- last.ckpt (ep15) ~= best on all metrics (type 52.2, traj_dir 82.5).

**Why omega still hedges (measured, not conjectured):** the bundles
specialized along the DOMINANT distortion dimensions — the field term at
p0 and the 4.0-weighted trajectory — i.e. cells split by motion
direction/shape, not by omega. Within a cell, hinge-side/axis ambiguity
still averages: with rho=0.25 a full omega-hedge costs only
0.5*rho^2*1 ~ 0.03 of distortion, too cheap to force cell boundaries
along omega, and screw_self is direction-only so a near-prismatic twist
whose constant field direction matches the arc chord is almost free.

**Decision:** keep the body metric, the bundle architecture, the fast
anneal, and the trajectory result. The omega fix needs re-pricing or
capacity: (a) raise rho (0.5-1.0) so omega errors shape the Voronoi
cells, and/or (b) K=8 so cells can subdivide omega modes, and/or (c) a
magnitude-aware screw_self residual. Decide after inspecting renders
(the trajectory head is now the reliable articulation readout; the twist
orbit renders will still show flat arcs).

Eval logs: logs/test_best.log, logs/test_last.log. Diagnostics:
tools/diag_wta_hyps.py, tools/diag_twist_radius.py (both vs best ckpt,
numbers above). Specs: docs/superpowers/specs/2026-08-11-*.md.
