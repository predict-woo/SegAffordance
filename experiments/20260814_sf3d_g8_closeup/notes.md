# 20260814_sf3d_g8_closeup — close-up split + relative direct trajectory

**Goal:** two changes on the gen-7 lift stack: (a) the CLOSE-UP-ONLY
data experiment (user decision) — `min_mask_area_frac: 0.0025` on top
of the radius filter: GT mask must cover >0.25% of the image; measured
first (tools + notes 2026-08-14): SF3D masks are functional-element
splats, median 0.028% (~5x5 px at 256^2), so this keeps only 19,296 of
458k records (4.2%), ~17.4k train / ~1.9k val. (b) trajectory back to
the Euler-era RELATIVE DIRECT readout (trajectory_absolute false,
delta_cumsum false) after gen-7's absolute head regressed traj_dir
92->84 — the user's Euler-era render showed the relative direct head
was historically clean; the August zigzag evidence came from a weak
early checkpoint, and the absolute frame (not directness) was gen-7's
amplifier.

**Setup:** config.yaml (= config/sf3d_train_runpod_g8_closeup.yaml).
RTX PRO 4500, NVMe LMDBs, 24 workers, batch 128, 60 epochs (~85s each,
~1h, <$1), milestones [48, 56]. Teardown hang killed as usual;
last.ckpt md5-deduped to best.

**Best: epoch 59 (the final epoch), val 1.0173 — still improving at
the end; a longer run has headroom.** ~2k close-up test samples.
**FRESH BASELINE — comparable to NO earlier generation** (different
split, different task difficulty: close elements are bigger, nearer,
narrower depth range).

**Result:**

- **Masks work when targets are resolvable: mIoU 0.178, PDet 9.4** —
  ~1.8x the best any full-split arm reached (g4 0.118) and ~3x PDet.
  Frozen CLIP was NOT the binding constraint here — target size was.
  This reframes the mask problem: on the full split the median
  element is ~5x5 px at 256^2; no head fixes that, resolution or
  cropping does.
- **Relative direct trajectory validated: traj_dir 94.8% / cos 0.783
  (best ever), smooth connected curves in the panels** — no zigzag,
  no delta-cumsum, no WTA. val L_trajectory 0.0097 by ep16 (below the
  historic 0.018 zero-motion signature; split-specific baseline not
  re-derived). The Euler formulation holds up when the model around
  it is healthy.
- Geometry on the easier split: point_err_3d 0.300 m, origin_err
  0.347 m (line 0.287 m), radius_err 0.160 m, 2D point 0.165.
- **Weak spots:** type 92.0% (below g7's 98.3 — the close-up mix is
  harder or class-shifted; val type-CE stayed ~0.41 vs g7's 0.24);
  axis err_all 32.8 deg with matched at 17.2 — a tail of unusual
  articulations (viz highlight: a top-loader lid hinge predicted as a
  side-swing).
- point_traj0_gap_m logs 0.0 by design (metric only defined for the
  absolute-trajectory mode).

**Decision:** (a) relative direct trajectory is the trajectory design
going forward — carry it back to the FULL split (a full-split rerun of
the g7 recipe with only the trajectory flag flipped would cleanly
separate the two variables mixed here). (b) The mask result argues the
full-split mask problem is RESOLUTION, not architecture — candidates:
larger input size, a zoom/crop second stage, or accepting close-up as
the task scope. (c) Axis-orientation tail and type on this split are
the open weaknesses.

vis: viz/20260814_sf3d_g8_closeup_panels

Eval log: logs/test_best.log (best=last, ckpt_md5.txt).
mean_origin_error_m (1.02) is the stale legacy metric — ignore.
Cache: /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac00025.pkl.
