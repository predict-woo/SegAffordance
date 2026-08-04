# clip_g3 trajectory points: the 1-cos before/after — zigzag GONE

Same 12 samples as [[20260804_sf3d_gen2_traj_points]], clip gen-3 @
best-epoch08, --traj-only mode.

**The sin^2 fold-back hypothesis is CONFIRMED.** Gen-2's scribbly
fold-back arcs (e.g. 08 counter door) and clumped sweeps (05 closet) are
now clean, monotone, evenly-spaced point sequences sweeping the GT
direction at full extent. Mechanism, per the analysis in the gen-3
notes: under sin^2 consistency a segment reversing along the screw
direction cost nothing, so folding was the loss-optimal way to hedge
amplitude; 1-cos charges each backward segment maximally. Under-
extension disappeared together with the folding, as predicted.

Also visible: |omega| commits to rot on true rotations here (0.66-0.74
-> rot) where gen-2 under-committed (0.48-0.57 -> trans) — the aggregate
type-from-|omega| regression (60%) must come from elsewhere in the
distribution (likely the prismatic side / borderline cases).

Tool: tools/sf3d_vis_predictions.py --traj-only (manifest.yaml).
Rendered 2026-08-04, checkpoint 20260804_sf3d_twist_g3/best-epoch08.
