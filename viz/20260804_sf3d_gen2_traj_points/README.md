# gen-2 trajectory-only point panels: the delta-cumsum before/after

Same 12 samples as [[20260803_sf3d_twist_traj_points]] (the batch that
exposed the old head's jittery unordered clouds), now for the three
gen-2 arms at last.ckpt, --traj-only mode.

The clip arm's predicted trajectories are now CONNECTED, ORDERED sweeps
with the start ring at one end, heading the GT direction (e.g. the
closet door: a clean rightward path where gen-1 produced a bidirectional
cloud straddling the anchor). Extent is still somewhat short. dinov3
remains cloud-like; 2donly produces straight ordered lines (omega
collapsed to 0 — trans-shaped for everything) that are often
mislocalised.

Tool: tools/sf3d_vis_predictions.py --traj-only (manifest.yaml).
Rendered 2026-08-04.
