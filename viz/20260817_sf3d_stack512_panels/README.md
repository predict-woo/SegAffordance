# 20260817_sf3d_stack512_panels — the dinov3 stack at 512

16 val samples (seed 42421, family-standard picks), rows: GT | g13res512 |
g14taps | g15costmap, all fed 512-px inputs from sf3d_frames_512.lmdb
(v3 GT). Regen: manifest.yaml (note --input-size 512 --frame-cache-path).

Interpretation, matching the metrics: g13's axes sit ON hinge edges
(08_rot_val1684: 16° vs g14's floating 38° — the taps' axis regression made
visible; g15 recovers to 14°). Masks are dramatically fuller than any
256-px arm (compare viz/20260817_sf3d_g12_256_panels, same samples).
On the "middle drawer in the third row" grounding probe (09_trans), the
512 arms put the point on/near the correct handle where 256 arms wandered.
