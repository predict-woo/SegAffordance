# HOI4D "put it in / take it out of the drawer" clips — what they look like

Three non-furniture sequences (Mug C2, Bowl C7, Knife C13) from the full
2,973-seq package whose task is "Put it in the drawer" / "Take it out of the
drawer". Per image: 10 evenly spaced frames, RGB row with the official
action-segment label (from `hoi4d_action_segments.csv`), 2Dseg row below.

Also one real-time (15 fps, 20 s) MP4 per clip — `<Category>_<seq>.mp4` —
with the live action label in a colored bar at the bottom, a segment
timeline with playhead, and the 2Dseg mask blended at 35% (scratchpad
`drawer_videos.py`, x264).

Rendered on segaff-probe16 by a throwaway script (scratchpad
`drawer_samples.py`) that pulls `align_rgb/image.mp4` + `2Dseg/` for the
picked sequences straight out of `hoi4d_raw/HOI4D_release.zip` /
`HOI4D_annotations.zip`; extraction left at `/workspace/ext_drawer_inout/`.

Findings (drive the verb-filter decision for the full-package sweep):
- The drawer is already OPEN for the whole clip in all three; there is no
  `open`/`close` segment — the clips are plain pick → carry → put-down with
  the drawer as the destination (the CSV confirms: only
  rest/Reachout/Grasp/Pickup/carry/putdown/Stop).
- 2Dseg for these rigid categories has exactly TWO classes: the labeled
  object (BGR 0,0,128) and the hand (BGR 0,128,0). The drawer / furniture
  is NOT segmented. So for rigid categories the "manipulated part" is
  always the single object class — no VLM choice needed (auto-forced).
- The person walks between tables while carrying (camera moves); a
  `carry` window's hand trajectory is therefore ego-motion + hand motion.
