# 20260901_hoi4d_2d_dct — g17_2d_dct from scratch on real HOI4D hands (IN FLIGHT)

**Question:** does real hand-interaction video (WiLoR wrist tracks +
moving-part masks, zero articulation GT) teach the 2D pipeline what
SF3D's derived 2D tracks taught it? First run of the real-video line
the 2D path was designed for. User-approved init: from scratch.

**Data:** 354 HOI4D furniture seqs (187 drawer C4 / 167 safe C6) ->
~19k samples (tools/hoi4d_process_2d.py; spec 2026-08-31). Per sample:
moving-part 2Dseg mask (per-window motion-energy color selection),
wrist point + wrist-track 2D trajectory, FFV1 16-bit depth, action-
template text, official K. Known data gotchas handled: 1-based WiLoR
frames; 13/354 action JSONs on a 10s clock (scaled; found by sibling
session); gather-grid mask coords. Type labels EVAL-ONLY (C4=trans,
C6=rot). Scene split = physical object, 15% val.

**Recipe:** identical to 20260822_sf3d_g17_2d_dct (projection 0.5,
depth anchor 0.5, L_pp 0.1, p_rev prior 0.5, DCT-6 head, all 3D losses
0). Filters zeroed (SF3D gates don't apply). 30 epochs, seed 42.

**Watch vs the SF3D 2D arm (mIoU 0.2655, shape 0.0947, traj_dir ~84,
p_rev self-organized):** does traj_dir emerge from real tracks; does
p_rev separate along the C4/C6 proxy; mask quality on only 354 objects.

Result: PENDING
