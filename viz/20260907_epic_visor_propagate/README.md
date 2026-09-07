# VISOR mask -> contact-onset frame by SAM2 video propagation — test on 4 EPIC videos

Regen (dev pod, SAM2.1 hiera-large in /opt/venv, ckpt /workspace/models/sam2.1_hiera_large.pt,
full-HD videos in /workspace/datasets/epic_videos/):
`python tools/epic_visor_propagate.py --videos P28_103,P04_05,P22_07,P01_09 --per-video 4 --validate`
No trained checkpoint involved (dataset-construction audit).

**Method.** For an interaction, take the nearest VISOR sparse frame with a
fixture mask (from `/workspace/datasets/visor/coverage.json`), cut the video
frames between it and the contact onset (EPIC-55 ids on the 60 fps grid,
extension ids at 50 fps; 1920x1080 = VISOR polygon space), seed SAM2's video
predictor with the VISOR polygon(s) as a mask prompt, propagate to the onset
(forward or reverse), keep the onset mask.

**Sample panels** `<noun>_<narration_id>_d<offset>.jpg`: LEFT = VISOR frame +
GT polygon (red); RIGHT = onset frame + propagated mask (green) + the
collaborator's middle-knuckle track projected with the video intrinsics
(cyan; cyan dot = onset, magenta = end). `<narration_id>_onset_mask.png` =
the propagated mask. Offsets tested: -15 .. +17 frames.

**Validation panels** `validate_<video>_<noun>_<f1>_<f2>_iou*.jpg`: propagate
from one human-labelled sparse frame to the NEXT one (gaps 22-40 frames,
harder than the onset offsets), score against the second frame's human
polygon: red = GT only, green = propagated only, yellow = overlap.

| video | validation IoU (n=6) | notes |
|---|---|---|
| P28_103 (50 fps) | mean 0.81, min 0.55 | 3 near-perfect (0.98-1.00); the low ones are VISOR labelling the opened door + interior while SAM2 keeps the door |
| P04_05 | mean 0.77, min 0.15 | 4 at 0.93-0.98; cupboard 0.70 (interior added by VISOR); dishwasher 0.15 = REAL failure: door swings 90 deg to horizontal over 40 frames, SAM2 keeps only the front edge |
| P22_07 | mean 0.85, min 0.55 | 4 at 0.96-1.00; the two lows are a motion-blurred seed where VISOR's "cupboard" = whole carcass + doors |
| P01_09 | (pending) | |

**Interpretation.** Propagation over the onset offsets we need (median ~15
frames, mostly toward the closed state) is clean in every sample: tight
edges, hand excluded, area ratio 0.6-1.2. The low validation IoUs are
dominated by VISOR's own inconsistency for cupboards/fridges (door vs door +
interior), where SAM2 tracks the moving part — the behaviour we want. One
genuine failure mode: a door rotating ~90 deg across the gap (dishwasher);
the pipeline should flag large area changes (e.g. ratio < 0.4 or > 2.5) for
review. Cost: ~1 min per sample on the PRO 4000 (24 GB), so the ~420
covered interactions are ~7 GPU-hours worst case; the expensive part is the
video fetch (6-8 GB per video; data.bris single stream 2.7 MB/s, 16 parallel
byte ranges 36 MB/s — use the chunked fetch; the HF mirror
a1raman/epic_kitchens_100 is slower from EU-RO-1 at 1.3 MB/s).
