# ARCTIC builder test panels — rendered moving-part masks + re-anchored trajectories

8 strokes from s01 box_use_02 / laptop_use_01 / microwave_use_01, produced by
`python tools/arctic_process_2d.py --out /workspace/tmp_arctic_test --seqs ... --viz 8`
(2026-09-08, after the geometric part-label fix; 1/3-resolution copies of the
2800x2000 ego frames). No trained checkpoint involved (dataset audit).

Per panel: red = the moving part ("top": lid / screen / door) rendered from
ARCTIC's mesh.obj with the GT object pose at the stroke start (separate
top/bottom z-buffers, 12 mm tolerance); green = middle-knuckle track of the
hand nearest the moving part over the stroke, re-anchored into this frame's
ego camera (world-fixed); cyan dot = stroke start; orange dot = the hinge
axis origin projected into the image; header = key, template description,
hand, number of points, angle at start -> end.

Read: masks sit on the objects (laptop lid fully covered, microwave door
seen edge-on, box lid standing open); hinge origins land on the hinge lines
(back edge of the laptop, left edge of the microwave door). Hands are NOT
cut out of the masks. Trajectories that swing a lid toward the head-mounted
camera leave the frame (laptop s00: opening to 132 deg).
