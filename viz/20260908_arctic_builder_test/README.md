# ARCTIC builder test panels — rendered moving-part masks + re-anchored trajectories

10 strokes from s01 laptop_use_01 / box_use_02 / microwave_use_01, produced by
`python tools/arctic_process_2d.py --out /workspace/tmp_arctic_test2 --seqs ... --viz 10`
(2026-09-08 evening, FINAL renderer; 1/3-resolution copies of the 2800x2000 ego
frames). Two bugs were found on these panels and fixed: (1) the simplified
mesh.obj + a vertex labelling left faces straddling the hinge -> triangles
smeared across the keyboard when the lid was open; now the dense per-part
top.obj / bottom.obj are rendered; (2) the articulation SIGN was flipped
(user spotted the laptop screen at 132 deg rendering as a strip on the
hinge): the moving part rotates by -angle about canonical +z — verified by
reproducing the collaborator's fingertip-to-moving-part distances (-angle
matches to ~0.3 cm, +angle is off by up to 14 cm on the box/microwave).

Per panel: red = the moving part ("top": lid / screen / door) rendered from
ARCTIC's mesh.obj with the GT object pose at the stroke start (separate
top/bottom z-buffers, 12 mm tolerance); green = middle-knuckle track of the
hand nearest the moving part over the stroke, re-anchored into this frame's
ego camera (world-fixed); cyan dot = stroke start; orange dot = the hinge
axis origin projected into the image; header = key, template description,
hand, number of points, angle at start -> end.

Read: masks sit on the objects (closed and standing laptop screens fully
covered; microwave door closed edge-on and open at 92 deg as its frame; box
lid closed and standing); hinge origins land on the hinge lines
(back edge of the laptop, left edge of the microwave door). Hands are NOT
cut out of the masks. Trajectories that swing a lid toward the head-mounted
camera leave the frame (laptop s00: opening to 132 deg).
