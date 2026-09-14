# SF3D-only control vs the final model on the user's four phone photos (paper Fig. 6 source)

`tools/predict_image.py --f35 26`, same four photos as `20260911_iphone_probe/inputs` (NOT committed), prompts
"open door" / "open closet" / "close laptop" / "push chair forward". Panels: photo | sf3d_only | dense.

**Read (2026-09-14).** Door and closet: BOTH models call rot, find the handle and draw a near-vertical hinge
line inside the door (neither on the edge) — doors / cabinets are in SceneFun3D's distribution, so the
control generalises there too. Laptop: the control segments the external keyboard and says trans; dense
segments (part of) the screen and says rot, hinge line diagonal (not at the screen base). Chair: both fail.
Paper reading: human video widens the object set (laptop), not hinge accuracy. Rows in the paper: door, laptop.
