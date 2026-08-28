# 20260829_sf3d_cf_h1_noaxis — H1 alone: no L2, no axis anchor (IN FLIGHT)

**Question:** completes the 2x2 (L2 on/off x anchor on/off). Does H1
carry articulation entirely by itself? Separates "the anchor matters
because L2 is blurry" from "the anchor matters, period."

**Recipe:** cf_h1only with `vae_weight 0.5 -> 0.0`. Only trajectory-
derived supervision left: the H1 derivative quadratic (1.0). Everything
else classical (type CE, 3D origin/point, masks/heatmaps). 30 epochs,
seed 42.

**Prediction (registered before launch, 2026-08-29):** trains; MA
28.5-29.5 (additive ~29.2 minus a bite); rot flips 16+ (H1's flip
penalty 2.0 < L2's 2.75, no scale-free anchor); matched possibly <16.6
(both removals sharpened precision); origin ~0.255; masks >= cf_h1only.

**The 2x2 so far:** closedform (L2+H1+anchor) 29.19 | cf_noaxis
(L2+H1) 27.71 | cf_h1only (H1+anchor) 30.64 RECORD | this arm (H1).

Result: PENDING
