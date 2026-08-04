# 20260804_sf3d_twist_g3

**Goal:** gen-3 CLIP twist arm — the twist as the SOLE motion
representation, plus direction-everywhere. vs gen-2
(20260804_sf3d_twist_clip): (1) 1-cos consistency (L_screw_gt/self are
now direction-sensitive — tests the hypothesis that sin^2 licensed the
fold-back zigzag and the under-extension), (2) axis head removed,
(3) type head removed (type emergent from |omega|; NOTE eval M-Pass/type
metrics now MEASURE the emergent type — expect lower than gen-2's 95%
CE-head number by construction, gen-2's emergent figure was 67%),
(4) structurally pitch-free twists (TwistMLP output map projects v
against omega — no helices representable).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).

**Result:** (pending)

**Decision:** (pending)
