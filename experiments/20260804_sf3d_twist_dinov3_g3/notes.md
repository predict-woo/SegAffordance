# 20260804_sf3d_twist_dinov3_g3

**Goal:** gen-3 twist arm on DINOv3 ViT-L + dino.txt, backbone UNFROZEN
(the frozen gen-2 run was weak everywhere — this tests whether the
features or the freezing was the problem). Same gen-3 stack as
20260804_sf3d_twist_g3 (1-cos, no axis/type heads, pitch-free twists),
same lr 1e-5 as the CLIP arm which also fine-tunes its backbone —
backbone is the only variable vs the g3 CLIP run.

**Setup:** config.yaml (= config/sf3d_train_runpod_twist_dinov3.yaml @ launch).

**Result:** (pending)

**Decision:** (pending)
