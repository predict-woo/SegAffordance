# 20260907_hoi4d_2d_v2_tf

**Goal:** does anchoring the projection loss at the GT knuckle (teacher forcing) beat the detached predicted-point/input-depth anchor? Also the first HOI4D arm whose trajectory term covers ALL categories (the pred_depth chain skipped the 11 depth-less categories in the sweep). Recipe: e100_lr3e5 + trajectory_proj_anchor=gt_point, depth_anchor_source=gt_point; DCT head kept.

**Result:** (pending)
