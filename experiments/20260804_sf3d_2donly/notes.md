# 20260804_sf3d_2donly

**Goal:** the 2D-only pretraining proof: NO 3D GT in training (twist L2,
3D-trajectory MSE, screw-gt, axis loss, type CE/input all off; type head
not built). The twist head learns only through
GT-2D-track -> L_traj_proj -> 3D trajectory head -> L_screw_self(1-cos)
-> twist, plus the |omega| Occam prior. SF3D's 3D GT is eval-only, so the
twist metrics measure exactly what 2D supervision taught.

**Setup:** config.yaml (= config/sf3d_train_runpod_2donly.yaml @ launch).
Same backbone/heads/schedule as 20260804_sf3d_twist_clip otherwise.

**Result:** (pending)

**Decision:** (pending)
