# 20260804_sf3d_2donly

**Goal:** the 2D-only pretraining proof: NO 3D GT in training (twist L2,
3D-trajectory MSE, screw-gt, axis loss, type CE/input all off; type head
not built). The twist head learns only through
GT-2D-track -> L_traj_proj -> 3D trajectory head -> L_screw_self(1-cos)
-> twist, plus the |omega| Occam prior. SF3D's 3D GT is eval-only, so the
twist metrics measure exactly what 2D supervision taught.

**Setup:** config.yaml (= config/sf3d_train_runpod_2donly.yaml @ launch).
Same backbone/heads/schedule as 20260804_sf3d_twist_clip otherwise.

**Result:** 16/16 epochs @ ~650-676 samples/s (leanest loss stack; best
val 0.3911 @ ep10; best/last eval near-identical). THE diagnostic: the
unweighted L_twist column (3D twist error, never trained on) fell 0.69 ->
0.60 (-14%) during pure 2D training, and twist_dir_acc reached 60.3% —
the projection -> trajectory -> screw-self -> twist chain DOES transmit
3D signal. But it is weak: twist axis err 57.0 deg (~random for
sign-agnostic angles), type-from-|omega| 48.4% = the all-trans base rate
(the omega Occam prior shrank |omega| with nothing pushing it up for
rotations), line dist ~200 m, traj_dir_cos 0.09 (image-plane direction is
supervised; the depth component of direction is unconstrained by
projection). Axis head 90.0 deg = untrained constant, as designed
(vae_weight 0). Eval logs: logs/eval2d_best.log, eval2d_last.log.

**Decision:** infrastructure proof PASSED (trains end to end, no
divergence, signal flows to the twist head); capability proof NOT yet —
2D-only-from-scratch does not produce usable articulation at these
settings. Next steps if pursued: use 2D as PRETRAINING before 3D
finetune (the intended regime anyway), tune/anneal screw_omega_shrink
(it collapsed |omega| toward trans), and consider depth-gradient cues to
constrain the out-of-plane direction component.
