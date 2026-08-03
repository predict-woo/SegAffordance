# GT twist sanity check: stored sign vs GT trajectory sweep

Verification batch before the sign-sensitive retrain: for 12 stratified val
samples, the GT twist is built exactly as `TwistLoss` builds it
(`twist_from_gt` on the batch's motion/type/origin fields, data loaded
through the training pipeline incl. fast_pipeline), anchored at the GT
interaction point lifted with the input depth map, and its orbit is drawn
in BOTH directions against the GT trajectory:

- yellow = `+t` orbit (the sweep the stored axis sign implies;
  t_max = pi/2 rot / 0.1 m trans, the preprocessor's constants)
- gray = `-t` orbit (the opposite sign)
- cyan = GT trajectory points, start ringed white

Per-sample `sign_check` verdict (velocity-field agreement score) is in the
header AND the filename.

**Result: 12/12 OK — rot and trans.** The cyan GT trajectory lies on the
yellow forward branch in every sample; the stored sign is canonical, as
`tools/sf3d_process.py` implies (it derives the trajectory FROM the signed
axis). The sign-sensitive twist loss (`twist_sign_agnostic: false`) is
therefore well-posed, and the depth-lifted anchor conversion produces
plausible metric depths (0.9-1.9 m in these frames).

- **Tool:** `tools/sf3d_vis_gt_twist.py` (manifest.yaml has the command)
- **Rendered:** 2026-08-04 on the dev pod
