# External A/B — Particulate (CVPR 2026) + closed-form screw loss

Spec: `docs/specs/2026-08-29_external_screw_loss_ab.md`. Scripts: `runpod/external/particulate/`.
Upstream: github.com/RuiningLi/particulate @ dee37a7. Released ckpt `rayli/Particulate/model.pt`,
PartField ckpt `mikaelaangel/partfield-ckpt`.

Slot-in (apply_patch.py): per revolute point, H1 quadratic with p = point, q = predicted
foot point, n = predicted part direction (GT foot points already in their dataloader);
per-part 1−cos anchor (0.5λ); prismatic L1 → 2(1−cos). Their per-point foot L1 (absolute
origin term) is KEPT in both arms; their L1 direction terms are OFF in "ours".
Eval: their evaluate.py (gIoU/Chamfer) + `eval_axis_particulate.py` (Hungarian on
centroids; signed/unsigned angle, flip, origin line distance) on the 77-object PM test
split (SINGAPO ids, 7 categories).

## Status
- 2026-08-29: scripts written. **Blocked on `sapien-sim/PartNetMobility` access** (HF,
  manual gate; requested). Plan: CPU pod for download + `process_urdf`/`cache_points`
  (2,350 assets, multi-core), then GPU pod for PartField features + 4k-step fine-tune A/B.

- 2026-08-30 04:05 CEST: access check with the local HF token (`andyye`, primary email
  gmail) → still 403. The gate auto-approves verifiable academic primary emails; set the
  ETH address as primary on the HF account if the request stalls.

## Results
(pending)


## Volumes deleted (2026-08-31)
All campaign volumes removed at the user's direction; checkpoints/data existed only there. Results survive in bundle/.
