# 20260912_mopd_rgb — MOPD (Locate n' Rotate, ACCV'24), RGB, fine-tuned on SF3D

**Goal / setup.** MOPD = OPDFormer-P + EfficientSAM ViT-S image encoder + EfficientNet-B5 "normal"
encoder + extra cross-attention layers in the transformer decoder; their recipe is a 1,000-iteration
fine-tune (lr 5e-6, batch 16, `configs/opd_p_real.yaml` with their `opd_base.yaml`, eval every 50
iters) from a FULL released checkpoint that is Baidu-only. We rebuilt the init from the same
sources (`tools/baselines_sf3d/mopd_compose_ckpt.py`): our `20260912_opdformer_p_rgb`
`model_final.pth` (576 keys) + the public EfficientSAM ViT-S release (153 keys) + geffnet
`tf_efficientnet_b5_ap` ImageNet weights (852 keys); MOPD's own 120 added parameters
(`seg_cross_attention_layers`, normal cross-attention, `input_proj2`) exist in no public checkpoint
and start from their random init (non-strict load, `runpod/baselines/mopd/patch_model_load.py`).
Other fit-to-env patches: `patch_matcher.py` (skips the `uotod` Sinkhorn block whose result their
matcher discards; the package does not import on our pods), the OPD bitmask + numpy patches.
Pod bl-opd-prgb (A100), 11:36-15:10 UTC 2026-09-12 (~3.5 h, dominated by 20 validation passes of
3,495 images; ~$6). Final model = `model_final.pth`.

**Their evaluator (validation, segm, iters 900..1000):** AP50 6.31 / 6.44 / 6.17; axis50 0.81-1.19.
Test split (segm): AP 1.07, AP50 4.99, AP75 0.04, all_motion50 0.65, type50 5.78, origin50 0.01,
axis50 0.73; bbox AP50 8.09.

**Our protocol (oracle best-IoU instance; 3,856/5,088 with any overlap, 267 with conf > 0.5):**

| PDet | mIoU | type % | MA | MA signed | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m |
|---|---|---|---|---|---|---|---|---|
| 30.9 | 0.321 | 67.6 | 13.3 | 12.2 | 47.8 / 32.2 deg | 13.6 / 22.6 | 0.715 | 0.664 |

vs its init OPDFormer-P RGB: 30.9 / 0.320 / 67.9 / 15.6 / 31.7 deg / 0.743.

**Reading.** Within noise of its initialisation on every column (masks identical, MA -2.3, origin
-3 cm): 1,000 iterations at lr 5e-6 with randomly initialised MOPD-specific layers cannot show the
foundation-prior gains the paper reports on OPDMulti. A faithful MOPD number needs their released
checkpoint (Baidu) or a full-length training of the added layers.
