# 20260914_fig5_handvideo_50 — Fig. 5 candidates: 50 held-out frames per hand-video source, both our models, dumps

**What.** 50 held-out records from each 2D source (HOI4D / EPIC / ARCTIC; the joint4 scene split, ratio 0.15 seed 42;
spread over sequences, seed 5, up to 6-8 picks per sequence; EPIC's val split has only 53 records so it is nearly all of
them) rendered by `tools/hoi4d_predict_articulation.py` as `[GT | sf3d_only | dense]` at 2x for picking Fig. 5 samples.
Models: `sf3d_only` = the SceneFun3D-only control `20260913_sf3d_decoder_l2anchor_dense` (best-epoch13),
`dense` = the final model `20260913_joint4_decoder_l2anchor_dense` (best-epoch13); test config
`config/sf3d_test_decoder_rgb_scalefree_dense.yaml`. Old panel style (mask red, point ring, decoded trajectory light
green, axis red, 90-deg orbit yellow, text); the paper figure will be re-rendered from the dumps in the Fig. 3 style once
samples are picked.

**Dumps (tracked).** `<src>/preds.jsonl`: one `tools/sf3d_preds_io.py` record per model x sample (+ `dataset`, `sample`),
i.e. mask RLE, type logits, point 3D/uv, axis, hinge 3D/uv, decoded trajectory -> any re-render without a GPU.

**Exports for the baselines session (volume only, 151 MB).** `/workspace/datasets/handvideo_fig5_samples/<src>/`:
`NN_<src>_frame_512.png` (the stretched 512x512 model input), `NN_<src>_gt_mask_512.png`, `NN_<src>_depth_512.npy`
(HOI4D only), `samples.jsonl` (key, val_idx, desc, gt_type, native_wh, K_native, K_norm, point_uv, 2D track, ARCTIC GT
axis/origin). Requested from the peer session 2026-09-14 ~22:00 UTC: OPDFormer-C 512 (HOI4D, depth), OPDFormer-P 512,
MOPD 512, A3VLM chain, 3DOI (2D line where no depth) -> `results/<model>/handvideo_preds.jsonl`; USDNet not applicable.

**Files.** `<src>/NN_<src>_<rot|trans>_<key>.png`. Sampler fix in this batch: `--per-seq` now means rounds of one pick
per sequence while records remain (the old condition halved the yield on small sources).

**Regen (pod).**
```
DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
CKD=experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt
CKC=experiments/20260913_sf3d_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval1.1144.ckpt
for ds in hoi4d epic arctic; do   # per-seq 2 for hoi4d, 8 for epic, 6 for arctic
  python tools/hoi4d_predict_articulation.py --dataset $ds --model sf3d_only $DENSE $CKC --model dense $DENSE $CKD \
    --num 50 --per-seq <see above> --seed 5 --scale 2 --out viz/20260914_fig5_handvideo_50/$ds \
    --dump viz/20260914_fig5_handvideo_50/$ds/preds.jsonl --export /workspace/datasets/handvideo_fig5_samples/$ds
done
```
Note: HOI4D was rendered before the sampler fix (per-seq 2, 80 sequences, unaffected: every sequence had >= 2 left).
