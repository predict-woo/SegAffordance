# 20260817_sf3d_g13_res512 — stack stage R: input resolution 512

**Recipe:** gen-12 (dinov3 + dino.txt, frozen, v3, trajw 0.15) with input
512 (data.input_size + backbone_image_size), point_sigma 16, bs 64, the
39 G 512-px frame cache. Best = epoch 27 (val 0.8288). RTX PRO 6000, ~3.7 h,
~$8. Ran in parallel with g14/g15 (chain attribution is pairwise).

## Test (best-epoch27, 5,088 samples; Δ vs gen-12 @256)

| metric | g13 | g12 | Δ |
|---|---|---|---|
| mIoU | **0.2655** | 0.1599 | **+66% — project record** |
| PDet | **21.38** | 5.09 | **4.2× — project record** (prev best 9.4, gen-8) |
| 2D point | **0.0998** | 0.1138 | −12% (first sub-0.10) |
| 3D point (m) | 0.2379 | 0.2529 | −6% |
| origin vs q* / line (m) | 0.2939 / 0.2663 | 0.3259 / 0.2964 | −3.2 / −3.0 cm |
| radius err (m) | 0.1332 | 0.1734 | −23% |
| axis° all / matched | 27.90 / **16.72** | 37.13 / 29.08 | **−9.2 / −12.4°** |
| MA pass | 26.18 | 12.11 | 2.2× (family best) |
| type acc | 93.04 | 91.45 | +1.6 |
| traj_dir acc | 86.18 | 84.65 | +1.5 (still w=0.15-suppressed) |
| L_pp_norm (probe, 512) | 0.1041 | 0.199 | halved |

## Reading

The survey's "resolution is the binding constraint" call is spectacularly
confirmed: token density 16×16 → 32×32 transformed masks (+66% mIoU, 4×
PDet), repaired most of gen-12's axis degradation, and improved every
localization metric — the close-up mask problem was resolution, full stop.
2–3× the compute per run is the price.

vis: (stack viz batch pending)
