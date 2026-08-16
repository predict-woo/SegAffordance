# 20260817_sf3d_g14_taps — stack stage T: 4-layer DPT-style taps

**Recipe:** g13 + `dinov3_multilayer_taps` (ViT-L blocks 4/11/17 by hook +
final tokens; /8 from tap4, /16 from taps 11+17, /32 from the aligned
tokens as before). Best = epoch 25 (val 0.8192). Throughput ≈ g13 (taps
nearly free).

## Test (best-epoch25; Δ vs g13 isolates the taps)

| metric | g14 | g13 | Δ |
|---|---|---|---|
| PDet | **23.05** | 21.38 | +1.7 (family best) |
| 3D point (m) | **0.2230** | 0.2379 | −1.5 cm (family best) |
| mIoU | 0.2681 | 0.2655 | ≈flat |
| origin / line (m) | 0.2927 / 0.2616 | 0.2939 / 0.2663 | ≈flat |
| radius (m) | 0.1380 | 0.1332 | ≈flat |
| 2D point | 0.1040 | 0.0998 | slightly worse |
| type acc | 90.72 | 93.04 | **−2.3** |
| axis° matched | 19.79 | 16.72 | **+3.1 worse** (all: 27.1 vs 27.9, flat) |
| MA pass | 21.42 | 26.18 | **−4.8** |
| traj_dir acc | 86.52 | 86.18 | flat |
| L_pp_norm | 0.1434 | 0.1041 | worse |

## Reading

Mixed at best: a small localization edge (PDet, 3D point) against real
semantic costs (type −2.3, matched-axis +3°, MA −4.8, consistency worse).
At 512 the final layer evidently retains enough spatial detail that the
early-layer taps mostly dilute the semantically-aligned features. NOT
recommended as default; keep the flag for future revisit (e.g. if
resolution drops or the /16 tap composition is tuned).
