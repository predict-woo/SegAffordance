# 20260817_sf3d_g15_costmap — stack stage C: patch-text cost map + gate fix

**Recipe:** g14 + `text_cost_map` (dual half-similarity channels at every
FPN level + the gate reads only the patch-aligned local half). Best =
epoch 28 (val 0.8145, family-lowest). NOTE: carries the taps too — Δ vs
g14 isolates the cost map, but "cost map without taps" was not run.

## Test (best-epoch28; Δ vs g14 isolates the cost map)

| metric | g15 | g14 | Δ |
|---|---|---|---|
| origin / line (m) | **0.2794 / 0.2515** | 0.2927 / 0.2616 | **−1.3 / −1.0 cm (family bests)** |
| radius (m) | **0.1285** | 0.1380 | **family best** |
| traj_dir acc / cos | **89.21 / 0.684** | 86.52 / 0.623 | **+2.7 — best of the w=0.15 arms** |
| axis° matched | 18.10 | 19.79 | −1.7 better |
| 2D / 3D point | 0.1015 / 0.2258 | 0.1040 / 0.2230 | ≈flat |
| mIoU / PDet | 0.2585 / 21.97 | 0.2681 / 23.05 | −1.0 / −1.1 |
| type acc | 90.23 | 90.72 | ≈flat |
| L_pp_norm | 0.1288 | 0.1434 | better |

## Reading

The cost map earns its keep on GEOMETRY: family-best origin/radius, +2.7
traj_dir, better consistency — the explicit text-similarity prior seems to
anchor the articulation triad. Small mask cost vs g14. Since the taps are
themselves questionable, the natural follow-up is **g13 + cost map without
taps** (recorded, not scheduled).
