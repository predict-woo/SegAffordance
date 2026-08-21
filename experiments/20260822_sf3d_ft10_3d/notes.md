# 20260822_sf3d_ft10_3d — label-efficiency ARM B: 2D pretrain → 10% 3D

**Recipe:** the g17 split-axis 3D recipe (+ DCT head, matching the
pretrain) on the ~10% scene partition, initialized from the p90 2D-DCT
pretrain (`20260822_sf3d_p90_2d` best-epoch06; all weights loaded, zero
skipped keys). Spec: 2026-08-22-2d-pretrain-label-efficiency-design.md.
Early-stopped at epoch 24, best = epoch 19 (val 1.4474 — vs scratch-10's
1.8192 floor: the pretrain is worth most of the scratch→full gap in val).

## The headline table — A / B / C (test, 5,088)

| metric | A: 10% scratch | **B: 90% 2D → 10% 3D** | C: g17 (100% 3D) |
|---|---|---|---|
| mIoU | 0.021 | **0.217** | 0.265 |
| PDet | 0.37 | **10.9** | 20.6 |
| axis matched / all | 25.0° / 44.4° | 28.5° / **37.1°** | 16.9° / 25.8° |
| MA | 4.9 | **8.7** | 25.9 |
| origin / radius (m) | 0.555 / 0.253 | 0.534 / 0.249 | 0.257 / 0.122 |
| 3D point | 0.400 | **0.278** | 0.232 |
| traj_dir acc / cos | 80.3 / 0.477 | 81.0 / 0.541 | 94.9 / 0.811 |
| roughness (m) | 0.263 | **0.0176** (best ever) | 0.091 |
| axis flip rate | 21.1 | **13.9** | 13.3 |

## Reading

**The trunk transfers; articulation only partly.**

- B ≫ A exactly where predicted: masks 10× (0.021→0.217), detection 30×
  (0.4→10.9), 3D point 0.40→0.28, roughness 15×. The 2D pretrain
  delivers the perception trunk that 10% of 3D data cannot buy.
- B vs C: masks reach ~82% of full-3D (0.217 vs 0.265) but PDet only
  ~53% and articulation stays far short — MA 8.7 vs 25.9, matched axis
  28.5° vs 16.9°, origin 2× worse. 10% of 3D labels teaches the
  articulation heads *something* (flip rate and axis-all clearly beat
  scratch) but nowhere near the full-data level.
- Verdict on the feasibility question: **partial yes.** 2D pretraining +
  10% 3D produces a working model an order of magnitude beyond 10%
  alone, but it does NOT "somewhat compare" to full 3D on articulation.
- Known confounds before reading this as a ceiling: (1) the p90 pretrain
  early-stopped undertrained (its trunk was 14% below the full-data 2D
  arm — a longer pretrain lifts B's starting point); (2) single 10%
  sample, scene-unbalanced by construction; (3) B keeps the DCT head
  (helps roughness/masks, slightly hurts matched axis per g19_dct).
- Curiosity: B's roughness 0.0176 is the smoothest trajectory of ANY
  model to date (DCT head + 3D supervision compound).

test pass: logs/test.log on the volume (ckpt best-epoch19-valloss1.4474);
pod deleted after the pass — no pods left running.
