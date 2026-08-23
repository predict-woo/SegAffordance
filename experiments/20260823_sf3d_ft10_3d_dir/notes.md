# 20260823_sf3d_ft10_3d_dir — label-efficiency v2 arm B'2: pretrain → 10% 3D

**Recipe:** g21 (g17 + DCT head + dir term 0.1) on the ~10% partition,
init = p90_2d_dir's epoch-26 checkpoint (all weights loaded). Training
was ENDED BY THE VOLUME QUOTA, not by convergence: the epoch-30 best-save
truncated at EDQUOT (3.76G/4.35G, removed) and the process died silently;
a resume died the same way at its next write. Final = **epoch 29**
(val 1.4845; val was moving ~0.001/epoch — effectively at plateau, so
the quota cut costs little). Test log lives in the experiment dir copy
only (volume was read-only-in-practice at test time). metrics.csv =
csv/version_0 (the main run; the brief resume logged to version_1).

## The v2 headline table — A' / B' / C' (test, 5,088; all g21 recipe)

| metric | A': 10% scratch | **B': 90% 2D → 10% 3D** | C': 100% 3D |
|---|---|---|---|
| mIoU | 0.029 | **0.188** | 0.271 |
| PDet | 0.98 | **13.4** | 23.2 |
| MA / MA_signed | 4.9 / 4.3 | **11.4 / 10.4** | 26.6 / 26.4 |
| axis matched / all | 38.4° / 44.3° | **23.0° / 34.5°** | 20.8° / 27.0° |
| flips all / rot | 19.4 / 25.6 | 13.2 / 33.1 | 11.6 / 15.4 |
| origin (m) | 0.516 | 0.571 | 0.278 |
| 3D point | 0.408 | **0.296** | 0.244 |
| traj_dir acc / cos | 80.2 / 0.477 | 82.1 / 0.550 | 88.8 / 0.724 |
| roughness (m) | 0.018 | 0.016 | **0.0085** |

## Reading — v2 vs v1's verdict

Same shape, stronger articulation transfer: B' ≫ A' everywhere that
matters and B' < C'. Vs v1's arm B (g17 recipe, undertrained pretrain):

- **Articulation clearly better**: MA 8.7 → 11.4 (now 43% of the
  100%-data level vs 34% in v1), matched axis 28.5° → 23.0° — within
  2.2° of C'. The better-trained trunk + DCT head pay off where v1
  lagged most.
- **Mask transfer slightly weaker**: mIoU 0.217 → 0.188 (69% of C' vs
  82% in v1) despite the better pretrain trunk — partly the quota cut,
  partly C' itself moved up (0.265 → 0.271). PDet transfer improved
  (10.9 → 13.4).
- The dir term's role in B' is ambiguous given its measured harms in
  C'/B'1 — v2's B-vs-C gaps are NOT attributable to label efficiency
  alone anymore; the term is a confound the v1 comparison didn't have.

**Verdict unchanged: 2D pretraining + 10% 3D is far beyond 10% alone
(MA 2.3×, masks 6×, detection 14×) and still short of full 3D.** The
v2 twist: the articulation gap narrows notably with the DCT head and a
fully-trained pretrain.

test pass: test.log here (ckpt best-epoch29-valloss1.4845; ep30 best was
quota-truncated and deleted). Pod deleted after the pass.
