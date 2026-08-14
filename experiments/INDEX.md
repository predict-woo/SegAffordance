# Experiment index

One row per experiment, newest last. Details in each experiment's `notes.md`.
Eval columns are on 300 fixed OPDMulti-val samples (seed 0) unless noted;
"det%" = fraction with mask IoU > 0.5.

| id | dataset | recipe | best val loss | mIoU | det% | type% | axis° | verdict |
|---|---|---|---|---|---|---|---|---|
| [20260721_opdreal_base](20260721_opdreal_base/) | OPDReal | from scratch, 30 ep, lr 2e-5 | 0.4069 (ep15) | 0.56* | 67%* | 100%* | 7.0* | pretrain ckpt for all OPDMulti runs |
| [20260721_opdmulti_headsonly](20260721_opdmulti_headsonly/) | OPDMulti | freeze backbone+depth+neck, lr 1e-5 | 0.4917 (ep8) | 0.566 | 65.7% | 96.7% | 18.2 | worst of the three recipes |
| [20260721_opdmulti_ft_full](20260721_opdmulti_ft_full/) | OPDMulti | full fine-tune, lr 1e-5 | 0.4601 (ep0!) | 0.592 | 68.0% | 97.7% | 17.3 | best val loss; overfits after 1 epoch |
| [20260721_opdmulti_ft_lowlr](20260721_opdmulti_ft_lowlr/) | OPDMulti | full fine-tune, lr 3e-6, 8 ep | 0.4654 (ep2) | 0.587 | 70.3% | 97.3% | 16.8 | superseded by tune_lr2e6 |
| [20260721_opdreal_frozenclip](20260721_opdreal_frozenclip/) | OPDReal | frozen CLIP, else = base | 0.4138 (ep11) | 0.525* | 54%* | 100%* | 9.5* | CLIP-ft buys only ~0.007 val; 40% faster |
| [20260721_opdmulti_frozenclip](20260721_opdmulti_frozenclip/) | OPDMulti | frozen CLIP, neck+dec+heads train, lr 1e-5 | 0.4683 (ep0) | 0.581 | 67.3% | 95.7% | 17.5 | overfit isn't CLIP's fault; not better |
| [20260721_opdmulti_tune_lr1e6](20260721_opdmulti_tune_lr1e6/) | OPDMulti | full FT, lr 1e-6, 12 ep | 0.4686 (ep6) | 0.585 | 69.3% | 97.0% | 16.5 | undertrains |
| [20260721_opdmulti_tune_lr2e6](20260721_opdmulti_tune_lr2e6/) | OPDMulti | full FT, lr 2e-6, 10 ep | 0.4698 (ep5) | 0.599 | **71.3%** | 96.7% | 16.6 | **recommended** (69.6% vs 67.9% on 1000-sample holdout vs ft_lowlr) |
| [20260721_opdmulti_tune_lr5e6](20260721_opdmulti_tune_lr5e6/) | OPDMulti | full FT, lr 5e-6, 8 ep | 0.4661 (ep2) | 0.595 | 69.7% | 96.3% | 17.2 | between 3e-6 and 1e-5 |
| [20260721_opdmulti_tune_lr2e6_wd1e3](20260721_opdmulti_tune_lr2e6_wd1e3/) | OPDMulti | lr 2e-6 + wd 1e-3 | 0.4707 (ep5) | 0.599 | 71.3% | 96.7% | 16.6 | wd: no effect |
| [20260721_opdmulti_tune_lr2e6_pdrop25](20260721_opdmulti_tune_lr2e6_pdrop25/) | OPDMulti | lr 2e-6 + proj_dropout 0.25 | 0.4697 (ep5) | 0.598 | 71.3% | 96.7% | 16.5 | dropout: no effect |

| [20260726_opdreal_siglip2l](20260726_opdreal_siglip2l/) | OPDReal | **SigLIP 2 Large** backbone, frozen, else = frozenclip | 0.4087 (ep17) | 0.579† | 68.4%† | 98.2%† | 12.5† | beats frozen CLIP +1.8pt det; ties unfrozen CLIP |
| [20260726_opdreal_dinov3l](20260726_opdreal_dinov3l/) | OPDReal | **DINOv3 ViT-L/16 + dino.txt** backbone, frozen, else = frozenclip | **0.4030** (ep25) | **0.603**† | **71.3%**† | 98.2%† | **10.4**† | **best backbone tested** — beats fine-tuned CLIP on every metric while frozen; likely under-trained at 30 ep |

\* OPDReal numbers are on 24 OPDReal-val samples (its own val set), not the
300-sample OPDMulti eval.

† Backbone-comparison rows (2026-07-26) are on **1000** fixed OPDReal-*valid*
samples, seed 0, via `tools/eval_checkpoint.py` — not comparable to the
24-sample `*` rows above. Re-measured baselines on that same draw:
CLIP RN50 unfrozen (20260721_opdreal_base, ep15) mIoU 0.578 / det 67.9% /
type 98.2% / axis 11.9°; CLIP RN50 frozen (20260721_opdreal_frozenclip, ep11)
mIoU 0.566 / det 66.6% / type 98.1% / axis 11.5°. Raw JSON in
`experiments/eval_results/`.

‡ All rows select the checkpoint by **best val loss**, never by det@0.5 —
det is a reporting metric, not a selection one. Note the two disagree: frozen
CLIP ep26 reaches det 69.3% vs ep11's 66.6% despite worse val loss, and
SigLIP 2 ep24 reaches 69.2% vs ep17's 68.4%. Treat cross-model det gaps under
~1.5pt as noise.
| [20260728_sf3d_twist](20260728_sf3d_twist/) | SF3D v2 | twist+screw, element point, type-input, 16 ep | 0.9891 (ep4) | 0.083† | 2.9%† | 95.1%† | 26.2† | twist metrics: type-from-ω 68%, axis 38.9°, line-dist 4.18 m |
| [20260728_sf3d_2d_twist](20260728_sf3d_2d_twist/) | SF3D v2 | + 2D head + screw track term, 16 ep | 1.0906 (ep15) | 0.093† | 3.7%† | 95.1%† | 26.9† | beats twist arm on ALL twist metrics (75%, 34.9°, 2.63 m); still improving at ep15 |

† SF3D rows: full 43,870-sample val split, not the OPD 300-sample protocol.
| 20260804_sf3d_twist_clip | SF3D | twist arm gen-2: no CVAE, delta-cumsum traj, sign-sensitive twist L2 | val 1.183@ep13, still improving; twist axis 39.6deg, dir 64.8%, traj_dir 75.2%, pass_ma 23.0%, mIoU 0.10 | fix round validated; reference ckpt = last (ep15) |
| 20260804_sf3d_twist_dinov3 | SF3D | gen-2 twist arm on frozen DINOv3 ViT-L + dino.txt | val plateaued ep2 (1.346); twist axis 49.7deg, dir 60.4%, pass_ma 8.1%, mIoU 0.007 | weak — CLIP stays the backbone |
| 20260804_sf3d_2donly | SF3D | 2D-only proof: no 3D GT in training (projection loss + screw-self 1-cos + omega prior) | L_twist fell 0.69->0.60 untrained, twist_dir 60%; axis ~random, type collapsed to trans | mechanism works, signal weak — use as pretraining before 3D finetune; tune omega prior |
| 20260804_sf3d_twist_g3 | SF3D | gen-3 clip: 1-cos, twist-only heads, pitch-free | dir 71.5%/79.7% (up), zigzag gone, mIoU 0.103; type-from-omega 60% (down) | 1-cos validated; type ablation pending |
| 20260804_sf3d_twist_dinov3_g3 | SF3D | gen-3 DINOv3 UNFROZEN, backbone-only variable | best-ever articulation: axis 36.95deg, dir 80.2%, type 71%, line 2.14m; masks broken (0.013) | freezing was the gen-2 problem; mask projector needs dino.txt fix |
| 20260811_sf3d_twist_g4 | SF3D | gen-4 clip: body-metric twist loss + K=4 WTA (twist,traj) bundles, traj weight 4.0 | trajectory FIXED: MSE 0.0099 vs 0.018 zero-motion baseline, traj_dir 82.7%; mIoU 0.118 (best clip); WTA works (spread 0.45, selector 56%) but omega still hedges within cells (|omega| 0.10, type 52%) | keep metric+bundles+fast anneal; omega needs rho bump and/or K=8 (see notes) |
| 20260812_sf3d_twist_g5 | SF3D | gen-5: + radius filter (no knobs <0.10m), rho 0.75, FROZEN CLIP | omega commitment works: rev |omega| 0.59 vs pris 0.10, radius ratio 2.1x (was 9.6), type 89.4%, dir 87.5%/92.6%, axis 29.4deg (all best-ever; filtered eval); masks down (mIoU 0.090, frozen backbone) | new articulation baseline; next: unfreeze ablation for masks, saturate omega |
| 20260813_sf3d_split_g6 | SF3D | gen-6: SPLIT heads (type CE + dir 1-cos + abs-3D origin/point), classical losses, L_pp consistency, no teacher forcing, unfrozen CLIP, K=1 | semantics best-ever: axis 22.1deg (15.1 matched), type 97.5%, MA-pass 42.2, traj_dir 92.0% with K=1 (no zero-motion collapse); localization weak: point3d 0.58m, origin 0.69m, radius_err 0.37m, mIoU 0.098 — pooled-vector 3D regressions, as risked | split validated for semantics; gen-7 = heatmap+depth lifts for both points (proposal in docs/slides) |
| 20260814_sf3d_g7 | SF3D | gen-7: heatmap+depth LIFTS for point+origin (3-channel projector, composed 3D losses), ABSOLUTE trajectory, GT-mask pooling back, FROZEN CLIP | point3d 0.454m (-22%), 2D point 0.138 (gen-4 par), type 98.3%, axis held 23.6deg; origin means flat (grounding-tail), mIoU 0.083 worst (frozen), traj_dir 84% REGRESSED (absolute head; gap p-traj0 0.29m) | keep lift stack; gen-8: traj back to relative+cumsum (or geometric decode), attack masks/grounding |
| 20260814_sf3d_g8_closeup | SF3D | gen-8: CLOSE-UP split (mask>0.25% of image, 19.3k records, FRESH baseline) + Euler-style RELATIVE DIRECT trajectory, else = g7 | mIoU 0.178 + PDet 9.4 (~1.8x/3x any full-split arm — masks work when targets are resolvable); traj_dir 94.8%/0.783 best-ever, curves smooth (no cumsum needed); pt3d 0.30m, origin 0.35m, radius 0.16m; type 92.0 and axis-all 32.8 are the weak spots | relative-direct traj adopted; full-split mask problem reframed as RESOLUTION; best=final epoch (headroom) |
| 20260815_sf3d_g9_closeup010 | SF3D | gen-9: g8 recipe on RELAXED close-up split (mask>0.1% + 5% edge margin, 59.2k records, FRESH baseline; 30 ep) | mIoU 0.146/PDet 5.4 (harder split than g8, between g8 and full); semantics UP on 3x data: type 95.0, axis-all 27.6deg (matched 15.8); pt3d 0.29m, origin 0.36m, traj_dir 93.1% | 0.1% band still resolution-limited at 256^2; serves as ARM A (joint) of the supervision ablation |
| 20260815_sf3d_g9abl_artonly | SF3D | ablation ARM B: g9 recipe MINUS trajectory head + L_traj + L_pp (same 59.2k split) | vs joint: type 92.0 (-3.0), axis-matched 20.4deg (+4.6); origin/mask/point flat (mIoU 0.148, pt3d 0.31m, origin 0.35m) | trajectory co-training regularizes type+axis; origin gains nothing |
| 20260815_sf3d_g9abl_trajonly | SF3D | ablation ARM C: g9 recipe MINUS all articulation heads/losses (2-ch projector, same split) | vs joint: traj_dir 87.6 (-5.5) / cos 0.631 (-0.105); masks BEST of 3 arms (mIoU 0.157, PDet 7.1); pt3d 0.30m flat | articulation supervision strongly helps trajectory; mild task competition on masks; JOINT WINS on both specialized tasks |
