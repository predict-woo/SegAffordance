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
