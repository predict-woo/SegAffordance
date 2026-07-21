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

\* OPDReal numbers are on 24 OPDReal-val samples (its own val set), not the
300-sample OPDMulti eval.
