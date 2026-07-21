# Experiment index

One row per experiment, newest last. Details in each experiment's `notes.md`.
Eval columns are on 300 fixed OPDMulti-val samples (seed 0) unless noted;
"det%" = fraction with mask IoU > 0.5.

| id | dataset | recipe | best val loss | mIoU | det% | type% | axis° | verdict |
|---|---|---|---|---|---|---|---|---|
| [20260721_opdreal_base](20260721_opdreal_base/) | OPDReal | from scratch, 30 ep, lr 2e-5 | 0.4069 (ep15) | 0.56* | 67%* | 100%* | 7.0* | pretrain ckpt for all OPDMulti runs |
| [20260721_opdmulti_headsonly](20260721_opdmulti_headsonly/) | OPDMulti | freeze backbone+depth+neck, lr 1e-5 | 0.4917 (ep8) | 0.566 | 65.7% | 96.7% | 18.2 | worst of the three recipes |
| [20260721_opdmulti_ft_full](20260721_opdmulti_ft_full/) | OPDMulti | full fine-tune, lr 1e-5 | 0.4601 (ep0!) | 0.592 | 68.0% | 97.7% | 17.3 | best val loss; overfits after 1 epoch |
| [20260721_opdmulti_ft_lowlr](20260721_opdmulti_ft_lowlr/) | OPDMulti | full fine-tune, lr 3e-6, 8 ep | 0.4654 (ep2) | 0.587 | 70.3% | 97.3% | 16.8 | **recommended** |

\* OPDReal numbers are on 24 OPDReal-val samples (its own val set), not the
300-sample OPDMulti eval.
