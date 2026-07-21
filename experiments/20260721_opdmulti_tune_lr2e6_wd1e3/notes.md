# 20260721_opdmulti_tune_lr2e6_wd1e3

**Goal:** Round 2 — does stronger weight decay (1e-3, 10x) improve the
lr 2e-6 winner?

**Setup:** = tune_lr2e6 with weight_decay 0.001.

**Result:** best val 0.4707 @ ep5. 300-sample: mIoU 0.599, det 71.3%,
type 96.7%, axis 16.6° — indistinguishable from wd 1e-4.

**Decision:** no effect at this run length; keep wd 1e-4.
