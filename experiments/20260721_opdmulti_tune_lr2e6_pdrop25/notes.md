# 20260721_opdmulti_tune_lr2e6_pdrop25

**Goal:** Round 2 — does halving projector dropout (0.25 vs 0.5) help the
lr 2e-6 winner?

**Setup:** = tune_lr2e6 with proj_dropout 0.25.

**Result:** best val 0.4697 @ ep5. 300-sample: mIoU 0.598, det 71.3%,
type 96.7%, axis 16.5° — indistinguishable from 0.5.

**Decision:** no effect; keep proj_dropout 0.5. LR is the only knob that
moved detection rate in this campaign.
