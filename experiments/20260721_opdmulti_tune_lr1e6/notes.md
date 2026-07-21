# 20260721_opdmulti_tune_lr1e6

**Goal:** LR sweep (round 1) for the OPDMulti full fine-tune — low end.

**Setup:** = ft_full recipe, lr 1e-6, 12 epochs. Init from opdreal_base best.

**Result:** best val 0.4686 @ ep6. 300-sample: mIoU 0.585, det 69.3%,
type 97.0%, axis 16.5°.

**Decision:** undertrains slightly; lr 2e-6 is better.
