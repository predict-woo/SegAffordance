# 20260721_opdmulti_tune_lr5e6

**Goal:** LR sweep (round 1) for the OPDMulti full fine-tune — high end.

**Setup:** = ft_full recipe, lr 5e-6, 8 epochs. Init from opdreal_base best.

**Result:** best val 0.4661 @ ep2. 300-sample: mIoU 0.595, det 69.7%,
type 96.3%, axis 17.2°.

**Decision:** overfits faster than 2e-6 without gains. LR curve peaks at
~2-3e-6 (1e-5: 68.0%, 5e-6: 69.7%, 3e-6: 70.3%, 2e-6: 71.3%, 1e-6: 69.3%).
