# 20260912_sf3d_decoder_l2anchor_ft_multi3dec — SF3D post-training from the multi3 decoder arm (chain stage 2)

**Goal.** The decoder CHAIN counterpart of the DCT chain (MA 32.80): SF3D post-training (30 ep, lr 1e-5, RGB-only scale-free) with the analytic decoder, closed-form L2 at 2pi + the direct axis loss 0.5, from `20260912_multi3_decoder_rgb_scalefree` best.

**Comparison rows.** DCT chain 32.80 / 32.43, origin 0.256, mIoU 0.254; joint decoder base 31.29, origin 0.273; joint decoder l2anchor (running).

**Status.** prepared 2026-09-12 (autonomous night).
