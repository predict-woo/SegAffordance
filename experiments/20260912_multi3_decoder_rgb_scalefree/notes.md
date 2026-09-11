# 20260912_multi3_decoder_rgb_scalefree — the multi-source 2D arm with the analytic decoder (chain stage 1)

**Goal.** multi3 recipe, decoder instead of the DCT head, projection loss on the decoded curve + type CE 0.5, 12 ep. Stage 1 of the decoder chain; stage 2 = `20260912_sf3d_decoder_l2anchor_ft_multi3dec`.

**Comparison rows.** multi3 DCT arm (HOI4D 0.753 / 89.8 / shape 0.0337, EPIC 0.520 / 58.5, ARCTIC 0.694 / 79.6); joint decoder per-source (HOI4D 0.615 / 74.4).

**Status.** prepared 2026-09-12 (autonomous night).
