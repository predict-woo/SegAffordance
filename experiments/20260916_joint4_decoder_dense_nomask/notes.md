# 20260916_joint4_decoder_dense_nomask — ablation: the final recipe WITHOUT the part mask

**Goal (supervisor's question 2026-09-15, user 2026-09-16).** Why predict the mask M at all? The dense readout reads every
articulation quantity as a mask-weighted mean of per-pixel votes (GT mask in training, predicted mask at test), so
"remove M" has to say what replaces the mask as the readout support. Chosen (option 2 of three): pool the votes and
the part-pooled features under the PREDICTED INTERACTION-POINT HEATMAP instead, train and test alike (detached in
train mode), and switch the mask loss off on every source. Tests whether reading over the part beats reading around the
contact point, with no mask supervision anywhere.

**Setup.** `config/joint4_decoder_dense_nomask.yaml` = the final recipe (`config/joint4_decoder_l2anchor_dense.yaml`) +
`model_params.vote_weight_source: point` (new switch in `model/segmenter.py` / `config/opd_train.py`: "mask" default,
"point", "uniform") + `loss_params.{mask,bce,dice}_weight: 0` (the 2d profile inherits them, so hand video loses its mask
loss too). The mask head stays in the graph but is unsupervised: its mIoU / PDet and the +M / +MA / +MAO rates are
meaningless for this row (report MA / axis / origin, dash for masks). Test calls carry
`--model.model_params.vote_weight_source point`.

**Smoke (dev pod).** `fast_dev_run 4`, batch 8: exit 0, 4 train + 4 val batches (loss 11.4 -> 9.2 in four steps).

**Run.** Pod jdec-nomask (RTX PRO 6000), chain `run_joint4dec_dense_nomask_chain.sh`, ~4.5 h + tests. Launched in parallel
with the depth arm (jdec-depth).

**Result.** (pending)
