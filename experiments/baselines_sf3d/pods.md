# Baseline pods — launch record (2026-09-11/12, session ethz-workspace-34)

All on the main volume `bckt1t9uuf` (EU-RO-1). A100 stock was empty most of the
night; `runpod/baselines/launch_poll.sh` retried every 3 min and fell back to
RTX PRO 6000 Server (Blackwell) after 10 attempts. Delete = by
`runpod/baselines/watch_chain.sh` on `CHAIN_DONE` (verify with `runpodctl pod list`).

| pod | id | GPU | image / torch | $/hr | runs |
|---|---|---|---|---|---|
| bl-usdnet | 2djanft93rd8se | A100 80GB PCIe | pytorch:2.1.1-cu121 (torch 2.1.1) | 1.59 | scan download, both dataset conversions, USDNet smoke + 200-epoch train |
| bl-opd-c | mbpohysjdyaoj0 | RTX PRO 6000 Blackwell Server | pytorch:1.0.3-cu1281 (ships torch 2.12+cu130; reinstalled torch 2.8.0+cu128 to match the 12.8 toolkit) | 2.09 | OPDFormer-C RGB-D (launched 00:29 UTC, ETA ~5 h) |
| bl-opd-p | 5cwiwjm8mcsa8r | RTX PRO 6000 Blackwell Server | same as bl-opd-c | 2.09 | OPDFormer-P RGB-D |
| bl-opd-prgb | vbxchs9n8z4xog | A100 80GB PCIe | pytorch:2.1.1-cu121 | 1.59 | OPDFormer-P RGB, then MOPD |

Data on the volume (sync-ignored): `/workspace/datasets/baselines/{data/sf3d_scans (17 GB),
data/opd_sf3d (20 GB), data/usdnet_sf3d, repos, runs, logs, ckpt}`.
