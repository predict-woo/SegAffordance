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

Status 12:20 UTC 2026-09-12: bl-opd-c deleted 07:10, bl-opd-p deleted 08:05, bl-usdnet deleted 12:13
(all by their watchers after CHAIN_DONE); bl-opd-prgb still running the MOPD fine-tune. The 1 cm
USDNet variant conversion (`data/usdnet_sf3d_v1cm`) was paused during training and died with the
pod at 87/182 train scenes; `sf3d_to_usdnet.py --voxel 0.01` resumes it (idempotent per scene).

## A3VLM + 3DOI pods (2026-09-12/13)

| pod | id | GPU | DC / volume | image / torch | $/hr | runs |
|---|---|---|---|---|---|---|
| bl-a3vlm-smoke | 0mrzcrjfoe37m7 | 2 x H100 SXM | EU-FR-1 / bl-eufr (5pw4vigftc) | pytorch:2.0.1-cu118 | 6.98 | env + SPHINX download; smoke OOMed (DP=1 shards nothing); deleted 2026-09-12 23:0x |
| bl-a3vlm | f2sxrh69pz4gfp | 4 x H200 141 GB | AP-JP-1 / bl-apjp (18bdh0pzec, grown 150->400->700 GB) | pytorch:2.0.1-cu118 | 18.36 | A3VLM smoke + 2-epoch run (from 22:59 UTC 2026-09-12), eval, chained-answer regeneration; final model (38 GB) + eval/logs copied to the main volume `results/a3vlm/`; DELETED 00:39 UTC 2026-09-14 |
| bl-3doi | 92w42sw5njjf6r | 2 x H200 141 GB | AP-JP-1 / bl-apjp | pytorch:1.0.3-cu1281 (torch 2.9.1) | 9.18 | 3DOI 62-epoch budget w/ early stopping (from 13:45 UTC 2026-09-13); no EU-RO-1 stock at 2 or 4 GPUs for A100/PRO 6000. Early-stopped after epoch 18 (best val 0.5917 @ epoch 10), exported, results + best checkpoint copied to `results/3doi_runpod/`; DELETED 20:33 UTC 2026-09-14 (~31 h, ~$283) |

Volumes to delete once results are verified and copied to the main volume: bl-apjp (18bdh0pzec,
700 GB, ~$105/month) and the unused bl-eufr (5pw4vigftc, 150 GB, ~$22/month).

## Resolution-matched reruns (2026-09-14, user-approved via the paper session)

Main volume `bckt1t9uuf` (1.5 TB, EU-RO-1), torch 2.1.1 cu121 image, `runpod/baselines/launch_poll.sh`
(A100/H100 SKUs, PRO 6000 fallback). Watchers `runpod/baselines/watch_chain.sh` copy results and
DELETE the pod on CHAIN_DONE.

| pod | id | GPU | $/hr | run | started (UTC) |
|---|---|---|---|---|---|
| bl-opd512-c | 6w98k0sr3v4cma | A100-SXM4-80GB | 1.59 | opd512_c_rgbd (OPDFormer-C RGB-D 512x384) | 00:26; CHAIN_DONE 17:21, deleted 17:22 UTC (~$27) |
| bl-mopd512 | az2ad8iqoruhd5 | A100 80GB PCIe | 1.59 | mopd512_rgb (MOPD, OPDFormer schedule, 512x384) | ~00:33 Sep 14; killed at iter ~32.5k 02:12 UTC Sep 15 (moved to bl-mopd-h200 from model_0029999), deleted (~$41) |
| bl-mopd-h200 | t0ht94598bv263 | H200 (AP-JP-1, volume bl-apjp; Xeon 8460Y+ host) | 4.59 | mopd512_rgb RESUMED from model_0029999.pth at 02:02 UTC Sep 15: 1.9-2.0 s/iter vs 2.73 on the A100; data `opd_sf3d_512` train/valid/test h5 copied to the JP volume (md5-verified), depth not needed for RGB | 01:1x Sep 15; ETA training end ~18:00 UTC, results copied back to the main volume afterwards |
| bl-opd512-prgb | 9r6p29cxu5gwtt | A100 80GB PCIe | 1.59 | opd512_p_rgb (OPDFormer-P RGB 512x384) | ~00:36; CHAIN_DONE 13:39, deleted 13:40 UTC (~$21) |
| bl-usdnet1cm | d2z37fxh1uwutg | A100 80GB PCIe | 1.59 | usdnet_v1cm (USDNet 1 cm; conversion on the dev pod) | ~00:39 |

## Fig. 5 hand-video inference pods (2026-09-15, paper-session request, user pre-authorised)

| pod | id | GPU | $/hr | did | deleted (UTC) |
|---|---|---|---|---|---|
| bl-hv-opd | iemymhvt8aty3p | A100 80GB PCIe (EU-RO-1) | 1.59 | OPDFormer-P RGB 512 + OPDFormer-C 512 on the 150 hand-video frames (`runpod/baselines/handvideo_opd.sh`) | 00:46 (~$0.5) |
| bl-hv-3doi | zdot2q7d8vnhw1 | A100 80GB PCIe | 1.59 | 3DOI on the frames, point_uv and GT-centroid prompts (`handvideo_3doi.sh`) | 00:58 (~$0.6) |
| bl-hv-a3vlm | 3kkh0wx7ifanhk | 2 x H200 (AP-JP-1, bl-apjp) | 9.18 | A3VLM REC -> REG-Joint chain on the frames (`chain.sh MODE=eval_hv`) | 01:04 (~$3.5) |

Outputs: `results/<run>/handvideo_preds*.jsonl`; stages under `/workspace/datasets/baselines/handvideo/`.
