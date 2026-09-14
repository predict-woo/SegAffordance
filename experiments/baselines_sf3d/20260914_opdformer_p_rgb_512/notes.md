# 20260914_opdformer_p_rgb_512 — OPDFormer-P (RGB) at 512x384, retrained on SF3D

**Goal.** Resolution-matched rerun of `20260912_opdformer_p_rgb` (see
`20260914_opdformer_c_rgbd_512/notes.md` for the motivation and the shared data/config facts).

**Setup.** OPDMulti `configs/opd_p_real.yaml`, `--input-format RGB`, recipe unchanged; data
`data/opd_sf3d_512`; `IMG_SIZE=512` (`runpod/baselines/opd/chain.sh p_rgb`).

**Run.** Pod `bl-opd512-prgb` (launch polling for A100/H100 stock from 00:21 UTC 2026-09-14). Run
dir `runs/opd512_p_rgb`, results `results/opd512_p_rgb/`.

**Result.** pending.
