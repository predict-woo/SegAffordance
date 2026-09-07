# 20260907_xeval_sf3d_on_hoi4d — SF3D-trained checkpoints evaluated on the HOI4D v2 held-out split

**Question (user, 2026-09-07):** the reverse of the post-training comparison — how do the pure-SF3D model and the HOI4D-pretrained-then-SF3D-post-trained models do on the HOI4D held-out split (110 objects, ~460 windows)?

**Protocol:** `train_SF3D_better.py test` with `config/hoi4d_v2_teacher_forcing.yaml` (HOI4D v2 depth-complete data, GT-anchored 2D trajectory metric) and each checkpoint via `--ckpt_path`; dev pod (PRO 4000), `--data.batch_size_val 16 --data.num_workers_val 8`. Only the 2D metrics are meaningful on HOI4D (no articulation GT; motion_info is a stub). Logs: `logs/test_<exp>.log` (volume + mirror, gitignored).

| checkpoint | trained on | mIoU | PDet | point err | traj shape | pred roughness |
|---|---|---|---|---|---|---|
| 20260821_sf3d_g19_dct (best-epoch20) | SF3D only | 0.044 | 0.2 | 0.197 | 0.228 | 0.0115 |
| 20260907_sf3d_g19_dct_ft_hoi4d (best-epoch25) | HOI4D dct_baseline → SF3D 30 ep | 0.131 | 2.4 | 0.084 | 0.164 | 0.0101 |
| 20260907_sf3d_g19_dct_ft_hoi4d_tf (best-epoch25) | HOI4D teacher_forcing → SF3D 30 ep | 0.113 | 2.0 | 0.105 | 0.141 | 0.0100 |
| 20260907_hoi4d_2d_v2_teacher_forcing (best-epoch88) | HOI4D only (reference) | 0.727 | 88.0 | 0.0147 | 0.0376 | 0.0085 |

**Result:** (1) pure SF3D does not transfer to HOI4D at all (mIoU 0.044, PDet 0.2) — the domain gap (top-down RGB-D room scans with SF3D descriptions vs hand-held tabletop video with VLM part descriptions) is total. (2) SF3D post-training forgets HOI4D almost completely: from 0.727 / 88.0 to 0.11–0.13 / 2–2.4 after 30 SF3D epochs at lr 1e-5; the residual is still 2–3× the pure-SF3D model on masks and half its point error, so a trace of the HOI4D init survives, but nothing usable. The HOI4D benefit measured on SF3D (MA +5.1, PDet +1.5) is therefore a better INIT, not a model that serves both domains. If a joint model is wanted, the routes are mixed-domain training (HOI4D records in the SF3D batches) or a much shorter/lower-LR post-training with a HOI4D replay term — not run.
