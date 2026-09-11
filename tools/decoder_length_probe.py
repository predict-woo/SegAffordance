"""Decoder arc-length probe: predicted metric arc length (L~ * z_p) vs the writer's constants on SF3D val."""
import sys, math, os
sys.path.insert(0, '/workspace/SegAffordance'); sys.path.insert(0, '/workspace/SegAffordance/tools')
import torch, numpy as np
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene
from model.losses.geometric import normalized_intrinsics
from sf3d_vis_predictions import load_model
cfg, ck = sys.argv[1], sys.argv[2]
r, m, d = get_default_transforms(image_size=(512, 512))
ds = SF3DDataset(lmdb_data_root='/workspace/datasets/sf3d_processed_v3', lmdb_path='/workspace/datasets/sf3d_processed_v3/data.lmdb', rgb_transform=r, mask_transform=m, depth_transform=d, image_size_for_mask_reconstruction=(512, 512), point_source='element', key_cache_path='/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl', return_trajectory_2d=True, frame_cache_path='/workspace/datasets/sf3d_frames_512.lmdb', fast_pipeline=True, load_depth=False, min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05)
_, va = split_dataset_by_scene(ds, 0.1, 42)
model, mp = load_model(cfg, ck, 'cuda')
rng = np.random.default_rng(0); idx = rng.choice(len(va), 400, replace=False)
rows = []
with torch.no_grad():
    for j in idx:
        it = va[int(j)]
        img_t, depth_t, desc, mask_t, _b, point_gt, motion_gt, type_gt, img_size, fname, origin_3d, K, traj3d, t2d, v2d = it
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float()).cuda()
        out = model(img_t[None].cuda(), depth_t[None].cuda(), model.tokenize([desc], 77).cuda(), None, None, None, None, K_norm)
        zp = float(out.point_3d_pred[0, 2]); L = float(out.trajectory_length[0]) * zp
        gt = traj3d.float().numpy(); gt_len = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
        rows.append((int(type_gt), L, gt_len, int(out.motion_type_logits.argmax(-1)), zp))
rows = np.array(rows)
for t, name in ((0, 'trans'), (1, 'rot')):
    s = rows[rows[:, 0] == t]
    print(f"{name}: n={len(s)} pred arc length metres: median {np.median(s[:,1]):.3f} mean {s[:,1].mean():.3f} p10 {np.percentile(s[:,1],10):.3f} p90 {np.percentile(s[:,1],90):.3f} | GT writer arc: median {np.median(s[:,2]):.3f} | ratio pred/GT median {np.median(s[:,1]/s[:,2]):.2f} | corr {np.corrcoef(s[:,1], s[:,2])[0,1]:.2f}")
print("type acc on the sample:", (rows[:,0] == rows[:,3]).mean())
