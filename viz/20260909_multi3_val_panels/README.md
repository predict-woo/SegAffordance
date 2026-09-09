# 20260909_multi3_val_panels — the multi-source 2D arm on its three held-out splits

`20260909_multi3_rgb_scalefree` best-epoch06-valloss0.3901.ckpt (HOI4D + EPIC +
ARCTIC, RGB-only scale-free plain-TF recipe, augmentation x8) rendered on 12
held-out records of each source: `hoi4d_v2/` (6 C4 trans + 6 C6 rot, seed
42421), `epic_v1/` and `arctic_v1/` (random, seed 42421). Left GT (moving-
part mask green, 2D track cyan, first-point ring), right prediction (mask
red, point_uv ring, projected trajectory magenta anchored at point_uv at
depth 1 — the unit convention — and p_rev). `contact_sheet.jpg` per source.

Regen (dev pod), one per source:
`python tools/hoi4d_vis_2d_panels.py --config config/<src>_rgb_scalefree.yaml
--ckpt experiments/20260909_multi3_rgb_scalefree/checkpoints/best-epoch06-valloss0.3901.ckpt
--out viz/20260909_multi3_val_panels/<src> --num 12` with src in hoi4d_v2, epic_v1, arctic_v1
(manifest.yaml in each dir).

What it shows: HOI4D 12/12 masks on the moving part, knuckle points,
trajectories in the GT direction — on par with or better than the
HOI4D-only arm (held-out mIoU 0.754 vs 0.733). EPIC: masks on the right
fixture in most panels (oven door, fridge doors, cupboard doors, drawer
fronts, dishwasher door) with some spill onto counters/hands; p_rev 0.84-0.98
on the door-type fixtures and ~0.4 on drawers — the type gate self-organised
on EPIC, which it never did on HOI4D. ARCTIC: masks on the moving part in
12/12 (scissors blade, espresso lever, waffle-iron lid, notebook cover,
laptop screen, microwave door). On both new sources the predicted
trajectories are short and jittery next to the long GT sweeps (2D shape
0.09 / 0.08 vs 0.036 on HOI4D) — the trajectory term is where the multi-
source arm is weakest (its val projection loss overfits from epoch ~8).
