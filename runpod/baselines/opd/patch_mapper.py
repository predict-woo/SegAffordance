"""Fit-to-data patch for OPDMulti/MOPD's MotionDatasetMapper: accept bitmask GT.

Our SF3D masks are sparse laser-splat footprints stored as COCO RLE; upstream
assumes polygon GT and calls ``convert_coco_poly_to_mask(gt_masks.polygons, ...)``
unconditionally (motion_dataset_mapper.py ~309). With ``INPUT.MASK_FORMAT bitmask``
detectron2 already builds ``BitMasks`` from RLE; this patch just stops the
polygon conversion for that case. Idempotent.

    python patch_mapper.py <repo>/opdformer/mask2former/data/motion_dataset_mapper.py
"""
import sys
from pathlib import Path

OLD = "gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)"
NEW = ("gt_masks = (convert_coco_poly_to_mask(gt_masks.polygons, h, w) if hasattr(gt_masks, 'polygons')"
       " else gt_masks.tensor.to(torch.uint8))  # SF3D baselines: bitmask GT")

for path in sys.argv[1:]:
    p = Path(path)
    src = p.read_text()
    if NEW in src:
        print(f"already patched: {p}")
        continue
    assert src.count(OLD) == 1, f"expected exactly one occurrence in {p}, found {src.count(OLD)}"
    p.write_text(src.replace(OLD, NEW))
    print(f"patched: {p}")
