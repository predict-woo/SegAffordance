# Adapted from detectron2.data.datasets.coco.load_coco_json (OPD upstream),
# stripped of the detectron2 dependency: we only need the plain list-of-dicts
# loader; dataset registration / DatasetCatalog machinery was removed.
import contextlib
import io
import logging
import os

import pycocotools.mask as mask_util

logger = logging.getLogger(__name__)


def load_motion_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None
):
    """
    Load a json file with COCO instances annotation format + motionnet hacks

    Args:
            json_file (str): full path to the json file in COCO instances annotation format.
            image_root (str or path-like): the directory where the images in this json file exists.
            dataset_name (str): if given, category ids are remapped to contiguous
                    ids sorted by original id (the detectron2 convention).

    Returns:
            list[dict]: one dict per image with keys file_name/height/width/
            image_id/camera/depth_file_name/annotations.

    Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    id_map = None
    if dataset_name is not None:
        cat_ids = sorted(coco_api.getCatIds())
        id_map = {v: i for i, v in enumerate(cat_ids)}

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    imgs_anns = list(zip(imgs, anns))

    logger.info(f"Loaded {len(imgs_anns)} images in COCO format from {json_file}")

    dataset_dicts = []

    ann_keys = [
        "iscrowd",
        "bbox",
        "keypoints",
        "category_id",
        "motion",
        "object_key",
        "image_id",
    ] + (extra_annotation_keys or [])

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        # 2DMotion changes: add extra annotations from img_dict
        record["camera"] = img_dict["camera"]
        record["depth_file_name"] = img_dict["depth_file_name"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            assert (
                anno.get("ignore", 0) == 0
            ), '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:  # filter out invalid polygons (< 3 points)
                    segm = [p for p in segm if len(p) % 2 == 0 and len(p) >= 6]
                    if len(segm) == 0:
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = 1  # detectron2 BoxMode.XYWH_ABS, kept for compat
            if id_map:
                obj["category_id"] = id_map[obj["category_id"]]
            obj["description"] = anno["description"]
            objs.append(obj)
        record["annotations"] = objs

        dataset_dicts.append(record)

    return dataset_dicts
