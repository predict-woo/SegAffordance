import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from torch.utils.data import DataLoader

import utils.config as config_loader
from datasets.opdreal import OPDRealDataset, get_default_transforms
from train_OPDReal import OPDRealTrainingModule
from evaluation.motion_coco_eval import MotionCocoEval
from utils.dataset import tokenize


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate CRIS model on OPDReal data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the OPDReal dataset"
    )
    parser.add_argument(
        "--dataset_key",
        type=str,
        required=True,
        help="Dataset key, e.g., 'opd_v_real_test'",
    )
    parser.add_argument(
        "--is-multi",
        action="store_true",
        default=False,
        help="indication if the dataset is OPDMulti or OPDReal",
    )
    parser.add_argument(
        "--axis_threshold",
        type=float,
        default=10.0,
        help="Axis angle threshold in degrees for mAP_ADir calculation",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=10,
        help="Interval for running evaluation and updating progress bar",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_output",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the evaluation on"
    )
    return parser


def run_evaluation(
    coco_gt_annos, coco_dt_annos, images_info, output_dir, axis_threshold
):
    """Runs COCO-style evaluation and returns mAP and error for axis direction."""
    if not coco_dt_annos:
        return -1.0, -1.0

    # --- Create COCO-like objects ---
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "images": images_info,
        "annotations": coco_gt_annos,
        "categories": [{"id": 1, "name": "part", "supercategory": "part"}],
    }
    coco_gt.createIndex()

    # --- Run Evaluation ---
    os.makedirs(output_dir, exist_ok=True)

    coco_dt = coco_gt.loadRes(coco_dt_annos)

    dummy_attr_path = os.path.join(output_dir, "dummy_model_attr.json")
    if not os.path.exists(dummy_attr_path):
        with open(dummy_attr_path, "w") as f:
            json.dump({}, f)

    coco_eval = MotionCocoEval(
        coco_gt,
        coco_dt,
        iouType="segm",
        MODELATTRPATH=dummy_attr_path,
        AxisThres=axis_threshold,
    )
    coco_eval.evaluate()
    coco_eval.accumulate()

    p = coco_eval.params
    # First IoU threshold is 0.5
    iou_idx = np.where(p.iouThrs == 0.5)[0][0]
    area_idx = p.areaRngLbl.index("all")
    max_det_idx = p.maxDets.index(100)

    s_err = coco_eval.eval["axis_scores"][iou_idx, :, area_idx, max_det_idx]
    err_adir = np.mean(s_err[s_err > -1]) if np.any(s_err > -1) else -1.0

    s_map = coco_eval.eval["mDP"][iou_idx, :, :, area_idx, max_det_idx]
    map_adir = np.mean(s_map[s_map > -1]) if np.any(s_map > -1) else -1.0

    return map_adir, err_adir


def main():
    parser = get_parser()
    args = parser.parse_args()

    # --- Load Config and Model ---
    cfg = config_loader.load_cfg_from_cfg_file(args.config)

    # Override batch size for evaluation
    cfg.batch_size_val = 1
    cfg.num_workers_val = 0

    model = OPDRealTrainingModule.load_from_checkpoint(
        args.checkpoint, hparams_file=None, cfg=cfg
    )
    model.to(args.device)
    model.eval()

    # --- Setup Dataset ---
    rgb_transform, mask_transform, depth_transform = get_default_transforms(
        image_size=(cfg.input_size[0], cfg.input_size[1])
    )

    dataset = OPDRealDataset(
        data_path=args.data_path,
        dataset_key=args.dataset_key,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        depth_transform=depth_transform,
        is_multi=args.is_multi,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_val,
        shuffle=False,
        num_workers=cfg.num_workers_val,
    )

    # --- Prepare for Evaluation ---
    coco_gt_annos = []
    coco_dt_annos = []
    images_info = []
    ann_id_counter = 1

    # --- Run Inference and Format Results ---
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            (
                img,
                depth,
                word_str_list,
                mask_gt,
                point_gt_norm,
                motion_gt,
            ) = batch

            img = img.to(args.device)
            depth = depth.to(args.device)
            mask_gt = mask_gt.to(args.device)
            point_gt_norm = point_gt_norm.to(args.device)
            motion_gt = motion_gt.to(args.device)

            tokenized_words = tokenize(word_str_list, cfg.word_len, truncate=True).to(
                args.device
            )

            (
                mask_pred_logits,
                point_pred_logits,
                coords_hat,
                motion_pred,
                _,
                _,
            ) = model(img, depth, tokenized_words, None, None, None)

            # --- Ground Truth Annotation ---
            mask_gt_np = mask_gt.cpu().numpy().squeeze()
            h, w = mask_gt_np.shape
            images_info.append({"id": i, "height": h, "width": w})

            rle_gt = mask_util.encode(np.asfortranarray(mask_gt_np.astype(np.uint8)))
            if isinstance(rle_gt["counts"], bytes):
                rle_gt["counts"] = rle_gt["counts"].decode("utf-8")

            gt_anno = {
                "id": ann_id_counter,
                "image_id": i,
                "category_id": 1,
                "segmentation": rle_gt,
                "motion": {
                    "type": "rotation",  # Dummy type for compatibility
                    "current_axis": motion_gt.cpu().numpy().squeeze().tolist(),
                },
                "score": 1.0,
                "area": int(mask_util.area(rle_gt)),
                "iscrowd": 0,
                "ignore": 0,
            }
            coco_gt_annos.append(gt_anno)

            # --- Prediction Annotation ---
            mask_pred_sigmoid = torch.sigmoid(mask_pred_logits).squeeze()
            # Resize predicted mask to match GT mask size if different
            if mask_pred_sigmoid.shape != mask_gt_np.shape:
                mask_pred_sigmoid = torch.nn.functional.interpolate(
                    mask_pred_sigmoid.unsqueeze(0).unsqueeze(0),
                    size=mask_gt_np.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            mask_pred_np = (mask_pred_sigmoid > 0.5).cpu().numpy()
            rle_dt = mask_util.encode(np.asfortranarray(mask_pred_np.astype(np.uint8)))
            if isinstance(rle_dt["counts"], bytes):
                rle_dt["counts"] = rle_dt["counts"].decode("utf-8")

            dt_anno = {
                "id": ann_id_counter,
                "image_id": i,
                "category_id": 1,
                "segmentation": rle_dt,
                "score": 1.0,  # Using a dummy score
                "mtype": 0,  # Dummy motion type (0=rotation)
                "maxis": motion_pred.cpu().numpy().squeeze().tolist(),
                "morigin": [0, 0, 0],  # Dummy origin
                "area": int(mask_util.area(rle_dt)),
            }
            coco_dt_annos.append(dt_anno)

            ann_id_counter += 1

            # --- Periodic Evaluation ---
            if (i + 1) % args.eval_interval == 0 or (i + 1) == len(dataloader):
                map_adir, err_adir = run_evaluation(
                    coco_gt_annos,
                    coco_dt_annos,
                    images_info,
                    args.output_dir,
                    args.axis_threshold,
                )
                pbar.set_description(
                    f"mAP_ADir: {map_adir*100:.2f}% | ERR_ADir: {err_adir:.3f}"
                )

    # --- Save Final Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, "gt.json")
    dt_path = os.path.join(args.output_dir, "dt.json")

    final_gt_dataset = {
        "info": {},
        "images": images_info,
        "annotations": coco_gt_annos,
        "categories": [{"id": 1, "name": "part", "supercategory": "part"}],
    }
    with open(gt_path, "w") as f:
        json.dump(final_gt_dataset, f)
    with open(dt_path, "w") as f:
        json.dump(coco_dt_annos, f)

    # --- Final Full Evaluation ---
    map_adir, err_adir = run_evaluation(
        coco_gt_annos,
        coco_dt_annos,
        images_info,
        args.output_dir,
        args.axis_threshold,
    )

    print("\n--- Final Evaluation Results ---")
    print(f"ERR_ADir @ IoU=0.50: {err_adir:.3f} degrees")
    print(f"mAP_ADir @ IoU=0.50 (Thres={args.axis_threshold} deg): {map_adir*100:.2f}%")


if __name__ == "__main__":
    main()
