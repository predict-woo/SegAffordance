import cv2
import argparse
import datetime
import os
import json
import numpy as np
import random
import h5py
from tqdm import tqdm

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from mask2former import MotionVisualizer

from OPDReal.motion_data import register_motion_instances
from OPDReal.config import add_motionnet_config, add_maskformer2_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Output directory
    cfg.OUTPUT_DIR = args.output_dir

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Render motion visualizations")
    parser.add_argument(
        "--config-file",
        default="configs/opd_c_real.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        metavar="DIR",
        help="path containing motion datasets",
    )
    parser.add_argument(
        "--output-dir",
        default=f"train_output/{datetime.datetime.now().isoformat()}",
        metavar="DIR",
        help="path for training output and visualizations",
    )
    parser.add_argument(
        "--output-json",
        default="image_groups.json",
        metavar="FILE",
        help="path to output JSON file for image groups",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of random samples to visualize. Default is 0 for all.",
    )
    parser.add_argument(
        "--is-real",
        action="store_true",
        help="indicating whether to visualize the gt for the MotionREAL dataset",
    )
    parser.add_argument(
        "--is-multi",
        action="store_true",
        default=False,
        help="indication if the dataset is OPDMulti or OPDReal",
    )
    return parser


def register_datasets(data_path, cfg):
    dataset_keys = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
    for dataset_key in dataset_keys:
        json_path = f"{data_path}/annotations_wd/{dataset_key}.json"
        imgs = f"{data_path}/{dataset_key.split('_')[-1]}"
        register_motion_instances(dataset_key, {}, json_path, imgs)


def main():
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    register_datasets(args.data_path, cfg)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    dataset_key = cfg.DATASETS.TEST[0]
    motion_metadata = MetadataCatalog.get(dataset_key)
    dataset_dicts = list(DatasetCatalog.get(dataset_key))

    if args.num_samples > 0 and len(dataset_dicts) > args.num_samples:
        dataset_dicts = random.sample(dataset_dicts, args.num_samples)

    # --- Load H5 files ---
    split_name = dataset_key.split("_")[-1]
    h5_path = f"{args.data_path}/{split_name}.h5"
    h5_file = h5py.File(h5_path, "r")
    images_dset = h5_file[f"{split_name}_images"]
    filenames_dset = h5_file[f"{split_name}_filenames"]
    filenames_map = {
        name.decode("utf-8"): i for i, name in enumerate(list(filenames_dset))
    }
    # --- End Load H5 files ---

    # --- Image Generation Phase ---
    print("Generating visualization images...")
    image_groups = {}
    for d in tqdm(dataset_dicts, desc="Generating Images"):
        img_filename = os.path.basename(d["file_name"])
        if img_filename not in filenames_map:
            logger.warning(f"{img_filename} not in H5 file. Skipping.")
            continue

        img = images_dset[filenames_map[img_filename]][:, :, :3]

        if args.is_real:
            intrinsic_matrix = (
                np.reshape(d["camera"]["intrinsic"], (3, 3), order="F")
                if args.is_multi
                else np.reshape(d["camera"]["intrinsic"]["matrix"], (3, 3), order="F")
            )
            line_length = 0.2
        else:
            intrinsic_matrix = None
            line_length = 1

        for i, anno in enumerate(d["annotations"]):
            base_name = f'{(d["file_name"].split("/")[-1]).split(".")[0]}'
            instance_name = f"{base_name}__{i}.png"
            image_path = os.path.join(cfg.OUTPUT_DIR, instance_name)

            visualizer = MotionVisualizer(img.copy(), metadata=motion_metadata, scale=2)

            current_anno = anno.copy()
            description = current_anno[
                "description"
            ]  # this is the description of the instance

            if args.is_real and not args.is_multi:
                current_anno["motion"]["origin"] = current_anno["motion"][
                    "current_origin"
                ]
                current_anno["motion"]["axis"] = current_anno["motion"]["current_axis"]

            vis = visualizer.draw_gt_instance_clean(
                current_anno,
                {},
                is_real=args.is_real,
                intrinsic_matrix=intrinsic_matrix,
                line_length=line_length,
            )
            cv_out = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, cv_out)

            if base_name not in image_groups:
                image_groups[base_name] = []
            image_groups[base_name].append({"path": image_path, "instance": i})

    print(
        f"Generated {sum(len(g) for g in image_groups.values())} images in {len(image_groups)} groups."
    )

    output_json_path = os.path.join(cfg.OUTPUT_DIR, args.output_json)
    with open(output_json_path, "w") as f:
        json.dump(image_groups, f, indent=4)

    print(f"Image groups metadata saved to {output_json_path}")

    h5_file.close()


if __name__ == "__main__":
    main()
