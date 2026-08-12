import io
import os
import json
from pathlib import Path

# from torch._tensor import Tensor # Not directly used, can be removed if not needed elsewhere
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Any, Optional, Callable, Dict, List, Tuple
import tqdm
import lmdb  # Added for LMDB
import pickle  # Added for deserializing LMDB data
import cv2
import shutil
import random

LMDB_DATASET_VERSION_COMPATIBLE = "1.0"  # For checking compatibility

# Dataloader workers each fork their own cv2, whose default per-op thread
# pool oversubscribes the (quota-limited) CPUs N-workers-fold and thrashes.
# All cv2 calls here are small (256-480px) — single-threaded is fastest.
# Set at import time so forked workers inherit it.
cv2.setNumThreads(0)

# ImageNet stats, matching get_default_transforms. The fast pipeline ships
# raw uint8 RGB and the model normalizes on GPU (model/segmenter.py uses
# these same constants).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SF3DDataset(Dataset):
    """
    PyTorch Dataset for loading processed SceneFun3D items from an LMDB database.
    Each sample corresponds to a specific item visible in a specific frame.
    """

    def __init__(
        self,
        lmdb_data_root: str,  # Changed from processed_data_root
        rgb_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        # skip_items_without_motion: bool = True, # This logic is now handled during LMDB creation
        image_size_for_mask_reconstruction: Tuple[int, int] = (
            224,
            224,
        ),  # Needed if original size not stored
        lmdb_path: Optional[str] = None,
        sensor_max_occluded_frac: Optional[float] = 0.5,
        min_revolute_radius: float = 0.0,
        key_cache_path: Optional[str] = None,
        return_trajectory_2d: bool = False,
        point_source: str = "motion_origin",
        frame_cache_path: Optional[str] = None,
        fast_pipeline: bool = False,
    ):
        """
        Args:
            lmdb_data_root (str): Path to the root directory of the LMDB dataset
                                     (output of process_frames_with_items.py).
            rgb_transform (callable, optional): Optional transform to be applied on the RGB image.
            mask_transform (callable, optional): Optional transform for the reconstructed mask.
            depth_transform (callable, optional): Optional transform for the placeholder depth map.
            image_size_for_mask_reconstruction (Tuple[int, int]): The target size (height, width)
                                                                  for reconstructing the mask if original
                                                                  dimensions aren't available per sample.
                                                                  This should match the size your model expects
                                                                  if you're not resizing the mask later.
        """
        self.lmdb_data_root = Path(lmdb_data_root)
        self.rgb_transform = rgb_transform
        self.mask_transform = (
            mask_transform  # Retained for potential use after mask reconstruction
        )
        self.depth_transform = depth_transform
        self.image_size_for_mask_reconstruction = (
            image_size_for_mask_reconstruction  # (height, width)
        )

        # This used to be hardcoded to Path("/dev/shm/data.lmdb"), which
        # silently ignored lmdb_data_root: whatever happened to be staged in
        # shm was loaded instead of the configured dataset, with no warning.
        # Default to the configured root; pass lmdb_path explicitly to use a
        # shm copy.
        #
        # Staging in shm is still worth it for training. A full pass over the
        # keys touches every leaf page (LMDB stores values inline), and on the
        # MooseFS-backed volume that runs at ~1.4 MB/s -- hours for a 13 GB
        # database -- while a sequential copy runs at ~155 MB/s:
        #     cp /workspace/datasets/sf3d_processed_v2/data.lmdb/data.mdb \
        #        /dev/shm/data.lmdb/
        self.lmdb_path = (
            Path(lmdb_path) if lmdb_path else self.lmdb_data_root / "data.lmdb"
        )
        self.sensor_max_occluded_frac = sensor_max_occluded_frac
        # Drop revolute records whose element rotates about an axis closer
        # than this (metres): the knob/dial/faucet mode. Measured on the v2
        # train split the revolute radius distribution is BIMODAL — 40% sit
        # under 3 cm (p25 = 1.4 cm), then a gap to the door/cabinet mode at
        # 0.3-0.5 m — and the knob mode supervises omega through pure
        # ambiguity (element ON the axis: no field signal, stub trajectory,
        # unknowable sign), which drove the gen-4 omega hedge. 0.10 sits in
        # the gap: it removes exactly the knob mode. Prismatic records are
        # never dropped. 0.0 disables (legacy behaviour).
        self.min_revolute_radius = min_revolute_radius
        self.key_cache_path = Path(key_cache_path) if key_cache_path else None

        # OFF by default: the extra columns cost decode time and only the 2D
        # arm consumes them. The 15-tuple is understood end to end
        # (model/targets.unpack_batch normalises the pixels and derives
        # anchor_depth; the test paths tolerate the extra columns), so turning
        # this on is the whole switch for the 2D arm.
        self.return_trajectory_2d = return_trajectory_2d

        # What the interaction point (tuple element 5) is supervised with:
        #   "motion_origin" — the projected motion origin (historical). For
        #       rotation that is the HINGE, which sits outside the element's
        #       mask 63% of the time and gets clamped to the image border when
        #       off-screen — an actively wrong target whenever the hinge is
        #       not on a visible surface.
        #   "element" — the projected element centroid (trajectory_2d[0] in
        #       the v2 LMDB): a graspable on-element point for BOTH motion
        #       types. Pair with the twist head, which carries the rotation
        #       axis as a line and so no longer needs a hinge point at all.
        if point_source not in ("motion_origin", "element"):
            raise ValueError(
                f"point_source must be 'motion_origin' or 'element', got {point_source!r}"
            )
        self.point_source = point_source

        # Fast per-sample path (profiled 2026-08-03: legacy ~25 ms/sample of
        # pure CPU, which under RunPod's ~10-core cgroup quota caps the whole
        # pipeline at ~310 samples/s — below the training GPU): cv2 JPEG
        # decode straight to target size, raw uint8 CHW RGB (the model
        # normalizes on GPU — model/segmenter.py), numpy->tensor depth with
        # no PIL round-trip, mask splatted into a reused per-worker buffer
        # with the bbox taken from the coordinate list instead of a full-res
        # np.where. Masks, depth, bbox and metadata are BIT-IDENTICAL to the
        # legacy path (nearest gather on the exact PIL grid); only the RGB
        # resize kernel differs (cv2 INTER_AREA vs PIL bilinear, mean abs
        # diff ~0.005 normalized), so runs that must stay exactly comparable
        # to pre-2026-08 checkpoints keep this off. Requires frame_cache_path.
        self.fast_pipeline = fast_pipeline
        if fast_pipeline and frame_cache_path is None:
            raise ValueError("fast_pipeline=True requires frame_cache_path")
        self._mask_bufs: Dict[Tuple[int, int], np.ndarray] = {}
        self._nearest_grids: Dict[Tuple[int, int], tuple] = {}

        # Optional frame cache (tools/sf3d_build_frame_cache.py): one LMDB of
        # training-sized frame bytes. Without it every sample pulls ~826 KB of
        # full-res files through the FUSE mount, whose single daemon caps the
        # whole pipeline at ~73 samples/s no matter how many workers
        # (profiled 2026-07-28); with it, ~81 KB from one mmap-served file.
        self.frame_env = None
        if frame_cache_path is not None:
            frame_cache = Path(frame_cache_path)
            if not frame_cache.exists():
                raise FileNotFoundError(f"frame cache not found at {frame_cache}")
            self.frame_env = lmdb.open(
                str(frame_cache), readonly=True, lock=False,
                readahead=False, meminit=False,
            )
            with self.frame_env.begin(write=False) as ftxn:
                meta = pickle.loads(ftxn.get(b"__metadata__"))
            cache_size = meta["depth_size"]
            th, tw = self.image_size_for_mask_reconstruction
            if (th, tw) != (cache_size, cache_size):
                raise ValueError(
                    f"frame cache was built for input size {cache_size}, "
                    f"dataset wants {(th, tw)} — rebuild with "
                    f"tools/sf3d_build_frame_cache.py --depth-size"
                )

        if not self.lmdb_path.exists():
            raise FileNotFoundError(f"LMDB database not found at {self.lmdb_path}")

        print(f"Opening LMDB database at {self.lmdb_path}")
        self.env = lmdb.open(
            str(self.lmdb_path),
            readonly=True,
            lock=False,  # Important for multi-process reading if not careful
            readahead=False,  # Usually not beneficial for many small random reads
            meminit=False,  # Only if you trust the DB file completely or manage memory manually
        )
        print(f"LMDB database opened")

        print(f"Getting item keys")
        self.item_keys = self._get_item_keys()
        print(f"Item keys: length {len(self.item_keys)}")

    def _get_item_keys(self) -> List[bytes]:
        """Keys to train on, minus records the ARKit sensor says are occluded.

        ``sensor_check`` is written by tools/sf3d_process.py: it compares the
        annotated element's laser points against the real ARKit LiDAR frame.
        hires_depth is RENDERED from the laser scan, so the writer's own
        visibility test cannot see a surface that is missing from the scan or
        that moved between the scan and the video (a closed door, a glossy
        tiled wall). Measured over the full 458,264-record database, 8.97% of
        records sit behind a real measured surface at a 0.5 cutoff -- and the
        number barely moves with the threshold (8.26% at 0.7, 7.58% at 0.9),
        so these are decisive, not borderline.

        Records with ``sensor_check`` None (1.5%: no ARKit frame paired within
        the writer's time window) are KEPT, since there is no evidence against
        them. Set sensor_max_occluded_frac=None to disable filtering entirely.
        """
        cutoff = self.sensor_max_occluded_frac
        min_rad = self.min_revolute_radius

        # Cache validity is keyed on the record COUNT, not the path, so a cache
        # built from a /dev/shm copy stays valid for the same database on the
        # volume (and vice versa).
        with self.env.begin(write=False) as txn:
            entry_count = txn.stat()["entries"]

        if self.key_cache_path and self.key_cache_path.is_file():
            cached = pickle.loads(self.key_cache_path.read_bytes())
            if (
                cached.get("cutoff") == cutoff
                and cached.get("entries") == entry_count
                and cached.get("min_revolute_radius", 0.0) == min_rad
            ):
                print(
                    f"Loaded {len(cached['keys'])} keys from cache "
                    f"{self.key_cache_path} (cutoff={cutoff}, min_rev_radius={min_rad})"
                )
                return cached["keys"]

        keys: List[bytes] = []
        dropped = 0
        dropped_radius = 0
        unverified = 0
        scan = cutoff is not None or min_rad > 0.0
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            if not scan:
                for key, _ in cursor:
                    if key != b"__metadata__":
                        keys.append(key)
            else:
                for key, value in cursor:
                    if key == b"__metadata__":
                        continue
                    record = pickle.loads(value)
                    if cutoff is not None:
                        sensor = record.get("sensor_check")
                        if sensor is None:
                            unverified += 1
                        elif sensor.get("sensor_occluded_frac", 0.0) > cutoff:
                            dropped += 1
                            continue
                    if min_rad > 0.0 and self._revolute_radius_below(record, min_rad):
                        dropped_radius += 1
                        continue
                    keys.append(key)

        if scan:
            total = len(keys) + dropped + dropped_radius
            print(
                f"Sensor filter (occluded_frac > {cutoff}): dropped {dropped} of "
                f"{total} records ({100.0 * dropped / max(1, total):.2f}%); "
                f"{unverified} kept without a sensor verdict"
            )
            if min_rad > 0.0:
                print(
                    f"Radius filter (revolute radius < {min_rad} m): dropped "
                    f"{dropped_radius} ({100.0 * dropped_radius / max(1, total):.2f}%)"
                )

        if self.key_cache_path:
            self.key_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_cache_path.write_bytes(
                pickle.dumps(
                    {
                        "cutoff": cutoff,
                        "entries": entry_count,
                        "min_revolute_radius": min_rad,
                        "keys": keys,
                    }
                )
            )
            print(f"Cached key list -> {self.key_cache_path}")

        return keys

    @staticmethod
    def _revolute_radius_below(record: dict, min_rad: float) -> bool:
        """True iff the record is revolute AND its element rotates about an
        axis closer than ``min_rad`` metres (the knob/dial mode). Records
        missing any needed field are KEPT (no evidence against them)."""
        motion_info = record.get("motion_info") or {}
        original = motion_info.get("original_motion_data") or {}
        if original.get("motion_type", "trans") not in ("rot", "rotation"):
            return False
        frame = motion_info.get("frame_specific_motion_data") or {}
        axis_dir = frame.get("motion_dir_3d_camera_coords")
        origin = frame.get("motion_origin_3d_camera_coords")
        traj = record.get("trajectory_3d_camera_coords")
        if not axis_dir or not origin or not traj:
            return False
        d = np.asarray(axis_dir, dtype=np.float64)
        n = np.linalg.norm(d)
        if n < 1e-8:
            return False
        d /= n
        rel = np.asarray(traj[0], dtype=np.float64) - np.asarray(origin, dtype=np.float64)
        radius = float(np.linalg.norm(rel - np.dot(rel, d) * d))
        return radius < min_rad

    def __len__(self) -> int:
        return len(self.item_keys)

    def __getitem__(self, idx: int):
        item_key_bytes = self.item_keys[idx]
        with self.env.begin(write=False) as txn:
            item_data_bytes = txn.get(item_key_bytes)
            if item_data_bytes is None:
                raise IndexError(f"Key {item_key_bytes.decode()} not found in LMDB.")

        item_data = pickle.loads(item_data_bytes)

        # --- Load RGB Image ---
        # Path is relative to lmdb_data_root/images
        rgb_image_filename = item_data["rgb_image_path"]
        rgb_image_actual_path = self.lmdb_data_root / "images" / rgb_image_filename
        target_h, target_w = self.image_size_for_mask_reconstruction

        frame_blob = None
        if self.frame_env is not None:
            with self.frame_env.begin(write=False) as ftxn:
                raw = ftxn.get(rgb_image_filename.encode())
            if raw is None:
                raise KeyError(
                    f"frame cache has no entry for {rgb_image_filename} — "
                    "rebuild with tools/sf3d_build_frame_cache.py"
                )
            frame_blob = pickle.loads(raw)

        if self.fast_pipeline:
            # cv2 decode + INTER_AREA resize, out as uint8 CHW RGB. Skips the
            # whole PIL/ToTensor/Normalize stack (~11 ms/sample); the model
            # normalizes on GPU (CRIS.forward, on img.dtype == uint8).
            original_width, original_height = frame_blob["orig_size"]
            bgr = cv2.imdecode(
                np.frombuffer(frame_blob["jpeg"], np.uint8), cv2.IMREAD_COLOR
            )
            if bgr.shape[:2] != (target_h, target_w):
                bgr = cv2.resize(
                    bgr, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
            rgb_image_tensor = torch.from_numpy(
                np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))
            )
        elif frame_blob is not None:
            # Cached frames are already draft-sized; the original dimensions
            # (which mask coordinates and point normalisation live in) were
            # recorded at build time because the small JPEG no longer knows them.
            original_width, original_height = frame_blob["orig_size"]
            rgb_image_pil = Image.open(io.BytesIO(frame_blob["jpeg"])).convert("RGB")
        else:
            # One sequential read, then decode from memory: on the MooseFS
            # volume per-file metadata ops (open/stat) and buffered small reads
            # are the dominant cost, not bytes (profiled 2026-07-28).
            rgb_image_pil = Image.open(io.BytesIO(rgb_image_actual_path.read_bytes()))
            # The ORIGINAL dimensions come from the header, before any decode —
            # mask coordinates and point normalisation are in original pixels.
            original_width, original_height = rgb_image_pil.size
            # Draft-mode JPEG decode: per-sample CPU is dominated by decoding
            # 1920x1440 JPEGs that are immediately resized to 256. draft()
            # decodes DCT-downscaled (here 1/4 -> 480x360), ~6x cheaper, and
            # the follow-up resize is from a 4x smaller source. No-op for
            # non-JPEG files.
            rgb_image_pil.draft("RGB", (target_w, target_h))
            rgb_image_pil = rgb_image_pil.convert("RGB")

        if not self.fast_pipeline:
            if self.rgb_transform:
                rgb_image_tensor = self.rgb_transform(rgb_image_pil)
            else:
                rgb_image_tensor = transforms.ToTensor()(
                    rgb_image_pil
                )  # Default if no transform
        
        # --- Load Depth Image (or create placeholder) ---
        depth_image_filename = item_data.get("depth_image_path")
        depth_pil = None
        depth_image_tensor = None
        if frame_blob is not None:
            depth_np_uint16 = cv2.imdecode(
                np.frombuffer(frame_blob["depth_png"], np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            if depth_np_uint16.shape != (target_h, target_w):
                depth_np_uint16 = cv2.resize(
                    depth_np_uint16, (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            if self.fast_pipeline:
                # Already target-sized: tensor directly, mm -> m. The legacy
                # PIL "F" round-trip only fed a Resize that was a no-op here.
                depth_image_tensor = torch.from_numpy(
                    depth_np_uint16.astype(np.float32)
                ).unsqueeze(0).div_(1000.0)
            else:
                depth_pil = Image.fromarray(
                    depth_np_uint16.astype(np.float32) / 1000.0, mode="F"
                )
        elif depth_image_filename:
            depth_image_actual_path = self.lmdb_data_root / "depth" / depth_image_filename
            # No exists() probe: that is one more FUSE stat per sample; the
            # read below raises the same information.
            try:
                # SceneFun3D depth is a 16-bit PNG in millimetres. Read as one
                # sequential blob, decode in memory (cv2.imread's access
                # pattern is slow over FUSE), and nearest-resize the uint16
                # BEFORE the float conversion so the mm->m cast runs on 256^2
                # pixels rather than the full 1920x1440. The old PIL-decode +
                # full-res float path was ~56 ms/sample, the single largest
                # per-sample cost (profiled 2026-07-28).
                depth_blob = np.frombuffer(
                    depth_image_actual_path.read_bytes(), dtype=np.uint8
                )
                depth_np_uint16 = cv2.imdecode(depth_blob, cv2.IMREAD_UNCHANGED)
                if depth_np_uint16 is None:
                    raise IOError("cv2.imdecode returned None")
                depth_np_uint16 = cv2.resize(
                    depth_np_uint16,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_np_float32 = depth_np_uint16.astype(np.float32) / 1000.0
                depth_pil = Image.fromarray(depth_np_float32, mode="F")
            except Exception as e:
                print(f"Warning: could not load depth image {depth_image_actual_path}. Using zero depth. Error: {e}")
        
        if depth_image_tensor is None:
            if depth_pil is None:
                # Create a placeholder if depth image is not found or fails to load
                # zero_depth = np.zeros((original_height, original_width), dtype=np.float32)
                # depth_pil = Image.fromarray(zero_depth, mode="F")
                raise ValueError(f"Depth image not found at {depth_image_actual_path}")

            if self.depth_transform:
                depth_image_tensor = self.depth_transform(depth_pil)
            else:
                depth_image_tensor = transforms.ToTensor()(depth_pil)


        mask_coords_yx = item_data.get("mask_coordinates_yx", [])
        if self.fast_pipeline:
            # bbox straight from the coordinate list (identical to the legacy
            # full-res np.where — the coords ARE the set pixels), splat into a
            # REUSED per-worker buffer (no 2.7 MB alloc per sample), then
            # nearest-downsample as a direct gather on PIL's sampling grid
            # (src = floor((dst+0.5)*scale), verified bit-identical to
            # PIL NEAREST — cv2.INTER_NEAREST uses a different grid and
            # disagreed at IoU 0.4-0.6 on these sparse splat masks). Zero only
            # the touched pixels afterwards so the buffer stays reusable.
            if mask_coords_yx:
                coords = np.asarray(mask_coords_yx, dtype=np.int64)
                rows_a, cols_a = coords[:, 0], coords[:, 1]
                bbox_tensor = torch.tensor(
                    [cols_a.min(), rows_a.min(),
                     cols_a.max() - cols_a.min(), rows_a.max() - rows_a.min()],
                    dtype=torch.float32,
                )
                buf = self._mask_bufs.get((original_height, original_width))
                if buf is None:
                    buf = np.zeros((original_height, original_width), dtype=np.uint8)
                    self._mask_bufs[(original_height, original_width)] = buf
                grid = self._nearest_grids.get((original_height, original_width))
                if grid is None:
                    r_idx = np.minimum(
                        ((np.arange(target_h) + 0.5) * (original_height / target_h)).astype(np.int64),
                        original_height - 1,
                    )
                    c_idx = np.minimum(
                        ((np.arange(target_w) + 0.5) * (original_width / target_w)).astype(np.int64),
                        original_width - 1,
                    )
                    grid = np.ix_(r_idx, c_idx)
                    self._nearest_grids[(original_height, original_width)] = grid
                buf[rows_a, cols_a] = 255
                small = buf[grid]
                buf[rows_a, cols_a] = 0
                mask_tensor = torch.from_numpy(
                    (small > 127).astype(np.float32)
                ).unsqueeze(0)
            else:
                bbox_tensor = torch.zeros(4, dtype=torch.float32)
                mask_tensor = torch.zeros((1, target_h, target_w), dtype=torch.float32)
        else:
            mask_np = np.zeros((original_height, original_width), dtype=np.uint8)
            if mask_coords_yx:  # Ensure there are coordinates
                rows, cols = zip(*mask_coords_yx)  # Separate y and x
                mask_np[np.array(rows), np.array(cols)] = 255  # Fill in the mask

            mask_pil = Image.fromarray(
                mask_np, mode="L"
            )  # Convert to PIL Image (Grayscale)

            # --- Bounding Box from Mask ---
            rows, cols = np.where(mask_np > 0)
            if rows.size > 0:
                x_min, x_max = cols.min(), cols.max()
                y_min, y_max = rows.min(), rows.max()
                bbox_tensor = torch.tensor([x_min, y_min, x_max - x_min, y_max - y_min], dtype=torch.float32)
            else:
                bbox_tensor = torch.zeros(4, dtype=torch.float32)

            if self.mask_transform:
                mask_tensor = self.mask_transform(mask_pil)
            else:
                # Default: resize to a fixed size (e.g., same as RGB transform if any) and convert to binary tensor
                # This should align with what get_default_transforms provided previously.
                # Let's use self.image_size_for_mask_reconstruction for consistency.
                default_mask_processing = transforms.Compose(
                    [
                        transforms.Resize(
                            self.image_size_for_mask_reconstruction,
                            interpolation=transforms.InterpolationMode.NEAREST,
                        ),
                        transforms.ToTensor(),
                        lambda x: (x > 0.5).float(),  # Ensure binary {0., 1.}
                    ]
                )
                mask_tensor = default_mask_processing(mask_pil)

        # --- Load Description ---
        description = item_data["description"]

        # --- Load Motion Info & Interaction Point ---
        motion_info = item_data["motion_info"]
        origin_2d_image_coord_norm = torch.zeros(2, dtype=torch.float32)  # Default
        motion_dir_3d_camera_coords = torch.zeros(3, dtype=torch.float32)
        motion_origin_3d_camera_coords = torch.zeros(3, dtype=torch.float32)
        motion_type = "trans" # default

        frame_specific_motion = motion_info.get("frame_specific_motion_data")
        if frame_specific_motion:
            origin_2d = frame_specific_motion.get(
                "motion_origin_2d_image_coords"
            )  # These are (x,y) from original file
            if origin_2d is not None and len(origin_2d) == 2:
                # Normalize origin_2d_image_coord (x, y)
                # The original code normalized by original_width, original_height
                # item_data["motion_info"] contains original, non-normalized coordinates.
                norm_x = origin_2d[0] / original_width if original_width > 0 else 0.0
                norm_y = origin_2d[1] / original_height if original_height > 0 else 0.0
                origin_2d_image_coord_norm = torch.tensor(
                    [norm_x, norm_y], dtype=torch.float32
                )

            motion_vec = frame_specific_motion.get("motion_dir_3d_camera_coords")
            if motion_vec is not None and len(motion_vec) == 3:
                motion_dir_3d_camera_coords = torch.tensor(
                    motion_vec, dtype=torch.float32
                )

            origin_3d = frame_specific_motion.get("motion_origin_3d_camera_coords")
            if origin_3d is not None and len(origin_3d) == 3:
                motion_origin_3d_camera_coords = torch.tensor(
                    origin_3d, dtype=torch.float32
                )
        
        if motion_info.get("original_motion_data"):
            motion_type = motion_info["original_motion_data"].get("motion_type", "trans")

        # Retarget the interaction point to the element itself (see __init__).
        # trajectory_2d_image_coords[0] is the projected element centroid —
        # index 0 of the stored polyline is also index 0 after the linspace
        # subsample, so no index mapping is needed. Falls back to the motion
        # origin when the centroid projects outside the frame (partially
        # visible element), which the valid flag marks.
        if self.point_source == "element":
            coords_2d = item_data.get("trajectory_2d_image_coords")
            valid_2d = item_data.get("trajectory_2d_valid")
            if coords_2d is None or valid_2d is None:
                raise ValueError(
                    f"point_source='element' but item {item_key_bytes.decode()} has no "
                    "2D trajectory. Records written before 2026-07-28 lack these fields; "
                    "rebuild with tools/sf3d_process.py or use sf3d_processed_v2."
                )
            if len(coords_2d) > 0 and bool(valid_2d[0]):
                origin_2d_image_coord_norm = torch.tensor(
                    [
                        coords_2d[0][0] / original_width if original_width > 0 else 0.0,
                        coords_2d[0][1] / original_height if original_height > 0 else 0.0,
                    ],
                    dtype=torch.float32,
                )

        # Clamp the normalized coordinates to be within [0, 1]
        origin_2d_image_coord_norm = torch.clamp(origin_2d_image_coord_norm, 0.0, 1.0)
        
        motion_type_map = {"trans": 0, "translation": 0, "rot": 1, "rotation": 1}
        motion_type_tensor = torch.tensor(motion_type_map.get(motion_type, 0), dtype=torch.long)

        image_size_tensor = torch.tensor([original_width, original_height], dtype=torch.float32)

        # --- Load Camera Intrinsics ---
        intrinsics_list = item_data.get("camera_intrinsics")
        if intrinsics_list is None:
            raise ValueError(f"Camera intrinsics not found in LMDB item {item_key_bytes.decode()}. This indicates a data processing error.")
        camera_intrinsic_matrix = torch.tensor(intrinsics_list, dtype=torch.float32)

        # --- Load Trajectory Data ---
        trajectory_3d_camera_coords = item_data.get("trajectory_3d_camera_coords", [])
        trajectory_indices = None
        if trajectory_3d_camera_coords:
            trajectory_tensor = torch.tensor(trajectory_3d_camera_coords, dtype=torch.float32)
            # Sample 20 points uniformly from the trajectory to match model output
            num_points = 20
            if len(trajectory_tensor) > 0:
                indices = torch.linspace(0, len(trajectory_tensor) - 1, num_points).long()
                trajectory_indices = indices
                trajectory_tensor = trajectory_tensor[indices]
            else:
                # If no trajectory, return zeros but with the correct shape
                trajectory_tensor = torch.zeros((num_points, 3), dtype=torch.float32)
        else:
            # Create empty trajectory if not available
            raise ValueError(f"Trajectory not found in LMDB item {item_key_bytes.decode()}")
            # trajectory_tensor = torch.zeros((20, 3), dtype=torch.float32)

        # --- 2D trajectory (only when explicitly requested; see __init__) ---
        # Written by tools/sf3d_process.py as the SAME curve projected into
        # this frame, so it is subsampled with the SAME indices as the 3D
        # trajectory and 2D point i corresponds to 3D point i.
        #
        # Coordinates are PIXELS in the original frame, left unnormalised on
        # purpose: image_size_tensor (element 8 of this tuple) carries the
        # frame size, so the head can normalise however it likes. Note the
        # existing interaction point is normalised AND clamped to [0,1], which
        # would be the wrong convention here -- a trajectory legitimately
        # leaves the frame, and clamping would invent border pixels.
        #
        # trajectory_2d_valid marks points that are in front of the camera and
        # inside the image. Invalid points carry [0, 0]; a 2D loss should mask
        # on this rather than regress the placeholders.
        if self.return_trajectory_2d:
            num_points = 20
            coords_2d = item_data.get("trajectory_2d_image_coords")
            valid_2d = item_data.get("trajectory_2d_valid")
            if coords_2d is None or valid_2d is None:
                raise ValueError(
                    f"return_trajectory_2d=True but item {item_key_bytes.decode()} has no "
                    "2D trajectory. Records written before 2026-07-28 lack these fields; "
                    "rebuild with tools/sf3d_process.py or use sf3d_processed_v2."
                )
            trajectory_2d_tensor = torch.tensor(coords_2d, dtype=torch.float32)
            trajectory_2d_valid_tensor = torch.tensor(valid_2d, dtype=torch.bool)
            if trajectory_indices is not None and len(trajectory_2d_tensor) == len(
                trajectory_3d_camera_coords
            ):
                trajectory_2d_tensor = trajectory_2d_tensor[trajectory_indices]
                trajectory_2d_valid_tensor = trajectory_2d_valid_tensor[trajectory_indices]
            else:
                trajectory_2d_tensor = torch.zeros((num_points, 2), dtype=torch.float32)
                trajectory_2d_valid_tensor = torch.zeros(num_points, dtype=torch.bool)

            return (
                rgb_image_tensor,
                depth_image_tensor,
                description,
                mask_tensor,
                bbox_tensor,
                origin_2d_image_coord_norm,
                motion_dir_3d_camera_coords,
                motion_type_tensor,
                image_size_tensor,
                rgb_image_filename,
                motion_origin_3d_camera_coords,
                camera_intrinsic_matrix,
                trajectory_tensor,
                trajectory_2d_tensor,        # (20, 2) pixels
                trajectory_2d_valid_tensor,  # (20,) bool
            )

        # Match return signature of OPDRealDataset + trajectory + additional fields
        return (
            rgb_image_tensor,           # Shape: (3, H, W) - RGB image tensor, normalized with ImageNet stats
            depth_image_tensor,         # Shape: (1, H, W) - Depth map in meters, single channel
            description,                # str - Text description of the interaction
            mask_tensor,                # Shape: (1, H, W) - Binary mask {0., 1.} for the object
            bbox_tensor,               # Shape: (4,) - Bounding box [x_min, y_min, width, height] in pixels
            origin_2d_image_coord_norm, # Shape: (2,) - Normalized interaction point [x_norm, y_norm] in [0,1]
            motion_dir_3d_camera_coords, # Shape: (3,) - Motion direction vector in camera coordinates
            motion_type_tensor,         # Shape: () - Motion type: 0=translation, 1=rotation
            image_size_tensor,          # Shape: (2,) - Original image dimensions [width, height] in pixels
            rgb_image_filename,         # str - Filename of the RGB image
            motion_origin_3d_camera_coords, # Shape: (3,) - Motion origin point in camera coordinates
            camera_intrinsic_matrix,    # Shape: (3, 3) - Camera intrinsic matrix
            trajectory_tensor,         # Shape: (20, 3) - 3D trajectory points in camera coordinates [x,y,z]
        )


    def __del__(self):
        if hasattr(self, 'env') and self.env:
            self.env.close()


def get_default_transforms(
    image_size: Tuple[int, int] = (256, 256)  # (height, width)
) -> Tuple[Callable, Callable, Callable]:
    """Returns a default set of transforms for RGB images and masks."""
    rgb_transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # (h,w) for Resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.NEAREST,  # (h,w) for Resize
            ),
            transforms.ToTensor(),
            lambda x: (x > 0.5).float(),  # Ensure binary mask {0., 1.}
        ]
    )
    depth_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
    )
    return rgb_transform, mask_transform, depth_transform


def split_dataset_by_scene(
    dataset: "SF3DDataset",
    val_split_ratio: float,
    manual_seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Splits the SF3DDataset into training and validation subsets based on scene IDs.
    This ensures that all frames from a particular scene belong to only one split,
    preventing data leakage between the train and validation sets.
    Args:
        dataset (SF3DDataset): The full dataset instance to be split.
        val_split_ratio (float): The proportion of scenes to allocate to the validation set.
        manual_seed (int): A random seed to ensure reproducible splits.
    Returns:
        Tuple[Subset, Subset]: A tuple containing the training subset and validation subset.
    """
    print(
        f"Splitting dataset by scene with val_split_ratio={val_split_ratio} and seed={manual_seed}"
    )

    # 1. Group item indices by their scene ID.
    scene_to_indices: Dict[str, List[int]] = {}
    for i, key_bytes in enumerate(dataset.item_keys):
        # Key format is assumed to be 'scene_id/...'
        key_str = key_bytes.decode("utf-8")
        scene_id = key_str.split("/")[0]
        if scene_id not in scene_to_indices:
            scene_to_indices[scene_id] = []
        scene_to_indices[scene_id].append(i)

    # 2. Shuffle and split the list of unique scene IDs.
    unique_scene_ids = sorted(list(scene_to_indices.keys()))
    rng = random.Random(manual_seed)
    rng.shuffle(unique_scene_ids)

    num_val_scenes = int(round(len(unique_scene_ids) * val_split_ratio))
    val_scenes = set(unique_scene_ids[:num_val_scenes])
    train_scenes = set(unique_scene_ids[num_val_scenes:])

    print(f"Total scenes: {len(unique_scene_ids)}")
    print(f"Train scenes: {len(train_scenes)}, Validation scenes: {len(val_scenes)}")

    # 3. Create lists of indices for the training and validation sets.
    train_indices: List[int] = []
    val_indices: List[int] = []
    for scene_id, indices in scene_to_indices.items():
        if scene_id in val_scenes:
            val_indices.extend(indices)
        else:
            # If not in val_scenes, it must be in train_scenes
            train_indices.extend(indices)

    # 4. Create Subset wrappers for the splits.
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset


if __name__ == "__main__":
    print("Testing SF3DDataset by loading random elements and generating debug visualizations...")
    
    # Set up paths
    lmdb_data_root = "/cluster/work/cvg/students/andrye/sf3d_processed"  # Contains data.lmdb and images/
    
    # Get default transforms
    rgb_transform, mask_transform, depth_transform = get_default_transforms(image_size=(256, 256))
    
    # Create dataset
    dataset = SF3DDataset(
        lmdb_data_root=lmdb_data_root,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        depth_transform=depth_transform,
    )
    
    print(f"\nDataset size: {len(dataset)} items")
    
    # Create debug output directory
    debug_dir = Path("./debug_vis_dataset")
    debug_dir.mkdir(exist_ok=True)
    print(f"Debug images will be saved to: {debug_dir}")
    
    # Generate 10 debug images from random samples
    import random
    random.seed(42)  # For reproducibility
    random_indices = random.sample(range(len(dataset)), min(10, len(dataset)))
    
    for i, idx in enumerate(random_indices):
        print(f"\nProcessing sample {i+1}/10 (index {idx})...")
        
        try:
            # Load sample
            (
                rgb_tensor,
                depth_tensor,
                description,
                mask_tensor,
                bbox_tensor,
                origin_2d_norm,
                motion_dir_3d,
                motion_type,
                image_size,
                rgb_filename,
                motion_origin_3d,
                camera_intrinsic,
                trajectory_3d,
            ) = dataset[idx]
            
            # Convert tensors back to numpy for visualization
            # RGB: (3, H, W) -> (H, W, 3) and denormalize
            rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
            rgb_np = rgb_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            rgb_np = np.clip(rgb_np, 0, 1)
            rgb_np = (rgb_np * 255).astype(np.uint8)
            
            # Depth: (1, H, W) -> (H, W)
            depth_np = depth_tensor.squeeze(0).numpy()
            
            # Mask: (1, H, W) -> (H, W)
            mask_np = mask_tensor.squeeze(0).numpy()
            
            # Convert to OpenCV format (BGR)
            debug_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            h, w = debug_image.shape[:2]
            
            # Get original image dimensions from the dataset
            original_w, original_h = image_size.numpy()
            
            # Calculate scaling factors for coordinate transformation
            scale_x = w / original_w
            scale_y = h / original_h
            
            # Convert normalized coordinates back to pixel coordinates in resized image
            origin_2d_pixels = origin_2d_norm.numpy() * np.array([w, h])
            
            # Convert camera intrinsics to numpy and scale for resized image
            K_matrix = camera_intrinsic.numpy()
            # Scale intrinsic matrix for the resized image
            K_matrix_scaled = K_matrix.copy()
            K_matrix_scaled[0, 0] *= scale_x  # fx
            K_matrix_scaled[1, 1] *= scale_y  # fy
            K_matrix_scaled[0, 2] *= scale_x  # cx
            K_matrix_scaled[1, 2] *= scale_y  # cy
            
            # Convert trajectory to numpy
            trajectory_np = trajectory_3d.numpy()
            
            # Project trajectory points to image coordinates
            def project_camera_to_image(points_camera, intrinsic_matrix, width, height):
                """Project 3D points in camera coordinates to 2D image coordinates."""
                points_camera = np.asarray(points_camera)
                if points_camera.ndim == 1:
                    points_camera = points_camera.reshape(1, -1)
                
                points_homo = intrinsic_matrix @ points_camera.T  # (3, N)
                z = points_homo[2, :]
                valid_mask = z > 0
                u = points_homo[0, :] / (z + 1e-8)
                v = points_homo[1, :] / (z + 1e-8)
                in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height) & valid_mask
                
                result = np.zeros((points_camera.shape[0], 3))
                result[:, 0] = v  # row (y)
                result[:, 1] = u  # col (x)
                result[:, 2] = in_bounds.astype(float)
                return result
            
            # Project and draw trajectory points (blue dots) using scaled intrinsics
            traj_map = project_camera_to_image(trajectory_np, K_matrix_scaled, w, h)
            vis_mask = traj_map[:, 2] == 1
            visible_traj_points = traj_map[vis_mask, :2]
            
            # Draw trajectory points as blue dots
            for pt in visible_traj_points:
                cv2.circle(debug_image, (int(pt[1]), int(pt[0])), 2, (255, 0, 0), -1)
            
            # Draw interaction point (green dot, larger)
            if origin_2d_pixels[0] >= 0 and origin_2d_pixels[1] >= 0:
                cv2.circle(debug_image, (int(origin_2d_pixels[0]), int(origin_2d_pixels[1])), 5, (0, 255, 0), -1)
            
            # Draw mask points (red dots, subsampled for clarity)
            mask_coords = np.where(mask_np > 0.5)
            if len(mask_coords[0]) > 0:
                # Subsample mask points for visibility
                step = max(1, len(mask_coords[0]) // 100)
                for y, x in zip(mask_coords[0][::step], mask_coords[1][::step]):
                    cv2.circle(debug_image, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # Draw motion arrow(s)
            motion_type_str = "translation" if motion_type.item() == 0 else "rotation"
            motion_origin_3d_np = motion_origin_3d.numpy()
            motion_dir_3d_np = motion_dir_3d.numpy()
            
            arrow_scale_m = 0.2
            if motion_type_str == "translation":
                # Cyan arrow for translation motion direction
                end_pt_3d = motion_origin_3d_np + motion_dir_3d_np * arrow_scale_m
                origin_proj = project_camera_to_image(motion_origin_3d_np, K_matrix_scaled, w, h)
                end_proj = project_camera_to_image(end_pt_3d, K_matrix_scaled, w, h)
                if origin_proj[0, 2] == 1 and end_proj[0, 2] == 1:
                    start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                    end = (int(end_proj[0, 1]), int(end_proj[0, 0]))
                    cv2.arrowedLine(debug_image, start, end, (0, 255, 255), 3, tipLength=0.1)
            elif motion_type_str == "rotation":
                # Yellow double arrow for rotation axis direction
                end1 = motion_origin_3d_np + motion_dir_3d_np * arrow_scale_m
                origin_proj = project_camera_to_image(motion_origin_3d_np, K_matrix_scaled, w, h)
                end1_proj = project_camera_to_image(end1, K_matrix_scaled, w, h)
                if origin_proj[0, 2] == 1 and end1_proj[0, 2] == 1:
                    start = (int(origin_proj[0, 1]), int(origin_proj[0, 0]))
                    end = (int(end1_proj[0, 1]), int(end1_proj[0, 0]))
                    cv2.arrowedLine(debug_image, start, end, (255, 255, 0), 3, tipLength=0.1)
                    cv2.arrowedLine(debug_image, end, start, (255, 255, 0), 3, tipLength=0.1)
            
            # Text overlays with legend information
            cv2.putText(debug_image, f"{motion_type_str}: {description[:50]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, f"Trajectory points: {int(vis_mask.sum())}/{trajectory_np.shape[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add legend
            legend_y = h - 120
            cv2.putText(debug_image, "Blue dots: Trajectory points", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(debug_image, "Green dot: Interaction point", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(debug_image, "Red dots: Mask pixels", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if motion_type_str == "translation":
                cv2.putText(debug_image, "Cyan arrow: Translation direction", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            elif motion_type_str == "rotation":
                cv2.putText(debug_image, "Yellow arrows: Rotation axis", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Save debug image
            debug_filename = f"dataset_debug_{i+1:02d}_{rgb_filename.replace('.jpg', '')}.png"
            debug_image_path = debug_dir / debug_filename
            cv2.imwrite(str(debug_image_path), debug_image)
            
            print(f"  Saved: {debug_filename}")
            print(f"  Description: {description[:100]}...")
            print(f"  Motion type: {motion_type_str}")
            print(f"  Trajectory points: {int(vis_mask.sum())}/{trajectory_np.shape[0]}")
            
        except Exception as e:
            print(f"  ERROR processing sample {idx}: {e}")
            continue
    
    print(f"\nDebug visualization complete! Check {debug_dir} for {len(random_indices)} debug images.")
