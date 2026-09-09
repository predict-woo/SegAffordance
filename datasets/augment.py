"""Geometry-consistent training augmentation for SF3D-format samples.

A sample is the 15-tuple SF3DDataset emits with return_trajectory_2d=True:

  0 img       (3, Ht, Wt) uint8 (fast pipeline) or float
  1 depth     (1, Ht, Wt)
  2 desc      str
  3 mask      (1, Ht, Wt) float {0, 1}
  4 bbox      (4,) [x, y, w, h] in ORIGINAL pixels
  5 point     (2,) normalised [u, v] in the ORIGINAL frame
  6 motion    (3,) camera-frame axis direction (prismatic) / hinge axis (revolute)
  7 type      () long, 0 = trans, 1 = rot
  8 img_size  (2,) [W, H] ORIGINAL pixels — the frame the 2D fields live in
  9 filename  str
 10 origin3d  (3,) camera-frame hinge origin (m)
 11 K         (3, 3) intrinsics in ORIGINAL pixels
 12 traj3d    (N, 3) camera-frame trajectory (m)
 13 traj2d    (N, 2) ORIGINAL pixels
 14 valid     (N,) bool

Three families, each transforming EVERY coupled field so that the
projection identity  traj2d == project(K, traj3d)  and the mask/point/
track alignment hold after the transform exactly as before:

* photometric — RGB only (brightness / contrast / saturation / hue jitter,
  grayscale, blur, noise). Nothing else is touched.
* horizontal flip — image/depth/mask flipped; x -> W - x for the 2D
  track, the point and cx; camera-frame x negated for traj3d / origin3d;
  the axis follows the right-hand rule under a reflection (prismatic
  direction reflects, a revolute axis reflects AND negates so the stored
  positive-angle sweep convention survives); "left"/"right" swapped in
  the description (or the flip is skipped when they appear — policy knob).
* scale + translate crop — a window of size s*(W, H), s in [scale_min, 1],
  containing the WHOLE mask bbox, the point and the first track point
  (with a margin) is resampled back to the input size. Geometrically this
  is a change of intrinsics (K' = diag(1/s, 1/s, 1) * [K with the
  principal point shifted]) with the camera pose unchanged, so every 3D
  field stays as is; 2D fields map with p' = (p - o) / s. If the must-keep
  box does not fit even at s = 1, the crop is skipped. Out-of-frame track
  points stay valid (a track legitimately leaves the frame; the losses
  already handle that).

Not included on purpose: in-plane rotation (a camera roll — it would have
to rotate every camera-frame 3D field about the principal point and
resample the mask), and anything that moves the mask relative to the
image.

Randomness comes from torch's global RNG, which the DataLoader seeds per
worker from the (seeded) loader generator, so an augmented epoch is
reproducible for a fixed manual_seed.
"""
import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:  # photometric ops; torchvision is a project dependency (transforms)
    from torchvision.transforms import functional as TVF
except ImportError:  # pragma: no cover
    TVF = None


@dataclass
class AugmentSpec:
    enabled: bool = True
    # --- photometric (RGB only) ---
    brightness: float = 0.3      # factor in [1-b, 1+b]
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.05            # in [-h, h] (torchvision units, max 0.5)
    photometric_p: float = 0.8   # probability the jitter is applied at all
    gray_p: float = 0.05
    blur_p: float = 0.15
    noise_p: float = 0.15
    noise_std: float = 6.0       # on the 0-255 scale
    # --- horizontal flip ---
    hflip_p: float = 0.5
    flip_text: str = "swap"      # "swap" left<->right in the text | "skip" the flip when they occur
    # --- scale + translate crop ---
    crop_p: float = 0.7
    scale_min: float = 0.6       # crop side as a fraction of the frame (1 = no zoom)
    keep_margin: float = 0.03    # margin around the must-keep box, fraction of W/H

    def __post_init__(self):
        if self.flip_text not in ("swap", "skip"):
            raise ValueError("flip_text must be 'swap' or 'skip'")
        if not (0.0 < self.scale_min <= 1.0):
            raise ValueError("scale_min must be in (0, 1]")


# ----------------------------------------------------------------------------
# pure per-sample transforms (deterministic given their parameters)
# ----------------------------------------------------------------------------

_LR = re.compile(r"\b(left|right|Left|Right|LEFT|RIGHT)\b")
_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def swap_left_right(text: str) -> str:
    return _LR.sub(lambda m: _SWAP[m.group(1)], text)


def has_left_right(text: str) -> bool:
    return _LR.search(text) is not None


def hflip_sample(s: Sequence) -> tuple:
    (img, depth, desc, mask, bbox, point, motion, mtype, img_size, fname,
     origin3d, K, traj3d, traj2d, valid) = s
    W = float(img_size[0])
    img = img.flip(-1)
    depth = depth.flip(-1)
    mask = mask.flip(-1)
    # bbox on discrete pixel columns 0..W-1: column c -> W-1-c
    bbox = bbox.clone()
    if bbox[2] > 0 or bbox[3] > 0:
        bbox[0] = (W - 1.0) - (bbox[0] + bbox[2])
    point = point.clone()
    point[0] = 1.0 - point[0]
    traj2d = traj2d.clone()
    traj2d[:, 0] = W - traj2d[:, 0]
    K = K.clone()
    K[0, 2] = W - K[0, 2]
    traj3d = traj3d.clone()
    traj3d[:, 0] = -traj3d[:, 0]
    origin3d = origin3d.clone()
    origin3d[0] = -origin3d[0]
    motion = motion.clone()
    if int(mtype) == 1:
        # reflection M = diag(-1,1,1): M R(n, th) M^-1 = R(-Mn, th)
        motion[1] = -motion[1]
        motion[2] = -motion[2]
    else:
        motion[0] = -motion[0]
    desc = swap_left_right(desc)
    return (img, depth, desc, mask, bbox, point, motion, mtype, img_size, fname,
            origin3d, K, traj3d, traj2d, valid)


def _resample(x: torch.Tensor, x0t: float, y0t: float, s: float, mode: str) -> torch.Tensor:
    """Resample (C, Ht, Wt) so that output pixel centre (i+.5, j+.5) reads
    input continuous coords (x0t + (j+.5) s, y0t + (i+.5) s)."""
    C, Ht, Wt = x.shape
    theta = torch.tensor(
        [[s, 0.0, 2.0 * x0t / Wt + s - 1.0], [0.0, s, 2.0 * y0t / Ht + s - 1.0]],
        dtype=torch.float32,
    )[None]
    grid = F.affine_grid(theta, (1, C, Ht, Wt), align_corners=False)
    out = F.grid_sample(x[None].float(), grid, mode=mode, padding_mode="zeros", align_corners=False)[0]
    return out


def crop_scale_sample(s_: Sequence, x0: float, y0: float, s: float) -> tuple:
    """Crop window [x0, x0 + sW] x [y0, y0 + sH] (ORIGINAL pixels) resampled
    back to the input size. A change of intrinsics; 3D untouched."""
    (img, depth, desc, mask, bbox, point, motion, mtype, img_size, fname,
     origin3d, K, traj3d, traj2d, valid) = s_
    W, H = float(img_size[0]), float(img_size[1])
    Ht, Wt = img.shape[-2:]
    x0t, y0t = x0 * Wt / W, y0 * Ht / H
    img_out = _resample(img, x0t, y0t, s, "bilinear")
    if img.dtype == torch.uint8:
        img_out = img_out.round().clamp(0, 255).to(torch.uint8)
    else:
        img_out = img_out.to(img.dtype)
    depth = _resample(depth, x0t, y0t, s, "nearest").to(depth.dtype)
    mask = (_resample(mask, x0t, y0t, s, "nearest") > 0.5).to(mask.dtype)
    bbox = bbox.clone()
    if bbox[2] > 0 or bbox[3] > 0:
        bbox[0] = (bbox[0] - x0) / s
        bbox[1] = (bbox[1] - y0) / s
        bbox[2] = bbox[2] / s
        bbox[3] = bbox[3] / s
    point = point.clone()
    point[0] = (point[0] - x0 / W) / s
    point[1] = (point[1] - y0 / H) / s
    traj2d = traj2d.clone()
    traj2d[:, 0] = (traj2d[:, 0] - x0) / s
    traj2d[:, 1] = (traj2d[:, 1] - y0) / s
    K = K.clone()
    K[0, 0] = K[0, 0] / s
    K[1, 1] = K[1, 1] / s
    K[0, 2] = (K[0, 2] - x0) / s
    K[1, 2] = (K[1, 2] - y0) / s
    return (img_out, depth, desc, mask, bbox, point, motion, mtype, img_size, fname,
            origin3d, K, traj3d, traj2d, valid)


def must_keep_box(s_: Sequence, margin: float) -> Optional[Tuple[float, float, float, float]]:
    """[x0, y0, x1, y1] in ORIGINAL pixels that any crop must contain: the
    mask bbox, the point, the first valid track point, plus a margin."""
    bbox, point, img_size, traj2d, valid = s_[4], s_[5], s_[8], s_[13], s_[14]
    W, H = float(img_size[0]), float(img_size[1])
    xs, ys = [], []
    if bbox[2] > 0 or bbox[3] > 0:
        xs += [float(bbox[0]), float(bbox[0] + bbox[2]) + 1.0]
        ys += [float(bbox[1]), float(bbox[1] + bbox[3]) + 1.0]
    xs.append(float(point[0]) * W); ys.append(float(point[1]) * H)
    if len(valid) > 0 and bool(valid[0]):
        xs.append(float(traj2d[0, 0])); ys.append(float(traj2d[0, 1]))
    mx, my = margin * W, margin * H
    return (max(0.0, min(xs) - mx), max(0.0, min(ys) - my), min(W, max(xs) + mx), min(H, max(ys) + my))


def choose_crop(s_: Sequence, scale_min: float, margin: float, rand: torch.Tensor) -> Optional[Tuple[float, float, float]]:
    """Pick (x0, y0, s) from three uniforms in `rand` so that the must-keep
    box lies inside the window; None when even s = 1 cannot contain it."""
    W, H = float(s_[8][0]), float(s_[8][1])
    bx0, by0, bx1, by1 = must_keep_box(s_, margin)
    s_need = max((bx1 - bx0) / W, (by1 - by0) / H)
    if s_need > 1.0 + 1e-6:
        return None
    lo = max(scale_min, s_need)
    s = lo + (1.0 - lo) * float(rand[0])
    if s > 1.0 - 1e-3:
        return None  # an (almost) identity crop is not worth a resample
    cw, ch = s * W, s * H
    x_lo, x_hi = max(0.0, bx1 - cw), min(bx0, W - cw)
    y_lo, y_hi = max(0.0, by1 - ch), min(by0, H - ch)
    if x_hi < x_lo - 1e-6 or y_hi < y_lo - 1e-6:
        return None
    x0 = x_lo + (x_hi - x_lo) * float(rand[1])
    y0 = y_lo + (y_hi - y_lo) * float(rand[2])
    return (x0, y0, s)


def photometric(img: torch.Tensor, spec: AugmentSpec, rand: torch.Tensor) -> torch.Tensor:
    """RGB-only jitter on a (3, H, W) uint8 tensor; `rand` = 9 uniforms."""
    if TVF is None or img.dtype != torch.uint8:
        return img
    out = img
    if float(rand[0]) < spec.photometric_p:
        b = 1.0 + spec.brightness * (2.0 * float(rand[1]) - 1.0)
        c = 1.0 + spec.contrast * (2.0 * float(rand[2]) - 1.0)
        sat = 1.0 + spec.saturation * (2.0 * float(rand[3]) - 1.0)
        h = spec.hue * (2.0 * float(rand[4]) - 1.0)
        out = TVF.adjust_brightness(out, max(b, 0.05))
        out = TVF.adjust_contrast(out, max(c, 0.05))
        out = TVF.adjust_saturation(out, max(sat, 0.0))
        if spec.hue > 0:
            out = TVF.adjust_hue(out, float(max(-0.5, min(0.5, h))))
    if float(rand[5]) < spec.gray_p:
        out = TVF.rgb_to_grayscale(out, num_output_channels=3)
    if float(rand[6]) < spec.blur_p:
        sigma = 0.3 + 1.2 * float(rand[7])
        out = TVF.gaussian_blur(out, kernel_size=[5, 5], sigma=[sigma, sigma])
    if float(rand[8]) < spec.noise_p and spec.noise_std > 0:
        noise = torch.randn(out.shape, dtype=torch.float32) * spec.noise_std
        out = (out.float() + noise).round().clamp(0, 255).to(torch.uint8)
    return out


def augment_sample(s_: Sequence, spec: AugmentSpec) -> tuple:
    """Apply the enabled families to one 15-tuple sample (draws from torch's RNG)."""
    if not spec.enabled:
        return tuple(s_)
    if len(s_) != 15:
        raise ValueError(f"augment_sample expects the 15-tuple sample, got {len(s_)}")
    out = tuple(s_)
    r = torch.rand(16)
    # 1. geometric first (crop), on the un-jittered image
    if float(r[0]) < spec.crop_p:
        crop = choose_crop(out, spec.scale_min, spec.keep_margin, r[1:4])
        if crop is not None:
            out = crop_scale_sample(out, *crop)
    # 2. flip
    if float(r[4]) < spec.hflip_p:
        if not (spec.flip_text == "skip" and has_left_right(out[2])):
            out = hflip_sample(out)
    # 3. photometric (RGB only)
    img = photometric(out[0], spec, r[5:14])
    return (img,) + tuple(out[1:])


class AugmentedDataset(Dataset):
    """Wraps any dataset of 15-tuple samples; augments on every access."""

    def __init__(self, base: Dataset, spec: AugmentSpec):
        self.base = base
        self.spec = spec

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return augment_sample(self.base[i], self.spec)
