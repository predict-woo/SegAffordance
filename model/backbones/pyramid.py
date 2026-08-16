"""Turning a single-scale ViT feature map into the 3-level pyramid FPN wants.

CLIP RN50 hands the neck features at strides 8/16/32 for free. A ViT only
produces one map (stride = patch size), so we rebuild the other two levels the
ViTDet way: deconvolve up for the fine level, pool down for the coarse one.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class SimpleFeaturePyramid(nn.Module):
    """(B, C, h, w) at stride 16 -> maps at strides 8, 16, 32.

    Follows the "simple feature pyramid" of ViTDet (Li et al., 2022): each
    level is built from the *same* single-scale map rather than by lateral
    top-down fusion, which is what makes it usable with a plain ViT.
    """

    def __init__(self, in_dim: int, out_channels: List[int]):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError(f"expected 3 output channel counts, got {out_channels}")

        # stride 16 -> 8
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        # stride 16 -> 16
        self.same = nn.Sequential(
            nn.Conv2d(in_dim, out_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        # stride 16 -> 32
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, out_channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
        )

    def forward(
        self, x: torch.Tensor, x_deep: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x builds all levels; x_deep, if given, replaces the source of /32.

        CLIP RN50 hands the neck raw ResNet features at /8 and /16 but an
        attnpool-projected, text-aligned map at /32 — and the FPN's gate and
        the dynamic kernel both rely on that deepest level sharing a space
        with the text state. x_deep lets a backbone reproduce that split.
        """
        deep = x if x_deep is None else x_deep
        return self.up(x), self.same(x), self.down(deep)


class MultiTapPyramid(nn.Module):
    """4 intermediate-layer taps -> strides 8/16/32 (frozen-DINO standard).

    DPT-style reassembly (DINOv2 "lin-4" / Depth Anything / SegDINO): the
    fine level comes from an EARLY layer, which still holds the high-
    frequency spatial detail the final layer has abstracted away —
    SimpleFeaturePyramid's deconv-from-final-layer cannot recover it (its
    ViTDet evidence base is fine-tuned MAE, a different regime; see
    knowledge/dinov3-dense-adapter-survey.md).

    forward(taps, x_deep=None): taps are 4 maps (B, in_dim, h, w) ordered
    shallow->deep (ViT-L blocks 4/11/17/23). x_deep, if given, replaces
    the deep tap as the /32 source (the dino.txt-aligned map — the same
    contract as SimpleFeaturePyramid).
    """

    def __init__(self, in_dim: int, out_channels: List[int]):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError(f"expected 3 output channel counts, got {out_channels}")
        self.up = nn.Sequential(                       # tap[0] -> /8
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.mid = nn.Sequential(                      # cat(tap[1], tap[2]) -> /16
            nn.Conv2d(2 * in_dim, out_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.down = nn.Sequential(                     # tap[3] (or x_deep) -> /32
            nn.Conv2d(in_dim, out_channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
        )

    def forward(self, taps, x_deep=None):
        if len(taps) != 4:
            raise ValueError(f"expected 4 taps, got {len(taps)}")
        deep = taps[3] if x_deep is None else x_deep
        return (
            self.up(taps[0]),
            self.mid(torch.cat([taps[1], taps[2]], dim=1)),
            self.down(deep),
        )


def tokens_to_map(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """(B, N, C) patch tokens -> (B, C, grid_h, grid_w)."""
    b, n, c = tokens.shape
    if n != grid_h * grid_w:
        raise ValueError(f"{n} tokens do not fill a {grid_h}x{grid_w} grid")
    return tokens.transpose(1, 2).reshape(b, c, grid_h, grid_w).contiguous()
