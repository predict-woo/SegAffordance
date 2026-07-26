"""Pluggable visual+text backbones.

``build_backbone`` is the only entry point the model uses. Selection is driven
by ``ModelParams.backbone`` ("clip_rn50" | "siglip2" | "dinov3").
"""

from typing import List

from .base import BackboneBase


def _id_or(model_params, default: str) -> str:
    value = getattr(model_params, "backbone_id", "") or ""
    return value if value else default


def build_backbone(model_params, fpn_in: List[int]) -> BackboneBase:
    name = getattr(model_params, "backbone", "clip_rn50")

    if name == "clip_rn50":
        from .clip_rn50 import ClipRN50Backbone

        return ClipRN50Backbone(
            clip_pretrain=model_params.clip_pretrain,
            word_len=model_params.word_len,
            fpn_in=fpn_in,
        )

    if name == "siglip2":
        from .siglip2 import SigLIP2Backbone

        return SigLIP2Backbone(
            model_id=_id_or(model_params, "google/siglip2-large-patch16-256"),
            fpn_in=fpn_in,
            image_size=getattr(model_params, "backbone_image_size", 256),
        )

    if name == "dinov3":
        from .dinov3 import DINOv3Backbone

        return DINOv3Backbone(
            model_id=_id_or(model_params, "facebook/dinov3-vitl16-pretrain-lvd1689m"),
            fpn_in=fpn_in,
            text_source=getattr(model_params, "text_source", "dinotxt"),
            clip_pretrain=model_params.clip_pretrain,
            word_len=model_params.word_len,
            dinotxt_weights=getattr(model_params, "dinotxt_weights", ""),
            dinov3_backbone_weights=getattr(model_params, "dinov3_backbone_weights", ""),
            dinov3_repo_dir=getattr(model_params, "dinov3_repo_dir", ""),
        )

    raise ValueError(f"unknown backbone {name!r}")


__all__ = ["BackboneBase", "build_backbone"]
