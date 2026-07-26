"""Common interface every visual+text backbone must satisfy.

The rest of the model (FPN -> TransformerDecoder -> heads) only ever sees:

  * a three-level visual pyramid ``(v2, v3, v4)`` at strides 8/16/32,
  * per-token text features ``word`` and a pooled text vector ``state``,
  * a padding mask for the text tokens.

Anything backbone-specific (patch size, tokenizer, image normalisation,
how a single-scale ViT is turned into a pyramid) lives behind this class.
"""

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


# What datasets/opdreal.py:36 bakes into every RGB tensor it emits.
DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)


class BackboneBase(nn.Module):
    """Vision+text encoder pair.

    Subclasses must set ``fpn_in``, ``word_dim`` and ``state_dim`` before the
    first forward, and implement ``encode_image`` / ``encode_text`` /
    ``tokenize``.
    """

    #: channel counts of (v2, v3, v4) — what FPN receives
    fpn_in: List[int]
    #: dim of the per-token text features fed to the decoder's cross-attention
    word_dim: int
    #: dim of the pooled text state used for FPN gating and the dynamic kernel
    state_dim: int
    #: token id used for padding (``pad_mask`` marks these positions)
    pad_token_id: int = 0
    #: longest token sequence the text tower accepts
    max_context_length: int = 77

    def tokenize(self, texts: Sequence[str], context_length: int) -> torch.Tensor:
        raise NotImplementedError

    def pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) bool — True where the position is padding."""
        return tokens.eq(self.pad_token_id)

    def encode_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, 3, H, W) -> (v2, v3, v4) at H/8, H/16, H/32."""
        raise NotImplementedError

    def encode_text(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, L) -> word (B, L, word_dim), state (B, state_dim)."""
        raise NotImplementedError

    # --- image normalisation -------------------------------------------------
    # The dataloader hands us ImageNet-normalised tensors. A backbone trained
    # under different statistics re-normalises here rather than in the data
    # pipeline, so existing datamodules/configs keep working untouched.

    def _register_renorm(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Set up an ImageNet-normalised -> (mean, std)-normalised conversion.

        Composing un-normalise then re-normalise into a single affine op:
            x' = (x * s_d + m_d - m_b) / s_b
        """
        d_mean = torch.tensor(DATASET_MEAN).view(1, 3, 1, 1)
        d_std = torch.tensor(DATASET_STD).view(1, 3, 1, 1)
        b_mean = torch.tensor(tuple(mean)).view(1, 3, 1, 1)
        b_std = torch.tensor(tuple(std)).view(1, 3, 1, 1)
        self.register_buffer("_renorm_scale", d_std / b_std, persistent=False)
        self.register_buffer("_renorm_shift", (d_mean - b_mean) / b_std, persistent=False)

    def renormalize(self, img: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "_renorm_scale"):
            return img
        return img * self._renorm_scale.to(img.dtype) + self._renorm_shift.to(img.dtype)
