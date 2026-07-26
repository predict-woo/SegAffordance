"""CLIP RN50 backbone — the original SegAffordance encoder.

Behaviour here is deliberately identical to the pre-refactor code path in
``model/segmenter.py``: same JIT load, same ``build_model``, same tensors out.
Checkpoints trained before the backbone refactor stay loadable, so the
20260721_* baselines remain comparable.
"""

from typing import List, Sequence, Tuple

import torch

from model.clip import build_model
from utils.dataset import tokenize as clip_tokenize

from .base import BackboneBase


class ClipRN50Backbone(BackboneBase):
    def __init__(self, clip_pretrain: str, word_len: int, fpn_in: List[int]):
        super().__init__()
        clip_model = torch.jit.load(clip_pretrain, map_location="cpu").eval()  # type: ignore
        self.clip = build_model(clip_model.state_dict(), word_len).float()

        # RN50: word features are pre-projection (512), state is post (1024).
        self.word_dim = int(self.clip.text_projection.shape[0])
        self.state_dim = int(self.clip.text_projection.shape[1])
        self.fpn_in = list(fpn_in)
        self.pad_token_id = 0
        self.max_context_length = word_len
        # No renormalisation: the baselines were trained on the dataloader's
        # ImageNet statistics, and changing that would break comparability.

        self._register_load_state_dict_pre_hook(self._remap_legacy_keys)

    @staticmethod
    def _remap_legacy_keys(state_dict, prefix, *_args, **_kwargs):
        """Accept checkpoints written before the backbone refactor.

        Pre-refactor the CLIP module sat directly at ``model.backbone.*``;
        it now lives at ``model.backbone.clip.*``. Rewrite on the way in so
        the 20260721_* checkpoints keep loading.
        """
        legacy = [
            k for k in state_dict
            if k.startswith(prefix) and not k.startswith(prefix + "clip.")
        ]
        for key in legacy:
            state_dict[prefix + "clip." + key[len(prefix):]] = state_dict.pop(key)

    def tokenize(self, texts: Sequence[str], context_length: int) -> torch.Tensor:
        return clip_tokenize(list(texts), context_length, truncate=True)

    def encode_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.clip.encode_image(img)

    def encode_text(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.clip.encode_text(tokens)
