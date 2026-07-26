"""SigLIP 2 backbone (vision + text, jointly trained, aligned embedding space).

Unlike CLIP RN50 the vision tower is a plain ViT, so the 3-level pyramid the
FPN expects is rebuilt by ``SimpleFeaturePyramid``. Text tokens come from a
Gemma sentencepiece tokenizer with a 64-token context, not CLIP's 77-token BPE.

Note on padding: SigLIP was pretrained on fixed-length padded sequences with no
attention mask, so the text tower is fed exactly that. The pad mask we compute
is for *our* decoder's cross-attention, which does need it.
"""

from typing import List, Sequence, Tuple

import torch

from .base import BackboneBase
from .pyramid import SimpleFeaturePyramid, tokens_to_map

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class SigLIP2Backbone(BackboneBase):
    def __init__(
        self,
        model_id: str,
        fpn_in: List[int],
        image_size: int = 256,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(model_id)
        self.vision_model = model.vision_model
        self.text_model = model.text_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)

        vis_cfg = model.config.vision_config
        txt_cfg = model.config.text_config
        self.patch_size = int(vis_cfg.patch_size)
        self.image_size = int(image_size)
        self._native_image_size = int(vis_cfg.image_size)

        self.fpn_in = list(fpn_in)
        self.adapter = SimpleFeaturePyramid(int(vis_cfg.hidden_size), self.fpn_in)

        self.word_dim = int(txt_cfg.hidden_size)
        self.state_dim = int(txt_cfg.hidden_size)

        pad_id = self._tokenizer.pad_token_id
        self.pad_token_id = int(pad_id) if pad_id is not None else 0
        self.max_context_length = int(txt_cfg.max_position_embeddings)

        self._register_renorm(SIGLIP_MEAN, SIGLIP_STD)

    # --- freezing ---------------------------------------------------------
    # Only the pretrained towers freeze; the pyramid adapter is new and must
    # always train, otherwise nothing connects the ViT to the neck.

    def pretrained_modules(self):
        return [self.vision_model, self.text_model]

    # --- text -------------------------------------------------------------

    def tokenize(self, texts: Sequence[str], context_length: int) -> torch.Tensor:
        length = min(context_length, self.max_context_length)
        out = self._tokenizer(
            list(texts),
            padding="max_length",
            max_length=length,
            truncation=True,
            return_tensors="pt",
        )
        return out["input_ids"]

    def encode_text(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.text_model(input_ids=tokens)
        word = out.last_hidden_state          # (B, L, hidden)
        state = out.pooler_output             # (B, hidden)
        return word, state

    # --- vision -----------------------------------------------------------

    def encode_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.renormalize(img)
        h, w = img.shape[-2:]
        kwargs = {}
        if h != self._native_image_size or w != self._native_image_size:
            kwargs["interpolate_pos_encoding"] = True
        out = self.vision_model(pixel_values=img, **kwargs)
        grid_h, grid_w = h // self.patch_size, w // self.patch_size
        feat = tokens_to_map(out.last_hidden_state, grid_h, grid_w)
        return self.adapter(feat)
