"""DINOv3 backbone.

DINOv3 is vision-only, so the text tower has to come from somewhere else:

  * ``text_source="dinotxt"`` — the dino.txt encoder, LiT-trained against this
    exact frozen backbone, so image and text share a 2048-d space. Published
    only for ViT-L/16, behind Meta's gated licence (weights come from the
    signed URLs mailed by ai.meta.com/resources/models-and-libraries/dinov3-downloads).
    This is the configuration worth comparing against SigLIP 2.
  * ``text_source="clip"`` — the RN50 text tower already in this repo. The two
    spaces are *not* aligned; the FPN gate and dynamic kernel must learn the
    correspondence from scratch. An ablation, not a default.

DINOv3 uses ImageNet normalisation, which is what the dataloader already
emits, so no renormalisation is needed.
"""

from typing import List, Sequence, Tuple

import torch

from .base import BackboneBase
from .pyramid import MultiTapPyramid, SimpleFeaturePyramid, tokens_to_map


class DINOv3Backbone(BackboneBase):
    def __init__(
        self,
        model_id: str,
        fpn_in: List[int],
        text_source: str = "dinotxt",
        clip_pretrain: str = "pretrain/RN50.pt",
        word_len: int = 77,
        dinotxt_weights: str = "",
        dinov3_backbone_weights: str = "",
        dinov3_repo_dir: str = "",
        dinotxt_hub_entry: str = "dinov3_vitl16_dinotxt_tet1280d20h24l",
        multilayer_taps: bool = False,
    ):
        super().__init__()
        self.text_source = text_source
        self.fpn_in = list(fpn_in)
        self.multilayer_taps = multilayer_taps

        if text_source == "dinotxt":
            self._init_dinotxt(
                dinotxt_weights, dinov3_backbone_weights, dinov3_repo_dir, dinotxt_hub_entry
            )
        elif text_source == "clip":
            self._init_hf_vision(model_id)
            self._init_clip_text(clip_pretrain, word_len)
        else:
            raise ValueError(f"unknown text_source {text_source!r}")

    # --- dino.txt: aligned vision head + text encoder ----------------------

    def _init_dinotxt(self, weights, backbone_weights, repo_dir, hub_entry):
        if not (weights and backbone_weights and repo_dir):
            raise ValueError(
                "text_source='dinotxt' needs dinotxt_weights, dinov3_backbone_weights "
                "and dinov3_repo_dir (the gated .pth files + a local dinov3 checkout)"
            )
        model, tokenizer = torch.hub.load(
            repo_dir,
            hub_entry,
            source="local",
            weights=weights,
            backbone_weights=backbone_weights,
        )
        self.dinotxt = model
        self._dinotxt_tokenizer = tokenizer

        vis = model.visual_model
        self.patch_size = int(getattr(vis.backbone, "patch_size", 16))
        embed_dim = int(model.model_config.embed_dim)
        backbone_dim = int(vis.backbone.embed_dim)
        # VisionHead projects tokens to embed_dim // multiplier, where the
        # multiplier counts how many things get concatenated into the final
        # embedding (class token and/or pooled patch tokens). Read the real
        # projection rather than assuming the multiplier.
        proj = getattr(vis.head, "linear_projection", None)
        aligned_dim = int(proj.out_features) if isinstance(proj, torch.nn.Linear) else backbone_dim

        # /8 and /16 from raw backbone tokens; /32 from the aligned head tokens.
        if self.multilayer_taps:
            self.adapter = MultiTapPyramid(backbone_dim, self.fpn_in)
        else:
            self.adapter = SimpleFeaturePyramid(backbone_dim, self.fpn_in)
        self.deep_proj = torch.nn.Conv2d(aligned_dim, backbone_dim, kernel_size=1, bias=False)

        self.word_dim = embed_dim
        self.state_dim = embed_dim
        self.pad_token_id = 0
        self.max_context_length = 77

        if self.multilayer_taps:
            self._install_tap_hooks(vis.backbone.blocks)

    def _install_tap_hooks(self, blocks):
        # Blocks 4/11/17 of ViT-L (the "lin-4" spread minus the final
        # layer, which the existing pass already returns). Hook outputs are
        # the post-block hidden states incl. prefix tokens.
        self._tap_cache = {}
        for i in (4, 11, 17):
            def _mk(idx):
                def hook(_m, _inp, out):
                    self._tap_cache[idx] = out
                return hook
            blocks[i].register_forward_hook(_mk(i))

    # --- HF vision tower (used with the CLIP text ablation) ----------------

    def _init_hf_vision(self, model_id: str):
        if self.multilayer_taps:
            # The clip-text ablation runs the HF vision tower, whose
            # encode_image path never assembles taps — half-wiring hooks
            # there would fail confusingly at forward time instead.
            raise ValueError(
                "dinov3_multilayer_taps is dinotxt-only for now "
                "(text_source='clip' uses the single-layer pyramid)"
            )
        from transformers import AutoModel

        self.vision_model = AutoModel.from_pretrained(model_id)
        self.patch_size = int(self.vision_model.config.patch_size)
        hidden = int(self.vision_model.config.hidden_size)
        self.adapter = SimpleFeaturePyramid(hidden, self.fpn_in)

    def _init_clip_text(self, clip_pretrain: str, word_len: int):
        from model.clip import build_model
        from utils.dataset import tokenize as clip_tokenize

        clip_model = torch.jit.load(clip_pretrain, map_location="cpu").eval()  # type: ignore
        self.text_tower = build_model(clip_model.state_dict(), word_len).float()
        # CLIP's visual tower is unused here, but it cannot be deleted:
        # CLIP.dtype is a property reading self.visual.conv1.weight.dtype,
        # and encode_text depends on it. 38M idle params is the cheaper bug.
        self._clip_tokenize = clip_tokenize
        self.word_dim = int(self.text_tower.text_projection.shape[0])
        self.state_dim = int(self.text_tower.text_projection.shape[1])
        self.pad_token_id = 0
        self.max_context_length = word_len

    # --- interface ---------------------------------------------------------

    def pretrained_modules(self):
        if self.text_source == "dinotxt":
            return [self.dinotxt]
        return [self.vision_model, self.text_tower]

    def tokenize(self, texts: Sequence[str], context_length: int) -> torch.Tensor:
        if self.text_source == "clip":
            return self._clip_tokenize(list(texts), context_length, truncate=True)
        return self._dinotxt_tokenizer.tokenize(list(texts), context_length)

    def encode_text(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.text_source == "clip":
            return self.text_tower.encode_text(tokens)
        # TextTower.forward returns only the pooled vector, so run its two
        # stages directly to keep the per-token features the decoder needs.
        text = self.dinotxt.text_model
        word = text.head(text.backbone(tokens))          # (B, L, embed_dim)
        state = word[torch.arange(word.shape[0], device=word.device), tokens.argmax(dim=-1)]
        return word, state

    def encode_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._encode(img)[0]

    def encode_image_full(self, img: torch.Tensor):
        return self._encode(img)

    def _encode(self, img: torch.Tensor):
        """(v2, v3, v4) plus extras. On the dinotxt path the extras carry
        the RAW psi patch tokens as a map (before deep_proj) — their channel
        width is the text-embedding HALF (dino.txt: text = [CLS-half;
        patch-half]), which is what the gen-15 cost map compares against."""
        h, w = img.shape[-2:]
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        if self.text_source == "clip":
            out = self.vision_model(pixel_values=img)
            # Strip CLS + register tokens by taking the trailing patch grid,
            # which is robust to however many prefix tokens the variant carries.
            tokens = out.last_hidden_state[:, -(grid_h * grid_w):, :]
            return self.adapter(tokens_to_map(tokens, grid_h, grid_w)), {}

        _feats, aligned_tokens, backbone_tokens = self.dinotxt.encode_image_with_patch_tokens(img)
        raw_map = tokens_to_map(backbone_tokens, grid_h, grid_w)
        aligned_raw = tokens_to_map(aligned_tokens, grid_h, grid_w)
        aligned_map = self.deep_proj(aligned_raw)
        extras = {"aligned_map": aligned_raw}
        if self.multilayer_taps:
            # Applying the backbone's final LayerNorm to each tap mirrors
            # get_intermediate_layers(norm=True) — the DINOv2/DPT convention.
            norm = self.dinotxt.visual_model.backbone.norm
            taps = []
            for i in (4, 11, 17):
                t = self._tap_cache.pop(i)
                if isinstance(t, tuple):
                    t = t[0]
                t = norm(t)[:, -(grid_h * grid_w):, :]      # final LN + strip prefix
                taps.append(tokens_to_map(t, grid_h, grid_w))
            taps.append(raw_map)                             # block-23 tokens, already normed
            return self.adapter(taps, x_deep=aligned_map), extras
        return self.adapter(raw_map, x_deep=aligned_map), extras
