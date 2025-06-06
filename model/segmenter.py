import torch
import torch.nn as nn
import torch.nn.functional as F


from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder, Projector_Mult

def soft_argmax2d(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: Bx1xHxW  - unnormalised.
    returns: Bx2  - (x, y) in pixel units.
    """
    B, _, H, W = logits.shape
    prob = logits.view(B, -1).softmax(-1)          # B × (H·W)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=logits.device),
        torch.linspace(0, W - 1, W, device=logits.device),
        indexing="ij",
    )
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], 0)  # 2×(H·W)
    return (prob @ grid.T)                


class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ## Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval() # type: ignore
        
        # encode_image (B, 3, H, W) -> v2: (B, fpn_in[0], H/8, W/8), v3: (B, fpn_in[1], H/16, W/16), v4: (B, fpn_in[2], H/32, W/32)
        # encode_text (B, L: word_len) -> word_features: (B, L, D_text: transformer_width), state: (B, fpn_in[2])
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float() 
        
        
        ## Multi-Modal FPN
        # forward: v2, v3, v4 -> fq: (B, fpn_out[1], H/16, W/16)
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        ## Decoder
        # fq: (B, fpn_out[1], H/16, W/16), word: (B, L, D_text: transformer_width), pad_mask: (B, L) -> fq: (B, fpn_out[1], H/16, W/16)
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,
            d_model=cfg.fpn_out[1],
            nhead=cfg.num_head,
            dim_ffn=cfg.dim_ffn,
            dropout=cfg.dropout,
            return_intermediate=cfg.intermediate,
        )
        
        ## Projector
        # fq: (B, fpn_out[1], H/16, W/16), state: (B, fpn_in[2]) -> maps: (B, 2, H/4, W/4)
        self.proj = Projector_Mult(cfg.fpn_in[2], cfg.fpn_out[1]//2, 3, out_channels=2)

    def forward(self, img, word, mask, interaction_point):
        """
        img: (B, 3, H, W)
        word: (B, L)
        mask: (B, 1, H, W)
        interaction_point: (B, 2) -> value in [0,1]
        """
        # padding mask used in decoder
        
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: x2, x3, x4
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img) # v2, v3, v4
        word, state = self.backbone.encode_text(word) # (B, L, D_text: transformer_width) (B, fpn_in[2])

        # b, 512, 26, 26 (C4)
        fq = self.neck(vis, state) # fq: (B, fpn_out[1], H/16, W/16)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        maps = self.proj(fq, state)          # B×2×H_map×W_map (where H_map=h*4, W_map=w*4)
        mask_pred  = maps[:,0:1]             # B×1×H_map×W_map
        point_pred = maps[:,1:2]             # B×1×H_map×W_map, logits, not sigmoided yet

        # soft-argmax → differentiable coordinates in pixel space of the map
        # coords_px will have x in [0, W_map-1] and y in [0, H_map-1]
        coords_px  = soft_argmax2d(point_pred)      # B×2
        
        # Get the actual H_map, W_map from point_pred for normalization
        _, _, H_map, W_map = point_pred.shape
        coords_hat = coords_px / torch.tensor([W_map, H_map], dtype=coords_px.dtype, device=coords_px.device)
        
        return mask_pred, point_pred, coords_hat