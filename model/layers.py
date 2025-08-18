import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True),
    )


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias), nn.BatchNorm1d(out_dim), nn.ReLU(True)
    )


class DepthEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=[128, 256]):
        super().__init__()
        # os4 -> os8
        self.conv1 = nn.Sequential(
            conv_layer(in_channels, 64, 3, padding=1),
            conv_layer(64, out_channels[0], 3, padding=1),
        )
        # os8 -> os16
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            conv_layer(out_channels[0], out_channels[0], 3, padding=1),
            conv_layer(out_channels[0], out_channels[1], 3, padding=1),
        )

    def forward(self, x):
        # Input x is (B, 1, H, W)
        # Downsample to /4
        x = F.max_pool2d(x, 2, 2)
        x = F.max_pool2d(x, 2, 2)

        # The original implementation produced a /4 feature map here.
        # We need to downsample it further to /8 to match the visual features.
        x_os4 = x
        x_os8 = F.max_pool2d(x_os4, 2, 2)

        # feat_8 is now at 1/8 resolution
        feat_8 = self.conv1(x_os8)

        # feat_16 is now at 1/16 resolution, as conv2 contains a maxpool layer.
        feat_16 = self.conv2(feat_8)

        return feat_8, feat_16


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv1 = conv_layer(
            in_channels + 2, out_channels, kernel_size, padding, stride
        )

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class KeypointProjector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.txt = nn.Linear(word_dim, in_dim * kernel_size * kernel_size + 1)


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1),
        )
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        """
        x: b, 512, 26, 26
        word: b, 512
        """
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(
            x, weight, padding=self.kernel_size // 2, groups=weight.size(0), bias=bias
        )
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class Projector_Mult(nn.Module):
    """
    Dynamic-kernel projector.
    out_channels lets the same weight generator produce several maps
    (e.g. mask + point).
    """

    def __init__(self, word_dim, in_dim, kernel_size, out_channels, proj_dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.proj_dropout = proj_dropout

        # visual tower (unchanged)
        self.vis = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1),
        )

        # Text processing with dropout
        self.txt_fc = nn.Sequential(
            nn.Linear(
                word_dim,
                out_channels * in_dim * kernel_size * kernel_size + out_channels,
            ),
            nn.Dropout(self.proj_dropout),
        )

    def forward(self, x, word):
        """
        x    : B × C × h × w
        word : B × word_dim
        """
        x = self.vis(x)
        B, C, H, W = x.shape  # x ← B×C×H×W

        # dynamic weights from text
        w_and_b = self.txt_fc(word)  # B × …
        weight, bias = (
            w_and_b[:, : -self.out_channels],
            w_and_b[:, -self.out_channels :],
        )
        weight = weight.contiguous().view(
            B * self.out_channels, C, self.kernel_size, self.kernel_size
        )  # (B·out)×C×k×k
        bias = bias.flatten()  # (B·out)

        x = x.reshape(1, B * C, H, W)  # 1 × (B·C) × H × W
        y = F.conv2d(
            x, weight, bias=bias, padding=self.kernel_size // 2, groups=B
        )  # 1 × (B·out) × H × W
        y = y.view(B, self.out_channels, H, W)  # B × out × H × W
        return y


class MotionVAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        condition_dim,
        latent_dim=32,
        hidden_dim=256,
        num_motion_types=2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_motion_types = num_motion_types

        # motion_dim is 3 for the 3D motion vector
        motion_dim = 3

        # Encoder
        self.enc_mlp = nn.Sequential(
            nn.Linear(feature_dim + motion_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec_mlp = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.motion_head = nn.Linear(hidden_dim, motion_dim)
        self.type_head = nn.Linear(hidden_dim, num_motion_types)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, motion_gt, features, condition):
        """
        motion_gt: (B, 3) - ground truth motion vector
        features: (B, feature_dim) - features for the encoder, e.g., from grid_sample
        condition: (B, condition_dim) - condition for the decoder, e.g., features + coords
        """
        # Encode
        enc_input = torch.cat([features, motion_gt], dim=1)
        h = self.enc_mlp(enc_input)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)

        # Reparameterize
        z = self.reparameterize(mean, log_var)

        # Decode
        dec_input = torch.cat([z, condition], dim=1)
        h_dec = self.dec_mlp(dec_input)
        motion_pred = self.motion_head(h_dec)
        motion_type_logits = self.type_head(h_dec)

        return motion_pred, motion_type_logits, mean, log_var

    def inference(self, condition):
        B = condition.shape[0]
        z = torch.randn(B, self.latent_dim, device=condition.device)
        dec_input = torch.cat([z, condition], dim=1)
        h_dec = self.dec_mlp(dec_input)
        motion_pred = self.motion_head(h_dec)
        motion_type_logits = self.type_head(h_dec)
        return motion_pred, motion_type_logits


class MotionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_motion_types: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )
        self.type_head = nn.Linear(hidden_dim, num_motion_types)

    def forward(self, condition: torch.Tensor):
        h = self.backbone(condition)
        motion_pred = self.motion_head(h)
        motion_type_logits = self.type_head(h)
        return motion_pred, motion_type_logits


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_layers, d_model, nhead, dim_ffn, dropout, return_intermediate=False
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, d_model, 2, dtype=torch.float)
                * -(math.log(10000.0) / d_model)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dimension (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = (
            torch.sin(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[1:d_model:2, :, :] = (
            torch.cos(pos_w * div_term)
            .transpose(0, 1)
            .unsqueeze(1)
            .repeat(1, height, 1)
        )
        pe[d_model::2, :, :] = (
            torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )
        pe[d_model + 1 :: 2, :, :] = (
            torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        )

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        """
        vis: b, 512, h, w
        txt: b, L, 512
        pad_mask: b, L
        """
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=9, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, kdim=d_model, vdim=d_model
        )
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, d_model),
        )
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        """
        vis: 26*26, b, 512
        txt: L, b, 512
        vis_pos: 26*26, 1, 512
        txt_pos: L, 1, 512
        pad_mask: b, L
        """
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(
            query=self.with_pos_embed(vis2, vis_pos),
            key=self.with_pos_embed(txt, txt_pos),
            value=txt,
            key_padding_mask=pad_mask,
        )[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]), nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(
            out_channels[2] + out_channels[1], out_channels[1], 1, 0
        )
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(
            out_channels[0] + out_channels[1], out_channels[1], 1, 0
        )
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1),
        )

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode="bilinear")
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode="bilinear")
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq
