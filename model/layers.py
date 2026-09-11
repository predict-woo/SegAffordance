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
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_motion_types: int = 2,
                 with_type_head: bool = True, with_motion_head: bool = True,
                 split_axis_heads: bool = False):
        super().__init__()
        if split_axis_heads and not with_motion_head:
            raise ValueError(
                "split_axis_heads needs with_motion_head: there is no axis "
                "readout to split"
            )
        self.split_axis_heads = split_axis_heads
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        # Optional (twist arms: the twist head carries the axis).
        # NOTE: the Sigmoid is NOT a sign constraint — CRIS.forward rescales
        # this output with (x - 0.5) * 2 to (-1, 1) before it becomes
        # outputs.motion_pred, so every axis direction is representable
        # (verified empirically 2026-08-18, tools/diag_axis_sign.py).
        def _axis_readout():
            return nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        if split_axis_heads:
            # Gen-17: per-type readouts on the shared trunk. The hinge axis
            # (⊥ to the motion) and the slide direction (= the motion) are
            # different physical quantities; a single readout blends them
            # under type ambiguity (2026-08-18 spec).
            self.motion_head = None
            self.motion_head_rot = _axis_readout()
            self.motion_head_trans = _axis_readout()
        else:
            self.motion_head = _axis_readout() if with_motion_head else None
            self.motion_head_rot = None
            self.motion_head_trans = None
        # Optional: no parameters at all when off (2D-only pretraining has
        # no type labels; type is emergent from the twist's |omega|).
        self.type_head = (
            nn.Linear(hidden_dim, num_motion_types) if with_type_head else None
        )

    def forward(self, condition: torch.Tensor):
        """Split mode -> (motion_rot, motion_trans, type_logits); legacy ->
        (motion_pred, type_logits)."""
        h = self.backbone(condition)
        motion_type_logits = self.type_head(h) if self.type_head is not None else None
        if self.split_axis_heads:
            return self.motion_head_rot(h), self.motion_head_trans(h), motion_type_logits
        motion_pred = self.motion_head(h) if self.motion_head is not None else None
        return motion_pred, motion_type_logits


class TrajectoryMLP(nn.Module):
    # dct_scale_split: softplus(bias) at init ~ 0.40 — a typical scale-free
    # path length (0.6 m arc / 1.5 m anchor depth on SF3D doors, ~0.4 on
    # hand video), so the unit shape starts at a sensible magnitude.
    SCALE_BIAS_INIT = -0.7

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_points: int = 20,
                 delta_cumsum: bool = False, num_hypotheses: int = 1,
                 absolute: bool = False, dct_coeffs: int = 0,
                 dct_pin_start: bool = False, dct_scale_split: bool = False):
        super().__init__()
        assert not (absolute and delta_cumsum), (
            "trajectory_absolute and trajectory_delta_cumsum are mutually "
            "exclusive: absolute mode is the direct (non-cumsum) readout"
        )
        if dct_coeffs > 0 and delta_cumsum:
            raise ValueError(
                "trajectory_dct_coeffs and trajectory_delta_cumsum are "
                "competing smoothing parameterizations — pick one"
            )
        if dct_coeffs > num_points:
            raise ValueError("trajectory_dct_coeffs cannot exceed num_points")
        if (dct_pin_start or dct_scale_split) and dct_coeffs <= 0:
            raise ValueError(
                "trajectory_dct_pin_start / trajectory_dct_scale_split are "
                "DCT-readout conventions — they need trajectory_dct_coeffs > 0"
            )
        if dct_pin_start and dct_coeffs + 1 > num_points:
            raise ValueError("pinned DCT readout needs dct_coeffs + 1 <= num_points")
        if (dct_pin_start or dct_scale_split) and absolute:
            raise ValueError("pinned / scale-split DCT readouts are relative-frame only")
        self.num_points = num_points
        # absolute: the num_points outputs are ABSOLUTE camera-frame points
        # (gen-7), not positions relative to the trajectory's own first point.
        # Computationally this IS the direct non-cumsum readout below — only
        # the semantics of the supervision target change.
        self.absolute = absolute
        # num_hypotheses > 1: K trajectories, one per WTA articulation bundle
        # (selected jointly with the twist by TwistMLP's logits — see the
        # 2026-08-11 WTA spec). Forward returns (B, K, N, 3).
        self.num_hypotheses = num_hypotheses
        # delta_cumsum: predict num_points-1 per-step displacement vectors and
        # integrate (cumsum) into relative positions, with point 0 pinned to
        # exactly 0 — the interaction point supplies the absolute anchor. The
        # loss stays in POSITION space (an off delta shifts every downstream
        # point and is penalised accordingly — supervising deltas themselves
        # would let drift hide). Same function class as the direct readout
        # (cumsum of a linear map is a linear map), but the sequence is a
        # connected path by construction: per-point errors can't decorrelate
        # into the zigzag clouds the direct head produced (see
        # viz/20260803_sf3d_twist_traj_points).
        self.delta_cumsum = delta_cumsum
        # Gen-19: truncated-DCT readout — the head emits dct_coeffs
        # low-frequency DCT-II coefficients per axis and a fixed IDCT
        # matrix decodes them to num_points positions. Jitter above the
        # K-th frequency is UNREPRESENTABLE by construction (the
        # motion-forecasting standard: Mao'19 / siMLPe / HumanMAC —
        # knowledge/trajectory-parameterization-survey.md). Losses stay on
        # the decoded points; forward output shape is unchanged.
        self.dct_coeffs = int(dct_coeffs)
        # DCT readout conventions v2 (2026-09-10, literature sweep
        # knowledge/2026-09-10_trajectory_head_synthesis_v2.md): the basis
        # is the small lever; the readout conventions carry the gains.
        #   pin_start: point 0 is EXACTLY 0 (the curve is relative to its
        #     first point by definition, so the head no longer spends
        #     coefficients learning that; BEAST's pinned c0, siMLPe's
        #     residual-to-anchor readout). The pin cancels the DC basis
        #     row, so the K coefficients are the first K AC frequencies.
        #   scale_split: the decoded curve is normalised to unit path
        #     length (shape) and multiplied by a separately predicted
        #     softplus scale (General Flow's shape/scale decomposition) —
        #     one head no longer has to cover 5 cm slides and 70 cm sweeps
        #     inside the same coefficient range.
        self.dct_pin_start = bool(dct_pin_start)
        self.dct_scale_split = bool(dct_scale_split)
        if self.dct_coeffs > 0:
            N, K = num_points, self.dct_coeffs
            n = torch.arange(N, dtype=torch.float64)
            k = torch.arange(N, dtype=torch.float64)
            dct_m = torch.cos(math.pi * (n[None, :] + 0.5) * k[:, None] / N)
            dct_m *= math.sqrt(2.0 / N)
            dct_m[0] /= math.sqrt(2.0)
            # Orthonormal: inverse == transpose. Keep K rows' transpose as
            # the (N, K) decode matrix: rows 0..K-1, or 1..K when pinned
            # (the DC row is cancelled by the pin and would be dead weight).
            rows = dct_m[1:K + 1] if self.dct_pin_start else dct_m[:K]
            self.register_buffer("idct_m", rows.T.contiguous().float())
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        # Each point has 3 coordinates (x, y, z)
        if self.dct_coeffs > 0:
            out_points = self.dct_coeffs
        else:
            out_points = num_points - 1 if delta_cumsum else num_points
        self.trajectory_head = nn.Linear(
            hidden_dim, num_hypotheses * out_points * 3
        )
        if self.dct_scale_split:
            # One log-ish scale per hypothesis; softplus keeps it positive
            # without exp's blow-up. Zero weights + a bias at a typical
            # magnitude: the shape branch trains first, the scale follows.
            self.scale_head = nn.Linear(hidden_dim, num_hypotheses)
            nn.init.zeros_(self.scale_head.weight)
            nn.init.constant_(self.scale_head.bias, self.SCALE_BIAS_INIT)

    def forward(self, condition: torch.Tensor):
        """-> (B, K, num_points, 3); K = num_hypotheses (1 for the legacy head)."""
        h = self.backbone(condition)
        trajectory_pred = self.trajectory_head(h)
        K = self.num_hypotheses
        if self.dct_coeffs > 0:
            coeffs = trajectory_pred.view(-1, K, self.dct_coeffs, 3)
            # (N, C) @ (B, K, C, 3) -> (B, K, N, 3)
            pts = torch.einsum("nc,bkcd->bknd", self.idct_m, coeffs)
            if self.dct_pin_start:
                pts = pts - pts[:, :, :1]
            if self.dct_scale_split:
                pts32 = pts.float()
                seg = pts32[:, :, 1:] - pts32[:, :, :-1]
                length = seg.norm(dim=-1).sum(-1)[..., None, None]      # (B, K, 1, 1)
                unit = pts32 / length.clamp(min=1e-4)
                scale = F.softplus(self.scale_head(h).float()).view(-1, K, 1, 1)
                pts = (unit * scale).to(pts.dtype)
            return pts
        if self.delta_cumsum:
            deltas = trajectory_pred.view(-1, K, self.num_points - 1, 3)
            rel = torch.cumsum(deltas, dim=2)
            zero = rel.new_zeros(rel.size(0), K, 1, 3)
            return torch.cat([zero, rel], dim=2)
        # Direct readout — no cumsum, no zero-pin. In absolute mode this is
        # the raw (B, K, num_points, 3) camera-frame prediction; in the
        # classical relative mode it is the same computation read as
        # relative-to-first-point positions.
        return trajectory_pred.view(-1, K, self.num_points, 3)


class TrajectoryLengthHead(nn.Module):
    """Arc length of the interaction point's path (2026-09-11 decoder design):
    one positive scalar per sample, the only trajectory-specific learned
    quantity when the analytic decoder replaces the trajectory head. Same
    construction as the v2 DCT scale head (2-layer MLP -> softplus, bias at
    softplus(-0.7) ~ 0.40, a typical scale-free path length)."""

    BIAS_INIT = -0.7

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True),
        )
        self.length_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.length_head.weight)
        nn.init.constant_(self.length_head.bias, self.BIAS_INIT)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """-> (B,) positive arc length."""
        return F.softplus(self.length_head(self.backbone(condition)).float()).view(-1)


class Trajectory2DMLP(nn.Module):
    """Image-plane track, normalised to [0, 1], relative to its own first point.

    Mirrors TrajectoryMLP's shape and convention but outputs 2D. It is a
    SEPARATE head rather than a reinterpretation of the 3D one because the two
    predict different physical quantities: this is the hand/contact path mined
    from video, while TrajectoryMLP is the functional element's swept path.
    Sharing one head would ask it to learn one curve in 2D pretraining and a
    different curve in 3D finetuning.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_points: int = 20,
                 delta_cumsum: bool = False):
        super().__init__()
        self.num_points = num_points
        # Same delta-cumsum construction as TrajectoryMLP (see its comment).
        self.delta_cumsum = delta_cumsum
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        out_points = num_points - 1 if delta_cumsum else num_points
        self.trajectory_head = nn.Linear(hidden_dim, out_points * 2)

    def forward(self, condition: torch.Tensor):
        h = self.backbone(condition)
        out = self.trajectory_head(h)
        if self.delta_cumsum:
            deltas = out.view(-1, self.num_points - 1, 2)
            rel = torch.cumsum(deltas, dim=1)
            zero = rel.new_zeros(rel.size(0), 1, 2)
            return torch.cat([zero, rel], dim=1)
        return out.view(out.size(0), self.num_points, 2)


class TwistMLP(nn.Module):
    """se(3) twist (omega, v) in R^6, camera frame — see model/losses/twist.py.

    Unconstrained linear output: revolute targets have |omega| = 1 and
    prismatic targets |omega| = 0, and both live in the interior of R^6, so
    there is no manifold to project onto. Normalising omega here would destroy
    exactly the |omega| -> 0 limit that makes the parameterisation unify the
    two motion types.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 pitch_free: bool = False, pitch_eps: float = 0.05,
                 num_hypotheses: int = 1):
        super().__init__()
        # pitch_free: the output map ends in a smoothed orthogonal
        # projection of v against omega, making the output space exactly
        # the pitch-free variety {omega . v = 0} — no helical motions.
        # Exact for |omega| >> eps (revolute), smoothly the identity as
        # omega -> 0 (prismatic, where pitch is vacuous). See
        # ModelParams.twist_pitch_free for why a literal 5-parameter chart
        # cannot exist.
        # num_hypotheses > 1: K winner-takes-all articulation hypotheses +
        # K selection logits (see the WTA spec). The logits select the WHOLE
        # bundle including the matching TrajectoryMLP hypothesis.
        self.pitch_free = pitch_free
        self.pitch_eps = pitch_eps
        self.num_hypotheses = num_hypotheses
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
        )
        self.twist_head = nn.Linear(hidden_dim, num_hypotheses * 6)
        self.logit_head = (
            nn.Linear(hidden_dim, num_hypotheses) if num_hypotheses > 1 else None
        )

    def forward(self, condition: torch.Tensor):
        """-> (twists (B, K, 6), logits (B, K) | None)."""
        h = self.backbone(condition)
        out = self.twist_head(h).view(-1, self.num_hypotheses, 6)
        if self.pitch_free:
            omega, v = out[..., :3], out[..., 3:]
            axial = (v * omega).sum(-1, keepdim=True)
            v = v - axial * omega / (
                omega.pow(2).sum(-1, keepdim=True) + self.pitch_eps ** 2
            )
            out = torch.cat([omega, v], dim=-1)
        logits = self.logit_head(h) if self.logit_head is not None else None
        return out, logits


class OriginDepthHead(nn.Module):
    """Metric depth (metres) of the 3D joint origin.

    Combined with ``point_uv`` and the intrinsics this yields the 3D origin
    the model otherwise never predicts — ``motion_origin_3d`` has always been
    ground-truth-only, which is why the cross-GT geometric loss teacher-forces
    it.

    Predicts an ABSOLUTE depth rather than an offset against the input depth
    map, because the depth maps are not on a common scale: SF3D converts to
    metres (``datasets/scenefun3d.py:133``) while OPD passes the raw array
    through (``datasets/opdreal.py:211``). Absolute metres is supervised
    unambiguously by ``motion_origin_3d[2]``.

    Softplus keeps the origin in front of the camera; a negative depth would
    make the projection meaningless.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, min_depth: float = 0.1):
        super().__init__()
        self.min_depth = min_depth
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, condition: torch.Tensor):
        return F.softplus(self.mlp(condition).squeeze(-1)) + self.min_depth


def sine_positions_2d(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """(h*w, dim) fixed 2D sine/cosine positions (DETR layout: half the
    channels encode y, half x), row-major to match ``fq.flatten(2)``."""
    if dim % 4 != 0:
        raise ValueError(f"2D sine positions need dim % 4 == 0, got {dim}")
    quarter = dim // 4
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(quarter, device=device, dtype=torch.float32) / quarter
    )
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] * freqs[None]  # (h, q)
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] * freqs[None]  # (w, q)
    py = torch.cat([ys.sin(), ys.cos()], dim=1)                                  # (h, 2q)
    px = torch.cat([xs.sin(), xs.cos()], dim=1)                                  # (w, 2q)
    pos = torch.cat([py[:, None, :].expand(h, w, -1), px[None, :, :].expand(h, w, -1)], dim=2)
    return pos.reshape(h * w, dim).to(dtype)


class _MaskedCrossAttention(nn.Module):
    """Multi-head cross-attention of queries over map tokens with a per-token
    additive bias on the logits (the part mask). Explicit projections + SDPA
    so the float bias, autocast and torch.compile behave."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} not divisible by nhead {nhead}")
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                bias: torch.Tensor) -> torch.Tensor:
        # q (B, K, C), keys/values (B, S, C), bias (B, S) -> (B, K, C)
        B, K, C = q.shape
        S = keys.shape[1]
        hd = C // self.nhead
        qh = self.q_proj(q).view(B, K, self.nhead, hd).transpose(1, 2)
        kh = self.k_proj(keys).view(B, S, self.nhead, hd).transpose(1, 2)
        vh = self.v_proj(values).view(B, S, self.nhead, hd).transpose(1, 2)
        attn_bias = bias[:, None, None, :].expand(B, self.nhead, K, S).to(qh.dtype)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_bias)
        return self.out_proj(out.transpose(1, 2).reshape(B, K, C))


class ArticulationReadout(nn.Module):
    """Learned queries reading the decoded feature map for the articulation
    heads (2026-09-12; ModelParams.articulation_readout).

    Mask-mean pooling hands the heads one average vector of the part, which
    keeps nothing of WHERE the hinge sits relative to the part. Here the
    queries attend over the stride-16 map with 2D sine positions on the keys
    and ``log(mask + eps)`` added to every attention logit, so tokens on the
    part dominate while the surroundings (the seam, the frame) stay reachable.

    mode "attnpool": a single query and a single cross-attention — attention
    pooling in place of the mask mean, nothing else changes.
    mode "query": ``num_queries`` queries through ``num_layers`` pre-norm
    layers (self-attention among the queries, masked cross-attention, FFN),
    LayerNormed output (B, num_queries, d_model).
    """

    def __init__(self, d_model: int, mode: str, num_queries: int = 4, num_layers: int = 2,
                 nhead: int = 8, dim_ffn: int = 1024, mask_eps: float = 0.01):
        super().__init__()
        if mode not in ("attnpool", "query"):
            raise ValueError(f"mode must be attnpool|query, got {mode!r}")
        if mask_eps <= 0:
            raise ValueError("mask_eps must be > 0 (log bias floor)")
        self.mode = mode
        self.mask_eps = float(mask_eps)
        self.d_model = d_model
        if mode == "attnpool":
            num_queries, num_layers = 1, 1
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        if mode == "attnpool":
            self.pool = _MaskedCrossAttention(d_model, nhead)
        else:
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    "ln1": nn.LayerNorm(d_model),
                    "self_attn": nn.MultiheadAttention(d_model, nhead, batch_first=True),
                    "ln2": nn.LayerNorm(d_model),
                    "cross_attn": _MaskedCrossAttention(d_model, nhead),
                    "ln3": nn.LayerNorm(d_model),
                    "ffn": nn.Sequential(
                        nn.Linear(d_model, dim_ffn), nn.GELU(), nn.Linear(dim_ffn, d_model)
                    ),
                }))
            self.out_norm = nn.LayerNorm(d_model)

    def forward(self, fq: torch.Tensor, mask_w: torch.Tensor) -> torch.Tensor:
        """fq (B, C, h, w) decoded map; mask_w (B, 1, h, w) part weights in
        [0, 1] (GT mask in teacher-forced training, predicted mask otherwise)
        -> (B, num_queries, C)."""
        B, C, h, w = fq.shape
        tok = fq.flatten(2).transpose(1, 2)                        # (B, S, C)
        pos = sine_positions_2d(h, w, C, tok.device, tok.dtype)    # (S, C)
        keys = tok + pos[None]
        bias = torch.log(mask_w.flatten(1).to(torch.float32) + self.mask_eps)  # (B, S)
        q = self.queries[None].expand(B, -1, -1).to(tok.dtype)
        if self.mode == "attnpool":
            return self.pool(q, keys, tok, bias)
        for layer in self.layers:
            x = layer["ln1"](q)
            q = q + layer["self_attn"](x, x, x, need_weights=False)[0]
            q = q + layer["cross_attn"](layer["ln2"](q), keys, tok, bias)
            q = q + layer["ffn"](layer["ln3"](q))
        return self.out_norm(q)


class DenseArticulationHead(nn.Module):
    """Dense hinge voting (2026-09-12; ModelParams.articulation_readout
    "dense"). Every pixel of the decoded map votes: a unit-ish rot axis, a
    trans direction (both tanh, in (-1, 1) like the rescaled MLP readouts),
    type logits, and a 2D offset from its own image position to the hinge
    origin (normalised [0, 1] coordinates, unbounded — the hinge may lie
    off-image). The part-mask-weighted mean of each field is the prediction:
    the classical ANCSH / Shape2Motion / OPD design, where the origin is
    located by the part's pixels rather than read out of one pooled vector.
    """

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 10, 1),   # rot 3 | trans 3 | type 2 | offset 2
        )
        # offsets start at zero: the initial origin vote is the part centroid
        nn.init.zeros_(self.net[-1].weight[8:])
        nn.init.zeros_(self.net[-1].bias[8:])

    def forward(self, fq: torch.Tensor, mask_w: torch.Tensor):
        """fq (B, C, h, w), mask_w (B, 1, h, w) in [0, 1] -> dict with
        rot (B, 3), trans (B, 3), type_logits (B, 2), origin_uv (B, 2),
        offset_field (B, 2, h, w), vote_uv (B, 2, h, w)."""
        B, _, h, w = fq.shape
        f = self.net(fq).float()
        wgt = mask_w.float()
        wsum = wgt.sum(dim=(-1, -2)) + 1e-6                          # (B, 1)

        def wmean(x):                                                # (B, k, h, w) -> (B, k)
            return (x * wgt).sum(dim=(-1, -2)) / wsum

        rot = wmean(torch.tanh(f[:, 0:3]))
        trans = wmean(torch.tanh(f[:, 3:6]))
        type_logits = wmean(f[:, 6:8])
        ys = (torch.arange(h, device=fq.device, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, device=fq.device, dtype=torch.float32) + 0.5) / w
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=0)   # (2, h, w): u, v
        offset = f[:, 8:10]
        vote = grid[None] + offset                                   # (B, 2, h, w)
        origin_uv = wmean(vote)
        return {
            "rot": rot, "trans": trans, "type_logits": type_logits, "origin_uv": origin_uv,
            "offset_field": offset, "vote_uv": vote,
        }


class Point3DHead(nn.Module):
    """Absolute 3D point in camera coordinates (metres).

    Used twice by the gen-6 split arm: the interaction point (graspable
    element centroid, GT = trajectory_3d[0]) and the revolute joint origin
    (GT = q*, the axis point perpendicular to the interaction point).
    Unconstrained linear output — camera-frame positions have no box to
    project onto, and canonicalized targets keep the regression local.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.mlp(condition)


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
    def __init__(self, in_channels, out_channels, text_dim=None):
        super(FPN, self).__init__()
        # text projection. For CLIP RN50 the pooled text state happens to have
        # the same width as v5 (both 1024, via attnpool), which is why this used
        # to read in_channels[2]; other backbones need it stated explicitly.
        if text_dim is None:
            text_dim = in_channels[2]
        self.txt_proj = linear_layer(text_dim, out_channels[2])
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
