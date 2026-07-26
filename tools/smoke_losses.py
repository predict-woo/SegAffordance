"""Smoke-check the loss wiring end to end, on synthetic batches.

    python tools/smoke_losses.py
    python tools/smoke_losses.py --batch 4 --device cpu

Builds the real model (CLIP RN50 + heads) once, then drives `_common_step`
through every combination of batch shape and geometric-loss variant:

    OPD   (9-tuple)  x cross_gt   -> no geometric term (OPD has no trajectory)
    SF3D  (13-tuple) x cross_gt   -> both historical cross-GT terms
    SF3D  (13-tuple) x pred_pred  -> the single symmetric term
    OPD   (9-tuple)  x pred_pred  -> still a no-op
    video (dict)     x projected  -> the 2D-pretraining alignment term,
                                     with the element-sweep head removed

Checks each total loss is finite, that gradients actually flow, and that the
logged term names are exactly the expected set — the last one is what catches
a variant silently no-opping or leaking terms into the wrong dataset.

Complements tests/test_geometric_losses.py, which covers the loss maths on
CPU with no model. This covers the wiring: batch unpack, forward, every loss
term, backward. It needs a GPU and pretrain/RN50.pt, which is why it lives
here rather than under tests/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from model.losses import build_geometric_loss
from train_OPDReal_better import OPDRealTrainingModule


def build_params(geometric_loss: str, stage: str = "3d"):
    """stage="2d" mirrors the pretraining config: no element-sweep head, a 2D
    hand-track head, and a metric origin-depth head."""
    model_params = ModelParams(
        clip_pretrain="pretrain/RN50.pt",
        word_len=77,
        depth_feat_channels=[128, 256],
        fpn_in=[512, 1024, 1024],
        fpn_out=[256, 512, 1024],
        num_layers=3,
        num_head=8,
        dim_ffn=1024,
        dropout=0.1,
        intermediate=False,
        proj_dropout=0.5,
        vae_latent_dim=32,
        vae_hidden_dim=256,
        num_motion_types=2,
        use_depth=True,
        use_cvae=True,
        use_trajectory_head=(stage != "2d"),
        use_2d_trajectory_head=(stage == "2d"),
        predict_origin_depth=(stage == "2d"),
    )
    loss_params = LossParams(
        bce_weight=0.5,
        dice_weight=0.5,
        mask_weight=0.5,
        point_map_weight=0.5,
        coord_weight=0.3,
        vae_weight=0.2,
        motion_type_weight=0.5,
        point_sigma=8.0,
        vae_beta=0.01,
        trajectory_weight=0.5,
        geometric_loss=geometric_loss,
        geometric_weight=0.5,
        trajectory_to_motion_weight=0.5,
        pred_pred_weight=0.1,
        projected_weight=1.0,
        projected_radius_weight=0.1,
        projected_radius_ref=1.0,
        trajectory_2d_weight=1.0,
    )
    config = Config(
        log_image_interval_steps=100,
        input_size=[256, 256],
        enable_wandb=False,
        val_vis_samples=4,
        manual_seed=42,
    )
    optimizer_params = OptimizerParams(
        lr=2e-5, weight_decay=1e-4, scheduler_milestones=[25, 27], scheduler_gamma=0.1
    )
    return model_params, loss_params, optimizer_params, config


def opd_batch(batch: int, size: int, device: str):
    """The 9-tuple OPDReal/OPDMulti emits."""
    return (
        torch.randn(batch, 3, size, size, device=device),
        torch.rand(batch, 1, size, size, device=device),
        ["open the top drawer of the cabinet"] * batch,
        (torch.rand(batch, 1, size, size, device=device) > 0.5).float(),
        torch.tensor([[10.0, 10.0, 50.0, 50.0]] * batch, device=device),
        torch.rand(batch, 2, device=device),
        # non-zero: MotionVAELoss asserts the GT axis has magnitude
        torch.nn.functional.normalize(torch.randn(batch, 3, device=device), dim=1),
        torch.randint(0, 2, (batch,), device=device),
        torch.tensor([[size, size]] * batch, device=device),
    )


def sf3d_batch(batch: int, size: int, device: str):
    """The 13-tuple SF3D emits: + filename, motion origin, intrinsics, trajectory."""
    return opd_batch(batch, size, device) + (
        ["scene0.png"] * batch,
        torch.randn(batch, 3, device=device),
        torch.eye(3, device=device).expand(batch, 3, 3),
        torch.randn(batch, 20, 3, device=device),
    )


def video_batch(batch: int, size: int, device: str):
    """A mined 2D source: hand track + intrinsics, NO element sweep and no 3D
    motion origin. Supplied as a dict so no positional layout is invented."""
    img, depth, words, mask, _bbox, point, motion, mtype, img_sz = opd_batch(batch, size, device)
    track = torch.rand(batch, 20, 2, device=device) * 0.5 + 0.25
    K = torch.tensor([[500.0, 0, 256.0], [0, 500.0, 256.0], [0, 0, 1.0]], device=device)
    return {
        "img": img, "depth": depth, "word": words, "mask": mask,
        "point_norm": point, "motion": motion, "motion_type": mtype,
        "img_size": torch.tensor([[512.0, 512.0]] * batch, device=device),
        "trajectory_2d": track,
        "anchor_depth": torch.full((batch,), 2.0, device=device),
        "camera_intrinsic": K.expand(batch, 3, 3).contiguous(),
    }


def switch_to_2d_heads(module, device):
    """Reconfigure an already-built model for 2D pretraining: drop the
    element-sweep head, add the 2D track and origin-depth heads."""
    from model.layers import OriginDepthHead, Trajectory2DMLP

    condition_dim = module.model.trajectory_predictor.backbone[0].in_features
    module.model.trajectory_predictor = None
    module.model.trajectory_2d_predictor = Trajectory2DMLP(condition_dim).to(device)
    module.model.origin_depth_head = OriginDepthHead(condition_dim).to(device)
    print(f"   switched to 2D heads (condition_dim={condition_dim})", flush=True)


def run_case(module, name, geometric_loss, batch, expected_terms):
    _, loss_params, _, _ = build_params(geometric_loss)
    module.loss_params = loss_params
    module.geometric_loss = build_geometric_loss(loss_params)
    module.zero_grad(set_to_none=True)

    # No Trainer is attached, so self.log would raise; capture instead. This
    # also lets us assert on which terms each variant actually emits.
    logged = {}
    module.log = lambda k, v, **kw: logged.__setitem__(k, float(v))

    loss = module._common_step(batch, 0, "train")
    loss.backward()

    grads = [p for p in module.model.parameters() if p.requires_grad and p.grad is not None]
    nonzero = sum(1 for p in grads if p.grad.abs().sum() > 0)
    terms = sorted(k.split("/", 1)[1] for k in logged if "geo" in k.lower())

    shape = "dict" if isinstance(batch, dict) else f"{len(batch)}-tuple"
    print(f"\n=== {name} (geometric_loss={geometric_loss!r}, {shape}) ===")
    print(f"   loss      {loss.item():.6f}  finite={bool(torch.isfinite(loss))}")
    print(f"   grads     {len(grads)} tensors, {nonzero} non-zero")
    print(f"   geometric {terms or '(none)'}")

    ok = True
    if not torch.isfinite(loss):
        print("   ❌ loss is not finite"); ok = False
    if nonzero == 0:
        print("   ❌ no gradients flowed"); ok = False
    if terms != sorted(expected_terms):
        print(f"   ❌ expected geometric terms {sorted(expected_terms)}"); ok = False
    if ok:
        print("   ✅ ok")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"device={args.device} batch={args.batch}", flush=True)
    model_params, loss_params, optimizer_params, config = build_params("cross_gt")
    module = OPDRealTrainingModule(
        model_params, loss_params, optimizer_params, config
    ).to(args.device).train()
    print("✅ built model once (reused across cases)", flush=True)

    opd = opd_batch(args.batch, args.size, args.device)
    sf3d = sf3d_batch(args.batch, args.size, args.device)
    cross_gt_terms = [
        "L_geometric_pred_vector_gt_traj",
        "L_geometric_pred_traj_gt_vector",
    ]

    video = video_batch(args.batch, args.size, args.device)

    results = [
        run_case(module, "OPD  / default", "cross_gt", opd, []),
        run_case(module, "SF3D / cross_gt", "cross_gt", sf3d, cross_gt_terms),
        run_case(module, "SF3D / pred_pred", "pred_pred", sf3d, ["L_geo_pred_pred"]),
        run_case(module, "OPD  / pred_pred", "pred_pred", opd, []),
    ]

    # Stage-1 (2D pretraining) head configuration. Swapped in place rather
    # than rebuilding CRIS, which would mean reloading CLIP off the volume.
    switch_to_2d_heads(module, args.device)
    results.append(
        run_case(module, "video / projected", "projected", video, ["L_geo_projected"])
    )

    if all(results):
        print("\n✅ all loss-wiring checks passed")
        return 0
    print(f"\n❌ {results.count(False)}/{len(results)} cases failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
