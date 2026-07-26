"""Smoke-check a backbone end to end before spending GPU hours on it.

    python tools/smoke_backbone.py --backbone siglip2
    python tools/smoke_backbone.py --backbone clip_rn50 --legacy-ckpt <path>

Verifies: construction, tokenizer, forward shapes, the frozen/trainable split,
a backward pass, and peak memory at the real training batch size.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.opd_train import ModelParams

DINOTXT_W = "/workspace/cache/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"
DINOV3_BB_W = "/workspace/cache/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DINOV3_REPO = "/workspace/cache/dinov3repo"
from model.segmenter import CRIS


def build_params(backbone: str, backbone_id: str, word_len: int, text_source: str) -> ModelParams:
    return ModelParams(
        clip_pretrain="pretrain/RN50.pt",
        word_len=word_len,
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
        backbone=backbone,
        backbone_id=backbone_id,
        backbone_image_size=256,
        text_source=text_source,
        dinotxt_weights=DINOTXT_W,
        dinov3_backbone_weights=DINOV3_BB_W,
        dinov3_repo_dir=DINOV3_REPO,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="siglip2")
    ap.add_argument("--backbone-id", default="")
    ap.add_argument("--word-len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--freeze", action="store_true", default=True)
    ap.add_argument("--legacy-ckpt", default="")
    ap.add_argument("--text-source", default="clip")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} backbone={args.backbone}")

    t0 = time.time()
    model = CRIS(build_params(args.backbone, args.backbone_id, args.word_len, args.text_source))
    print(f"✅ built in {time.time() - t0:.1f}s")

    bb = model.backbone
    print(f"   fpn_in={bb.fpn_in} word_dim={bb.word_dim} state_dim={bb.state_dim} "
          f"pad_id={bb.pad_token_id} max_ctx={bb.max_context_length}")
    print(f"   word_proj={'yes' if model.word_proj is not None else 'no (dims already match)'}")

    if args.legacy_ckpt:
        raw = torch.load(args.legacy_ckpt, map_location="cpu")
        sd = raw.get("state_dict", raw)
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bb_missing = [k for k in missing if k.startswith("backbone.")]
        bb_unexpected = [k for k in unexpected if k.startswith("backbone.")]
        print(f"   legacy ckpt: {len(bb_missing)} missing / {len(bb_unexpected)} unexpected backbone keys")
        if bb_missing or bb_unexpected:
            print(f"   ⚠️  sample missing={bb_missing[:3]} unexpected={bb_unexpected[:3]}")
        else:
            print("   ✅ legacy checkpoint maps cleanly onto the refactored model")

    # frozen/trainable split
    if args.freeze and hasattr(bb, "pretrained_modules"):
        for m in bb.pretrained_modules():
            for p in m.parameters():
                p.requires_grad = False
    train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n = sum(p.numel() for p in model.parameters())
    print(f"   params: {total_n/1e6:.1f}M total, {train_n/1e6:.1f}M trainable "
          f"({100*train_n/total_n:.1f}%)")

    model.to(device).train()
    texts = ["open the top drawer of the cabinet"] * args.batch
    tok = model.tokenize(texts, args.word_len).to(device)
    print(f"   tokens: {tuple(tok.shape)} dtype={tok.dtype} "
          f"pad_frac={(tok == bb.pad_token_id).float().mean():.2f}")

    img = torch.randn(args.batch, 3, args.size, args.size, device=device)
    dep = torch.rand(args.batch, 1, args.size, args.size, device=device)
    mask = (torch.rand(args.batch, 1, args.size, args.size, device=device) > 0.7).float()
    pt = torch.rand(args.batch, 2, device=device)
    mgt = torch.randn(args.batch, 3, device=device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model(img, dep, tok, mask, pt, mgt)
    names = ["mask", "point", "coords", "motion", "type", "mu", "logvar", "traj"]
    for n, o in zip(names, out):
        print(f"   {n:7s} {tuple(o.shape) if o is not None else None}")

    loss = sum(o.float().pow(2).mean() for o in out if o is not None)
    loss.backward()
    fwd_bwd = time.time() - t0

    grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    print(f"✅ backward ok: {len(grads)} tensors got gradients")
    if device == "cuda":
        print(f"   peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"| fwd+bwd {fwd_bwd:.2f}s for batch {args.batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
