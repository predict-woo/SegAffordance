"""Compose a full MOPD state dict from an OPDFormer checkpoint + EfficientSAM ViT-S.

MOPD (Locate n' Rotate, github.com/lisiqi-zju/MOPD) fine-tunes from a FULL
model checkpoint: its constructor does ``self.load_state_dict(torch.load(
cfg.MODEL.WEIGHTS))`` strictly, and the model contains, besides the OPDFormer
weights, an EfficientSAM ViT-S image encoder (``image_encoder.module.*``) and
an EfficientNet-B5 "normal" encoder (``normal_encoder.*``, geffnet
``tf_efficientnet_b5_ap`` ImageNet weights). The released MOPD checkpoint is
Baidu-only, so we rebuild the same composition: our OPDFormer-P RGB weights
trained on SF3D + the public EfficientSAM ViT-S release + geffnet's ImageNet
weights (the same sources MOPD's own checkpoint was assembled from).

    python tools/baselines_sf3d/mopd_compose_ckpt.py --opd <opdformer.pth> \
        --esam <efficient_sam_vits.pt> --mopd-repo <MOPD clone> --out init.pth
"""
import argparse
import sys
import zipfile
from pathlib import Path


def remap_esam(esam_state):
    """EfficientSAM release keys ``image_encoder.X`` -> MOPD ``image_encoder.module.X``."""
    out = {}
    for k, v in esam_state.items():
        if k.startswith("image_encoder."):
            out["image_encoder.module." + k[len("image_encoder."):]] = v
    return out


def unwrap_opd(opd_ckpt):
    """detectron2 checkpoints wrap the weights under ``model``."""
    return opd_ckpt["model"] if isinstance(opd_ckpt, dict) and "model" in opd_ckpt else opd_ckpt


def compose(opd_state, esam_state, normal_state=None):
    from collections import OrderedDict

    out = OrderedDict(opd_state)
    # Keep detectron2's per-module version metadata: without it MaskFormerHead's
    # legacy-format converter (_load_from_state_dict) re-prefixes the pixel-decoder
    # keys ("sem_seg_head.pixel_decoder." -> ".pixel_decoder.pixel_decoder.") and the
    # strict load in MOPD's constructor fails.
    if hasattr(opd_state, "_metadata"):
        out._metadata = dict(opd_state._metadata)
    out.update(remap_esam(esam_state))
    if normal_state:
        out.update({("normal_encoder." + k if not k.startswith("normal_encoder.") else k): v for k, v in normal_state.items()})
    return out


def load_esam(path):
    import torch

    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = [n for n in z.namelist() if n.endswith(".pt")][0]
            z.extract(name, p.parent)
            p = p.parent / name
    sd = torch.load(p, map_location="cpu")
    return sd["model"] if isinstance(sd, dict) and "model" in sd else sd


def normal_encoder_state(mopd_repo):
    """Instantiate MOPD's Encoder(B=5, pretrained=True) to get geffnet ImageNet weights."""
    sys.path.insert(0, str(Path(mopd_repo) / "opdformer"))
    from mask2former.segmen_anything.submodules import Encoder  # noqa: E402

    return {k: v for k, v in Encoder(B=5, pretrained=True).state_dict().items()}


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--opd", required=True)
    ap.add_argument("--esam", required=True)
    ap.add_argument("--mopd-repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    opd = unwrap_opd(torch.load(a.opd, map_location="cpu"))
    esam = load_esam(a.esam)
    normal = normal_encoder_state(a.mopd_repo)
    sd = compose(opd, esam, normal)
    n_opd = len(opd); n_esam = sum(k.startswith("image_encoder.module.") for k in sd); n_norm = sum(k.startswith("normal_encoder.") for k in sd)
    print(f"opd {n_opd} keys, esam {n_esam} keys, normal {n_norm} keys -> {len(sd)} total")
    torch.save(sd, a.out)


if __name__ == "__main__":
    main()
