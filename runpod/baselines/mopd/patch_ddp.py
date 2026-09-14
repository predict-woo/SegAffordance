"""Patch MOPD so a run can (1) RESUME from detectron2's `last_checkpoint` (their main() hard-codes
`resume_or_load(resume=False)` and the parser has no --resume) and (2) train data-parallel with the
SAME maths as one GPU: detectron2 splits IMS_PER_BATCH across GPUs and averages gradients, and the
only batch-dependent layers -- the geffnet EfficientNet-B5 "normal" encoder's 116 BatchNorm2d --
are converted to SyncBatchNorm so their statistics are still taken over the full batch of 16 (the
R50 backbone uses FrozenBN, EfficientSAM uses LayerNorm). Written 2026-09-14 to move the 512x384
MOPD run from 1 x A100 (2.67 s/iter, one CPU thread busy, GPU 18 %) to 2 GPUs. Idempotent.

Targets: <opdformer>/train.py and <opdformer>/mask2former/maskformer_model.py, which wraps the
EfficientSAM encoder in nn.DataParallel: its device_ids default to all visible GPUs with cuda:0
first, so under DDP rank 1 dies with "module must have its parameters ... on device cuda:0".
DataParallel is a no-op on one GPU (what the 1-GPU run had), so it becomes a plain pass-through
that keeps the `.module` attribute their forward reads (image_encoder.module.img_size).
DDP also needs find_unused_parameters=True: MOPD has branches that never reach the loss under this
config (no update on 1 GPU either); the flag only changes the reducer's bookkeeping.

  python patch_ddp.py <path to opdformer/train.py>
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])

# --- maskformer_model.py: DataParallel -> pass-through -------------------------------------
mm = p.parent / "mask2former" / "maskformer_model.py"
ms = mm.read_text()
if "SF3D_DDP" in ms:
    print("already patched", mm)
else:
    old = "        self.image_encoder=nn.DataParallel(self.image_encoder)\n"
    assert ms.count(old) == 1, old
    ms = ms.replace(old, "        self.image_encoder = _SF3DPassThrough(self.image_encoder)"
                         "  # SF3D_DDP: was nn.DataParallel (a no-op on 1 GPU)\n")
    anchor = "\n@META_ARCH_REGISTRY.register()\nclass MaskFormer("
    assert ms.count(anchor) == 1, "MaskFormer class anchor not found"
    helper = (
        "\n\nclass _SF3DPassThrough(nn.Module):\n"
        '    """SF3D_DDP: stands in for nn.DataParallel(encoder) -- same forward, keeps `.module`."""\n'
        "    def __init__(self, module):\n"
        "        super().__init__()\n"
        "        self.module = module\n\n"
        "    def forward(self, *args, **kwargs):\n"
        "        return self.module(*args, **kwargs)\n"
    )
    ms = ms.replace(anchor, helper + anchor, 1)
    mm.write_text(ms)
    print("patched", mm)

# --- train.py, stage 1: --resume and SyncBN ---------------------------------------------------
s = p.read_text()
if "SF3D_DDP" in s:
    print("already patched", p)
else:
    edits = [
        ('    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")\n',
         '    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")\n'
         '    parser.add_argument("--resume", action="store_true", help="SF3D_DDP: resume from OUTPUT_DIR/last_checkpoint")\n'),
        ('    trainer = OPDTrainer(cfg)\n    trainer.resume_or_load(resume=False)\n',
         '    trainer = OPDTrainerSyncBN(cfg)  # SF3D_DDP\n    trainer.resume_or_load(resume=args.resume)\n'),
        ('def main(args):\n',
         'class OPDTrainerSyncBN(OPDTrainer):\n'
         '    """SF3D_DDP: BatchNorm statistics over the whole IMS_PER_BATCH when the batch is split over GPUs."""\n'
         '    @classmethod\n'
         '    def build_model(cls, cfg):\n'
         '        import torch\n'
         '        from detectron2.utils import comm\n'
         '        model = super().build_model(cfg)\n'
         '        if comm.get_world_size() > 1:\n'
         '            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)\n'
         '        return model\n'
         '\n\n'
         'def main(args):\n'),
    ]
    for old, new in edits:
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", p)

# --- train.py, stage 2: DDP must tolerate parameters that get no gradient ---------------------
s = p.read_text()
if "SF3D_DDP_UNUSED" in s:
    print("already patched (unused params)", p)
else:
    old = "def main(args):\n"
    new = ('import detectron2.engine.defaults as _d2_defaults  # SF3D_DDP_UNUSED\n'
           '_orig_create_ddp_model = _d2_defaults.create_ddp_model\n\n\n'
           'def _create_ddp_model_unused(model, **kwargs):\n'
           '    kwargs.setdefault("find_unused_parameters", True)\n'
           '    return _orig_create_ddp_model(model, **kwargs)\n\n\n'
           '_d2_defaults.create_ddp_model = _create_ddp_model_unused\n\n\n' + old)
    assert s.count(old) == 1, old
    s = s.replace(old, new, 1)
    p.write_text(s)
    print("patched (unused params)", p)
