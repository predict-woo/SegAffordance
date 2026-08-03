"""Locate the SF3D training bottleneck: input pipeline vs GPU step.

Phases (--phase, default "all"):
  data — dataloader-only throughput: iterate the real train dataloader and
         report samples/s with the GPU untouched. If this is at or below the
         full-step rate, training is input-bound.
  item — single-process __getitem__ cProfile: where a worker's time goes per
         sample (LMDB read, JPEG decode, mask reconstruction, transforms).
  step — real training steps on the GPU with data-wait vs compute split, and
         compute further split into forward / loss / backward+opt
         (fp16 autocast + GradScaler, matching trainer precision 16).
  ops  — torch.profiler over a few steps; top ops by CUDA time.

Run on a pod (warm the LMDBs first, as train_pod.sh does):
    python tools/profile_sf3d_train.py \
        --config config/sf3d_train_runpod_twist.yaml --workers 40
"""

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.scenefun3d_datamodule import SF3DDataModule
from train_SF3D_better import SF3DTrainingModule


def build(cfg_path, workers, batch_size, fast=False, channels_last=False,
          compile_model=False, force_no_compile=False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    mp = dict(m["model_params"])
    if channels_last:
        mp["channels_last"] = True
    if compile_model:
        mp["compile_model"] = True
    if force_no_compile:
        # phase 'parts' wraps submodule forwards with CUDA events — that
        # instrumentation cannot live inside a compiled graph.
        mp["compile_model"] = False
    module = SF3DTrainingModule(
        model_params=ModelParams(**mp),
        loss_params=LossParams(**m["loss_params"]),
        optimizer_params=OptimizerParams(**m["optimizer_params"]),
        config=Config(**m["config"]),
    )
    module.log = lambda *a, **k: None  # no trainer attached
    d = dict(cfg["data"])
    if workers:
        d["num_workers_train"] = workers
    if batch_size:
        d["batch_size_train"] = batch_size
    if fast:
        d["fast_pipeline"] = True
    dm = SF3DDataModule(**d)
    dm.setup("fit")
    return module, dm


def to_cuda(batch):
    return tuple(
        t.cuda(non_blocking=True) if torch.is_tensor(t) else t for t in batch
    )


def phase_data(dm, n_batches):
    # Consume with zero compute so next() blocks on worker PRODUCTION. Run
    # long enough that the initial prefetch-queue head start (workers x
    # prefetch_factor x batch) is amortized; the last-half rate is the
    # steady-state pipeline ceiling.
    dl = dm.train_dataloader()
    bs = dl.batch_size
    t_start = time.perf_counter()
    it = iter(dl)  # spawns workers
    stamps = []
    for _ in range(n_batches):
        next(it)
        stamps.append(time.perf_counter())
    total = stamps[-1] - t_start
    mid = len(stamps) // 2
    half = stamps[-1] - stamps[mid - 1]
    print(f"[data] {n_batches} batches x {bs}: "
          f"overall {n_batches * bs / total:.0f} samples/s (incl. spin-up), "
          f"steady-state {(n_batches - mid) * bs / half:.0f} samples/s")


def phase_item(dm, n_samples):
    ds = dm.train_dataset
    idx = torch.randperm(len(ds))[:n_samples].tolist()
    for i in idx[:10]:
        ds[i]  # warm lmdb handles
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    for i in idx:
        ds[i]
    prof.disable()
    dt = time.perf_counter() - t0
    print(f"[item] {n_samples} samples in {dt:.1f}s -> {1000 * dt / n_samples:.1f} ms/sample single-proc")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(14)
    print("\n".join(s.getvalue().splitlines()[4:26]))


def phase_step(module, dm, n_batches):
    module.cuda().train()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    dl = dm.train_dataloader()
    it = iter(dl)
    bs = dl.batch_size

    def one_step(batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module._common_step(batch, 0, "train")
        if isinstance(loss, dict):
            loss = loss["loss"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # warmup (cudnn autotune, allocator)
    for _ in range(3):
        one_step(to_cuda(next(it)))
    torch.cuda.synchronize()

    t_data = t_fwd_loss = t_bwd = 0.0
    t_all0 = time.perf_counter()
    for _ in range(n_batches):
        t0 = time.perf_counter()
        batch = to_cuda(next(it))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module._common_step(batch, 0, "train")
        if isinstance(loss, dict):
            loss = loss["loss"]
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        t_data += t1 - t0
        t_fwd_loss += t2 - t1
        t_bwd += t3 - t2
    t_all = time.perf_counter() - t_all0

    n = n_batches
    print(f"[step] {n} steps x {bs}: {n * bs / t_all:.0f} samples/s end-to-end")
    print(f"[step]   data wait : {1000 * t_data / n:7.1f} ms/batch")
    print(f"[step]   fwd+loss  : {1000 * t_fwd_loss / n:7.1f} ms/batch")
    print(f"[step]   bwd+opt   : {1000 * t_bwd / n:7.1f} ms/batch")
    gpu_ms = 1000 * (t_fwd_loss + t_bwd) / n
    data_ms = 1000 * t_data / n
    verdict = "INPUT-BOUND (dataloader)" if data_ms > 0.3 * gpu_ms else "GPU-BOUND (compute)"
    print(f"[step] verdict: {verdict}  (gpu {gpu_ms:.0f} ms vs exposed data wait {data_ms:.0f} ms)")

    # forward-only vs full: isolates loss cost from the backbone. eval():
    # the train-mode forward pools with the GT mask, which this bench
    # doesn't thread through.
    batch = to_cuda(next(it))
    (img, depth, desc) = batch[0], batch[1], batch[2]
    word = module.model.tokenize(list(desc), module.model_params.word_len).cuda()
    module.model.eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(6):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            module.model(img, depth, word, None, None, None, None)
    torch.cuda.synchronize()
    module.model.train()
    print(f"[step]   model fwd only (no loss, no grad): {1000 * (time.perf_counter() - t0) / 6:7.1f} ms/batch")
    del it, dl


def phase_parts(module, dm, n_batches):
    """CUDA-event timing per model stage and per loss term.

    Wraps instance forwards, so nn.Module __call__ picks up the timed
    version. GPU time is attributed to whichever stage recorded it; glue
    (interpolates, soft-argmax, condition assembly) lands in 'fwd other'.
    """
    module.cuda().train()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    pending, sums = [], {}

    def wrap(obj, attr, key):
        if obj is None:
            return
        fn = getattr(obj, attr, None)
        if fn is None:
            return

        def timed(*a, **k):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*a, **k)
            e.record()
            pending.append((key, s, e))
            return out

        setattr(obj, attr, timed)

    m = module.model
    wrap(m.backbone, "encode_image", "backbone.encode_image")
    wrap(m.backbone, "encode_text", "backbone.encode_text")
    for name in ("word_proj", "depth_encoder", "neck", "decoder", "proj",
                 "motion_vae", "motion_mlp", "trajectory_predictor",
                 "trajectory_2d_predictor", "twist_predictor",
                 "origin_depth_head"):
        wrap(getattr(m, name, None), "forward", f"model.{name}")
    for name in ("mask_loss_fn", "point_map_loss_fn", "vae_loss_fn",
                 "motion_type_loss_fn", "trajectory_loss_fn",
                 "geometric_loss", "twist_loss", "traj_projection_loss"):
        wrap(getattr(module, name, None), "forward", f"loss.{name}")

    dl = dm.train_dataloader()
    it = iter(dl)
    bs = dl.batch_size

    def one_step(batch, measure=False):
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module._common_step(batch, 0, "train")
        if isinstance(loss, dict):
            loss = loss["loss"]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if measure:
            sums["fwd total"] = sums.get("fwd total", 0.0) + (t1 - t0) * 1000
            sums["bwd+opt"] = sums.get("bwd+opt", 0.0) + (t2 - t1) * 1000
            for key, s, e in pending:
                sums[key] = sums.get(key, 0.0) + s.elapsed_time(e)
        pending.clear()

    for _ in range(3):
        one_step(to_cuda(next(it)))
    for _ in range(n_batches):
        one_step(to_cuda(next(it)), measure=True)

    fwd_parts = sum(v for k, v in sums.items()
                    if k.startswith(("model.", "backbone.", "loss.")))
    sums["fwd other (glue)"] = sums["fwd total"] - fwd_parts
    total = sums["fwd total"] + sums["bwd+opt"]
    print(f"[parts] {n_batches} steps x {bs}; step total {total / n_batches:.0f} ms/batch")
    for k, v in sorted(sums.items(), key=lambda kv: -kv[1]):
        print(f"[parts]   {k:28s} {v / n_batches:8.1f} ms/batch  {100 * v / total:5.1f}%")
    del it, dl


def phase_ops(module, dm, n_batches):
    module.cuda().train()
    opt = torch.optim.AdamW(module.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    dl = dm.train_dataloader()
    it = iter(dl)
    batches = [to_cuda(next(it)) for _ in range(n_batches + 2)]

    def one_step(batch):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = module._common_step(batch, 0, "train")
        if isinstance(loss, dict):
            loss = loss["loss"]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for b in batches[:2]:
        one_step(b)
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for b in batches[2:]:
            one_step(b)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
    del it, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sf3d_train_runpod_twist.yaml")
    ap.add_argument("--phase", default="all",
                    choices=["all", "data", "item", "step", "parts", "ops"])
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--workers", type=int, default=0, help="override num_workers_train")
    ap.add_argument("--batch-size", type=int, default=0, help="override batch_size_train")
    ap.add_argument("--item-samples", type=int, default=300)
    ap.add_argument("--fast", action="store_true", help="enable data.fast_pipeline")
    ap.add_argument("--channels-last", action="store_true", help="override model channels_last")
    ap.add_argument("--compile", action="store_true", help="override model compile_model")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    if args.phase == "all":
        # One phase per PROCESS: persistent_workers keeps each phase's worker
        # pool alive for the life of the process, so stacking phases OOMs the
        # box (learned the hard way — the OOM killer takes the main silently).
        sys.exit("--phase all is not supported in-process; run phases separately")
    module, dm = build(args.config, args.workers, args.batch_size, args.fast,
                       args.channels_last, args.compile,
                       force_no_compile=(args.phase == "parts"))
    print(f"[{args.phase}] config={args.config} batch={dm.batch_size_train} "
          f"workers={dm.num_workers_train} fast={args.fast} "
          f"cl={args.channels_last} compile={args.compile}")

    if args.phase == "item":
        phase_item(dm, args.item_samples)
    elif args.phase == "data":
        phase_data(dm, args.batches)
    elif args.phase == "step":
        phase_step(module, dm, args.batches)
    elif args.phase == "parts":
        phase_parts(module, dm, args.batches)
    elif args.phase == "ops":
        phase_ops(module, dm, min(args.batches, 6))


if __name__ == "__main__":
    main()
