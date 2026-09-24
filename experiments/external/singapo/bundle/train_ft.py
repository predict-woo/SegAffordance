"""Matched fine-tune entry: load WEIGHTS ONLY from a released checkpoint
(no trainer/optimizer/scheduler state), start a fresh short run. Both arms
use this file; the arm is selected by config (system.name / screw_weight)."""
import os, argparse, torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)  # ckpts carry OmegaConf objects
import data, systems
import systems.system_screw  # noqa: F401 (registers sys_singapo_screw)
from utils.misc import load_config
from utils.callbacks import ConfigSnapshotCallback, GPUCacheCleanCallback
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def split_params(system):
    # replicate SingapoSystem.load_cage_weights' split without loading CAGE
    system.cage_params, system.adapter_params = [], []
    for name, param in system.model.named_parameters():
        (system.adapter_params if ("img" in name or "norm5" in name) else system.cage_params).append(param)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init_weights", required=True, help="released ckpt; weights only")
    args, extras = ap.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)
    pl.seed_everything(int(config.get("seed", 42)), workers=True)

    dm = data.make(config.data.name, config=config.data)
    system = systems.make(config.system.name, config=config.system)
    sd = torch.load(args.init_weights, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = system.load_state_dict(sd, strict=False)
    print(f"[init_weights] loaded {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    assert len(unexpected) == 0, unexpected[:5]
    split_params(system)

    os.makedirs(config.logger.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint.dirpath, exist_ok=True)
    logger = CSVLogger(save_dir=config.logger.save_dir, name="csv")
    callbacks = [ModelCheckpoint(**config.checkpoint), LearningRateMonitor(),
                 ConfigSnapshotCallback(config), GPUCacheCleanCallback()]
    trainer = pl.Trainer(devices=1, accelerator="auto", logger=logger,
                         callbacks=callbacks, **config.trainer)
    trainer.fit(system, datamodule=dm)


if __name__ == "__main__":
    main()
