"""Test entry tolerant of checkpoints without trainer loop state; otherwise
identical to upstream test.py (GT graph, 5 samples, retrieval + eval_metrics)."""
import os, argparse, torch
import functools
torch.load = functools.partial(torch.load, weights_only=False)  # ckpts carry OmegaConf objects
import data, systems
import systems.system_screw  # noqa: F401
import lightning.pytorch as pl
from utils.misc import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    args, extras = ap.parse_known_args()
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)
    config.data.test_which = "pm"; config.system.test_which = "pm"
    dm = data.make(config.data.name, config=config.data)
    system = systems.make(config.system.name, config=config.system)
    trainer = pl.Trainer(devices=1, accelerator="auto", logger=False, **config.trainer)
    ck = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = system.load_state_dict(ck["state_dict"], strict=False)
    print(f"[test_ft] loaded weights; missing={len(missing)} unexpected={len(unexpected)}")
    if "loops" in ck and "fit_loop" in ck["loops"]:
        try:
            trainer.fit_loop.load_state_dict(ck["loops"]["fit_loop"])
        except Exception as e:  # noqa
            print("[test_ft] could not restore fit_loop:", e)
    trainer.test(system, datamodule=dm)


if __name__ == "__main__":
    main()
