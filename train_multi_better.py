"""Multi-source 2D training: the SF3D trainer (train_SF3D_better.py) driven
by MultiSourceDataModule — HOI4D + EPIC + ARCTIC LMDBs in one seeded
shuffle. Same losses, metrics and checkpoints as train_SF3D_better.py; only
the datamodule class differs (LightningCLI binds it at import time, so a
separate entry point is the least invasive way to switch).

    python train_multi_better.py fit --config config/multi3_rgb_scalefree.yaml
"""
from pytorch_lightning.cli import LightningCLI

from datasets.multisource_datamodule import MultiSourceDataModule
from train_SF3D_better import SF3DTrainingModule

if __name__ == "__main__":
    LightningCLI(SF3DTrainingModule, MultiSourceDataModule, save_config_callback=None)
