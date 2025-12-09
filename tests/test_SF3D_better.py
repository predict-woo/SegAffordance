import os
import warnings

import torch
from pytorch_lightning.cli import LightningCLI

from datasets.scenefun3d_datamodule import SF3DDataModule
from train_SF3D_better import SF3DTrainingModule


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    # Usage:
    #   python test_SF3D_better.py test --config config/opd_train.yaml --config config/sf3d_train.yaml --ckpt_path /path/to.ckpt
    LightningCLI(SF3DTrainingModule, SF3DDataModule, save_config_callback=None)


