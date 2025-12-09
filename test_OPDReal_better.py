import os
import warnings

import torch
from pytorch_lightning.cli import LightningCLI

from datasets.opdreal_datamodule import OPDRealDataModule
from train_OPDReal_better import OPDRealTrainingModule


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


if __name__ == "__main__":
    # Usage:
    #   python test_OPDReal_better.py test --config config/opd_train.yaml --config config/opdreal_train.yaml --data.is_multi false --ckpt_path /path/to.ckpt
    LightningCLI(OPDRealTrainingModule, OPDRealDataModule, save_config_callback=None)
