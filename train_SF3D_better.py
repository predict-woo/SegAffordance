import os
import typing
import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.scenefun3d_datamodule import SF3DDataModule
from model.segmenter import CRIS
from train_OPDReal_better import OPDRealTrainingModule
from utils.dataset import tokenize
from utils.tools import create_composite_visualization, make_gaussian_map

torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SF3DTrainingModule(OPDRealTrainingModule):
    pass


if __name__ == "__main__":
    LightningCLI(SF3DTrainingModule, SF3DDataModule, save_config_callback=None)
