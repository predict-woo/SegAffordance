import os
import typing
import warnings

import torch
from pytorch_lightning.cli import LightningCLI

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.opdreal_datamodule import OPDRealDataModule
from train_OPDReal_better import OPDRealTrainingModule


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OPDMultiTrainingModule(OPDRealTrainingModule):
    def __init__(
        self,
        model_params: ModelParams,
        loss_params: LossParams,
        optimizer_params: OptimizerParams,
        config: Config,
        train_only_heads: bool,
        finetune_from_path: typing.Optional[str] = None,
    ):
        super().__init__(
            model_params=model_params,
            loss_params=loss_params,
            optimizer_params=optimizer_params,
            config=config,
            finetune_from_path=finetune_from_path,
        )
        self.train_only_heads = train_only_heads

    def configure_optimizers(self):
        if self.train_only_heads:
            print(
                "❄️ Freezing backbone, depth encoder, and neck. Training only decoder and heads."
            )
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            # depth encoder may be disabled
            if hasattr(self.model, "depth_encoder") and self.model.depth_encoder is not None:
                for param in self.model.depth_encoder.parameters():
                    param.requires_grad = False
            for param in self.model.neck.parameters():
                param.requires_grad = False

            # Sanity check
            for name, param in self.model.named_parameters():
                if name.startswith(("backbone.", "depth_encoder.", "neck.")):
                    assert not param.requires_grad

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
        from torch.optim.lr_scheduler import MultiStepLR

        scheduler = MultiStepLR(
            optimizer,
            milestones=self.optimizer_params.scheduler_milestones,
            gamma=self.optimizer_params.scheduler_gamma,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    LightningCLI(OPDMultiTrainingModule, OPDRealDataModule, save_config_callback=None)
