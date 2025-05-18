import os
import argparse
import datetime
import time
import warnings
from functools import partial

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import IoU, Precision
import wandb

import utils.config as config
from utils.dataset import RefDataset
from model import build_segmenter

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

class CRISLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model, self.param_list = build_segmenter(args)
        if args.sync_bn:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Initialize metrics
        self.train_iou = IoU(num_classes=2, threshold=0.35)
        self.val_iou = IoU(num_classes=2, threshold=0.35)
        self.train_precision = Precision(task='binary', threshold=0.5)
        self.val_precision = Precision(task='binary', threshold=0.5)
        
        # Save hyperparameters
        self.save_hyperparameters(args)
        
    def forward(self, img, text, mask=None):
        return self.model(img, text, mask)
    
    def training_step(self, batch, batch_idx):
        image, text, target = batch
        target = target.unsqueeze(1)
        
        # Forward pass
        pred, target, loss = self(image, text, target)
        
        # Calculate metrics
        pred_sigmoid = torch.sigmoid(pred)
        pred_binary = (pred_sigmoid > 0.35).float()
        target_binary = (target > 0.35).float()
        
        iou = self.train_iou(pred_binary, target_binary)
        precision = self.train_precision(pred_binary, target_binary)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/prec@50', precision, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, text, params = batch
        
        # Forward pass
        pred = self(image, text)
        pred = torch.sigmoid(pred)
        
        # Resize prediction if needed
        if pred.shape[-2:] != image.shape[-2:]:
            pred = F.interpolate(pred, size=image.shape[-2:], mode='bicubic', align_corners=True).squeeze(1)
            
        # Process each sample in batch
        iou_list = []
        for pred_i, mask_dir, mat, ori_size in zip(pred, params['mask_dir'], params['inverse'], params['ori_size']):
            h, w = ori_size
            mat = mat.numpy()
            pred_i = pred_i.cpu().numpy()
            pred_i = cv2.warpAffine(pred_i, mat, (w, h), flags=cv2.INTER_CUBIC, borderValue=0.)
            pred_i = np.array(pred_i > 0.35)
            
            mask = cv2.imread(mask_dir, flags=cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            
            # Calculate IoU
            inter = np.logical_and(pred_i, mask)
            union = np.logical_or(pred_i, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            
        return torch.tensor(iou_list, device=self.device)
    
    def validation_epoch_end(self, outputs):
        # Concatenate all IoU values
        iou_list = torch.cat(outputs)
        
        # Calculate precision at different thresholds
        prec_list = []
        for thres in torch.arange(0.5, 1.0, 0.1):
            tmp = (iou_list > thres).float().mean()
            prec_list.append(tmp)
            
        iou = iou_list.mean()
        
        # Log metrics
        self.log('val/iou', iou, on_epoch=True, prog_bar=True)
        for i, thres in enumerate(range(5, 10)):
            self.log(f'val/Pr@{thres*10}', prec_list[i], on_epoch=True)
            
        return iou
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.param_list,
                                   lr=self.args.base_lr,
                                   weight_decay=self.args.weight_decay)
        
        scheduler = MultiStepLR(optimizer,
                              milestones=self.args.milestones,
                              gamma=self.args.lr_decay)
        
        return [optimizer], [scheduler]

def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='path to checkpoint to resume from')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

@rank_zero_only
def setup_wandb(args):
    """Setup Weights & Biases logging"""
    wandb.init(
        project="CRIS",
        name=args.exp_name,
        config=args,
        tags=[args.dataset, args.clip_pretrain]
    )

def main():
    args = get_parser()
    
    # Setup output directory
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project="CRIS",
        name=args.exp_name,
        config=args,
        tags=[args.dataset, args.clip_pretrain]
    )
    
    # Build datasets
    train_data = RefDataset(lmdb_dir=args.train_lmdb,
                           mask_dir=args.mask_root,
                           dataset=args.dataset,
                           split=args.train_split,
                           mode='train',
                           input_size=args.input_size,
                           word_length=args.word_len)
    
    val_data = RefDataset(lmdb_dir=args.val_lmdb,
                         mask_dir=args.mask_root,
                         dataset=args.dataset,
                         split=args.val_split,
                         mode='val',
                         input_size=args.input_size,
                         word_length=args.word_len)
    
    # Build dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers_val,
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    model = CRISLightning(args)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename='{epoch}-{val/iou:.2f}',
        monitor='val/iou',
        mode='max',
        save_top_k=1
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # Use 1 GPU or CPU
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        precision=16 if args.fp16 else 32,
        gradient_clip_val=args.max_norm if hasattr(args, 'max_norm') else None,
        deterministic=True if args.manual_seed is not None else False,
        default_root_dir=args.output_dir
    )
    
    # Train
    if args.resume:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    else:
        trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    main() 