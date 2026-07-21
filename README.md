# SegAffordance

A multi-modal deep learning framework for **affordance segmentation** and **motion prediction** on articulated objects. The model takes RGB images, depth maps, and text descriptions as input, and predicts:

- **Segmentation mask** of the target object/part
- **Interaction point** (e.g., handle location)
- **Motion direction** (3D axis vector)
- **Motion type** (translation or rotation)
- **Trajectory prediction** (sequence of 3D points)

## 🏗️ Architecture Overview

The model is built on CLIP (Contrastive Language-Image Pre-training) and extends it with multi-modal fusion for affordance understanding:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input                                    │
│   RGB Image (3×H×W) + Depth Map (1×H×W) + Text Description      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP Backbone (RN50)                          │
│   ┌───────────────┐           ┌───────────────┐                 │
│   │ Visual Encoder│           │ Text Encoder  │                 │
│   │ (ModifiedResNet)          │ (Transformer) │                 │
│   └───────┬───────┘           └───────┬───────┘                 │
│           │                           │                          │
│     v2,v3,v4 features          word_features, state             │
└───────────┼───────────────────────────┼─────────────────────────┘
            │                           │
            ▼                           │
┌───────────────────────┐               │
│    Depth Encoder      │               │
│  (DepthEncoder)       │               │
│  feat_8, feat_16      │               │
└───────────┬───────────┘               │
            │                           │
            ▼                           │
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Modal FPN (Feature Pyramid Network)           │
│        Fuses visual features (v2+depth_8, v3+depth_16, v4)       │
│                     with text state                              │
└───────────────────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transformer Decoder (Cross-Attention)               │
│         Visual-Language fusion with positional encoding          │
└───────────────────────────────────────┬─────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Output Heads                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐ │
│  │  Projector_Mult │  │  MotionVAE/MLP │  │  TrajectoryMLP     │ │
│  │  (mask + point) │  │  (axis + type) │  │  (3D trajectory)   │ │
│  └────────────────┘  └────────────────┘  └────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Outputs                                  │
│   • Segmentation Mask (H/4 × W/4)                               │
│   • Interaction Point Heatmap (H/4 × W/4)                       │
│   • Soft-argmax Coordinates (x, y) ∈ [0,1]                      │
│   • Motion Direction (3D unit vector)                           │
│   • Motion Type (translation=0, rotation=1)                     │
│   • 3D Trajectory (20 points × 3 coords)                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
SegAffordance/
├── model/                          # Neural network components
│   ├── segmenter.py                # Main CRIS model
│   ├── clip.py                     # CLIP backbone (ViT/ResNet)
│   └── layers.py                   # Custom layers (FPN, Projector, VAE, etc.)
│
├── datasets/                       # Dataset implementations
│   ├── opdreal.py                  # OPDReal dataset (real-world)
│   ├── opdreal_datamodule.py       # PyTorch Lightning DataModule for OPDReal
│   ├── scenefun3d.py               # SceneFun3D dataset (indoor scenes)
│   └── scenefun3d_datamodule.py    # PyTorch Lightning DataModule for SF3D
│
├── config/                         # Configuration files
│   ├── opd_train.py                # Dataclass definitions for configs
│   ├── *_runpod*.yaml              # ACTIVE self-contained training configs (RunPod paths)
│   ├── opd_train.yaml              # Base model config (composed with overlays)
│   ├── opdreal_train.yaml          # Historical Euler-era training configs
│   ├── *_test.yaml                 # Test overlays for tests/*_better.py
│   └── ...
│
├── train_OPDReal_better.py         # Training script for OPDReal
├── train_OPDMulti_better.py        # Training script for OPDMulti (finetune)
├── train_SF3D_better.py            # Training script for SceneFun3D
│
├── tests/                          # Test/evaluation scripts
│   ├── test_OPDReal_better.py
│   ├── test_OPDMulti_better.py
│   └── test_SF3D_better.py
│
├── tools/                          # Data pipelines & utilities
│   ├── sf3d_process.py             # SceneFun3D -> LMDB builder (see docstring)
│   ├── gen_descriptions.py         # OPD description generation (Codex VLM)
│   ├── vis_predictions.py          # Checkpoint visualizer (GT vs pred panels)
│   └── show_opd_samples.py         # Dataset sample renderer
│
├── utils/                          # Utility functions
│   ├── tools.py                    # Loss functions, visualization
│   ├── dataset.py                  # CLIP tokenization
│   └── simple_tokenizer.py         # CLIP tokenizer
│
├── runpod/                         # RunPod infra: runbook, dev-pod ctl, setup
├── annotations_backup/             # Gzipped OPD annotations w/ generated descriptions (do not delete)
├── pretrain/RN50.pt                # CLIP weights (symlink on pod; see runpod/setup.sh)
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

### Setup

On a RunPod pod, `bash runpod/setup.sh` does all of this (installs
`requirements.txt` into the shared `/workspace/venv` and links the CLIP
weights). For a local/manual environment:

```bash
# Install PyTorch first (adjust for your CUDA version), then:
pip install -r requirements.txt
```

### Pretrained CLIP Weights

The model expects CLIP ResNet-50 weights at `pretrain/RN50.pt` (download from
[OpenAI CLIP](https://github.com/openai/CLIP)). On RunPod this is a symlink
to `/workspace/models/RN50.pt` created by `runpod/setup.sh` — locally it will
appear dangling; that's expected.

## 📊 Datasets

All datasets live on the RunPod network volume: OPDReal at
`/workspace/datasets/MotionDataset_h5_real`, OPDMulti at
`/workspace/datasets/OPDMulti/MotionDataset_h5`, SceneFun3D at
`/workspace/datasets/sf3d_processed`. The OPD `description` fields were
regenerated 2026-07-20 (image-conditioned VLM, see `runpod/README.md`);
gzipped copies with restore instructions are in `annotations_backup/`.
OPDSynth was dropped from the project (2026-07-19).

### OPDReal (Real-World Articulated Objects)

Real-world dataset with articulated objects (doors, drawers, cabinets, etc.).

**Data structure:**
```
MotionDataset_h5_real/
├── train.h5                    # Training images (HDF5)
├── valid.h5                    # Validation images
├── test.h5                     # Test images
├── depth.h5                    # Depth maps (all splits)
├── annotations_bwdf/           # Filtered annotations
│   ├── MotionNet_train.json
│   ├── MotionNet_valid.json
│   └── MotionNet_test.json
└── real-attr.json              # Object attributes for evaluation
```

### OPDMulti (Synthetic Multi-Object)

Synthetic dataset with multiple articulated objects per scene.

**Data structure:**
```
MotionDataset_h5/
├── train.h5
├── valid.h5  
├── test.h5
├── depth.h5
└── annotations_bwdf/
    └── ...
```

### SceneFun3D (Indoor Scenes with Trajectories)

Indoor scene dataset with interaction trajectories.

**Data structure:**
```
sf3d_processed/
├── data.lmdb                   # LMDB database with all data
├── images/                     # RGB images
└── depth/                      # Depth images (16-bit PNG, mm)
```

## 🎯 Training

**All GPU work happens on RunPod** — datasets, checkpoints, and the venv live
on a network volume mounted at `/workspace`. Read `runpod/README.md` (infra
runbook) and `CLAUDE.md` (workflow) first. The active configs are the
self-contained `config/*_runpod.yaml` files; the non-runpod YAMLs are
historical Euler-cluster configs kept for their hyperparameters.

### OPDReal Training

```bash
python train_OPDReal_better.py fit --config config/opdreal_train_runpod.yaml
```

### OPDMulti Training (Fine-tuning from OPDReal)

```bash
python train_OPDMulti_better.py fit --config config/opdmulti_train_runpod.yaml
```

Fine-tuning recipes compared 2026-07-21 (300 val samples): full fine-tune at
lr 3e-6 for ~2 epochs wins (`opdmulti_train_runpod_nofreeze_lowlr.yaml`,
IoU>0.5 70.3%) over heads-only (65.7%). OPDMulti overfits within ~2 epochs
of full fine-tuning — keep runs short.

### SceneFun3D Training

Copy the LMDB to shared memory first for I/O speed (the loader expects
`/dev/shm/data.lmdb`):

```bash
cp -r /workspace/datasets/sf3d_processed/data.lmdb /dev/shm/
python train_SF3D_better.py fit --config config/sf3d_train_runpod.yaml
```

## 📈 Testing & Evaluation

The `*_test.yaml` overlays still carry Euler-era data paths — override
`data.data_path` (and `origin_norm_json_path` where set) with the
`/workspace/datasets/...` locations when running on RunPod. For quick
qualitative checks use `tools/vis_predictions.py` instead.

### OPDReal Testing

```bash
python train_OPDReal_better.py test \
    --config config/opd_train.yaml \
    --config config/opdreal_train.yaml \
    --config config/opdreal_test.yaml \
    --ckpt_path /path/to/checkpoint.ckpt
```

### OPDMulti Testing

```bash
python train_OPDMulti_better.py test \
    --config config/opd_train.yaml \
    --config config/opdmulti_train.yaml \
    --config config/opdmulti_test.yaml \
    --ckpt_path /path/to/checkpoint.ckpt
```

### SceneFun3D Testing

```bash
python train_SF3D_better.py test \
    --config config/sf3d_train.yaml \
    --config config/sf3d_test.yaml \
    --ckpt_path /path/to/checkpoint.ckpt
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Mean IoU** | Intersection over Union for segmentation masks |
| **P_Det** | Detection rate (IoU > threshold) |
| **P_Det+M** | Detection + correct motion type |
| **P_Det+MA** | Detection + motion type + axis direction |
| **P_Det+MAO** | Detection + motion type + axis + origin (for rotation) |
| **ERR_ADir** | Mean axis direction error (degrees) |
| **Mean Point Error** | L2 error for interaction point prediction |
| **Mean Origin Error** | 3D origin error (normalized, for rotational motions) |

## ⚙️ Configuration

### Model Parameters (`ModelParams`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `clip_pretrain` | Path to CLIP weights | `pretrain/RN50.pt` |
| `word_len` | Text token length | 77 |
| `use_depth` | Enable depth fusion | `true` |
| `use_cvae` | Use CVAE for motion (vs MLP) | `true` |
| `fpn_in` | FPN input channels | `[512, 1024, 1024]` |
| `fpn_out` | FPN output channels | `[256, 512, 1024]` |
| `num_layers` | Transformer decoder layers | 3 |
| `num_head` | Attention heads | 8 |
| `dim_ffn` | FFN dimension | 1024 |
| `dropout` | Dropout rate | 0.2 |
| `vae_latent_dim` | VAE latent dimension | 32 |
| `vae_hidden_dim` | VAE hidden dimension | 256 |

### Loss Parameters (`LossParams`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mask_weight` | Mask loss weight | 0.5 |
| `point_map_weight` | Point heatmap loss weight | 1.0 |
| `coord_weight` | Coordinate regression weight | 0.5 |
| `vae_weight` | Motion VAE loss weight | 0.5 |
| `motion_type_weight` | Motion type classification weight | 0.5 |
| `trajectory_weight` | Trajectory prediction weight | 0.5 |
| `geometric_weight` | Geometric consistency weight | 0.5 |
| `bce_weight` | BCE weight in DiceBCE | 0.5 |
| `dice_weight` | Dice weight in DiceBCE | 0.5 |
| `point_sigma` | Gaussian heatmap sigma | 8.0 |
| `vae_beta` | KL divergence weight | 0.01 |

### Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `test_iou_threshold` | IoU threshold for matching | 0.5 |
| `test_motion_threshold_deg` | Axis error threshold (degrees) | 10.0 |
| `test_pred_threshold` | Mask binarization threshold | 0.5 |
| `test_match_metric` | Matching metric (`mask` or `bbox`) | `mask` |
| `test_visualize_debug` | Enable debug visualizations | `false` |

## 🔬 Model Components

### CRIS Model (`model/segmenter.py`)

Main model class that combines all components:

```python
class CRIS(nn.Module):
    def __init__(self, model_params):
        # CLIP backbone for vision and text encoding
        self.backbone = build_model(clip_state_dict, word_len)
        
        # Optional depth encoder
        self.depth_encoder = DepthEncoder(...)
        
        # Multi-modal FPN
        self.neck = FPN(...)
        
        # Transformer decoder for cross-attention
        self.decoder = TransformerDecoder(...)
        
        # Output projector (mask + point)
        self.proj = Projector_Mult(...)
        
        # Motion prediction (CVAE or MLP)
        self.motion_vae = MotionVAE(...)  # or self.motion_mlp
        
        # Trajectory prediction
        self.trajectory_predictor = TrajectoryMLP(...)
```

### Key Layers (`model/layers.py`)

- **`DepthEncoder`**: Encodes depth maps to multi-scale features
- **`FPN`**: Feature Pyramid Network for multi-scale visual-text fusion
- **`TransformerDecoder`**: Cross-attention between visual and text features
- **`Projector_Mult`**: Dynamic kernel convolution for multi-output prediction
- **`MotionVAE`**: Conditional VAE for motion vector prediction
- **`MotionMLP`**: Deterministic MLP alternative to VAE
- **`TrajectoryMLP`**: Predicts 3D trajectory points

### Loss Functions (`utils/tools.py`)

- **`DiceBCELoss`**: Combined Dice + BCE loss for segmentation
- **`MotionVAELoss`**: Reconstruction + KL divergence for motion VAE
- **Geometric Consistency Loss**: Ensures trajectory-motion vector consistency

## 🎨 Visualization

The RunPod configs log losses to CSV under `experiments/<id>/logs/` with wandb off;
`tools/vis_predictions.py` renders GT-vs-prediction panels for any checkpoint.
When wandb is enabled, training visualizations include:

- Predicted vs GT segmentation masks
- Interaction point heatmaps
- Motion direction arrows (3D projected)
- Trajectory points (GT: blue, Pred: cyan)
- Motion type annotations

Debug visualizations during testing save images to disk showing:
- Point prediction heatmap overlay
- Mask prediction overlay
- Ground truth annotations

## 📝 Training Tips

1. **Data Loading Speed**: For SceneFun3D, copy LMDB to `/dev/shm` for faster I/O
2. **Transfer Learning**: Train on OPDReal first, then finetune on OPDMulti
3. **Depth Usage**: Depth improves performance on OPDReal; can be disabled with `use_depth: false`
4. **CVAE vs MLP**: CVAE captures multi-modal motion distributions; MLP is simpler
5. **Batch Size**: Use larger batches (128-256) with mixed precision (`precision: 16`)

## 🔗 References

- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [CRIS: CLIP-Driven Referring Image Segmentation](https://arxiv.org/abs/2111.15174)
- [OPD Dataset](https://3dlg-hcvc.github.io/OPD/)
- [SceneFun3D](https://scenefun3d.github.io/)
