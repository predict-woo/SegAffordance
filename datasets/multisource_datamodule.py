"""Multi-source datamodule: HOI4D + EPIC + ARCTIC (or any SF3D-format
LMDBs) in ONE training stream, shuffled together with a fixed seed.

Every source is an SF3D-format LMDB (tools/hoi4d_process_2d.py,
epic_process_2d.py, arctic_process_2d.py write the same record layout), so
each is read by the unchanged SF3DDataset and split by scene with the SAME
seed and ratio as the single-source datamodule (a kitchen video / a
physical object / a subject-object pair never straddles train and val).
The train subsets are concatenated (optionally repeated per source to
rebalance a small source) and shuffled by a torch.Generator seeded with
manual_seed, so the interleaving of sources is reproducible run to run;
the val subsets are concatenated in source order and shuffled with the
same fixed seed (visualization sampling), test = val unshuffled.

Config shape (LightningCLI, see config/multi3_rgb_scalefree.yaml):

  data:
    sources:
      - name: hoi4d
        train_data_dir: /workspace/datasets/hoi4d_processed_2d_v2
        frame_cache_path: /workspace/datasets/hoi4d_processed_2d_v2/frames.lmdb
        key_cache_path: /workspace/cache/hoi4d_2d_keys_v2.pkl
      - name: epic
        ...
        repeat: 1
    manual_seed: 42
    val_split_ratio: 0.15
    ...

Entry point: train_multi_better.py (same SF3DTrainingModule; only the
datamodule class differs — LightningCLI binds it at import time).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from datasets.augment import AugmentSpec, AugmentedDataset
from datasets.scenefun3d import (
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)


@dataclass
class SourceSpec:
    """One SF3D-format LMDB. `repeat` > 1 duplicates the source's TRAIN
    subset that many times in the concatenation (val is never repeated) —
    a crude way to keep a 359-record source from being 6% of every epoch."""
    name: str
    train_data_dir: str
    frame_cache_path: Optional[str] = None
    key_cache_path: Optional[str] = None
    lmdb_path: Optional[str] = None
    repeat: int = 1


def seeded_train_loader(
    dataset: Dataset, batch_size: int, num_workers: int, seed: int,
) -> DataLoader:
    """The training loader: shuffled by a generator seeded with `seed`, so
    the order in which sources interleave is the same every run (and every
    epoch sequence: the generator advances deterministically)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def concat_with_repeats(subsets: List[Dataset], repeats: List[int]) -> ConcatDataset:
    parts: List[Dataset] = []
    for s, r in zip(subsets, repeats):
        if r < 1:
            raise ValueError(f"repeat must be >= 1, got {r}")
        parts.extend([s] * r)
    return ConcatDataset(parts)


class MultiSourceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sources: List[SourceSpec],
        val_split_ratio: float,
        input_size: Tuple[int, int],
        batch_size_train: int,
        batch_size_val: int,
        num_workers_train: int,
        num_workers_val: int,
        manual_seed: int,
        point_source: str = "element",
        return_trajectory_2d: bool = True,
        fast_pipeline: bool = True,
        load_depth: bool = False,
        augment: Optional[AugmentSpec] = None,
        epoch_multiplier: int = 1,
    ) -> None:
        """augment: geometry-consistent training augmentation (datasets/
        augment.py) applied to the TRAIN stream only — val/test stay raw.
        epoch_multiplier: k > 1 makes one epoch k passes over the (augmented)
        train set, i.e. k distinct augmented views of every record per
        epoch; the seeded shuffle keeps the order reproducible."""
        super().__init__()
        if epoch_multiplier < 1:
            raise ValueError("epoch_multiplier must be >= 1")
        if not sources:
            raise ValueError("MultiSourceDataModule needs at least one source")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")
        self.sources = list(sources)
        self.val_split_ratio = val_split_ratio
        self.input_size = tuple(input_size)
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.manual_seed = manual_seed
        self.point_source = point_source
        self.return_trajectory_2d = return_trajectory_2d
        self.fast_pipeline = fast_pipeline
        self.load_depth = load_depth
        self.augment = augment
        self.epoch_multiplier = epoch_multiplier
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        # name -> (train_len, val_len), filled by setup(); handy for logs/notes
        self.source_sizes: dict = {}

    def _build_source(self, spec: SourceSpec) -> SF3DDataset:
        rgb_t, mask_t, depth_t = get_default_transforms(image_size=self.input_size)
        return SF3DDataset(
            lmdb_data_root=spec.train_data_dir,
            rgb_transform=rgb_t,
            mask_transform=mask_t,
            depth_transform=depth_t,
            image_size_for_mask_reconstruction=self.input_size,
            point_source=self.point_source,
            lmdb_path=spec.lmdb_path,
            key_cache_path=spec.key_cache_path,
            return_trajectory_2d=self.return_trajectory_2d,
            frame_cache_path=spec.frame_cache_path,
            fast_pipeline=self.fast_pipeline,
            load_depth=self.load_depth,
            # the hand datasets carry no sensor verdicts and no SF3D-style
            # radius/mask/edge filters (same as the single-source hand configs)
            min_revolute_radius=0.0,
            min_mask_area_frac=0.0,
            edge_margin_frac=0.0,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ("fit", "test", "validate", None) or self.val_dataset is not None:
            return
        train_parts, val_parts, repeats = [], [], []
        for spec in self.sources:
            ds = self._build_source(spec)
            tr, va = split_dataset_by_scene(
                ds, val_split_ratio=self.val_split_ratio, manual_seed=self.manual_seed,
            )
            self.source_sizes[spec.name] = (len(tr), len(va))
            print(
                f"[multisource] {spec.name}: {len(ds)} records -> train {len(tr)}"
                f"{' x' + str(spec.repeat) if spec.repeat > 1 else ''}, val {len(va)}"
            )
            train_parts.append(tr)
            val_parts.append(va)
            repeats.append(spec.repeat)
        train = concat_with_repeats(train_parts, repeats)
        if self.augment is not None and self.augment.enabled:
            train = AugmentedDataset(train, self.augment)
        if self.epoch_multiplier > 1:
            train = ConcatDataset([train] * self.epoch_multiplier)
        self.train_dataset = train
        self.val_dataset = ConcatDataset(val_parts)
        print(
            f"[multisource] train {len(self.train_dataset)} samples/epoch "
            f"(augment={'on' if self.augment is not None and self.augment.enabled else 'off'}, "
            f"epoch_multiplier={self.epoch_multiplier}), val {len(self.val_dataset)}; "
            f"shuffle seed {self.manual_seed}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') first")
        return seeded_train_loader(
            self.train_dataset, self.batch_size_train, self.num_workers_train, self.manual_seed,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup('fit') first")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=True,  # fixed-seed shuffle: visualization sampling sees every source
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers_val > 0,
            generator=torch.Generator().manual_seed(self.manual_seed),
        )

    def test_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup('test') first")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.num_workers_val,
            pin_memory=True,
            drop_last=False,
        )
