"""Multi-source datamodule: several SF3D-format LMDBs (HOI4D, EPIC, ARCTIC,
SF3D itself) in ONE training stream of SOURCE-HOMOGENEOUS batches, shuffled
with a fixed seed.

Every source is read by the unchanged SF3DDataset and split by scene with
the shared seed (per-source ratio). The train subsets are optionally
augmented (datasets/augment.py, per source), repeated (`repeat` = k
augmented views per record per epoch — the knob that balances a 5k-record
hand set against 53k SF3D records), tagged with their source name (16th
tuple element), and concatenated. `SourceBatchSampler` then yields batches
that each come from ONE source, in a seeded random order, so the trainer
can pick that source's loss profile (2D projection recipe vs 3D SF3D
recipe) per batch — every existing loss stays untouched and the expected
gradient equals row-level mixing. Val = the union of the val subsets, also
in homogeneous batches (fixed-seed shuffle), so `val/<source>/loss_total`
is logged per source; test = val unshuffled.

Config shape (LightningCLI, see config/joint4_dct_rgb_scalefree.yaml):

  data:
    sources:
      - name: sf3d
        train_data_dir: /workspace/datasets/sf3d_processed_v3
        frame_cache_path: /workspace/datasets/sf3d_frames_512.lmdb
        key_cache_path: /workspace/cache/...
        min_revolute_radius: 0.10
        min_mask_area_frac: 0.001
        edge_margin_frac: 0.05
        val_split_ratio: 0.1
        augment: false
        loss_profile: "3d"
      - name: hoi4d
        ...
        repeat: 10
        hflip_p: 0.0
        loss_profile: "2d"
    augment: {...}          # shared augmentation spec (train only, per-source on/off)
    manual_seed: 42
    val_split_ratio: 0.15   # default per-source ratio

Entry point: train_multi_better.py (same SF3DTrainingModule; the trainer
maps `targets.source` -> loss profile via model.source_profiles).
"""
from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from datasets.augment import AugmentSpec, AugmentedDataset
from datasets.scenefun3d import (
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)


@dataclass
class SourceSpec:
    """One SF3D-format LMDB and how it enters the stream."""
    name: str
    train_data_dir: str
    frame_cache_path: Optional[str] = None
    key_cache_path: Optional[str] = None
    lmdb_path: Optional[str] = None
    # k > 1 duplicates the source's TRAIN subset k times in the concatenation
    # (val never) — with augmentation on, k distinct augmented views per
    # record per epoch. Balances small hand sets against SF3D.
    repeat: int = 1
    # Per-source override of the augmentation's horizontal-flip probability
    # (None = the shared data.augment value). HOI4D descriptions say "right
    # drawer" etc., so it runs with 0.0; EPIC/ARCTIC texts carry no
    # orientation words.
    hflip_p: Optional[float] = None
    # Whether the shared augmentation applies to this source's train subset
    # at all (SF3D: false — augmentation exists to balance the hand sets).
    augment: bool = True
    # SF3DDataset filters (the hand sets use 0 / 0 / 0; SF3D's g19 recipe uses
    # 0.10 / 0.001 / 0.05) and the sensor-occlusion cutoff (None disables).
    min_revolute_radius: float = 0.0
    min_mask_area_frac: float = 0.0
    edge_margin_frac: float = 0.0
    sensor_max_occluded_frac: Optional[float] = 0.5
    # Per-source val ratio (None = the datamodule's default).
    val_split_ratio: Optional[float] = None
    # Which loss profile the trainer applies to this source's batches
    # (model.source_profiles maps name -> profile; kept here for the record).
    loss_profile: str = "2d"
    # 2026-09-11: leave records with motion_type "none" out of this source's key list
    # (the LMDB keeps them). Set False to train on them (read as translation).
    skip_unlabeled_motion: bool = True


class SourceTagDataset(Dataset):
    """Appends the source name as the 16th tuple element of every sample."""

    def __init__(self, base: Dataset, name: str):
        self.base = base
        self.name = name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return tuple(self.base[i]) + (self.name,)


class SourceBatchSampler(Sampler[List[int]]):
    """Batches of global ConcatDataset indices, each batch from ONE part.

    Within a part the indices are shuffled; the batch order across parts is
    shuffled too; both from a generator seeded with (seed + epoch), so the
    interleaving is reproducible and changes every epoch. `set_epoch` is
    what Lightning calls when it exists; the iterator also advances its
    own epoch counter as a fallback.
    """

    def __init__(self, part_lengths: Sequence[int], batch_size: int, seed: int,
                 shuffle: bool = True, drop_last: bool = True):
        self.part_lengths = list(part_lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self._calls = 0
        self.offsets = [0]
        for n in self.part_lengths:
            self.offsets.append(self.offsets[-1] + n)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> List[List[int]]:
        g = torch.Generator().manual_seed(self.seed + self.epoch + self._calls * 100003)
        batches: List[List[int]] = []
        for p, n in enumerate(self.part_lengths):
            order = torch.randperm(n, generator=g) if self.shuffle else torch.arange(n)
            order = (order + self.offsets[p]).tolist()
            for s in range(0, n, self.batch_size):
                b = order[s:s + self.batch_size]
                if len(b) < self.batch_size and self.drop_last:
                    continue
                batches.append(b)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._batches()
        if self.shuffle:
            self._calls += 1  # next epoch reshuffles even without set_epoch
        return iter(batches)

    def __len__(self) -> int:
        total = 0
        for n in self.part_lengths:
            total += n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        return total


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
        augment.py) applied to the TRAIN stream of the sources with
        augment=true — val/test stay raw. epoch_multiplier: k > 1 repeats
        EVERY source's train part k more times (on top of per-source repeat)."""
        super().__init__()
        if not sources:
            raise ValueError("MultiSourceDataModule needs at least one source")
        names = [s.name for s in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names: {names}")
        if epoch_multiplier < 1:
            raise ValueError("epoch_multiplier must be >= 1")
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
        # name -> (train_len, val_len) of the RAW subsets, filled by setup()
        self.source_sizes: Dict[str, Tuple[int, int]] = {}
        self._train_part_lengths: List[int] = []
        self._val_part_lengths: List[int] = []

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
            sensor_max_occluded_frac=spec.sensor_max_occluded_frac,
            min_revolute_radius=spec.min_revolute_radius,
            min_mask_area_frac=spec.min_mask_area_frac,
            edge_margin_frac=spec.edge_margin_frac,
            skip_unlabeled_motion=spec.skip_unlabeled_motion,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage not in ("fit", "test", "validate", None) or self.val_dataset is not None:
            return
        train_parts: List[Dataset] = []
        val_parts: List[Dataset] = []
        for spec in self.sources:
            ds = self._build_source(spec)
            ratio = self.val_split_ratio if spec.val_split_ratio is None else spec.val_split_ratio
            tr, va = split_dataset_by_scene(ds, val_split_ratio=ratio, manual_seed=self.manual_seed)
            self.source_sizes[spec.name] = (len(tr), len(va))
            aug_note = "off"
            if self.augment is not None and self.augment.enabled and spec.augment:
                src_aug = self.augment
                if spec.hflip_p is not None:
                    src_aug = replace(src_aug, hflip_p=spec.hflip_p)
                tr = AugmentedDataset(tr, src_aug)
                aug_note = f"on (hflip_p={src_aug.hflip_p})"
            k = spec.repeat * self.epoch_multiplier
            if k < 1:
                raise ValueError(f"{spec.name}: repeat must be >= 1")
            tagged = SourceTagDataset(tr, spec.name)
            train_parts.extend([tagged] * k)
            val_parts.append(SourceTagDataset(va, spec.name))
            print(
                f"[multisource] {spec.name}: {len(ds)} records -> train {len(tr)}"
                f"{' x' + str(k) if k > 1 else ''} = {len(tr) * k}/epoch, val {len(va)}, "
                f"augment {aug_note}, profile {spec.loss_profile}"
            )
        self.train_dataset = ConcatDataset(train_parts)
        self.val_dataset = ConcatDataset(val_parts)
        self._train_part_lengths = [len(p) for p in train_parts]
        self._val_part_lengths = [len(p) for p in val_parts]
        print(
            f"[multisource] train {len(self.train_dataset)} samples/epoch in source-homogeneous "
            f"batches, val {len(self.val_dataset)}; shuffle seed {self.manual_seed}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') first")
        return DataLoader(
            self.train_dataset,
            batch_sampler=SourceBatchSampler(
                self._train_part_lengths, self.batch_size_train, self.manual_seed,
                shuffle=True, drop_last=True,
            ),
            num_workers=self.num_workers_train,
            pin_memory=True,
            persistent_workers=self.num_workers_train > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup('fit') first")
        # fixed-seed shuffle (visualization sampling sees every source);
        # homogeneous batches so the per-source loss profile applies
        sampler = SourceBatchSampler(
            self._val_part_lengths, self.batch_size_val, self.manual_seed,
            shuffle=True, drop_last=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=_FixedOrderSampler(sampler),  # same order every epoch
            num_workers=self.num_workers_val,
            pin_memory=True,
            persistent_workers=self.num_workers_val > 0,
        )

    def test_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup('test') first")
        return DataLoader(
            self.val_dataset,
            batch_sampler=SourceBatchSampler(
                self._val_part_lengths, self.batch_size_val, self.manual_seed,
                shuffle=False, drop_last=False,
            ),
            num_workers=self.num_workers_val,
            pin_memory=True,
        )


class _FixedOrderSampler(Sampler[List[int]]):
    """A SourceBatchSampler frozen at one shuffle (val: same order every epoch)."""

    def __init__(self, inner: SourceBatchSampler):
        inner._calls = 0
        self.batches = inner._batches()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# kept for callers/tests of the first version
def seeded_train_loader(dataset: Dataset, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def concat_with_repeats(subsets: List[Dataset], repeats: List[int]) -> ConcatDataset:
    parts: List[Dataset] = []
    for s, r in zip(subsets, repeats):
        if r < 1:
            raise ValueError(f"repeat must be >= 1, got {r}")
        parts.extend([s] * r)
    return ConcatDataset(parts)
