"""MultiSourceDataModule: per-source scene splits, concatenation with
repeats, and the fixed-seed shuffle that makes the source interleaving
reproducible. SF3DDataset is stubbed (no LMDBs needed)."""
import torch
from torch.utils.data import Dataset

import datasets.multisource_datamodule as msd
from datasets.multisource_datamodule import (
    MultiSourceDataModule,
    SourceSpec,
    concat_with_repeats,
    seeded_train_loader,
)


class _Stub(Dataset):
    """Looks like SF3DDataset to split_dataset_by_scene: item_keys with a
    'scene/rest' layout; __getitem__ returns (source, index)."""
    def __init__(self, name, scenes, per_scene):
        self.name = name
        self.item_keys = [f"{name}_scene{s}/f{i}".encode() for s in range(scenes) for i in range(per_scene)]

    def __len__(self):
        return len(self.item_keys)

    def __getitem__(self, i):
        return (self.name, i)


def _dm(monkeypatch, seed=42, repeats=(1, 1, 1)):
    specs = [
        SourceSpec(name="hoi4d", train_data_dir="/x/hoi4d", repeat=repeats[0]),
        SourceSpec(name="epic", train_data_dir="/x/epic", repeat=repeats[1]),
        SourceSpec(name="arctic", train_data_dir="/x/arctic", repeat=repeats[2]),
    ]
    sizes = {"hoi4d": (20, 5), "epic": (10, 3), "arctic": (16, 4)}
    monkeypatch.setattr(
        MultiSourceDataModule, "_build_source",
        lambda self, spec: _Stub(spec.name, *sizes[spec.name]),
    )
    dm = MultiSourceDataModule(
        sources=specs, val_split_ratio=0.2, input_size=(64, 64), batch_size_train=4,
        batch_size_val=4, num_workers_train=0, num_workers_val=0, manual_seed=seed,
    )
    dm.setup("fit")
    return dm


def _order(dm):
    return [tuple(x) for b in dm.train_dataloader() for x in zip(*b)]


def test_split_per_source_no_scene_leak_and_all_sources_present(monkeypatch):
    dm = _dm(monkeypatch)
    assert set(dm.source_sizes) == {"hoi4d", "epic", "arctic"}
    for name, (ntr, nva) in dm.source_sizes.items():
        assert ntr > 0 and nva > 0
    total_train = sum(v[0] for v in dm.source_sizes.values())
    assert len(dm.train_dataset) == total_train
    assert len(dm.val_dataset) == sum(v[1] for v in dm.source_sizes.values())
    # every train item and every val item belongs to exactly one split
    train_items = {dm.train_dataset[i] for i in range(len(dm.train_dataset))}
    val_items = {dm.val_dataset[i] for i in range(len(dm.val_dataset))}
    assert not (train_items & val_items)
    assert {s for s, _ in train_items} == {"hoi4d", "epic", "arctic"}


def test_fixed_seed_gives_identical_interleaving_and_a_new_seed_changes_it(monkeypatch):
    a = _order(_dm(monkeypatch, seed=7))
    b = _order(_dm(monkeypatch, seed=7))
    c = _order(_dm(monkeypatch, seed=8))
    assert a == b
    assert a != c
    # sources are actually mixed within the stream (not blocked by source)
    first_batch_sources = {s for s, _ in a[:4]}
    assert len(a) > 4 and len({s for s, _ in a}) == 3
    assert any(a[i][0] != a[i + 1][0] for i in range(len(a) - 1))
    assert isinstance(first_batch_sources, set)


def test_repeat_duplicates_only_the_train_subset(monkeypatch):
    dm1 = _dm(monkeypatch, repeats=(1, 1, 1))
    dm3 = _dm(monkeypatch, repeats=(1, 3, 1))
    e_tr, e_va = dm1.source_sizes["epic"]
    assert len(dm3.train_dataset) == len(dm1.train_dataset) + 2 * e_tr
    assert len(dm3.val_dataset) == len(dm1.val_dataset)


def test_per_source_hflip_override_wraps_each_source_separately(monkeypatch):
    from datasets.augment import AugmentSpec, AugmentedDataset
    specs = [
        SourceSpec(name="hoi4d", train_data_dir="/x/hoi4d", hflip_p=0.0),
        SourceSpec(name="epic", train_data_dir="/x/epic"),
    ]
    monkeypatch.setattr(
        MultiSourceDataModule, "_build_source",
        lambda self, spec: _Stub(spec.name, 10, 3),
    )
    dm = MultiSourceDataModule(
        sources=specs, val_split_ratio=0.2, input_size=(64, 64), batch_size_train=4,
        batch_size_val=4, num_workers_train=0, num_workers_val=0, manual_seed=1,
        augment=AugmentSpec(hflip_p=0.5), epoch_multiplier=2,
    )
    dm.setup("fit")
    parts = dm.train_dataset.datasets  # epoch_multiplier=2 -> [concat, concat]
    assert len(parts) == 2 and parts[0] is parts[1]
    per_source = parts[0].datasets
    assert all(isinstance(p, AugmentedDataset) for p in per_source)
    assert per_source[0].spec.hflip_p == 0.0 and per_source[1].spec.hflip_p == 0.5
    assert per_source[0].spec.crop_p == per_source[1].spec.crop_p  # only the flip differs
    # val is never augmented
    assert not any(isinstance(p, AugmentedDataset) for p in dm.val_dataset.datasets)


def test_concat_and_loader_helpers():
    ds = [torch.utils.data.TensorDataset(torch.arange(3)), torch.utils.data.TensorDataset(torch.arange(3, 5))]
    cat = concat_with_repeats(ds, [2, 1])
    assert len(cat) == 8
    o1 = [int(x) for b in seeded_train_loader(cat, 2, 0, 3) for x in b[0]]
    o2 = [int(x) for b in seeded_train_loader(cat, 2, 0, 3) for x in b[0]]
    assert o1 == o2 and sorted(o1) == sorted([0, 1, 2, 0, 1, 2, 3, 4])
