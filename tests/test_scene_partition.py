"""Tests for the scene-level train partition behind the 2D-pretrain
label-efficiency experiments (spec:
docs/superpowers/specs/2026-08-22-2d-pretrain-label-efficiency-design.md).

partition_subset_by_scene splits the train Subset produced by
split_dataset_by_scene into a large "pretrain" and a small "finetune" part,
greedily by sample count at scene granularity. These tests run on a fake
dataset object (only .item_keys is consumed) — no LMDB needed.
"""

import pytest
from torch.utils.data import Subset

from datasets.scenefun3d import partition_subset_by_scene
from datasets.scenefun3d_datamodule import SF3DDataModule


class FakeDataset:
    """Only what partition_subset_by_scene touches: item_keys."""

    def __init__(self, scene_sizes):
        # scene_sizes: {scene_id: n_frames}
        self.item_keys = []
        for scene_id, n in scene_sizes.items():
            for i in range(n):
                self.item_keys.append(f"{scene_id}/frame_{i:04d}".encode())

    def __len__(self):
        return len(self.item_keys)


def make_train_subset(scene_sizes):
    ds = FakeDataset(scene_sizes)
    return ds, Subset(ds, list(range(len(ds))))


# Unequal scene sizes on purpose — the greedy-by-sample-count behavior is
# only distinguishable from scene-count splitting when sizes differ a lot.
SCENES = {f"scene{i:02d}": n for i, n in enumerate(
    [400, 350, 300, 250, 200, 150, 120, 100, 80, 50, 30, 20, 10, 5, 5]
)}


def test_partition_is_deterministic():
    _, subset = make_train_subset(SCENES)
    pre1, fine1 = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    pre2, fine2 = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    assert list(pre1.indices) == list(pre2.indices)
    assert list(fine1.indices) == list(fine2.indices)


def test_partition_changes_with_seed():
    _, subset = make_train_subset(SCENES)
    _, fine_a = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    _, fine_b = partition_subset_by_scene(subset, ratio=0.1, seed=7)
    assert set(fine_a.indices) != set(fine_b.indices)


def test_partition_disjoint_and_complete():
    _, subset = make_train_subset(SCENES)
    pre, fine = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    pre_set, fine_set = set(pre.indices), set(fine.indices)
    assert pre_set.isdisjoint(fine_set)
    assert pre_set | fine_set == set(subset.indices)


def test_no_scene_straddles_the_partition():
    ds, subset = make_train_subset(SCENES)
    pre, fine = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    scene = lambda idx: ds.item_keys[idx].decode().split("/")[0]
    pre_scenes = {scene(i) for i in pre.indices}
    fine_scenes = {scene(i) for i in fine.indices}
    assert pre_scenes.isdisjoint(fine_scenes)


def test_finetune_fraction_near_target():
    _, subset = make_train_subset(SCENES)
    total = len(subset.indices)
    max_scene = max(SCENES.values())
    for seed in (4242, 7, 99):
        _, fine = partition_subset_by_scene(subset, ratio=0.1, seed=seed)
        frac = len(fine.indices) / total
        # Greedy stops at the first scene crossing the target, so the
        # overshoot is bounded by the largest scene.
        assert 0.1 <= frac <= 0.1 + max_scene / total


def test_pretrain_indices_preserve_subset_order():
    # Pretrain indices are filtered from subset.indices, keeping order —
    # matters for reproducibility of anything indexing into the subset.
    _, subset = make_train_subset(SCENES)
    pre, _ = partition_subset_by_scene(subset, ratio=0.1, seed=4242)
    positions = {idx: i for i, idx in enumerate(subset.indices)}
    assert list(pre.indices) == sorted(pre.indices, key=lambda x: positions[x])


def _make_datamodule(train_scene_subset):
    return SF3DDataModule(
        train_data_dir="/nonexistent",
        val_split_ratio=0.1,
        input_size=(512, 512),
        batch_size_train=1,
        batch_size_val=1,
        num_workers_train=0,
        num_workers_val=0,
        manual_seed=42,
        train_scene_subset=train_scene_subset,
    )


def test_datamodule_rejects_unknown_subset_name():
    with pytest.raises(ValueError, match="train_scene_subset"):
        _make_datamodule("bogus")


def test_datamodule_accepts_valid_subset_names():
    for name in (None, "pretrain", "finetune"):
        dm = _make_datamodule(name)
        assert dm.train_scene_subset == name
        assert dm.train_subset_ratio == 0.1
        assert dm.train_subset_seed == 4242


def test_datamodule_setup_selects_partition(monkeypatch):
    # Drive setup() end-to-end with the SF3DDataset construction and the
    # val split stubbed out, and check which partition lands in
    # train_dataset — and that val_dataset is never touched.
    import datasets.scenefun3d_datamodule as dm_mod

    ds, full_train = make_train_subset(SCENES)
    val_marker = Subset(ds, [0, 1])

    monkeypatch.setattr(
        dm_mod, "get_default_transforms", lambda image_size: (None, None, None)
    )
    monkeypatch.setattr(dm_mod, "SF3DDataset", lambda **kw: ds)
    monkeypatch.setattr(
        dm_mod,
        "split_dataset_by_scene",
        lambda dataset, val_split_ratio, manual_seed: (full_train, val_marker),
    )

    expected_pre, expected_fine = partition_subset_by_scene(
        full_train, ratio=0.1, seed=4242
    )

    for name, expected in (
        ("pretrain", expected_pre),
        ("finetune", expected_fine),
    ):
        dm = _make_datamodule(name)
        dm.setup("fit")
        assert list(dm.train_dataset.indices) == list(expected.indices)
        assert dm.val_dataset is val_marker

    dm = _make_datamodule(None)
    dm.setup("fit")
    assert dm.train_dataset is full_train
