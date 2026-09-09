"""Joint 2D+3D training (2026-09-10): source-homogeneous batches, source tags,
and per-batch loss profiles in the trainer."""
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from datasets.multisource_datamodule import SourceBatchSampler, SourceTagDataset
from model.targets import unpack_batch


class _Items(Dataset):
    def __init__(self, name, n):
        self.name, self.n = name, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (self.name, i)


# ---- sampler --------------------------------------------------------------

def test_source_batch_sampler_is_homogeneous_seeded_and_covers_everything():
    lengths = [17, 5, 30]
    s = SourceBatchSampler(lengths, batch_size=4, seed=3, shuffle=True, drop_last=False)
    batches = list(iter(s))
    offsets = [0, 17, 22, 52]
    def part(i):
        return next(p for p in range(3) if offsets[p] <= i < offsets[p + 1])
    for b in batches:
        assert len({part(i) for i in b}) == 1        # one source per batch
        assert len(b) <= 4
    assert sorted(i for b in batches for i in b) == list(range(52))  # every index once
    assert len(batches) == len(s) == 5 + 2 + 8
    # same seed + epoch -> same order; the next epoch reshuffles; set_epoch reproduces
    a = list(iter(SourceBatchSampler(lengths, 4, seed=3)))
    b = list(iter(SourceBatchSampler(lengths, 4, seed=3)))
    assert a == b
    s2 = SourceBatchSampler(lengths, 4, seed=3)
    e0 = list(iter(s2)); e1 = list(iter(s2))
    assert e0 != e1
    s3 = SourceBatchSampler(lengths, 4, seed=3); s3.set_epoch(1)
    assert list(iter(s3)) != e0
    # drop_last drops the ragged tail of each part
    sd = SourceBatchSampler(lengths, 4, seed=3, drop_last=True)
    assert len(sd) == 4 + 1 + 7 and all(len(b) == 4 for b in iter(sd))
    # a part smaller than the batch contributes nothing under drop_last
    assert len(SourceBatchSampler([3, 8], 4, seed=0, drop_last=True)) == 2


def test_tagged_concat_batches_carry_one_source_each():
    parts = [SourceTagDataset(_Items("a", 9), "a"), SourceTagDataset(_Items("b", 6), "b")]
    ds = ConcatDataset(parts)
    loader = DataLoader(ds, batch_sampler=SourceBatchSampler([9, 6], 4, seed=1, drop_last=False), num_workers=0)
    seen = {"a": 0, "b": 0}
    for names, idx, src in loader:
        assert len(set(src)) == 1 and set(names) == set(src)
        seen[src[0]] += len(src)
    assert seen == {"a": 9, "b": 6}


# ---- unpack_batch ----------------------------------------------------------

def _tuple15(B=2, N=5):
    z = torch.zeros
    return (z(B, 3, 8, 8), z(B, 1, 8, 8), ["a", "b"], z(B, 1, 8, 8), z(B, 4), z(B, 2), z(B, 3),
            z(B, dtype=torch.long), torch.tensor([[8.0, 8.0]] * B), ["f0", "f1"], z(B, 3),
            torch.eye(3).expand(B, 3, 3), z(B, N, 3), z(B, N, 2), torch.ones(B, N, dtype=torch.bool))


def test_unpack_batch_reads_the_source_tag_and_rejects_mixed_batches():
    _, _, _, t = unpack_batch(_tuple15())
    assert t.source is None
    _, _, _, t = unpack_batch(_tuple15() + (["hoi4d", "hoi4d"],))
    assert t.source == "hoi4d" and t.trajectory_2d is not None
    try:
        unpack_batch(_tuple15() + (["hoi4d", "sf3d"],))
    except ValueError as e:
        assert "mixed-source" in str(e)
    else:
        raise AssertionError("mixed batch accepted")


# ---- trainer: per-batch loss profiles --------------------------------------

def test_trainer_applies_the_batch_source_loss_profile():
    from tests.test_g7_lift import _g7_batch as _base_batch
    from tests.test_origin_local_sample import _build_module

    m = _build_module(split_axis_heads=True, trajectory_absolute=False)

    def _g7_batch():
        # the shared helper is the 13-tuple; append the 2D track = the GT
        # curve projected with the batch intrinsics (pixels) + all-valid flags
        b = list(_base_batch())
        assert len(b) == 13
        K, traj = b[11].float(), b[12].float()
        uvw = torch.einsum("bij,bnj->bni", K, traj)
        uv = uvw[..., :2] / uvw[..., 2:3].clamp(min=1e-3)
        b += [uv, torch.ones(uv.shape[:2], dtype=torch.bool)]
        return tuple(b)
    # profile "2d": no 3D-GT terms, projection loss on the unit anchor
    over = {"trajectory_weight": 0.0, "point_3d_weight": 0.0, "origin_weight": 0.0,
            "origin_map_weight": 0.0, "vae_weight": 0.0, "motion_type_weight": 0.0,
            "trajectory_proj_weight": 0.5, "trajectory_proj_normalized": True,
            "trajectory_proj_anchor": "unit", "trajectory_scale_source": "unit"}
    m.loss_profiles = {"2d": over}
    m.source_profiles = {"hoi4d": "2d", "sf3d": "default"}
    import dataclasses
    from torch import nn
    m._profile_params = {"2d": dataclasses.replace(m.loss_params, **over)}
    geo, proj = m._build_loss_modules(m._profile_params["2d"])
    m._profile_modules = nn.ModuleDict({"2d_geometric": geo, "2d_projection": proj})

    logged = {}
    m.log = lambda name, value, **kw: logged.__setitem__(name, float(value) if torch.is_tensor(value) else value)
    base_lp, base_geo, base_proj = m.loss_params, m.geometric_loss, m.traj_projection_loss

    torch.manual_seed(0)
    loss_2d = m._common_step(_g7_batch() + (["hoi4d", "hoi4d"],), 0, "train")
    assert torch.isfinite(loss_2d)
    assert logged.get("train/L_traj_proj", 0.0) > 0.0        # the 2D data term fired
    # (the raw 3D terms are still LOGGED at weight 0 — the trainer logs the
    # unweighted value — so the profile is witnessed by the projection term)
    assert "train/hoi4d/loss_total" in logged
    # the swap is undone after the step
    assert m.loss_params is base_lp and m.geometric_loss is base_geo and m.traj_projection_loss is base_proj

    logged.clear()
    torch.manual_seed(0)
    loss_3d = m._common_step(_g7_batch() + (["sf3d", "sf3d"],), 0, "train")
    assert torch.isfinite(loss_3d)
    assert logged.get("train/L_trajectory", 0.0) > 0.0      # 3D recipe on the sf3d batch
    assert logged.get("train/L_traj_proj", 0.0) == 0.0      # projection off in the default profile
    assert "train/sf3d/loss_total" in logged

    logged.clear()
    torch.manual_seed(0)
    loss_plain = m._common_step(_g7_batch(), 0, "train")    # untagged batch = default profile
    assert torch.isfinite(loss_plain) and not any(k.count("/") == 2 for k in logged)
