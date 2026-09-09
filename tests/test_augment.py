"""datasets/augment.py: every transform must keep the sample's coupled
geometry consistent — traj2d == project(K, traj3d), the point and track
inside the mask region, axis conventions under reflection, text swap —
and the photometric family must touch RGB only."""
import torch

from datasets.augment import (
    AugmentSpec,
    AugmentedDataset,
    augment_sample,
    choose_crop,
    crop_scale_sample,
    hflip_sample,
    must_keep_box,
    photometric,
    swap_left_right,
)

W, H, T = 640.0, 480.0, 512


def _project(K, X):
    uv = (K @ X.T).T
    return uv[:, :2] / uv[:, 2:3]


def _sample(desc="Open the left door", mtype=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    # a metric curve in front of the camera whose projection is the track
    start = torch.tensor([0.1, 0.05, 1.2])
    traj3d = start[None] + torch.cumsum(torch.rand(20, 3, generator=g) * 0.02, 0)
    traj2d = _project(K, traj3d)
    valid = torch.ones(20, dtype=torch.bool)
    point = traj2d[0] / torch.tensor([W, H])
    # mask: a blob around the first track point, in target pixels
    mask = torch.zeros(1, T, T)
    cx, cy = int(point[0] * T), int(point[1] * T)
    mask[0, max(0, cy - 40):cy + 40, max(0, cx - 60):cx + 60] = 1.0
    ys, xs = torch.nonzero(mask[0], as_tuple=True)
    # bbox in ORIGINAL pixels, as the reader computes it from the coordinate list
    bbox = torch.tensor([xs.min() * W / T, ys.min() * H / T, (xs.max() - xs.min()) * W / T, (ys.max() - ys.min()) * H / T]).float()
    img = torch.randint(0, 255, (3, T, T), dtype=torch.uint8)
    depth = torch.zeros(1, T, T)
    motion = torch.nn.functional.normalize(torch.tensor([0.2, 0.9, -0.3]), dim=0)
    return (img, depth, desc, mask, bbox, point, motion, torch.tensor(mtype), torch.tensor([W, H]),
            "f.jpg", torch.tensor([0.3, -0.2, 1.5]), K, traj3d, traj2d, valid)


def _check_projection(s):
    assert torch.allclose(_project(s[11], s[12]), s[13], atol=1e-3)


def _check_point_in_mask(s):
    u, v = s[5]
    assert 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0
    assert s[3][0, int(v * T), int(u * T)] == 1.0


def test_hflip_keeps_projection_identity_and_is_an_involution():
    s = _sample()
    f = hflip_sample(s)
    _check_projection(f)
    _check_point_in_mask(f)
    assert torch.equal(f[0], s[0].flip(-1)) and torch.equal(f[3], s[3].flip(-1))
    assert torch.allclose(f[13][:, 0], W - s[13][:, 0]) and torch.allclose(f[13][:, 1], s[13][:, 1])
    assert torch.allclose(f[11][0, 2], W - s[11][0, 2])
    assert torch.allclose(f[12][:, 0], -s[12][:, 0]) and torch.allclose(f[10][0], -s[10][0])
    ff = hflip_sample(f)
    for i in (0, 3, 4, 5, 6, 10, 11, 12, 13):
        assert torch.allclose(ff[i].float(), s[i].float(), atol=1e-4), i
    assert ff[2] == s[2]


def test_hflip_axis_convention_and_text_swap():
    rot = hflip_sample(_sample(mtype=1))
    n = torch.nn.functional.normalize(torch.tensor([0.2, 0.9, -0.3]), dim=0)
    assert torch.allclose(rot[6], torch.tensor([n[0], -n[1], -n[2]]))   # R(-Mn, th)
    trans = hflip_sample(_sample(mtype=0))
    assert torch.allclose(trans[6], torch.tensor([-n[0], n[1], n[2]]))  # direction reflects
    assert rot[2] == "Open the right door"
    assert swap_left_right("the right-hand door on the left, RIGHT? leftover") == "the left-hand door on the right, LEFT? leftover"


def test_crop_keeps_projection_identity_mask_and_point():
    s = _sample()
    x0, y0, sc = choose_crop(s, 0.6, 0.03, torch.tensor([0.4, 0.5, 0.5]))
    c = crop_scale_sample(s, x0, y0, sc)
    _check_projection(c)
    _check_point_in_mask(c)
    assert c[8].tolist() == [W, H] and torch.equal(c[12], s[12]) and torch.equal(c[10], s[10])
    # intrinsics: focal scaled by 1/s, principal point shifted then scaled
    assert torch.allclose(c[11][0, 0], s[11][0, 0] / sc) and torch.allclose(c[11][0, 2], (s[11][0, 2] - x0) / sc)
    # the whole mask survived the crop (area scales by 1/s^2, resampling tolerance)
    ratio = c[3].sum() / s[3].sum()
    assert abs(ratio - 1.0 / sc ** 2) < 0.15 / sc ** 2
    # the analytic bbox agrees with the resampled mask's extent (original px)
    ys, xs = torch.nonzero(c[3][0], as_tuple=True)
    assert abs(float(xs.min()) * W / T - float(c[4][0])) < 4.0 and abs(float(ys.min()) * H / T - float(c[4][1])) < 4.0


def test_crop_never_cuts_the_must_keep_box():
    s = _sample()
    for k in range(50):
        r = torch.rand(3, generator=torch.Generator().manual_seed(k))
        crop = choose_crop(s, 0.5, 0.03, r)
        assert crop is not None
        x0, y0, sc = crop
        bx0, by0, bx1, by1 = must_keep_box(s, 0.03)
        assert x0 - 1e-4 <= bx0 and bx1 <= x0 + sc * W + 1e-4
        assert y0 - 1e-4 <= by0 and by1 <= y0 + sc * H + 1e-4
    # an element filling the frame cannot be zoomed: no crop
    big = list(_sample()); big[4] = torch.tensor([0.0, 0.0, W - 1, H - 1])
    assert choose_crop(tuple(big), 0.5, 0.03, torch.tensor([0.1, 0.5, 0.5])) is None


def test_photometric_touches_only_rgb_and_keeps_dtype():
    s = _sample()
    spec = AugmentSpec(photometric_p=1.0, gray_p=1.0, blur_p=1.0, noise_p=1.0)
    img = photometric(s[0], spec, torch.tensor([0.0, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.5, 1.0]))
    assert img.dtype == torch.uint8 and img.shape == s[0].shape and not torch.equal(img, s[0])
    assert torch.equal(img[0], img[1])  # grayscale applied (noise skipped: it is per-channel)
    noisy = photometric(s[0], spec, torch.tensor([1.0, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 0.5, 0.0]))
    assert noisy.dtype == torch.uint8 and not torch.equal(noisy, s[0])
    assert photometric(s[0].float(), spec, torch.zeros(9)).dtype == torch.float32  # float input: untouched


def test_augment_sample_consistency_and_determinism():
    spec = AugmentSpec(crop_p=1.0, hflip_p=1.0, photometric_p=1.0)
    s = _sample()
    torch.manual_seed(3); a = augment_sample(s, spec)
    torch.manual_seed(3); b = augment_sample(s, spec)
    for i in (0, 3, 4, 5, 6, 11, 13):
        assert torch.equal(a[i], b[i]), i
    _check_projection(a)
    _check_point_in_mask(a)
    assert a[2] == "Open the right door"
    torch.manual_seed(4); c = augment_sample(s, spec)
    assert not torch.equal(a[0], c[0])
    off = augment_sample(s, AugmentSpec(enabled=False))
    assert torch.equal(off[0], s[0]) and off[2] == s[2]


def test_flip_text_skip_policy_and_wrapper():
    spec = AugmentSpec(crop_p=0.0, hflip_p=1.0, photometric_p=0.0, gray_p=0.0, blur_p=0.0, noise_p=0.0, flip_text="skip")
    s = _sample(desc="Open the left door")
    assert augment_sample(s, spec)[2] == "Open the left door" and torch.equal(augment_sample(s, spec)[0], s[0])
    s2 = _sample(desc="Open the door")
    assert torch.equal(augment_sample(s2, spec)[0], s2[0].flip(-1))

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return 3
        def __getitem__(self, i): return _sample(seed=i)
    ds = AugmentedDataset(_DS(), AugmentSpec())
    assert len(ds) == 3 and len(ds[1]) == 15
