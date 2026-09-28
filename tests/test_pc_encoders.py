"""Unit tests for the point-cloud encoder stack (sampling, collate, PointViT).

Run with:  pixi run python -m pytest tests/ -q
CUDA-only checks are skipped automatically on CPU-only machines.
"""

import pytest
import torch

from pc_encoders.collate import collate_point_cloud
from pc_encoders.pointvit_encoder import PointViTEncoder
from pc_encoders.sampling import fps_index

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_packed(sizes, device, channels=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    coord = torch.rand(sum(sizes), channels, generator=g).to(device)
    batch = torch.repeat_interleave(
        torch.arange(len(sizes)), torch.tensor(sizes)
    ).to(device)
    return coord, batch


# ---------------------------------------------------------------- sampling --


@pytest.mark.parametrize("device", DEVICES)
def test_fps_exact_counts(device):
    sizes = [7000, 2048, 2047, 2049, 100, 31, 1, 4093]
    k = 2048
    coord, batch = make_packed(sizes, device)
    idx = fps_index(coord, batch, k)
    got = torch.bincount(batch[idx], minlength=len(sizes)).tolist()
    assert got == [min(s, k) for s in sizes]
    # grouped by cloud, no duplicate rows
    assert (batch[idx].diff() >= 0).all()
    assert idx.unique().numel() == idx.numel()


@pytest.mark.parametrize("device", DEVICES)
def test_fps_deterministic_and_seeded_at_first_row(device):
    sizes = [500, 300, 700]
    coord, batch = make_packed(sizes, device)
    a = fps_index(coord, batch, 128)
    b = fps_index(coord, batch, 128)
    assert torch.equal(a, b)
    # each cloud's FPS starts at its first packed row (random_start=False)
    firsts = torch.tensor([0, 500, 800], device=device)
    starts = a[torch.searchsorted(batch[a].contiguous(), torch.arange(3, device=device))]
    assert torch.equal(starts, firsts)


@pytest.mark.parametrize("device", DEVICES)
def test_fps_noop_below_budget(device):
    sizes = [10, 20, 30]
    coord, batch = make_packed(sizes, device)
    idx = fps_index(coord, batch, 64)
    assert torch.equal(idx, torch.arange(60, device=device))


def test_fps_empty_batch():
    coord = torch.empty(0, 3)
    batch = torch.empty(0, dtype=torch.long)
    assert fps_index(coord, batch, 8).numel() == 0


# ----------------------------------------------------------------- collate --


def test_collate_packs_and_drops_sentinels():
    T, N = 2, 50
    samples = []
    for _ in range(3):
        pts = torch.rand(T, N, 3, dtype=torch.float64)
        pts[:, -5:, :] = -1.0  # 5 sentinel points per frame
        samples.append({"points": pts, "action": torch.zeros(T, 2)})
    out = collate_point_cloud(samples, point_key="points", drop_value=-1.0)
    packed = out["points"]
    assert packed["coord"].dtype == torch.float32
    assert packed["coord"].shape == (3 * T * (N - 5), 3)
    assert packed["feat"] is None
    assert packed["batch"].max().item() == 3 * T - 1
    assert out["action"].shape == (3, T, 2)


def test_collate_feat_rows_stay_aligned():
    T, N = 1, 30
    pts = torch.rand(2, T, N, 3, dtype=torch.float64)
    rgb = torch.rand(2, T, N, 3, dtype=torch.float64)
    pts[0, 0, 0] = -1.0  # sentinel in sample 0 only
    samples = [{"points": pts[i], "points_rgb": rgb[i]} for i in range(2)]
    out = collate_point_cloud(
        samples, point_key="points", feat_key="points_rgb", drop_value=-1.0
    )
    packed = out["points"]
    assert packed["coord"].shape == packed["feat"].shape == (2 * N - 1, 3)
    # the surviving rows of sample 0 line up with their colors
    assert torch.allclose(packed["feat"][: N - 1], rgb[0, 0, 1:].float())


# ---------------------------------------------------------------- PointViT --


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_forward(device):
    sizes = [800, 300, 1500]
    coord, batch = make_packed(sizes, device)
    enc = PointViTEncoder(
        embed_dim=32, num_tokens=64, group_size=16, group_radius=0.2
    ).to(device)
    out = enc({"coord": coord, "batch": batch, "feat": None})
    assert out.shape == (3, 32)
    assert out.isfinite().all()
    # FPS is deterministic; the ball-query subsample is stochastic, so
    # forwards only repeat under the same torch seed
    torch.manual_seed(0)
    a = enc({"coord": coord, "batch": batch, "feat": None})
    torch.manual_seed(0)
    b = enc({"coord": coord, "batch": batch, "feat": None})
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_small_cloud_center_padding(device):
    # a cloud with fewer points than num_tokens gets its centers repeated
    sizes = [40, 500]  # 40 < num_tokens=64
    coord, batch = make_packed(sizes, device)
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=64, group_size=16, group_radius=0.3
    ).to(device)
    center_idx, nbr_idx = enc._sample_and_group(coord, batch, 2)
    assert center_idx.shape == (2, 64)
    assert nbr_idx.shape == (2, 64, 16)
    # cloud 0's centers are its own rows, cyclically repeated
    assert center_idx[0].unique().numel() == 40
    assert (center_idx[0] < 40).all() and (center_idx[1] >= 40).all()
    # group members never cross cloud boundaries
    assert (nbr_idx[0] < 40).all() and (nbr_idx[1] >= 40).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_group_members_within_radius(device):
    sizes = [600, 900]
    coord, batch = make_packed(sizes, device)
    r = 0.15
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=8, group_radius=r
    ).to(device)
    center_idx, nbr_idx = enc._sample_and_group(coord, batch, 2)
    centers = coord[center_idx.reshape(-1)].view(2, 32, 1, 3)
    members = coord[nbr_idx.reshape(-1)].view(2, 32, 8, 3)
    dist = (members - centers).norm(dim=-1)
    assert (dist <= r + 1e-6).all()


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_starved_ball_repeats_members(device):
    # one far-away isolated point: its ball holds only itself, so its group is
    # that point repeated group_size times (and the forward still works)
    coord, batch = make_packed([200], device)
    coord = torch.cat([coord, torch.full((1, 3), 10.0, device=device)])
    batch = torch.cat([batch, batch.new_zeros(1)])
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=8, group_size=4, group_radius=0.1
    ).to(device)
    center_idx, nbr_idx = enc._sample_and_group(coord, batch, 1)
    far = (center_idx[0] == 200).nonzero()
    assert far.numel() > 0  # FPS must pick the outlier
    assert (nbr_idx[0, far.squeeze(-1)] == 200).all()  # ball = itself, repeated
    out = enc({"coord": coord, "batch": batch, "feat": None})
    assert out.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_small_cloud_below_group_size(device):
    # clouds smaller than group_size are fine with the ball query (the earlier
    # kNN required >= group_size points per cloud)
    sizes = [8, 500]
    coord, batch = make_packed(sizes, device)
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=16, group_radius=0.3
    ).to(device)
    out = enc({"coord": coord, "batch": batch, "feat": None})
    assert out.shape == (2, 16)
    assert out.isfinite().all()


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_feat_path(device):
    sizes = [300, 400]
    coord, batch = make_packed(sizes, device)
    feat = torch.rand(sum(sizes), 3, device=device)
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=8, group_radius=0.2, use_feat=True
    ).to(device)
    out = enc({"coord": coord, "batch": batch, "feat": feat})
    assert out.shape == (2, 16)
    assert out.isfinite().all()


# ---------------------------------------------------------- ground removal --


def make_ground_batch(device):
    """Two clouds on the z=0 plane: cloud 0 has 60 plane + 40 elevated points,
    cloud 1 has 100 plane + 100 elevated. Elevated points sit at z >= 0.1."""
    g = torch.Generator().manual_seed(0)

    def cloud(n_plane, n_up):
        xy = torch.rand(n_plane + n_up, 2, generator=g)
        z = torch.cat([torch.zeros(n_plane), 0.1 + torch.rand(n_up, generator=g)])
        return torch.cat([xy, z.unsqueeze(1)], dim=1)

    clouds = [cloud(60, 40), cloud(100, 100)]
    coord = torch.cat(clouds).to(device)
    batch = torch.repeat_interleave(
        torch.arange(2), torch.tensor([100, 200])
    ).to(device)
    return coord, batch


@pytest.mark.parametrize("device", DEVICES)
def test_remove_ground_filters_inliers_and_keeps_feat_aligned(device):
    coord, batch = make_ground_batch(device)
    feat = coord.clone()  # per-point feature == its coord, to check alignment
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=8,
        ground_plane=(0, 0, 1, 0), ground_thresh=0.05, ground_min_points=8,
    ).to(device)
    out = enc.remove_ground({"coord": coord, "batch": batch, "feat": feat})
    assert (out["coord"][:, 2] >= 0.05).all()  # every survivor is off-plane
    assert torch.bincount(out["batch"], minlength=2).tolist() == [40, 100]
    assert torch.equal(out["coord"], out["feat"])  # rows stayed aligned


@pytest.mark.parametrize("device", DEVICES)
def test_remove_ground_non_unit_normal_is_normalized(device):
    coord, batch = make_ground_batch(device)
    kw = dict(
        embed_dim=16, num_tokens=32, group_size=8,
        ground_thresh=0.05, ground_min_points=8,
    )
    a = PointViTEncoder(ground_plane=(0, 0, 1, 0), **kw).to(device)
    b = PointViTEncoder(ground_plane=(0, 0, 7, 0), **kw).to(device)  # same plane, scaled
    data = {"coord": coord, "batch": batch, "feat": None}
    assert torch.equal(a.remove_ground(data)["coord"], b.remove_ground(data)["coord"])


@pytest.mark.parametrize("device", DEVICES)
def test_remove_ground_min_points_keeps_starved_cloud_whole(device):
    coord, batch = make_ground_batch(device)
    # cloud 0 would keep 40 < 64 -> stays whole; cloud 1 keeps 100 -> filtered
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=8,
        ground_plane=(0, 0, 1, 0), ground_thresh=0.05, ground_min_points=64,
    ).to(device)
    out = enc.remove_ground({"coord": coord, "batch": batch, "feat": None})
    assert torch.bincount(out["batch"], minlength=2).tolist() == [100, 100]
    assert torch.equal(out["coord"][:100], coord[:100])  # cloud 0 untouched


def test_remove_ground_disabled_is_noop():
    coord, batch = make_ground_batch("cpu")
    enc = PointViTEncoder(embed_dim=16, num_tokens=32, group_size=8)  # no ground_plane
    data = {"coord": coord, "batch": batch, "feat": None}
    assert enc.remove_ground(data) is data


@pytest.mark.parametrize("device", DEVICES)
def test_pointvit_forward_with_ground_removal(device):
    coord, batch = make_ground_batch(device)
    feat = torch.rand(coord.shape[0], 3, device=device)
    enc = PointViTEncoder(
        embed_dim=16, num_tokens=32, group_size=8, group_radius=0.2, use_feat=True,
        ground_plane=(0, 0, 1, 0), ground_thresh=0.05, ground_min_points=8,
    ).to(device)
    out = enc({"coord": coord, "batch": batch, "feat": feat})
    assert out.shape == (2, 16)
    assert out.isfinite().all()
