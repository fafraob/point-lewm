"""Unit tests for the Vox-WM (voxstats) encoder, pc_encoders/voxstats_encoder.py.

Covers the per-cell statistics themselves (known-point values, empty cells,
out-of-bounds clamping), the determinism guarantees the arm depends on
(bit-identical across batch compositions and under ambient autocast), the
LOCKSTEP with UtoniaEncoder's preprocessing (same canonical frame, same dedup,
same cell assignment), and the config mirror between the voxstats_* and
utoniawm_* train configs.

Run with:  pixi run python -m pytest tests/test_voxstats.py -q
"""

import math

import pytest
import torch

from pc_encoders.utonia_encoder import UtoniaEncoder
from pc_encoders.voxstats_encoder import FEATURE_DIM, VoxelStatsEncoder

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_encoder(**kw):
    """Encoder in an identity canonical frame on a 2x2x2 grid over [0, 1]^3
    (0.5-sized cells), so expected features can be written by hand."""
    args = dict(
        grid_dims=(2, 2, 2),
        grid_bounds=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        coord_scale=1.0,
        grid_size=1e-3,
        canonical_plane=(0.0, 0.0, 1.0, 0.0),  # z-up plane through the origin
        canonical_center=(0.0, 0.0, 1.0),      # -> R = I, t = 0
        count_norm=256,
    )
    args.update(kw)
    return VoxelStatsEncoder(**args)


def pack(coords, batch=None):
    coords = torch.as_tensor(coords, dtype=torch.float32)
    if batch is None:
        batch = torch.zeros(len(coords), dtype=torch.long)
    return {"coord": coords, "batch": torch.as_tensor(batch, dtype=torch.long), "feat": None}


# ----------------------------------------------------------------- basics --


def test_identity_frame_is_identity():
    enc = make_encoder()
    assert torch.allclose(enc.canon_R, torch.eye(3))
    assert torch.allclose(enc.canon_t, torch.zeros(3))


def test_output_shape_and_num_tokens():
    enc = make_encoder()
    assert enc.num_tokens == 8
    g = torch.Generator().manual_seed(0)
    coords = torch.rand(200, 3, generator=g)
    out = enc(pack(coords, batch=[0] * 120 + [1] * 80))
    assert out.shape == (2, 8, FEATURE_DIM)


def test_embed_dim_below_feature_width_is_rejected():
    with pytest.raises(AssertionError):
        make_encoder(embed_dim=16)


def test_raw_width_has_no_lift():
    enc = make_encoder(embed_dim=FEATURE_DIM)
    assert enc.lift is None
    assert "lift" not in enc.state_dict()


# ---------------------------------------------------------- 25 -> 192 lift --


def test_lift_is_orthonormal_and_exported():
    enc = make_encoder(embed_dim=192)
    Q = enc.lift
    assert Q.shape == (192, FEATURE_DIM)
    assert torch.allclose(Q.T @ Q, torch.eye(FEATURE_DIM), atol=1e-5)
    # persistent buffer: the weight export pins the lift a checkpoint was trained with
    assert "lift" in enc.state_dict()


def test_lift_output_shape_and_isometry():
    g = torch.Generator().manual_seed(1)
    coords = torch.rand(600, 3, generator=g)
    batch = torch.arange(600) // 200
    raw = make_encoder(embed_dim=FEATURE_DIM)(pack(coords, batch))        # (3, 8, 25)
    lifted = make_encoder(embed_dim=192)(pack(coords, batch))             # (3, 8, 192)
    assert lifted.shape == (3, 8, 192)
    # exactly the raw statistics mapped through the lift ...
    assert torch.allclose(lifted, raw @ make_encoder(embed_dim=192).lift.T, atol=1e-6)
    # ... so pairwise token distances (the planning cost) are unchanged
    d_raw = torch.cdist(raw.reshape(-1, FEATURE_DIM), raw.reshape(-1, FEATURE_DIM))
    d_lift = torch.cdist(lifted.reshape(-1, 192), lifted.reshape(-1, 192))
    assert torch.allclose(d_raw, d_lift, atol=1e-4)
    # empty cells stay exactly zero tokens
    empty = (raw.abs().sum(-1) == 0)
    assert torch.equal(lifted[empty], torch.zeros(int(empty.sum()), 192))


def test_lift_is_seeded():
    a = make_encoder(embed_dim=192, lift_seed=0).lift
    b = make_encoder(embed_dim=192, lift_seed=0).lift
    c = make_encoder(embed_dim=192, lift_seed=1).lift
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_single_point_features_exact():
    enc = make_encoder()
    p = (0.1, 0.2, 0.4)  # cell (0,0,0); o = p/0.5 - 0.5 = (-0.3, -0.1, 0.3)
    out = enc(pack([p]))[0]
    o = torch.tensor([-0.3, -0.1, 0.3])
    tok = out[0]
    assert tok[0] == 1.0                                            # occupancy
    assert torch.isclose(tok[1], torch.tensor(math.log1p(1) / math.log1p(256)))
    assert torch.allclose(tok[2:5], o, atol=1e-6)                   # mean
    assert torch.allclose(tok[5:8], torch.zeros(3), atol=1e-6)      # std
    assert torch.allclose(tok[8:11], torch.zeros(3), atol=1e-6)     # cov
    assert torch.allclose(tok[11:14], o, atol=1e-6)                 # min
    assert torch.allclose(tok[14:17], o, atol=1e-6)                 # max
    # sub-occupancy: signs (-,-,+) -> bit (0,0,1) -> flat 1
    sub = torch.zeros(8)
    sub[1] = 1.0
    assert torch.equal(tok[17:], sub)
    # every other cell is an all-zero token
    assert torch.equal(out[1:], torch.zeros(7, FEATURE_DIM))


def test_two_point_stats_exact():
    enc = make_encoder()
    # both in cell (0,0,0): o1 = (-0.4,-0.4,-0.4), o2 = (0.4, 0.4, -0.4)
    out = enc(pack([(0.05, 0.05, 0.05), (0.45, 0.45, 0.05)]))[0, 0]
    assert torch.isclose(out[1], torch.tensor(math.log1p(2) / math.log1p(256)))
    assert torch.allclose(out[2:5], torch.tensor([0.0, 0.0, -0.4]), atol=1e-6)   # mean
    assert torch.allclose(out[5:8], torch.tensor([0.4, 0.4, 0.0]), atol=1e-6)    # std
    # cov(x,y) = E[xy] - ExEy = 0.16; z is constant -> cov(x,z) = cov(y,z) = 0
    assert torch.allclose(out[8:11], torch.tensor([0.16, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[11:14], torch.tensor([-0.4, -0.4, -0.4]), atol=1e-6)
    assert torch.allclose(out[14:17], torch.tensor([0.4, 0.4, -0.4]), atol=1e-6)
    # sub-bits (-,-,-) -> 0 and (+,+,-) -> 6
    sub = torch.zeros(8)
    sub[0] = sub[6] = 1.0
    assert torch.equal(out[17:], sub)


def test_out_of_bounds_points_clamp_onto_edge_cells():
    enc = make_encoder()
    # x above hi, y below lo, z in-bounds: cell (1, 0, 1) = flat 5,
    # offsets clamp to the box faces: o = (0.5, -0.5, 0.7/0.5 - 1 - 0.5)
    out = enc(pack([(2.0, -1.0, 0.7)]))[0]
    occupied = out[:, 0].nonzero(as_tuple=True)[0]
    assert occupied.tolist() == [5]
    assert torch.allclose(out[5, 2:5], torch.tensor([0.5, -0.5, -0.1]), atol=1e-6)
    assert float(out[..., 2:17].abs().max()) <= 0.5


def test_dedup_collapses_same_voxel_points():
    enc = make_encoder()
    # second point within the 1e-3 dedup voxel of the first: count stays 1 and
    # the FIRST point in packed order is the surviving representative
    a = enc(pack([(0.1, 0.2, 0.4), (0.1002, 0.2002, 0.4002)]))
    b = enc(pack([(0.1, 0.2, 0.4)]))
    assert torch.equal(a, b)


def test_empty_middle_cloud_is_all_zero():
    enc = make_encoder()
    out = enc(pack([(0.1, 0.2, 0.4), (0.6, 0.7, 0.8)], batch=[0, 2]))
    assert out.shape[0] == 3
    assert torch.equal(out[1], torch.zeros(8, FEATURE_DIM))
    assert out[0, :, 0].sum() == 1 and out[2, :, 0].sum() == 1


# ----------------------------------------------------------- determinism --


@pytest.mark.parametrize("device", DEVICES)
def test_batch_composition_invariance_bitexact(device):
    enc = make_encoder().to(device)
    g = torch.Generator().manual_seed(1)
    a = torch.rand(500, 3, generator=g).to(device)
    b = torch.rand(300, 3, generator=g).to(device)
    alone = enc({"coord": a, "batch": torch.zeros(500, dtype=torch.long, device=device), "feat": None})
    together = enc({
        "coord": torch.cat([a, b]),
        "batch": torch.cat([torch.zeros(500, dtype=torch.long), torch.ones(300, dtype=torch.long)]).to(device),
        "feat": None,
    })
    assert torch.equal(alone[0], together[0])


def test_deterministic_and_permutation_invariant():
    enc = make_encoder()
    g = torch.Generator().manual_seed(2)
    coords = torch.rand(400, 3, generator=g)
    out1 = enc(pack(coords))
    out2 = enc(pack(coords))
    assert torch.equal(out1, out2)
    perm = torch.randperm(400, generator=g)
    out3 = enc(pack(coords[perm]))
    # permutation moves the dedup representative within a voxel and the fp
    # summation order, so exact equality is not guaranteed -- closeness is
    assert torch.allclose(out1, out3, atol=1e-3)


def test_ambient_autocast_does_not_leak():
    enc = make_encoder()
    g = torch.Generator().manual_seed(3)
    coords = torch.rand(300, 3, generator=g)
    ref = enc(pack(coords))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = enc(pack(coords))
    assert out.dtype == torch.float32
    assert torch.equal(ref, out)


# ------------------------------------------- lockstep with UtoniaEncoder --


def test_preprocessing_matches_utonia_encoder():
    """Same config -> same canonical frame, same dedup survivors, same cell
    assignment as the Utonia arm (the docstrings promise lockstep)."""
    ve = VoxelStatsEncoder()          # cube defaults, copied from UtoniaEncoder
    ue = UtoniaEncoder()              # never runs its backbone here
    assert torch.equal(ve.canon_R, ue.canon_R)
    assert torch.equal(ve.canon_t, ue.canon_t)
    assert torch.equal(ve.grid_lo, ue.grid_lo)
    assert torch.equal(ve.grid_cell, ue.grid_cell)

    g = torch.Generator().manual_seed(4)
    # raw sensor-frame coords near the cube workspace center
    coords = torch.tensor([1.27, 0.0, 0.25]) + (torch.rand(600, 3, generator=g) - 0.5)
    batch = torch.zeros(600, dtype=torch.long)
    canon_v, canon_u = ve._canonicalize(coords), ue._canonicalize(coords)
    assert torch.equal(canon_v, canon_u)
    keep_v = ve._voxel_dedup(canon_v, batch, 1)
    keep_u, _ = ue._voxel_dedup(canon_u, batch, 1)
    assert torch.equal(keep_v, keep_u)
    # occupied cells == the cells UtoniaEncoder would pool super-points into
    cells_u = ue._cell_index(canon_u[keep_u]).unique()
    occ = ve({"coord": coords, "batch": batch, "feat": None})[0, :, 0]
    assert torch.equal(occ.nonzero(as_tuple=True)[0], cells_u.sort().values)


# --------------------------------------------------- predictor integration --


def test_token_predictor_roundtrip():
    from module import TokenPredictor

    enc = make_encoder()
    g = torch.Generator().manual_seed(5)
    coords = torch.rand(4 * 100, 3, generator=g)  # B=2 x T=2 frames
    batch = torch.repeat_interleave(torch.arange(4), 100)
    emb = enc(pack(coords, batch)).view(2, 2, 8, FEATURE_DIM)
    pred = TokenPredictor(
        num_frames=2, num_tokens=8, input_dim=FEATURE_DIM,
        action_emb_dim=FEATURE_DIM, hidden_dim=FEATURE_DIM,
        depth=1, heads=2, dim_head=8, mlp_dim=32,
    )
    out = pred(emb, torch.zeros(2, 2, FEATURE_DIM))
    assert out.shape == (2, 2, 8, FEATURE_DIM)


# ------------------------------------------------------------ config sync --


PAIRS = [
    ("voxstats_cube", "utoniawm_cube"),
    ("voxstats_pusht", "utoniawm_pusht"),
    ("voxstats_reacher", "utoniawm_reacher"),
    ("voxstats_tworoom", "utoniawm_tworoom"),
]

# the one deliberate protocol difference: at lr 1e-3 the Push-T Vox-WM
# predictor diverged late in training, so its config (and the reported run)
# uses 3e-4 (see config/train/voxstats_pusht.yaml)
LR_OVERRIDES = {"voxstats_pusht": 3e-4}

# encoder keys that define the shared tokenization + preprocessing
GEOMETRY_KEYS = ("grid_bounds", "coord_scale", "grid_size",
                 "canonical_plane", "canonical_center")


@pytest.mark.parametrize("vox,uto", PAIRS)
def test_configs_mirror_utoniawm(vox, uto):
    """The voxstats_* configs must mirror their utoniawm_* siblings on
    everything except what a token holds: same token grid and canonical frame,
    same training protocol (epochs, per-rank batch, lr, horizon). The only
    exception is the lr listed in LR_OVERRIDES."""
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    with hydra.initialize(config_path="../config/train", version_base=None):
        cv = hydra.compose(config_name=vox)
        cu = hydra.compose(config_name=uto)

    for k in ("grid_x", "grid_y", "grid_z", "history_size", "num_preds", "seed"):
        assert cv[k] == cu[k], (k, cv[k], cu[k])
    for k in GEOMETRY_KEYS:
        assert OmegaConf.to_container(cv.model.encoder, resolve=True)[k] == \
               OmegaConf.to_container(cu.model.encoder, resolve=True)[k], k
    assert cv.trainer.max_epochs == cu.trainer.max_epochs
    assert cv.loader.batch_size == cu.loader.batch_size
    opt_v = OmegaConf.to_container(cv.optimizer, resolve=True)
    opt_u = OmegaConf.to_container(cu.optimizer, resolve=True)
    assert opt_v.pop("lr") == LR_OVERRIDES.get(vox, opt_u["lr"])
    opt_u.pop("lr")
    assert opt_v == opt_u
    assert cv.loss == cu.loss == {}
    # predictor identical up to the token width
    pv = OmegaConf.to_container(cv.model.predictor, resolve=True)
    pu = OmegaConf.to_container(cu.model.predictor, resolve=True)
    for k in ("depth", "heads", "dim_head", "mlp_dim", "dropout", "emb_dropout", "num_tokens"):
        assert pv[k] == pu[k], k
    # same token width as utoniawm (the 25 statistics are lifted to it)
    assert cv.embed_dim == cu.embed_dim == 192
    assert cv.model.encoder.lift_seed == 0
    assert cv.model.encoder._target_ == "pc_encoders.voxstats_encoder.VoxelStatsEncoder"
    # raw clouds live: the lidar data configs, never the utonia_emb caches
    assert cv.data.obs.get("emb_cache") is None
    assert cv.data.obs.source_key == "lidar" and cv.data.obs.in_channels == 3
