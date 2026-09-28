"""Unit tests for the Utonia-WM arm (frozen Utonia token grid + cached tokens).

Run with:  pixi run python -m pytest tests/ -q
Everything except the last test runs WITHOUT the 550 MB Utonia checkpoint
(the backbone is lazy-loaded only by live forwards); the live-forward test
skips itself when the checkpoint or CUDA is absent.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from module import TokenPredictor
from pc_encoders.utonia_encoder import UtoniaEncoder, _REPO_ROOT
from utils import LazyMemmapColumn, attach_embedding_cache, encoder_cache_signature

CKPT = _REPO_ROOT / "checkpoints/utonia/utonia.pth"


def _fp16_step(scale):
    """One fp16 quantization step at ``scale`` -- the tolerance the fp16 cache
    imposes, and the bound within which batching may move a feature."""
    h = torch.tensor(scale, dtype=torch.float16)
    return float(torch.nextafter(h, torch.tensor(1e4, dtype=torch.float16)) - h)


def make_encoder(**kw):
    # 1152 == the pooled width -> no projection; geometry/dedup/passthrough
    # tests stay projection-agnostic. Projection-specific tests pass 192.
    kw.setdefault("embed_dim", 1152)
    return UtoniaEncoder(**kw)


# --------------------------------------------------------- canonical frame --


def test_canonical_frame_geometry():
    enc = make_encoder(coord_scale=4.0)
    plane = torch.tensor([0.628104, 0.0, -0.778129, -0.638995])
    n = plane[:3] / plane[:3].norm()
    # points ON the fitted table plane -> z == 0 in the canonical frame
    seed = torch.tensor([[1.2, 0.0, 0.15]])
    on_plane = seed - ((seed @ n) + plane[3] / plane[:3].norm()) * n
    canon = enc._canonicalize(on_plane)
    assert canon[0, 2].abs() < 1e-5
    # the workspace center maps to the xy origin, above the table
    c = enc._canonicalize(torch.tensor([[1.27, 0.0, 0.25]]))
    assert c[0, :2].abs().max() < 1e-5
    assert c[0, 2] > 0
    # rigid up to the global scale: pairwise distances scale by coord_scale
    pts = torch.randn(32, 3, generator=torch.Generator().manual_seed(0))
    d_raw = torch.cdist(pts, pts).fill_diagonal_(0)  # cdist self-distance fp32 noise
    d_can = torch.cdist(enc._canonicalize(pts), enc._canonicalize(pts)).fill_diagonal_(0)
    assert torch.allclose(d_can, 4.0 * d_raw, rtol=1e-3, atol=1e-3)  # fp32 round-off


# ------------------------------------------------------------- voxel dedup --


def test_voxel_dedup_one_point_per_voxel_and_deterministic():
    enc = make_encoder(grid_size=0.05)
    g = torch.Generator().manual_seed(0)
    coord = torch.rand(500, 3, generator=g)
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor([200, 300]))
    keep, gc = enc._voxel_dedup(coord, batch, 2)
    assert torch.equal(keep, enc._voxel_dedup(coord, batch, 2)[0])  # deterministic
    # one point per (cloud, voxel)
    key = batch[keep] * 10**9 + gc[:, 0] * 10**6 + gc[:, 1] * 10**3 + gc[:, 2]
    assert key.unique().numel() == keep.numel()
    # every dropped point shares a voxel with a kept EARLIER row (first-wins)
    assert torch.equal(keep, keep.sort().values)
    # per-cloud rebase: each cloud's grid coords start at 0
    for b in range(2):
        assert int(gc[batch[keep] == b].min()) == 0


def test_voxel_dedup_identical_points_collapse():
    enc = make_encoder(grid_size=0.01)
    coord = torch.zeros(10, 3)
    batch = torch.zeros(10, dtype=torch.long)
    keep, _ = enc._voxel_dedup(coord, batch, 1)
    assert keep.tolist() == [0]  # first row wins


# -------------------------------------------------------------- token grid --


def test_num_tokens_is_grid_product():
    assert make_encoder(grid_dims=(8, 8, 4)).num_tokens == 256
    assert make_encoder(grid_dims=(4, 3, 2)).num_tokens == 24


def test_cell_index_layout_and_bounds():
    # unit cube split into 2x2x2 cells of 0.5, coord_scale 1 so canonical
    # meters == scaled units
    enc = make_encoder(grid_dims=(2, 2, 2), grid_bounds=((0, 0, 0), (1, 1, 1)),
                       coord_scale=1.0)
    # cell centers -> flat index x*Dy*Dz + y*Dz + z
    pts, want = [], []
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                pts.append([0.25 + 0.5 * ix, 0.25 + 0.5 * iy, 0.25 + 0.5 * iz])
                want.append((ix * 2 + iy) * 2 + iz)
    got = enc._cell_index(torch.tensor(pts))
    assert got.tolist() == want
    # out-of-bounds clamps into the edge cells (never a negative/overflow index)
    far = torch.tensor([[-99.0, -99.0, -99.0], [99.0, 99.0, 99.0]])
    assert enc._cell_index(far).tolist() == [0, 7]


def test_cell_index_respects_coord_scale():
    # grid_bounds are canonical METERS; the encoder bins SCALED coords, so a
    # point at 0.9 canonical m must land in the same cell at any coord_scale
    for scale in (1.0, 4.0):
        enc = make_encoder(grid_dims=(2, 1, 1), grid_bounds=((0, 0, 0), (2, 1, 1)),
                           coord_scale=scale)
        p = torch.tensor([[0.9, 0.5, 0.5]]) * scale  # canonical -> scaled
        assert enc._cell_index(p).tolist() == [0]
        q = torch.tensor([[1.1, 0.5, 0.5]]) * scale
        assert enc._cell_index(q).tolist() == [1]


def test_tokenize_pools_per_cell_and_zeros_empty():
    enc = make_encoder(grid_dims=(2, 1, 1), grid_bounds=((0, 0, 0), (2, 1, 1)),
                       coord_scale=1.0, pool="meanmax", feature_dim=2, embed_dim=4)
    # cloud 0: two super-points in cell 0, none in cell 1
    # cloud 1: one super-point in cell 1, none in cell 0
    f = torch.tensor([[1.0, 3.0], [5.0, 1.0], [2.0, 2.0]])
    cell = torch.tensor([0, 0, 1])
    ob = torch.tensor([0, 0, 1])
    tok = enc._tokenize(f, cell, ob, 2)
    assert tok.shape == (2, 2, 4)  # (clouds, tokens, mean||max)
    assert torch.allclose(tok[0, 0], torch.tensor([3.0, 2.0, 5.0, 3.0]))  # mean||max
    assert torch.equal(tok[0, 1], torch.zeros(4))  # empty cell -> zeros
    assert torch.allclose(tok[1, 1], torch.tensor([2.0, 2.0, 2.0, 2.0]))
    assert torch.equal(tok[1, 0], torch.zeros(4))


def test_tokenize_is_permutation_invariant_and_deterministic():
    enc = make_encoder(grid_dims=(2, 2, 1), grid_bounds=((0, 0, 0), (2, 2, 1)),
                       coord_scale=1.0, feature_dim=3, embed_dim=6)
    g = torch.Generator().manual_seed(0)
    f = torch.randn(64, 3, generator=g)
    cell = torch.randint(0, 4, (64,), generator=g)
    ob = torch.randint(0, 2, (64,), generator=g).sort().values  # batch-grouped
    a = enc._tokenize(f, cell, ob, 2)
    assert torch.equal(a, enc._tokenize(f, cell, ob, 2))  # deterministic
    # pooling is order-independent within a cell: shuffle rows inside each cloud
    for b in (0, 1):
        m = ob == b
        idx = torch.arange(len(f))
        perm = idx[m][torch.randperm(int(m.sum()), generator=g)]
        idx[m] = perm
        f, cell = f[idx], cell[idx]
    assert torch.allclose(a, enc._tokenize(f, cell, ob, 2), atol=1e-6)


# ------------------------------------------------------------- passthrough --


def test_cached_token_passthrough_reshapes():
    enc = make_encoder(grid_dims=(2, 2, 1))  # 4 tokens
    emb = torch.randn(3 * 4, 1152)
    batch = torch.repeat_interleave(torch.arange(3), 4)
    out = enc({"coord": emb, "batch": batch, "feat": None})
    assert out.shape == (3, 4, 1152)
    assert torch.equal(out.reshape(12, 1152), emb)  # row order preserved


def test_cached_passthrough_rejects_wrong_token_count():
    enc = make_encoder(grid_dims=(2, 2, 1))  # expects 4 rows/cloud
    emb = torch.randn(6, 1152)
    batch = torch.repeat_interleave(torch.arange(2), 3)
    with pytest.raises(AssertionError, match="num_tokens rows per"):
        enc({"coord": emb, "batch": batch, "feat": None})


def test_embed_dim_equal_in_channels_is_rejected():
    with pytest.raises(AssertionError):
        UtoniaEncoder(embed_dim=3, in_channels=3)


def test_grid_bounds_must_be_ordered():
    with pytest.raises(AssertionError):
        UtoniaEncoder(grid_bounds=((0, 0, 0), (0, 1, 1)))


# --------------------------------------------------------- fixed projection --


def test_frozen_projection_orthonormal_and_seed_deterministic():
    a = UtoniaEncoder(embed_dim=192)  # default config: 1152 pooled -> 192
    b = UtoniaEncoder(embed_dim=192)
    assert a.frozen_proj.shape == (1152, 192)
    assert torch.equal(a.frozen_proj, b.frozen_proj)  # same seed -> same matrix
    eye = a.frozen_proj.T @ a.frozen_proj
    assert torch.allclose(eye, torch.eye(192), atol=1e-5)  # orthonormal columns
    c = UtoniaEncoder(embed_dim=192, proj_seed=1)
    assert not torch.equal(a.frozen_proj, c.frozen_proj)
    # no projection when embed_dim matches the pooled width
    assert UtoniaEncoder(embed_dim=1152).frozen_proj is None
    assert UtoniaEncoder(embed_dim=576, pool="mean").frozen_proj is None


def test_frozen_projection_is_in_state_dict_and_untrainable():
    enc = UtoniaEncoder(embed_dim=192)
    assert "frozen_proj" in enc.state_dict()  # exported with the weights
    assert sum(p.numel() for p in enc.parameters()) == 0  # still zero params
    # the token grid is config-only, never persisted (config is the source of truth)
    assert not [k for k in enc.state_dict() if k.startswith("grid_")]


def test_cached_passthrough_uses_projected_width():
    enc = UtoniaEncoder(embed_dim=192, grid_dims=(2, 1, 1))
    emb = torch.randn(2 * 2, 192)  # the cache stores POST-projection tokens
    batch = torch.repeat_interleave(torch.arange(2), 2)
    out = enc({"coord": emb, "batch": batch, "feat": None})
    assert torch.equal(out, emb.view(2, 2, 192))


# ---------------------------------------------------------- cache plumbing --


class FakeDataset:
    """The slice of LanceDataset the cache attachment touches."""

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self._cache = {}
        self._keys = ["action"]
        self.updated = False

    def _update_fetch_columns(self):
        self.updated = True


OBS = {"live_source_key": "lidar", "live_invalid_value": -1.0}


def write_cache(tmp_path, rows, dim, enc_sig, episodes_done, num_tokens=1, obs=OBS):
    path = tmp_path / "emb.npy"
    arr = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(rows, dim * num_tokens)
    )
    arr[:] = np.arange(rows, dtype=np.float16)[:, None]
    arr.flush()
    meta = {"dataset": "x", "rows": rows, "embed_dim": dim, "num_tokens": num_tokens,
            "encoder": enc_sig, "obs": obs, "episodes_done": episodes_done}
    (tmp_path / "emb.npy.meta.json").write_text(json.dumps(meta))
    return path


def test_attach_embedding_cache_serves_rows(tmp_path):
    enc_cfg = {"_target_": "pc_encoders.utonia_encoder.UtoniaEncoder",
               "embed_dim": 8, "coord_scale": 4.0}
    sig = encoder_cache_signature(dict(enc_cfg))
    ds = FakeDataset([3, 2])
    path = write_cache(tmp_path, rows=5, dim=8, enc_sig=sig, episodes_done=2,
                       num_tokens=4)
    attach_embedding_cache(ds, "utonia_emb", path, enc_cfg, repo_root=tmp_path)
    assert "utonia_emb" in ds._keys and ds.updated
    got = ds._cache["utonia_emb"][1:4]
    assert isinstance(got, np.ndarray) and got.shape == (3, 32)  # tokens flattened
    assert np.allclose(got[:, 0], [1, 2, 3])


def test_attach_embedding_cache_refuses_different_ray_drop_rule(tmp_path):
    """The missed-ray sentinel decides which points reach the frozen backbone,
    so it changes every cached feature even though it lives in the DATA config
    rather than the encoder's."""
    sig = encoder_cache_signature({"embed_dim": 8})
    path = write_cache(tmp_path, 5, 8, sig, episodes_done=1)
    # same rule -> accepted
    attach_embedding_cache(FakeDataset([5]), "utonia_emb", path, {"embed_dim": 8},
                           repo_root=tmp_path, obs_cfg=dict(OBS))
    # misses KEPT instead of dropped -> refused
    with pytest.raises(AssertionError, match="raw-observation rule"):
        attach_embedding_cache(FakeDataset([5]), "utonia_emb", path, {"embed_dim": 8},
                               repo_root=tmp_path,
                               obs_cfg={"live_source_key": "lidar",
                                        "live_invalid_value": 0.0})
    # a different source column -> refused
    with pytest.raises(AssertionError, match="raw-observation rule"):
        attach_embedding_cache(FakeDataset([5]), "utonia_emb", path, {"embed_dim": 8},
                               repo_root=tmp_path,
                               obs_cfg={"live_source_key": "lidar_other",
                                        "live_invalid_value": -1.0})


def test_attach_embedding_cache_refuses_cache_without_obs_rule(tmp_path):
    sig = encoder_cache_signature({"embed_dim": 8})
    path = write_cache(tmp_path, 5, 8, sig, episodes_done=1, obs=None)
    with pytest.raises(AssertionError, match="raw-observation rule"):
        attach_embedding_cache(FakeDataset([5]), "utonia_emb", path, {"embed_dim": 8},
                               repo_root=tmp_path, obs_cfg=dict(OBS))


def test_attach_embedding_cache_refuses_wrong_token_count(tmp_path):
    sig = encoder_cache_signature({"embed_dim": 8})
    ds = FakeDataset([5])
    path = write_cache(tmp_path, 5, 8, sig, episodes_done=1, num_tokens=4)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    meta["num_tokens"] = 8  # lies about the width the file actually has
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(AssertionError):
        attach_embedding_cache(ds, "utonia_emb", path, {"embed_dim": 8},
                               repo_root=tmp_path)


def test_attach_embedding_cache_refuses_mismatched_encoder(tmp_path):
    sig = encoder_cache_signature({"embed_dim": 8, "coord_scale": 4.0})
    ds = FakeDataset([5])
    path = write_cache(tmp_path, 5, 8, sig, episodes_done=1)
    with pytest.raises(AssertionError, match="DIFFERENT encoder config"):
        attach_embedding_cache(ds, "utonia_emb", path,
                               {"embed_dim": 8, "coord_scale": 2.0}, repo_root=tmp_path)


def test_attach_embedding_cache_refuses_incomplete(tmp_path):
    sig = encoder_cache_signature({"embed_dim": 8})
    ds = FakeDataset([3, 2])
    path = write_cache(tmp_path, 5, 8, sig, episodes_done=1)
    with pytest.raises(AssertionError, match="incomplete"):
        attach_embedding_cache(ds, "utonia_emb", path, {"embed_dim": 8},
                               repo_root=tmp_path)


def test_cache_signature_drops_only_the_target():
    from pc_encoders.utonia_encoder import FEATURE_VERSION

    a = encoder_cache_signature({"embed_dim": 8, "_target_": "x", "bf16": False})
    b = encoder_cache_signature({"embed_dim": 8, "_target_": "y", "bf16": False})
    assert a == b == {"embed_dim": 8, "bf16": False,
                      "_feature_version": FEATURE_VERSION}
    c = encoder_cache_signature({"embed_dim": 8, "bf16": True})
    assert a != c  # precision IS part of the feature signature
    d = encoder_cache_signature({"embed_dim": 8, "grid_dims": [8, 8, 4]})
    e = encoder_cache_signature({"embed_dim": 8, "grid_dims": [8, 8, 8]})
    assert d != e  # the token grid IS part of it
    f = encoder_cache_signature({"embed_dim": 8, "grid_bounds": [[0, 0, 0], [1, 1, 1]]})
    g = encoder_cache_signature({"embed_dim": 8, "grid_bounds": [[0, 0, 0], [2, 1, 1]]})
    assert f != g  # so are the bounds


def test_cache_signature_tracks_the_code_version(monkeypatch):
    """A cache built by a different encoder IMPLEMENTATION must be refused even
    when the config is byte-identical."""
    import pc_encoders.utonia_encoder as ue

    before = encoder_cache_signature({"embed_dim": 8})
    monkeypatch.setattr(ue, "FEATURE_VERSION", ue.FEATURE_VERSION + 1)
    assert encoder_cache_signature({"embed_dim": 8}) != before


def test_batch_knob_is_excluded_from_the_cache_signature():
    """Since vendored patch #2 the chunk size is a pure throughput knob, so a
    cache built with a different one must still be accepted."""
    a = encoder_cache_signature({"embed_dim": 8, "max_clouds_per_forward": 1})
    b = encoder_cache_signature({"embed_dim": 8, "max_clouds_per_forward": 64})
    assert a == b


def test_lazy_memmap_column_pickles_as_path(tmp_path):
    path = tmp_path / "col.npy"
    np.save(path, np.arange(12, dtype=np.float16).reshape(6, 2))
    col = LazyMemmapColumn(path)
    assert col[2:4].shape == (2, 2)
    clone = pickle.loads(pickle.dumps(col))
    assert clone._mm is None  # memmap not shipped through the pickle
    assert np.allclose(clone[5], [10, 11])


# --------------------------------------------------------------- multi-GPU ---


def test_world_size_resolution():
    from omegaconf import OmegaConf

    import train as tm

    def cfg(**tr):
        return OmegaConf.create({"trainer": tr, "loss": {}})

    assert tm.resolve_world_size(cfg(devices=1)) == 1
    assert tm.resolve_world_size(cfg(devices=2)) == 2
    assert tm.resolve_world_size(cfg(devices=[0, 1, 2])) == 3
    assert tm.resolve_world_size(cfg(devices=2, num_nodes=3)) == 6
    # "auto" on the CPU accelerator is a single rank whatever the box has
    assert tm.resolve_world_size(cfg(devices="auto", accelerator="cpu")) == 1


def test_sigreg_syncs_batchnorm_only_when_multi_rank():
    """SIGReg is a whole-batch statistic: a multi-rank SIGReg run must switch
    the projector's BatchNorm to SyncBatchNorm (module.SIGReg syncs itself), so
    the objective matches a single-GPU run at the same effective batch. Single
    rank and the per-sample losses (prediction-only / LDAD) are left alone."""
    from omegaconf import OmegaConf

    import train as tm

    sigreg = OmegaConf.create({"loss": {"sigreg": {"weight": 1.0}}})
    plain = OmegaConf.create({"loss": {}})
    assert tm.multi_gpu_loss_trainer_kwargs(sigreg, 2) == {"sync_batchnorm": True}
    assert tm.multi_gpu_loss_trainer_kwargs(sigreg, 1) == {}   # single rank
    assert tm.multi_gpu_loss_trainer_kwargs(plain, 8) == {}    # prediction-only / LDAD


ENVS = ("cube", "tworoom", "pusht", "reacher")
# the trained-encoder arms (point encoder trained end to end) and the
# frozen-target arms (only the predictor trains), one config per environment
TRAINED_ARMS = tuple(f"{m}_{e}" for m in ("point_lewm", "point_delta_jepa") for e in ENVS)
FROZEN_ARMS = tuple(f"{m}_{e}" for m in ("utoniawm", "voxstats") for e in ENVS)


def _protocol():
    """(effective batch, max_epochs) of every shipped arm. The effective batch
    is ranks x per-rank batch x accumulation, the number of samples behind one
    optimizer step."""
    hydra = pytest.importorskip("hydra")

    import train as tm

    out = {}
    with hydra.initialize(config_path="../config/train", version_base=None):
        for name in TRAINED_ARMS + FROZEN_ARMS:
            c = hydra.compose(config_name=name)
            out[name] = (tm.resolve_world_size(c) * c.loader.batch_size
                         * c.get("accumulate_grad_batches", 1),
                         c.trainer.max_epochs)
    return out


def test_shipped_configs_agree_on_effective_batch():
    """Every trained-encoder arm steps the optimizer on the same number of
    samples (128), however that is split over ranks, per-rank batch and
    accumulation. The frozen-target arms share one deliberately larger batch,
    4x theirs (see config/train/utoniawm_cube.yaml)."""
    prot = _protocol()
    trained = {k: prot[k][0] for k in TRAINED_ARMS}
    frozen = {k: prot[k][0] for k in FROZEN_ARMS}
    assert set(trained.values()) == {128}, trained
    assert set(frozen.values()) == {4 * 128}, frozen


def test_epoch_budget_is_shared_except_for_the_frozen_arm():
    """Within an environment the trained-encoder arms share one epoch count
    (50 on cube, the paper's schedule, and one shared count on the other
    three); the frozen-target arms deliberately run longer because only their
    predictor trains (see config/train/utoniawm_cube.yaml). Their per-epoch
    checkpoints still allow an equal-budget comparison."""
    prot = _protocol()
    per_env = {}
    for env in ENVS:
        shared = {prot[f"{m}_{env}"][1] for m in ("point_lewm", "point_delta_jepa")}
        assert len(shared) == 1, (env, shared)
        per_env[env] = shared.pop()
        for m in ("utoniawm", "voxstats"):
            assert prot[f"{m}_{env}"][1] >= per_env[env], (env, m, prot)
    assert per_env["cube"] == 50, per_env
    assert len({per_env[e] for e in ENVS if e != "cube"}) == 1, per_env


# ------------------------------------------------------- sharded precompute --


def _shard_meta(tmp_path, out, a, b, done=None, ident=None):
    """Write one shard progress sidecar, as precompute_utonia.py would."""
    m = dict(ident or _IDENT)
    m.update(ep_start=a, ep_end=b, episodes_done=b if done is None else done)
    (tmp_path / f"{out.name}.shard{a}-{b}.meta.json").write_text(json.dumps(m))


_IDENT = {"dataset": "d", "rows": 40, "embed_dim": 8, "num_tokens": 4,
          "encoder": {"embed_dim": 8}, "obs": {"live_source_key": "lidar",
                                               "live_invalid_value": -1.0}}
_KEYS = ("dataset", "rows", "embed_dim", "num_tokens", "encoder", "obs")


def _finalize(tmp_path, n_episodes=4):
    from precompute_utonia import finalize_shards

    out = tmp_path / "tok.npy"
    meta_path = tmp_path / "tok.npy.meta.json"
    finalize_shards(out, meta_path, dict(_IDENT), _KEYS, n_episodes)
    return meta_path


def test_finalize_merges_complete_shards(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 2)
    _shard_meta(tmp_path, out, 2, 4)
    meta = json.loads(_finalize(tmp_path).read_text())
    # the canonical meta is what training accepts: fully done, no shard fields
    assert meta["episodes_done"] == 4
    assert "ep_start" not in meta and "ep_end" not in meta
    assert {k: meta[k] for k in _KEYS} == dict(_IDENT)


def test_finalize_refuses_a_gap(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 2)
    _shard_meta(tmp_path, out, 3, 4)
    with pytest.raises(SystemExit, match="covered by no shard"):
        _finalize(tmp_path)
    assert not (tmp_path / "tok.npy.meta.json").exists()


def test_finalize_refuses_an_overlap(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 3)
    _shard_meta(tmp_path, out, 2, 4)
    with pytest.raises(SystemExit, match="covered twice"):
        _finalize(tmp_path)


def test_finalize_refuses_an_unfinished_shard(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 2)
    _shard_meta(tmp_path, out, 2, 4, done=3)
    with pytest.raises(SystemExit, match="unfinished"):
        _finalize(tmp_path)


def test_finalize_refuses_short_coverage(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 4)
    with pytest.raises(SystemExit, match="shards stop at episode 4"):
        _finalize(tmp_path, n_episodes=10)


def test_finalize_refuses_a_shard_from_another_config(tmp_path):
    out = tmp_path / "tok.npy"
    _shard_meta(tmp_path, out, 0, 2)
    other = dict(_IDENT, encoder={"embed_dim": 16})
    _shard_meta(tmp_path, out, 2, 4, ident=other)
    with pytest.raises(SystemExit, match="different encoder/dataset/obs config"):
        _finalize(tmp_path)


def test_finalize_refuses_when_no_shards_exist(tmp_path):
    with pytest.raises(SystemExit, match="no shard progress files"):
        _finalize(tmp_path)


# ------------------------------------------------------------- predictor ----


def make_predictor(num_frames=3, num_tokens=4, dim=8, act=6, depth=2, heads=2):
    return TokenPredictor(
        num_frames=num_frames, num_tokens=num_tokens, input_dim=dim,
        action_emb_dim=act, hidden_dim=dim, output_dim=dim, depth=depth,
        heads=heads, dim_head=4, mlp_dim=16,
    )


def test_predictor_shapes_and_shorter_context():
    p = make_predictor().eval()
    x, c = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6)
    assert p(x, c).shape == (2, 3, 4, 8)
    # a shorter window must still work (the mask slice stays frame-aligned)
    assert p(x[:, :2], c[:, :2]).shape == (2, 2, 4, 8)


def test_predictor_rejects_token_count_mismatch():
    p = make_predictor(num_tokens=4)
    with pytest.raises(AssertionError):
        p(torch.randn(1, 3, 5, 8), torch.randn(1, 3, 6))


def test_predictor_mask_is_frame_block_causal():
    """Frame t sees frames <= t (all their tokens), never t+1."""
    p = make_predictor(num_frames=3, num_tokens=4).eval()
    x, c = torch.randn(1, 3, 4, 8), torch.randn(1, 3, 6)
    base = p(x, c)
    # perturbing the LAST frame must leave earlier frames' outputs untouched
    x2 = x.clone()
    x2[:, 2] += 5.0
    out2 = p(x2, c)
    assert torch.allclose(base[:, :2], out2[:, :2], atol=1e-6)
    assert not torch.allclose(base[:, 2], out2[:, 2], atol=1e-4)
    # perturbing the FIRST frame must change every later frame (causal, not blind)
    x3 = x.clone()
    x3[:, 0] += 5.0
    out3 = p(x3, c)
    assert not torch.allclose(base[:, 2], out3[:, 2], atol=1e-4)
    # WITHIN a frame, tokens see each other: perturbing token 0 of frame 0
    # changes token 3 of frame 0
    x4 = x.clone()
    x4[:, 0, 0] += 5.0
    out4 = p(x4, c)
    assert not torch.allclose(base[:, 0, 3], out4[:, 0, 3], atol=1e-4)


def test_predictor_action_is_tiled_per_token():
    """The action embedding conditions every token of its own frame only."""
    p = make_predictor().eval()
    x, c = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 6)
    base = p(x, c)
    c2 = c.clone()
    c2[:, 2] += 3.0  # last frame's action
    out2 = p(x, c2)
    assert torch.allclose(base[:, :2], out2[:, :2], atol=1e-6)
    assert not torch.allclose(base[:, 2], out2[:, 2], atol=1e-4)


def test_predictor_positional_embeddings_distinguish_tokens():
    p = make_predictor()
    assert p.pos_token.shape == (1, 1, 4, 8)
    assert p.pos_frame.shape == (1, 3, 1, 8)
    assert "attn_mask" not in p.state_dict()  # derived, not persisted


# --------------------------------------------------- train-forward integration --


def compose_utoniawm():
    hydra = pytest.importorskip("hydra")
    with hydra.initialize(config_path="../config/train", version_base=None):
        return hydra.compose(config_name="utoniawm_cube")


def test_utoniawm_config_grid_is_consistent():
    cfg = compose_utoniawm()
    import hydra

    assert cfg.num_tokens == cfg.grid_x * cfg.grid_y * cfg.grid_z == 256
    enc = hydra.utils.instantiate(cfg.model.encoder)
    assert enc.num_tokens == cfg.num_tokens
    assert cfg.model.predictor.num_tokens == cfg.num_tokens
    # the workspace really is inside the configured bounds (see the measured
    # canonical bbox in the model config's comments)
    lo, hi = cfg.model.encoder.grid_bounds
    assert lo[0] <= -0.794 and hi[0] >= 0.890
    assert lo[1] <= -0.814 and hi[1] >= 0.814
    assert lo[2] <= 0.0 and hi[2] >= 0.491


def test_utoniawm_forward_integration():
    """Compose the real utoniawm_cube config and run the real training forward on a
    fake CACHED-token batch (no Utonia checkpoint involved)."""
    import hydra
    from omegaconf import open_dict

    import train as train_mod
    from pc_encoders.collate import collate_point_cloud

    cfg = compose_utoniawm()
    assert cfg.loss == {}  # prediction loss ONLY (frozen targets cannot collapse)

    act_dim = 10
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = act_dim

    torch.manual_seed(0)
    model = hydra.utils.instantiate(cfg.model)
    assert all(p.requires_grad for p in model.parameters())  # frozen backbone hidden

    n_frames = cfg.data.dataset.num_steps
    D, K = cfg.embed_dim, cfg.num_tokens
    samples = [
        {"points": torch.randn(n_frames, K, D),  # cached token grids
         "action": torch.randn(n_frames, act_dim)}
        for _ in range(2)
    ]
    batch = collate_point_cloud(samples, point_key="points")

    class Shim:
        def __init__(self, model):
            self.model = model
            self.logged = {}

        def log_dict(self, d, **kw):
            self.logged.update(d)

    out = train_mod.lejepa_forward(Shim(model), batch, "fit", cfg)
    assert out["emb"].shape == (2, n_frames, K, D)  # token-shaped latents
    assert set(k for k in out if k.endswith("loss")) == {"pred_loss", "loss"}
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    pred_grads = [p.grad for p in model.predictor.parameters() if p.grad is not None]
    assert pred_grads and any(g.abs().sum() > 0 for g in pred_grads)


def test_utoniawm_get_cost_with_token_latents():
    """Planning path end-to-end on cached tokens: rollout + criterion must keep
    the token axis intact and return one cost per action candidate."""
    import hydra
    from omegaconf import open_dict

    from pc_encoders.collate import collate_point_cloud

    cfg = compose_utoniawm()
    act_dim, B, S, T = 10, 2, 5, 4
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = act_dim
    torch.manual_seed(0)
    model = hydra.utils.instantiate(cfg.model).eval()
    K, D, H = cfg.num_tokens, cfg.embed_dim, cfg.history_size

    def packed(n_clouds):
        s = [{"points": torch.randn(n_clouds, K, D)}]
        return collate_point_cloud(s, point_key="points")["points"]

    info = {
        "points": packed(B * H),          # B * history_size context clouds
        "goal": packed(B),                # one goal frame per batch element
    }
    cand = torch.randn(B, S, T, act_dim)
    with torch.no_grad():
        cost = model.get_cost(info, cand)
    assert cost.shape == (B, S)
    assert torch.isfinite(cost).all()
    assert info["predicted_emb"].shape[:2] == (B, S)
    assert info["predicted_emb"].shape[-2:] == (K, D)  # token axis survived
    assert info["goal_emb"].shape == (B, 1, 1, K, D)


def test_criterion_scores_last_step_over_all_tokens():
    """The cost is the summed MSE of the LAST predicted frame's whole token
    grid -- not a slice of it (a [..., -1:, :] index would keep one token)."""
    import hydra
    from omegaconf import open_dict

    cfg = compose_utoniawm()
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = 10
    model = hydra.utils.instantiate(cfg.model)
    B, S, T, K, D = 2, 3, 4, 5, 6
    pred = torch.zeros(B, S, T, K, D)
    goal = torch.zeros(B, 1, 1, K, D)
    pred[:, :, -1] = 2.0  # every token of the scored frame is off by 2
    pred[:, :, 0] = 99.0  # earlier frames are ignored
    cost = model.criterion({"predicted_emb": pred, "goal_emb": goal})
    assert cost.shape == (B, S)
    assert torch.allclose(cost, torch.full((B, S), 4.0 * K * D))


# ------------------------------------------------------------- live forward --


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_live_forward_with_real_checkpoint():
    enc = UtoniaEncoder(embed_dim=192).cuda().eval()
    g = torch.Generator().manual_seed(0)
    # two plausible tabletop clouds in the raw sensor frame
    pts = torch.rand(2, 800, 3, generator=g) * 0.4 + torch.tensor([1.1, -0.2, 0.05])
    coord = pts.reshape(-1, 3).cuda()
    batch = torch.repeat_interleave(torch.arange(2), torch.tensor([800, 800])).cuda()
    with torch.inference_mode():
        a = enc({"coord": coord, "batch": batch, "feat": None})
        b = enc({"coord": coord, "batch": batch, "feat": None})
    assert a.shape == (2, enc.num_tokens, 192) and a.isfinite().all()
    assert torch.equal(a, b)  # deterministic (no shuffle_orders, no sampling)
    assert (a != 0).any(dim=-1).sum() < 2 * enc.num_tokens  # empty cells exist
    # features are fp16-representable: the cache round-trip must be exact
    assert torch.equal(a, a.half().float())
    # the projection is the fixed orthogonal map of the raw per-cell features
    raw = UtoniaEncoder(embed_dim=1152).cuda().eval()
    raw._backbone_slot = enc._backbone_slot  # reuse the loaded backbone
    with torch.inference_mode():
        r = raw({"coord": coord, "batch": batch, "feat": None})
    assert ((r @ enc.frozen_proj.cuda()).half().float() - a).abs().max() < 1e-3


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_live_features_are_independent_of_batch_composition(chunk):
    """A cloud's frozen features must not depend on which clouds accompany it.

    This is the regression guard for vendored patch #2: upstream's non-flash
    attention used patch_size = min(points over the batch), so a 300-point
    neighbour moved a 4000-point cloud's features by ~56% (rel L2). With the
    patch, patch_size is fixed and only fp16-level rounding remains. The small
    cloud is present to exercise the masked sub-patch path; its OWN features are
    not asserted here (see
    test_attention_patch_size_is_independent_of_batch_composition for the
    invariant that covers it, and the encoder docstring for why a sub-patch
    cloud amplifies bit-level noise).
    """
    enc = UtoniaEncoder(embed_dim=192, max_clouds_per_forward=chunk).cuda().eval()
    g = torch.Generator().manual_seed(0)
    base = torch.tensor([1.1, -0.2, 0.05])
    target = (torch.rand(4000, 3, generator=g) * 0.4 + base).cuda()
    small = (torch.rand(300, 3, generator=g) * 0.4 + base).cuda()

    def encode(clouds):
        counts = torch.tensor([len(c) for c in clouds])
        with torch.inference_mode():
            return enc({
                "coord": torch.cat(clouds),
                "batch": torch.repeat_interleave(
                    torch.arange(len(clouds)), counts).cuda(),
                "feat": None,
            })

    alone = encode([target])[0]
    ulp = _fp16_step(alone.abs().max().item())
    for got, what in [(encode([target, small])[0], "small neighbour"),
                      (encode([small, target])[1], "reversed order"),
                      (encode([target] * 8)[3], "8 same-size clouds")]:
        assert (alone - got).abs().max().item() <= ulp, what


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_sparse_convs_are_pinned_to_a_timing_independent_algorithm():
    """spconv picks its implicit-GEMM variant by BENCHMARKING once per process,
    which made features differ between runs (up to 2 fp16 cache ULPs) even
    though they were stable within a run. Only a pinned algorithm makes the
    cache-equals-live guarantee hold, and no in-process comparison can catch a
    regression here -- so assert the pin itself."""
    import spconv.pytorch as spconv
    from spconv.core import ConvAlgo

    enc = UtoniaEncoder(embed_dim=192).cuda().eval()
    model = enc._backbone(torch.device("cuda"))
    convs = [m for m in model.modules() if isinstance(m, spconv.SubMConv3d)]
    assert convs, "no sparse convs found -- did the backbone layout change?"
    assert all(m.algo == ConvAlgo.Native for m in convs)


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_live_features_ignore_ambient_autocast():
    """Trainer precision must not reach the frozen features.

    Lightning runs the train configs under `precision: bf16`, and both the
    canonicalization and the token projection are autocast-eligible matmuls, so
    without forward()'s explicit `enabled=False` a live encode under the trainer
    produced features ~7% (rel L2) away from the fp32 ones the cache holds.
    """
    enc = UtoniaEncoder(embed_dim=192).cuda().eval()
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(2000, 3, generator=g) * 0.4
           + torch.tensor([1.1, -0.2, 0.05])).cuda()
    packed = {"coord": pts, "batch": torch.zeros(2000, dtype=torch.long).cuda(),
              "feat": None}
    with torch.inference_mode():
        plain = enc(packed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ambient = enc(packed)
    assert torch.equal(plain, ambient)


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_attention_patch_size_is_independent_of_batch_composition():
    """The invariant vendored patch #2 establishes, asserted directly.

    Upstream shrank the attention patch size to the smallest cloud in the batch,
    so one tiny cloud rewrote every other cloud's attention window (56% rel L2
    here). Feature-level tolerances cannot express this cleanly for a sub-patch
    cloud -- the backbone amplifies bit-level noise ~2x per block -- so assert
    the mechanism: the patch size must always be the configured one.
    """
    from utonia.model import SerializedAttention

    enc = UtoniaEncoder(embed_dim=192, max_clouds_per_forward=8).cuda().eval()
    model = enc._backbone(torch.device("cuda"))
    seen = []
    orig = SerializedAttention.forward
    try:
        def spy(self, point):
            out = orig(self, point)
            seen.append((self.patch_size, self.patch_size_max))
            return out
        SerializedAttention.forward = spy
        g = torch.Generator().manual_seed(0)
        base = torch.tensor([1.1, -0.2, 0.05])
        big = (torch.rand(4000, 3, generator=g) * 0.4 + base).cuda()
        tiny = (torch.rand(120, 3, generator=g) * 0.4 + base).cuda()
        counts = torch.tensor([len(big), len(tiny)])
        with torch.inference_mode():
            enc({"coord": torch.cat([big, tiny]),
                 "batch": torch.repeat_interleave(torch.arange(2), counts).cuda(),
                 "feat": None})
    finally:
        SerializedAttention.forward = orig
    assert seen, "no attention blocks ran"
    assert all(k == kmax for k, kmax in seen), sorted({k for k, _ in seen})


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_live_forward_tolerates_empty_cloud():
    """A frame whose every ray missed encodes as an all-zero token grid rather
    than crashing the backbone (patch_size would be 0)."""
    enc = UtoniaEncoder(embed_dim=192).cuda().eval()
    g = torch.Generator().manual_seed(0)
    pts = (torch.rand(500, 3, generator=g) * 0.4
           + torch.tensor([1.1, -0.2, 0.05])).cuda()
    # clouds 0 and 2 have points, cloud 1 is empty
    batch = torch.cat([torch.zeros(250), torch.full((250,), 2)]).long().cuda()
    with torch.inference_mode():
        out = enc({"coord": pts, "batch": batch, "feat": None})
    assert out.shape == (3, enc.num_tokens, 192)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert (out[0] != 0).any() and (out[2] != 0).any()


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_spconv_bias_stripped():
    """Every backbone sparse conv runs bias-free with the bias re-added by our
    own hook. spconv 2.3.8's Native inference path applies a conv bias only
    when the GEMM kernel its runtime tuner picked happens to fuse it and
    silently drops it otherwise, keyed on the exact row count -- so the same
    conv applied the bias for a batch-built cache row and dropped it for the
    single-cloud live encode of the same frame (~1.1 absolute after the
    backbone's ~2x-per-block amplification; the failed --verify of the first
    full cache). Asserting the mechanism: no conv may leave bias handling to
    spconv."""
    import spconv.pytorch as spconv

    enc = UtoniaEncoder(embed_dim=192).cuda().eval()
    model = enc._backbone(torch.device("cuda"))
    convs = [m for m in model.modules() if isinstance(m, spconv.SubMConv3d)]
    assert convs, "no sparse convs found in the backbone"
    for m in convs:
        assert m.bias is None, "conv still relies on spconv's fused bias"
        assert hasattr(m, "subm_bias"), "stripped conv lost its bias buffer"
        assert m._forward_hooks, "bias re-add hook missing"


@pytest.mark.skipif(
    not (CKPT.is_file() and torch.cuda.is_available()),
    reason="needs the Utonia checkpoint and a GPU",
)
def test_features_are_independent_of_batch_composition():
    """A cloud's tokens may move by at most one fp16 cache step when encoded
    beside other (same-size) clouds -- the invariant the embedding cache is
    built on: cache rows are encoded in chunks of max_clouds_per_forward,
    closed-loop eval encodes one cloud at a time, and the two must agree.
    The spconv fused-bias bug broke exactly this (bias applied at one row
    count, dropped at another) while every single-path test stayed green."""
    enc = UtoniaEncoder(embed_dim=192, max_clouds_per_forward=8).cuda().eval()
    g = torch.Generator().manual_seed(0)
    base = torch.tensor([1.1, -0.2, 0.05])
    clouds = [(torch.rand(4000, 3, generator=g) * 0.4 + base).cuda()
              for _ in range(3)]
    counts = torch.tensor([len(c) for c in clouds])
    with torch.inference_mode():
        batched = enc({"coord": torch.cat(clouds),
                       "batch": torch.repeat_interleave(torch.arange(3), counts).cuda(),
                       "feat": None})
        single = enc({"coord": clouds[1],
                      "batch": torch.zeros(len(clouds[1]), dtype=torch.long).cuda(),
                      "feat": None})[0]
    worst = float((batched[1] - single).abs().max())
    ulp = _fp16_step(float(single.abs().max()))
    assert worst <= ulp * 1.001, (worst, ulp)
