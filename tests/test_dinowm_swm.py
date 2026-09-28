"""Unit tests for dinowm_swm/h5_dset.py -- the .h5 TrajDataset behind the
vendored DINO-WM trainer -- on a tiny synthetic file with the swm layout
(blosc-compressed pixels, contiguous episodes, NaN terminal actions).

Run with:  pixi run python -m pytest tests/test_dinowm_swm.py -q
"""
import sys

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")


def _import_upstream_pieces():
    """Import dinowm_swm.h5_dset and upstream's TrajSlicerDataset without
    leaving the vendored tree's top-level names behind in this process.

    third_party/dino_wm is written to sit first on sys.path: it has its own
    top-level ``datasets``, ``utils`` and ``train`` packages, which collide
    with the Hugging Face ``datasets`` package (imported by lightning, so it is
    loaded as soon as the repository's utils.py / train.py are) and with the
    repository's own utils.py / train.py that the other test modules import.
    The upstream import therefore runs with the Hugging Face package set aside,
    and afterwards the vendored ``datasets`` modules and the sys.path entry are
    removed again. The classes stay bound (dinowm_swm.h5_dset holds them and
    has no lazy upstream imports), and every later ``import datasets`` /
    ``import utils`` / ``import train`` resolves as it would without this
    module, whatever order pytest collects the test files in.
    """
    from dinowm_swm import DINO_WM_ROOT

    def vendored_keys():
        return [k for k in sys.modules if k == "datasets" or k.startswith("datasets.")]

    hf_datasets = {k: sys.modules.pop(k) for k in vendored_keys()}
    try:
        import dinowm_swm.h5_dset as h5_dset  # calls add_upstream_to_path()
        from datasets.traj_dset import TrajSlicerDataset  # vendored dino_wm
    finally:
        for k in vendored_keys():
            del sys.modules[k]
        sys.modules.update(hf_datasets)
        sys.path[:] = [p for p in sys.path if p != str(DINO_WM_ROOT)]
    return h5_dset, TrajSlicerDataset


_h5_dset, TrajSlicerDataset = _import_upstream_pieces()
SWMH5SlicerDataset = _h5_dset.SWMH5SlicerDataset
SWMH5TrajDataset = _h5_dset.SWMH5TrajDataset
compute_norm_stats = _h5_dset.compute_norm_stats
load_swm_h5_slice_train_val = _h5_dset.load_swm_h5_slice_train_val

EP_LEN = [30, 25, 41]


@pytest.fixture(scope="module")
def h5file(tmp_path_factory):
    import hdf5plugin
    path = tmp_path_factory.mktemp("swm") / "toy.h5"
    rng = np.random.default_rng(0)
    n = sum(EP_LEN)
    off = np.concatenate([[0], np.cumsum(EP_LEN)[:-1]])
    action = rng.normal(size=(n, 2)).astype(np.float32) * 3 + 1
    for o, l in zip(off, EP_LEN):
        action[o + l - 1] = np.nan                  # swm: no action after the last frame
    with h5py.File(path, "w") as f:
        f["ep_len"] = np.array(EP_LEN, np.int32)
        f["ep_offset"] = off.astype(np.int64)
        f["action"] = action
        f["proprio"] = rng.normal(size=(n, 4)).astype(np.float32)
        f["observation"] = rng.normal(size=(n, 6))
        f.create_dataset("pixels", data=rng.integers(0, 255, size=(n, 8, 8, 3), dtype=np.uint8),
                         chunks=(10, 8, 8, 3), **hdf5plugin.Blosc(cname="lz4", clevel=5))
    return path


def test_stats_match_standard_scaler(h5file):
    sk = pytest.importorskip("sklearn.preprocessing")
    with h5py.File(h5file) as f:
        a = f["action"][:]
    a = a[~np.isnan(a).any(1)]
    sc = sk.StandardScaler().fit(a)                  # what eval.py fits on the Lance table
    st = compute_norm_stats(h5file, proprio_key="proprio")
    assert np.allclose(st["action_mean"], sc.mean_) and np.allclose(st["action_std"], sc.scale_)


def test_terminal_nan_actions_are_excluded_and_zero_filled(h5file):
    ds = SWMH5TrajDataset(h5file, proprio_key="proprio", state_key="observation")
    assert ds.get_seq_length(0) == 30 and ds.get_action_length(0) == 29
    assert torch.isfinite(ds.actions).all()
    sl = SWMH5SlicerDataset(ds, num_frames=4, frameskip=5)
    span = 20
    # every window's action rows are real: start + span <= action_length
    for i, s, e in sl.slices:
        assert e - s == span and e <= ds.get_action_length(int(i))
    assert len(sl) == sum((l - 1) - span + 1 for l in EP_LEN)
    obs, act, state = sl[0]
    assert obs["visual"].shape == (4, 3, 8, 8) and act.shape == (4, 10) and state.shape == (4, 6)
    assert torch.isfinite(act).all()


def test_slicer_matches_upstream_semantics(h5file):
    """Same (i, start, end) -> same frames / actions / states as upstream's
    TrajSlicerDataset, which loads the whole episode and slices it."""
    ds = SWMH5TrajDataset(h5file, proprio_key="proprio", state_key="observation")
    ours = SWMH5SlicerDataset(ds, num_frames=4, frameskip=5)
    np.random.seed(0)
    theirs = TrajSlicerDataset(ds, num_frames=4, frameskip=5)   # __getitem__ via ds[i] (whole episode)
    lookup = {tuple(int(v) for v in t): k for k, t in enumerate(theirs.slices)}
    for j in range(0, len(ours), max(1, len(ours) // 7)):
        key = tuple(int(v) for v in ours.slices[j])
        o_obs, o_act, o_state = ours[j]
        t_obs, t_act, t_state = theirs[lookup[key]]
        assert torch.equal(o_obs["visual"], t_obs["visual"])
        assert torch.equal(o_obs["proprio"], t_obs["proprio"])
        assert torch.equal(o_act, t_act) and torch.equal(o_state, t_state)


def test_contiguous_read_equals_per_frame_reads(h5file):
    ds = SWMH5TrajDataset(h5file)
    frames = np.array([3, 8, 13, 18])
    obs, _, _, info = ds.get_frames(2, frames)
    with h5py.File(h5file) as f:
        off = int(f["ep_offset"][2])
        ref = np.stack([f["pixels"][off + t] for t in frames])
    ref = torch.from_numpy(ref).permute(0, 3, 1, 2).float() / 255.0
    assert torch.equal(obs["visual"], ref) and info["episode"] == 2
    # noprop: constant zero 1-dim proprio
    assert obs["proprio"].shape == (4, 1) and torch.equal(obs["proprio"], torch.zeros(4, 1))


def test_loader_split_and_dims(h5file, tmp_path):
    datasets, traj = load_swm_h5_slice_train_val(
        transform=None, data_path=h5file, split_ratio=0.67, num_hist=3, num_pred=1, frameskip=5,
        stats_path=tmp_path / "stats.json")
    assert len(traj["train"]) == 2 and len(traj["valid"]) == 1
    assert set(traj["train"].episodes) | set(traj["valid"].episodes) == {0, 1, 2}
    assert datasets["train"].action_dim == 2 * 5 and datasets["train"].proprio_dim == 1
    assert (tmp_path / "stats.json").exists()


# --- resume on a non-zero rank -------------------------------------------------
def _fake_torch(cuda_available):
    """Stand-in torch module: records the kwargs torch.load receives."""
    import types
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return "ckpt"
    fake = types.SimpleNamespace(load=load, device=torch.device,
                                 cuda=types.SimpleNamespace(is_available=lambda: cuda_available))
    return fake, calls


def test_torch_load_defaults_to_local_rank_device(monkeypatch):
    from dinowm_swm.train import _patch_torch_load_to_local_device
    monkeypatch.setenv("LOCAL_RANK", "1")
    fake, calls = _fake_torch(cuda_available=True)
    orig = _patch_torch_load_to_local_device(fake)
    assert fake.load("model_latest.pth") == "ckpt"
    assert calls[-1] == {"map_location": torch.device("cuda", 1)}
    # an explicit map_location wins
    fake.load("x.pth", map_location="cpu")
    assert calls[-1] == {"map_location": "cpu"}
    # idempotent: patching again keeps one wrapper and returns the same original
    assert _patch_torch_load_to_local_device(fake) is fake.load
    assert fake.load.__wrapped__ is orig


def test_torch_load_untouched_without_cuda(monkeypatch):
    from dinowm_swm.train import _patch_torch_load_to_local_device
    monkeypatch.setenv("LOCAL_RANK", "1")
    fake, calls = _fake_torch(cuda_available=False)
    _patch_torch_load_to_local_device(fake)
    fake.load("model_latest.pth")
    assert calls[-1] == {}

