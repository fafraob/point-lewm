"""stable-worldmodel ``.lance`` tables for the vendored DINO-WM trainer.

The Lance tables hold the same rows as the ``.h5`` image datasets (one row per
frame, ``episode_idx`` / ``step_idx`` in order, episodes contiguous), with
``pixels`` as JPEG blobs instead of raw uint8 and the point-cloud columns next
to them. After ``scripts/data/scanview_from_lance.py`` the pixels are the
sensor-view renders, which only exist in the tables, so DINO-WM trains from
the table directly. This module is :mod:`dinowm_swm.h5_dset` over Lance: the
same ``TrajDataset`` contract, the same window reads (``get_frames`` fetches
exactly the rows of a window with one ``take``), the same normalisation
statistics over the whole table and the same stats file, so eval.py's planner
needs no change. :func:`h5_dset.load_swm_h5_slice_train_val` dispatches here
when ``data_path`` ends in ``.lance``.
"""
import io
import json
import os
from pathlib import Path

import lance
import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .h5_dset import STATS_FILENAME, SWMH5SlicerDataset
from . import add_upstream_to_path

add_upstream_to_path()
from datasets.traj_dset import TrajDataset  # noqa: E402  (vendored dino_wm)


def _column(ds, name, dtype=np.float32):
    col = ds.to_table(columns=[name])[name]
    return np.asarray(col.to_pylist(), dtype=dtype)


def compute_norm_stats_lance(path, action_key="action", proprio_key=None, state_key=None):
    """Same contract as :func:`h5_dset.compute_norm_stats`, over a Lance table."""
    ds = lance.dataset(str(path))
    stats = {}
    for name, key in (("action", action_key), ("proprio", proprio_key), ("state", state_key)):
        if key is None:
            continue
        x = _column(ds, key, np.float64).reshape(ds.count_rows(), -1)
        x = x[~np.isnan(x).any(axis=1)]
        mean, std = x.mean(0), x.std(0)
        std = np.where(std < 1e-8, 1.0, std)
        stats[f"{name}_mean"], stats[f"{name}_std"] = mean.tolist(), std.tolist()
        stats[f"{name}_key"] = key
    stats["source"] = str(path)
    return stats


class SWMLanceTrajDataset(TrajDataset):
    """One item = one whole episode; ``get_frames`` reads a window's rows with
    one ``take``. Arguments as :class:`h5_dset.SWMH5TrajDataset`."""

    def __init__(
        self,
        data_path,
        episodes=None,
        transform=None,
        action_key="action",
        proprio_key=None,
        state_key=None,
        normalize_action=True,
        stats=None,
    ):
        self.data_path = str(data_path)
        self.transform = transform
        self.action_key, self.proprio_key, self.state_key = action_key, proprio_key, state_key
        self.normalize_action = normalize_action
        ds = lance.dataset(self.data_path)
        n_rows = ds.count_rows()
        ep_idx = _column(ds, "episode_idx", np.int64)
        step = _column(ds, "step_idx", np.int64)
        # rows must be episode-contiguous and in step order (the writer's order)
        starts = np.flatnonzero(np.r_[True, ep_idx[1:] != ep_idx[:-1]])
        self.ep_offset = starts
        self.ep_len = np.diff(np.r_[starts, n_rows])
        assert np.array_equal(np.unique(ep_idx), np.arange(len(starts))), "episodes must be 0..N-1, contiguous"
        assert np.all(step == np.concatenate([np.arange(n) for n in self.ep_len])), "rows must be in step order"
        actions = _column(ds, action_key).reshape(n_rows, -1)
        proprios = _column(ds, proprio_key).reshape(n_rows, -1) if proprio_key else np.zeros((n_rows, 1), np.float32)
        states = _column(ds, state_key).reshape(n_rows, -1) if state_key else proprios.copy()
        blob = ds.take([0], columns=["pixels"])["pixels"][0].as_py()
        self.img_hw = Image.open(io.BytesIO(blob)).size[::-1]

        self.episodes = np.arange(len(self.ep_len)) if episodes is None else np.asarray(episodes, dtype=np.int64)
        self.action_dim, self.proprio_dim, self.state_dim = actions.shape[1], proprios.shape[1], states.shape[1]

        if stats is None:
            stats = compute_norm_stats_lance(self.data_path, action_key, proprio_key, state_key)
        self.stats = stats

        def _stat(name, dim):
            if f"{name}_mean" in stats:
                m = torch.tensor(stats[f"{name}_mean"], dtype=torch.float32)
                s = torch.tensor(stats[f"{name}_std"], dtype=torch.float32)
                assert m.numel() == dim, (name, m.numel(), dim)
                return m, s
            return torch.zeros(dim), torch.ones(dim)

        self.action_mean, self.action_std = _stat("action", self.action_dim)
        self.proprio_mean, self.proprio_std = _stat("proprio", self.proprio_dim)
        self.state_mean, self.state_std = _stat("state", self.state_dim)
        if not normalize_action:
            self.action_mean, self.action_std = torch.zeros(self.action_dim), torch.ones(self.action_dim)
            self.proprio_mean, self.proprio_std = torch.zeros(self.proprio_dim), torch.ones(self.proprio_dim)
            self.state_mean, self.state_std = torch.zeros(self.state_dim), torch.ones(self.state_dim)

        # leading rows of each episode with a REAL action (terminal rows are NaN)
        bad = np.isnan(actions).any(axis=1)
        self.n_valid_actions = self.ep_len.copy()
        for e in np.unique(np.searchsorted(self.ep_offset, np.where(bad)[0], side="right") - 1):
            rows = np.where(bad[self.ep_offset[e]: self.ep_offset[e] + self.ep_len[e]])[0]
            self.n_valid_actions[e] = int(rows.min())
        actions = np.nan_to_num(actions, nan=0.0)
        self.actions = (torch.from_numpy(actions) - self.action_mean) / self.action_std
        self.proprios = (torch.from_numpy(proprios) - self.proprio_mean) / self.proprio_std
        self.states = torch.from_numpy(states)
        self._ds = None
        self._ds_pid = None

    # -- table handle: opened lazily in EACH process (DataLoader workers fork) --
    def _table(self):
        if self._ds is None or self._ds_pid != os.getpid():
            self._ds = lance.dataset(self.data_path)
            self._ds_pid = os.getpid()
        return self._ds

    # -- TrajDataset contract ----------------------------------------------
    def __len__(self):
        return len(self.episodes)

    def get_seq_length(self, idx):
        return int(self.ep_len[self.episodes[idx]])

    def get_action_length(self, idx):
        return int(self.n_valid_actions[self.episodes[idx]])

    def rows(self, idx, start, end):
        off = int(self.ep_offset[self.episodes[idx]])
        return off + np.arange(start, end)

    def get_frames(self, idx, frames):
        """``frames``: sorted episode-local indices; exactly those rows are read."""
        frames = np.asarray(frames, dtype=np.int64)
        ep = int(self.episodes[idx])
        rows = int(self.ep_offset[ep]) + frames
        blobs = self._table().take(rows.tolist(), columns=["pixels"])["pixels"].to_pylist()
        image = torch.from_numpy(
            np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs])
        )
        image = rearrange(image, "t h w c -> t c h w").float() / 255.0
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[rows]}
        return obs, self.actions[rows], self.states[rows], {"episode": ep}

    def __getitem__(self, idx):
        return self.get_frames(idx, np.arange(self.get_seq_length(idx)))

    def get_all_actions(self):
        return torch.cat([self.actions[self.rows(i, 0, self.get_seq_length(i))] for i in range(len(self))])

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_swm_lance_slice_train_val(
    transform,
    data_path,
    split_ratio=0.9,
    split_seed=42,
    normalize_action=True,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    action_key="action",
    proprio_key=None,
    state_key=None,
    stats_path=None,
    n_rollout=None,
):
    """Lance twin of :func:`h5_dset.load_swm_h5_slice_train_val` (same split
    convention, same stats file)."""
    data_path = str(data_path)
    stats = compute_norm_stats_lance(data_path, action_key, proprio_key, state_key)
    stats_path = Path(stats_path or STATS_FILENAME)
    if not stats_path.exists():
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))

    n_eps = int(_column(lance.dataset(data_path), "episode_idx", np.int64).max()) + 1
    if n_rollout:
        n_eps = min(n_eps, int(n_rollout))
    perm = torch.randperm(n_eps, generator=torch.Generator().manual_seed(int(split_seed))).numpy()
    n_train = int(split_ratio * n_eps)
    train_eps, val_eps = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    kw = dict(transform=transform, action_key=action_key, proprio_key=proprio_key, state_key=state_key,
              normalize_action=normalize_action, stats=stats)
    train_dset = SWMLanceTrajDataset(data_path, episodes=train_eps, **kw)
    val_dset = SWMLanceTrajDataset(data_path, episodes=val_eps, **kw)
    print(f"[swm_lance] {data_path}: {n_eps} episodes -> {len(train_dset)} train / {len(val_dset)} val, "
          f"action_dim {train_dset.action_dim}, proprio_dim {train_dset.proprio_dim} "
          f"({'zeros, noprop' if proprio_key is None else proprio_key})")

    num_frames = num_hist + num_pred
    datasets = {
        "train": SWMH5SlicerDataset(train_dset, num_frames, frameskip),
        "valid": SWMH5SlicerDataset(val_dset, num_frames, frameskip),
    }
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
