"""stable-worldmodel ``.h5`` datasets for the vendored DINO-WM trainer.

The swm image datasets (pusht_expert_train.h5, reacher.h5, tworoom.h5,
cube_single_expert.h5) store every frame of every episode as one flat row:
``pixels (N, 224, 224, 3) uint8`` (blosc-lz4, chunks of 100 rows), per-row
``action`` / proprio / state columns, and ``ep_offset`` / ``ep_len`` per
episode (episodes are contiguous). Upstream DINO-WM datasets are per-episode
mp4s read with decord; this module gives the trainer the same
``TrajDataset`` interface over the h5 rows.

Two things differ from upstream on purpose:

* **Window reads.** Upstream's ``TrajSlicerDataset.__getitem__`` loads the
  WHOLE episode and then slices four frames out of it. On these files that is
  a ~30 MB decode per 4-frame sample. :class:`SWMH5SlicerDataset` reads only
  the contiguous ``[start, end)`` span of the window (one or two 100-row
  chunks, ~3 ms) and subsamples in memory. Strided hyperslabs
  (``pixels[s:e:5]``) are 30x slower than a contiguous read + numpy stride
  (measured 87 vs 3 ms), so the read is always contiguous.
* **Normalization statistics** are computed over the WHOLE file (not the
  train split), population std (ddof=0). eval.py z-scores actions with an
  sklearn StandardScaler fitted on the full Lance table of the same data, so
  the planner's z-scored candidates land in exactly the training action space.
  The stats are dumped to ``swm_h5_norm_stats.json`` in the run folder.

Terminal actions. swm stores the action taken AFTER each frame, so an
episode's last row carries a NaN action (there is none). The z-scoring stats
skip those rows (as eval.py's StandardScaler does), the stored actions have
them zero-filled, and the slicer only enumerates windows whose ``num_frames x
frameskip`` action rows are all real -- i.e. the last usable window start is
one frame earlier than upstream's ``T - num_frames * frameskip``.

Proprioception: ``proprio_key: null`` (the *noprop* arm) feeds a constant
zero 1-dim "proprio" so no proprioceptive information reaches the model while
upstream's code path (proprio encoder, tiled proprio block, proprio loss) is
left untouched -- upstream divides by ``num_proprio_repeat`` and cannot run
with the proprio stream removed.
"""
import json
import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  registers the blosc filter the pixels use (in every DataLoader worker too)
import numpy as np
import torch
from einops import rearrange

from . import add_upstream_to_path

add_upstream_to_path()
from datasets.traj_dset import TrajDataset, TrajSlicerDataset  # noqa: E402  (vendored dino_wm)

STATS_FILENAME = "swm_h5_norm_stats.json"
# chunk cache large enough for the two 15 MB chunks a window can straddle
_RDCC_NBYTES = 64 << 20


def compute_norm_stats(path, action_key="action", proprio_key=None, state_key=None):
    """Per-column mean / population-std over ALL rows of the file (see module
    docstring). Zero-variance dims get std 1 so they stay zero after scaling."""
    stats = {}
    with h5py.File(path, "r") as f:
        for name, key in (("action", action_key), ("proprio", proprio_key), ("state", state_key)):
            if key is None:
                continue
            x = np.asarray(f[key][:], dtype=np.float64)
            x = x.reshape(len(x), -1)
            x = x[~np.isnan(x).any(axis=1)]
            mean, std = x.mean(0), x.std(0)  # ddof=0 == sklearn StandardScaler
            std = np.where(std < 1e-8, 1.0, std)
            stats[f"{name}_mean"], stats[f"{name}_std"] = mean.tolist(), std.tolist()
            stats[f"{name}_key"] = key
    stats["source"] = str(path)
    return stats


class SWMH5TrajDataset(TrajDataset):
    """One item = one whole episode (upstream's contract); ``get_frames`` reads
    an arbitrary sorted subset of an episode's frames with ONE contiguous read.

    Args:
        data_path: the .h5 file.
        episodes: episode indices this dataset exposes (train / val split);
            None = all.
        transform: callable on ``(T, 3, H, W)`` float images in [0, 1]
            (upstream's ``datasets.img_transforms.default_transform``).
        action_key / proprio_key / state_key: h5 columns. ``proprio_key=None``
            -> constant zero (T, 1) proprio (noprop). ``state_key=None`` ->
            the proprio is reused as state (state is only logged upstream).
        normalize_action: z-score actions and proprio with ``stats``.
        stats: dict from :func:`compute_norm_stats`; computed if None.
    """

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
        with h5py.File(self.data_path, "r") as f:
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            n_rows = f["pixels"].shape[0]
            self.img_hw = tuple(int(v) for v in f["pixels"].shape[1:3])
            actions = np.asarray(f[action_key][:], dtype=np.float32).reshape(n_rows, -1)
            if proprio_key is not None:
                proprios = np.asarray(f[proprio_key][:], dtype=np.float32).reshape(n_rows, -1)
            else:
                proprios = np.zeros((n_rows, 1), dtype=np.float32)
            if state_key is not None:
                states = np.asarray(f[state_key][:], dtype=np.float32).reshape(n_rows, -1)
            else:
                states = proprios.copy()
        assert self.ep_offset[0] == 0, "episodes must start at row 0"
        assert np.array_equal(self.ep_offset[1:], self.ep_offset[:-1] + self.ep_len[:-1]), \
            "episodes must be contiguous"
        assert self.ep_offset[-1] + self.ep_len[-1] == n_rows, (n_rows, "rows vs ep_offset/ep_len")

        self.episodes = np.arange(len(self.ep_len)) if episodes is None else np.asarray(episodes, dtype=np.int64)
        self.action_dim, self.proprio_dim, self.state_dim = actions.shape[1], proprios.shape[1], states.shape[1]

        if stats is None:
            stats = compute_norm_stats(self.data_path, action_key, proprio_key, state_key)
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

        # number of leading rows of each episode with a REAL action (the
        # terminal row's action is NaN in swm datasets, see module docstring)
        bad = np.isnan(actions).any(axis=1)
        self.n_valid_actions = self.ep_len.copy()
        for e in np.unique(np.searchsorted(self.ep_offset, np.where(bad)[0], side="right") - 1):
            rows = np.where(bad[self.ep_offset[e]: self.ep_offset[e] + self.ep_len[e]])[0]
            self.n_valid_actions[e] = int(rows.min())
        actions = np.nan_to_num(actions, nan=0.0)
        # upstream keeps normalized actions/proprios in memory; states raw
        self.actions = (torch.from_numpy(actions) - self.action_mean) / self.action_std
        self.proprios = (torch.from_numpy(proprios) - self.proprio_mean) / self.proprio_std
        self.states = torch.from_numpy(states)
        self._h5 = None
        self._h5_pid = None

    # -- h5 handle: opened lazily in EACH process (DataLoader workers fork) -----
    def _pixels(self):
        if self._h5 is None or self._h5_pid != os.getpid():
            self._h5 = h5py.File(self.data_path, "r", rdcc_nbytes=_RDCC_NBYTES, rdcc_nslots=1009)
            self._h5_pid = os.getpid()
        return self._h5["pixels"]

    # -- TrajDataset contract ----------------------------------------------
    def __len__(self):
        return len(self.episodes)

    def get_seq_length(self, idx):
        return int(self.ep_len[self.episodes[idx]])

    def get_action_length(self, idx):
        """Frames of episode ``idx`` that have a real (non-NaN) action."""
        return int(self.n_valid_actions[self.episodes[idx]])

    def rows(self, idx, start, end):
        """Global row indices of episode ``idx``'s frames ``[start, end)``."""
        off = int(self.ep_offset[self.episodes[idx]])
        return off + np.arange(start, end)

    def get_frames(self, idx, frames):
        """``frames``: sorted episode-local indices. One contiguous read."""
        frames = np.asarray(frames, dtype=np.int64)
        ep = int(self.episodes[idx])
        off = int(self.ep_offset[ep])
        lo, hi = int(frames.min()), int(frames.max()) + 1
        block = self._pixels()[off + lo: off + hi]           # (hi-lo, H, W, 3) uint8
        image = torch.from_numpy(np.ascontiguousarray(block[frames - lo]))
        image = rearrange(image, "t h w c -> t c h w").float() / 255.0
        if self.transform:
            image = self.transform(image)
        rows = off + frames
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


class SWMH5SlicerDataset(TrajSlicerDataset):
    """Upstream's slicer semantics (every window of ``num_frames`` frames at
    stride ``frameskip``, the ``frameskip`` actions after each frame
    concatenated) but reading ONLY the window's rows, and enumerating only
    windows whose action rows are all real -- see module docstring."""

    def __init__(self, dataset, num_frames, frameskip=1, process_actions="concat"):
        self.dataset = dataset
        self.num_frames = num_frames
        self.frameskip = frameskip
        span = num_frames * frameskip
        self.slices = []
        n_short = 0
        for i in range(len(self.dataset)):
            T = self.dataset.get_action_length(i)   # upstream: get_seq_length
            if T < span:
                n_short += 1
            else:
                self.slices += [(i, start, start + span) for start in range(T - span + 1)]
        if n_short:
            print(f"[swm_h5] ignored {n_short} episodes shorter than {span} action steps")
        self.slices = np.random.permutation(self.slices)   # upstream: global-RNG shuffle
        self.proprio_dim = self.dataset.proprio_dim
        self.action_dim = self.dataset.action_dim * (self.frameskip if process_actions == "concat" else 1)
        self.state_dim = self.dataset.state_dim

    def __getitem__(self, idx):
        i, start, end = (int(v) for v in self.slices[idx])
        frames = np.arange(start, end, self.frameskip)  # num_frames frames
        obs, _, state, _ = self.dataset.get_frames(i, frames)
        act = self.dataset.actions[self.dataset.rows(i, start, end)]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
        return obs, act, state


def load_swm_h5_slice_train_val(
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
    """Hydra target for ``env.dataset`` (called by upstream ``train.Trainer``
    with ``num_hist``/``num_pred``/``frameskip``). Episode-level random split
    with a fixed seed (upstream's ``random_split_traj`` convention).

    Writes the normalization stats to ``stats_path`` (default
    ``./swm_h5_norm_stats.json`` = the Hydra run folder) for eval / publishing."""
    data_path = str(data_path)
    stats = compute_norm_stats(data_path, action_key, proprio_key, state_key)
    stats_path = Path(stats_path or STATS_FILENAME)
    if not stats_path.exists():
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2))

    with h5py.File(data_path, "r") as f:
        n_eps = int(f["ep_len"].shape[0])
    if n_rollout:
        n_eps = min(n_eps, int(n_rollout))
    perm = torch.randperm(n_eps, generator=torch.Generator().manual_seed(int(split_seed))).numpy()
    n_train = int(split_ratio * n_eps)
    train_eps, val_eps = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    kw = dict(transform=transform, action_key=action_key, proprio_key=proprio_key, state_key=state_key,
              normalize_action=normalize_action, stats=stats)
    train_dset = SWMH5TrajDataset(data_path, episodes=train_eps, **kw)
    val_dset = SWMH5TrajDataset(data_path, episodes=val_eps, **kw)
    print(f"[swm_h5] {data_path}: {n_eps} episodes -> {len(train_dset)} train / {len(val_dset)} val, "
          f"action_dim {train_dset.action_dim}, proprio_dim {train_dset.proprio_dim} "
          f"({'zeros, noprop' if proprio_key is None else proprio_key})")

    num_frames = num_hist + num_pred
    datasets = {
        "train": SWMH5SlicerDataset(train_dset, num_frames, frameskip),
        "valid": SWMH5SlicerDataset(val_dset, num_frames, frameskip),
    }
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
