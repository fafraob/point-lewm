"""Cache loading, splits, pair construction, conditioning assembly.

The caches (one per env, ~200 floats/frame) fit in GPU memory, so there is no
DataLoader: tensors are uploaded once and indexed with random permutations.

Splits -- two exclusions, both at **episode** granularity:

``eval`` episodes  every episode named by any ``starts_*.json`` window file used
                   for closed-loop evaluation is removed from training entirely,
                   so the closed-loop comparison runs on trajectories the goal
                   model has never seen. (The frozen encoder did see them; that
                   is identical for the baseline and every variant, so it cannot
                   confound the comparison.)
``val`` episodes   a fixed 10 % slice (``ep % 10 == 9``) held out for the
                   offline metrics and model selection.

Pair construction (``--use-z`` conditioning)
--------------------------------------------
The z-conditioned variants need ``(z_now, goal pose) -> z_goal`` triples with
the temporal gaps the planner actually faces. The goal is pinned at
``start + goal_offset_steps`` (25 env steps) but the planner replans every 25
env steps too, so later replans face smaller -- and once the objects have moved
past, effectively negative -- gaps. Pairs are therefore drawn over a spread of
offsets of both signs (default +-5..+-25). Offset 0 is excluded: there the
answer *is* ``z_now``, an identity shortcut that lets the head ignore the goal.
Pairs are matched on ``step`` values, never on row arithmetic (the cache stride
phase differs per episode).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_cache(path):
    with np.load(path, allow_pickle=True) as f:
        d = {k: f[k] for k in f.files}
    meta = {k[6:]: d.pop(k).item() for k in list(d) if k.startswith("_meta_")}
    n = len(d["z"])
    print(f"[data] {path}: {n} frames, keys={sorted(d)}, meta={meta}")
    return d, meta


def eval_episode_set(eval_dir, pattern="starts_*.json"):
    """Every episode index referenced by the closed-loop window files."""
    eps = set()
    files = sorted(Path(eval_dir).glob(pattern)) if eval_dir and Path(eval_dir).is_dir() else []
    for p in files:
        with open(p) as f:
            eps.update(json.load(f)["episodes"])
    print(f"[data] excluding {len(eps)} eval episodes from {len(files)} window files")
    return eps


def make_splits(ep, eval_eps, val_mod=10, val_rem=9):
    """Boolean masks ``(train, val)`` over cache rows; eval episodes in neither."""
    ep = np.asarray(ep)
    is_eval = np.isin(ep, np.fromiter(eval_eps, dtype=ep.dtype, count=len(eval_eps))) \
        if eval_eps else np.zeros(len(ep), bool)
    is_val = (ep % val_mod) == val_rem
    train = ~is_eval & ~is_val
    val = ~is_eval & is_val
    print(f"[data] rows: train {train.sum()}  val {val.sum()}  excluded {is_eval.sum()}")
    return train, val


def build_pairs(ep, step, offset):
    """Row-index pairs ``(now, goal)`` with ``step[goal] - step[now] == offset``.

    Vectorized: a hash of ``ep * BIG + step`` maps a wanted ``(ep, step +
    offset)`` to its row.
    """
    ep = np.asarray(ep, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    key = ep * 100000 + step
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    want = ep * 100000 + step + offset
    idx = np.searchsorted(key_sorted, want)
    idx = np.clip(idx, 0, len(key_sorted) - 1)
    goal_rows = order[idx]
    ok = key[goal_rows] == want
    now_rows = np.flatnonzero(ok)
    return now_rows, goal_rows[ok]


def build_pairs_multi(ep, step, offsets, max_pairs=None, seed=0):
    """Pairs over several ``(goal - now)`` step offsets, concatenated."""
    nows, goals = [], []
    for o in offsets:
        if o == 0:
            continue
        n, g = build_pairs(ep, step, offset=o)
        nows.append(n)
        goals.append(g)
    now = np.concatenate(nows)
    goal = np.concatenate(goals)
    if max_pairs and len(now) > max_pairs:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(now), max_pairs, replace=False)
        now, goal = now[sel], goal[sel]
    print(f"[data] {len(now)} pairs over offsets {sorted(o for o in offsets if o)}")
    return now, goal


def assemble(cache, use_z, rows=None, z_now_rows=None):
    """Conditioning matrix ``[coords, rot?, z_now?]`` -> ``(N, C)`` float32.

    ``coords`` and (when present) ``rot`` are stored by the precompute already
    in the sensor frame; ``z_now`` is the latent at the *context* rows -- a
    different row from the target.
    """
    rows = slice(None) if rows is None else rows
    parts = [cache["coords"][rows]]
    if "rot" in cache:
        parts.append(cache["rot"][rows])
    if use_z:
        assert z_now_rows is not None, "--use-z conditioning needs z_now_rows"
        parts.append(cache["z"][z_now_rows])
    return np.concatenate(parts, axis=1).astype(np.float32)


def to_gpu(x, device="cuda"):
    return torch.as_tensor(np.ascontiguousarray(x), device=device)


class TensorBank:
    """GPU-resident (cond, z) tensors plus a shuffling minibatch iterator."""

    def __init__(self, cond, z, device="cuda"):
        self.cond = to_gpu(cond, device)
        self.z = to_gpu(z, device)
        self.n = len(self.z)

    def batches(self, batch_size, generator=None, shuffle=True):
        idx = (
            torch.randperm(self.n, device=self.cond.device, generator=generator)
            if shuffle
            else torch.arange(self.n, device=self.cond.device)
        )
        for i in range(0, self.n - (batch_size - 1 if shuffle else 0), batch_size):
            j = idx[i : i + batch_size]
            yield self.cond[j], self.z[j]
