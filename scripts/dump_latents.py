#!/usr/bin/env python3
"""Dump a checkpoint's latents, privileged state, and predictor rollouts.

    pixi run python scripts/dump_latents.py \\
        --table data/reacher.lance \\
        --logs-root . --run . --policy reacher/point-lewm/weights.pt \\
        --out tmp/latents/reacher_point_lewm.npz

(``<logs-root>/<run>/checkpoints/<policy>``: the released checkpoints with
``--logs-root . --run .``, a training run of your own with ``--run <folder>
--policy <config name>/weights_final.pt``.)

One GPU pass per checkpoint; everything the figures need lands in one ``.npz``
so `scripts/latent_geometry.py` is pure numpy and reruns in seconds.

WHAT IS DUMPED, AND WHY EACH PIECE

``z`` -- ``projector(encoder(cloud))``, i.e. exactly the tensor
``jepa.JEPA.encode`` writes to ``info["emb"]`` and that the CEM cost scores
rollouts against. Not the pre-projector encoder output: the planner never sees
that one, so a claim about "the latent space" has to be made about this one.

Frames are taken at the TRAINING stride: every ``frameskip``-th env step, in
time order, within whole episodes. That is what makes the rollout panel
possible -- a strided i.i.d. sample of frames (what ``target2latent.precompute``
dumps) cannot be rolled out, because a rollout needs a contiguous run of model
steps plus the actions between them.

``act`` -- the frameskip-bundled macro-action of each frame, z-scored with the
statistics of the WHOLE action column. This reproduces the training pipeline
exactly: ``utils.get_column_normalizer`` fits mean/std on the full column, the
LanceDataset transform applies it to the raw per-env-step actions, and only
THEN reshapes the window to ``(num_steps, frameskip * action_dim)`` (see
``stable_worldmodel/data/formats/lance.py:_load_slice``). Normalizing after the
reshape would apply a 2-vector to a 10-vector and is the obvious way to get
this silently wrong.

``roll_pred`` / ``roll_true`` -- autoregressive predictor rollouts from
``history`` real frames under the TRUE actions, and the true latents of the
same future frames. Teacher-forced one-step error says nothing about planning;
CEM scores a latent that is ``horizon`` predictor applications deep, so that is
what gets measured. ``roll_last`` carries the last context frame's latent, the
no-change baseline every arm must beat -- a collapsed latent has a tiny
prediction error and a tiny no-change error, and only the ratio separates them.

``noise_*`` -- ``PointViTEncoder._group`` re-draws each ball's points on every
forward, so ``Enc(cloud)`` is stochastic. Encoding the same clouds twice gives
the irreducible floor that every error in the analysis is read against.

HISTORY LENGTH. Defaults to the checkpoint's own ``predictor.num_frames``,
which is NOT the same across the reacher arms (point_lewm_reacher trains with 3,
point_delta_jepa_reacher with 5 -- its history_size is derived from the LDAD horizon).
``jepa.JEPA.rollout`` hardcodes ``history_size=3`` at eval, so pass
``--history 3`` to reproduce planning conditions and the default to see the
model at its trained context length; the value used is recorded in the dump.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# LitePT's pooling projection falls over in fp16 cuBLASLt above ~2^22 points;
# the same guard target2latent.precompute sets before importing torch.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pyarrow as pa
import torch

# paths.py lives at the repo root, which is not on sys.path when this script
# is run as `python scripts/dump_latents.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LOGS_ROOT  # noqa: E402


def _ensure_repo_importable():
    """Put the repo root on sys.path.

    A checkpoint's config names ``jepa.JEPA`` and ``pc_encoders...``, which live
    at the repo root -- but running ``python scripts/dump_latents.py`` puts only
    ``scripts/`` on the path, so instantiate() cannot find them.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _ensure_swm_importable():
    """Fall back to the stable-worldmodel checkout shipped in the repo root.

    Same fallback ``scripts/viz_attention.py`` uses: the editable install
    hangs off a single ``stable_worldmodel.pth`` that a pixi sync can briefly
    invalidate, and the checkout is always there.
    """
    try:
        import stable_worldmodel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    vendored = Path(__file__).resolve().parents[1] / "stable-worldmodel"
    if (vendored / "stable_worldmodel" / "__init__.py").is_file():
        sys.path.insert(0, str(vendored))


# ------------------------------------------------------------------ lance I/O


def _to_2d(col) -> np.ndarray:
    """pyarrow column of ``fixed_size_list<float>[k]`` -> ``(n, k)`` float32."""
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate([_to_2d(c) for c in col.chunks], axis=0)
    if pa.types.is_fixed_size_list(col.type):
        k = col.type.list_size
        flat = col.flatten().to_numpy(zero_copy_only=False)
        return flat.reshape(-1, k).astype(np.float32)
    arr = col.to_numpy(zero_copy_only=False)
    if arr.dtype == object:
        arr = np.stack(arr)
    return np.asarray(arr, dtype=np.float32).reshape(len(col), -1)


def episode_index(ds):
    """(ep_id, offset, length) of every episode, in table order.

    Episodes are written contiguously; this asserts it rather than trusting it,
    because every window built below indexes by ``offset + step``.
    """
    tbl = ds.to_table(columns=["episode_idx", "step_idx"])
    ep = np.asarray(tbl.column("episode_idx").to_numpy(zero_copy_only=False), dtype=np.int64)
    step = np.asarray(tbl.column("step_idx").to_numpy(zero_copy_only=False), dtype=np.int64)

    change = np.flatnonzero(np.diff(ep)) + 1
    offsets = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([offsets, [len(ep)]]))
    ep_ids = ep[offsets]
    assert len(np.unique(ep_ids)) == len(ep_ids), "episodes are not contiguous in the table"
    for o, n in zip(offsets, lengths):
        assert step[o] == step[o : o + n].min(), "step_idx does not start at the episode offset"
    return ep_ids, offsets, lengths


# ------------------------------------------------------------------- encoding


def pack(clouds, invalid_value=-1.0, device="cuda"):
    """``(B, N, 3)`` float32 clouds -> packed ``{coord, batch, feat}``.

    Byte-identical to the training ``collate_point_cloud`` / eval
    ``LidarPlanAdapter._pack`` / ``target2latent.precompute.pack``: drop points
    whose coords are all ``invalid_value`` (missed rays), concatenate, index by
    cloud. ``feat`` is None -- these checkpoints are geometry-only.
    """
    pts = torch.as_tensor(clouds, device=device)
    keep = ~(pts == invalid_value).all(dim=-1)
    counts = keep.sum(dim=1)
    batch = torch.repeat_interleave(torch.arange(pts.shape[0], device=device), counts)
    return {"coord": pts[keep].float(), "batch": batch, "feat": None}


@torch.inference_mode()
def encode_clouds(model, lidar, batch_size, invalid_value, device):
    """``(n, N, 3)`` clouds -> ``(n, D)`` latents, in the planner's own space."""
    out = []
    for i in range(0, len(lidar), batch_size):
        packed = pack(lidar[i : i + batch_size], invalid_value, device)
        out.append(model.projector(model.encoder(packed)).float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


@torch.inference_mode()
def rollout_batch(model, z_ctx, acts, horizon, device):
    """Autoregressive rollout, mirroring ``jepa.JEPA.rollout``'s inner loop.

    Args:
        z_ctx: (B, HS, D) true latents of the context frames.
        acts:  (B, HS + horizon - 1, A) macro-actions from the first context
            frame onward. ``acts[:, j]`` is the action taken AT frame j (it
            carries frame j to frame j+1), so predicting ``horizon`` frames past
            the context needs ``HS + horizon - 1`` of them.
        horizon: number of frames to predict.

    Returns:
        (B, horizon, D) predicted latents.
    """
    hs = z_ctx.shape[1]
    emb = torch.as_tensor(z_ctx, device=device)
    act = torch.as_tensor(acts, device=device)

    preds = []
    for k in range(horizon):
        act_emb = model.action_encoder(act[:, : hs + k])
        pred = model.predict(emb[:, -hs:], act_emb[:, -hs:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        preds.append(pred)
    return torch.cat(preds, dim=1).float().cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", required=True, help="path to the .lance table")
    p.add_argument("--run", required=True, help="run folder under --logs-root")
    p.add_argument("--policy", required=True,
                   help="<name>/weights_final.pt (or weights_epoch_N.pt), relative to <run>/checkpoints/")
    p.add_argument("--logs-root", default=None, help="default: paths.LOGS_ROOT ($PLWM_LOGS_ROOT)")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default=None, help="label for this arm in the figures (default: --policy's folder)")
    p.add_argument("--episodes", type=int, default=600, help="episodes to sample")
    p.add_argument("--frameskip", type=int, default=5, help="MUST match the training data config")
    p.add_argument("--history", type=int, default=0, help="rollout context frames (0 = predictor.num_frames)")
    p.add_argument("--horizon", type=int, default=10, help="rollout steps to predict")
    p.add_argument("--roll-starts", type=int, default=6, help="rollout start points per episode")
    p.add_argument(
        "--state-cols",
        default="qpos,qvel,finger_pos,target_pos",
        help="privileged columns to carry along as probe targets",
    )
    p.add_argument("--batch-size", type=int, default=64, help="clouds per encoder forward")
    p.add_argument("--fetch-episodes", type=int, default=48, help="episodes per lance fetch (memory knob)")
    p.add_argument("--invalid-value", type=float, default=-1.0)
    p.add_argument("--noise-probe", type=int, default=2048, help="frames re-encoded to measure the sampling floor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    _ensure_repo_importable()
    _ensure_swm_importable()
    logs_root = args.logs_root or LOGS_ROOT
    name = args.name or args.policy.split("/")[0]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    import lance
    import stable_worldmodel as swm

    ds = lance.dataset(args.table)
    ep_ids, offsets, lengths = episode_index(ds)
    print(f"[dump] {args.table}: {ds.count_rows()} rows, {len(ep_ids)} episodes")

    # -- episode sample. Sorted so the lance fetches stay roughly sequential.
    n_pick = min(args.episodes, len(ep_ids))
    pick = np.sort(rng.choice(len(ep_ids), size=n_pick, replace=False))
    fs = args.frameskip
    # Episodes too short to yield a usable window are dropped by the rollout
    # loop below (which knows `history`); here only degenerate ones are cut.
    pick = np.asarray([i for i in pick if lengths[i] >= 2 * fs], dtype=np.int64)
    assert len(pick), "no episode is long enough for a single model step"
    print(f"[dump] sampled {len(pick)} episodes (lengths {lengths[pick].min()}-{lengths[pick].max()})")

    # -- action normalizer: fitted on the FULL column, exactly as
    #    utils.get_column_normalizer does (NaN rows dropped first).
    act_all = _to_2d(ds.to_table(columns=["action"]).column("action"))
    finite = act_all[~np.isnan(act_all).any(axis=1)]
    act_mean = finite.mean(axis=0, keepdims=True)
    act_std = finite.std(axis=0, keepdims=True)
    act_all = np.nan_to_num((act_all - act_mean) / act_std, nan=0.0).astype(np.float32)
    print(f"[dump] action normalizer: mean {act_mean.ravel()} std {act_std.ravel()}")

    # -- model
    model = swm.wm.utils.load_pretrained(args.policy, cache_dir=str(Path(logs_root) / args.run))
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    num_frames = int(model.predictor.pos_embedding.shape[1])
    history = args.history or num_frames
    assert history <= num_frames, f"--history {history} exceeds the predictor's {num_frames} position embeddings"
    print(f"[dump] {name}: predictor.num_frames={num_frames}, rollout history={history}, horizon={args.horizon}")

    state_cols = [c.strip() for c in args.state_cols.split(",") if c.strip()]
    schema = {f.name for f in ds.schema}
    missing = [c for c in state_cols if c not in schema]
    assert not missing, f"columns not in the table: {missing}"

    # -- encode every episode's model frames, in time order --------------------
    z_parts, ep_parts, step_parts, act_parts, state_parts = [], [], [], [], {c: [] for c in state_cols}
    ep_ptr = [0]
    t0 = time.time()
    for lo in range(0, len(pick), args.fetch_episodes):
        chunk = pick[lo : lo + args.fetch_episodes]
        rows, per_ep = [], []
        for i in chunk:
            local = np.arange(0, lengths[i], fs)
            per_ep.append(len(local))
            rows.extend((offsets[i] + local).tolist())

        tbl = ds.take(rows, columns=["lidar", "step_idx", *state_cols])
        lidar = _to_2d(tbl.column("lidar")).reshape(len(rows), -1, 3)
        z_parts.append(encode_clouds(model, lidar, args.batch_size, args.invalid_value, args.device))
        step_parts.append(np.asarray(tbl.column("step_idx").to_numpy(zero_copy_only=False), dtype=np.int32))
        for c in state_cols:
            state_parts[c].append(_to_2d(tbl.column(c)))

        for i, n in zip(chunk, per_ep):
            ep_parts.append(np.full(n, ep_ids[i], dtype=np.int32))
            ep_ptr.append(ep_ptr[-1] + n)
            # macro-action of model frame t: the fs env-step actions [t*fs, (t+1)*fs).
            # The tail frame of an episode has no full bundle; it is padded with
            # zeros and excluded from every rollout by construction (a rollout
            # never reads past its last usable action).
            a = np.zeros((n, fs * act_all.shape[1]), dtype=np.float32)
            for t in range(n):
                s = offsets[i] + t * fs
                e = min(s + fs, offsets[i] + lengths[i])
                a[t, : (e - s) * act_all.shape[1]] = act_all[s:e].reshape(-1)
            act_parts.append(a)

        done = ep_ptr[-1]
        el = time.time() - t0
        print(
            f"[dump] {lo + len(chunk)}/{len(pick)} episodes  {done} frames  "
            f"{done / max(el, 1e-9):.0f} frames/s",
            flush=True,
        )

    z = np.concatenate(z_parts)
    ep = np.concatenate(ep_parts)
    step = np.concatenate(step_parts)
    act = np.concatenate(act_parts)
    ep_ptr = np.asarray(ep_ptr, dtype=np.int64)
    state = {c: np.concatenate(v) for c, v in state_parts.items()}
    print(f"[dump] encoded {len(z)} frames -> z {z.shape}")

    # -- rollouts ---------------------------------------------------------------
    # A start s needs history context frames and horizon future frames, and the
    # actions at frames [s, s + history + horizon - 1) -- all inside the episode.
    ctx_idx, roll_ep, roll_t0 = [], [], []
    span = history + args.horizon
    for j in range(len(ep_ptr) - 1):
        lo, hi = ep_ptr[j], ep_ptr[j + 1]
        t_max = (hi - lo) - span
        if t_max < 0:
            continue
        starts = np.unique(np.linspace(0, t_max, args.roll_starts).round().astype(int))
        for s in starts:
            ctx_idx.append(lo + s)
            roll_ep.append(ep[lo])
            roll_t0.append(s + history - 1)

    ctx_idx = np.asarray(ctx_idx, dtype=np.int64)
    print(f"[dump] {len(ctx_idx)} rollouts x {args.horizon} steps")

    roll_pred, roll_true, roll_last = [], [], []
    rb = 256
    for i in range(0, len(ctx_idx), rb):
        base = ctx_idx[i : i + rb]
        z_ctx = np.stack([z[b : b + history] for b in base])
        acts = np.stack([act[b : b + history + args.horizon - 1] for b in base])
        roll_pred.append(rollout_batch(model, z_ctx, acts, args.horizon, args.device))
        roll_true.append(np.stack([z[b + history : b + span] for b in base]))
        roll_last.append(z_ctx[:, -1])
    roll_pred = np.concatenate(roll_pred) if roll_pred else np.zeros((0, args.horizon, z.shape[1]), np.float32)
    roll_true = np.concatenate(roll_true) if roll_true else np.zeros_like(roll_pred)
    roll_last = np.concatenate(roll_last) if roll_last else np.zeros((0, z.shape[1]), np.float32)

    # -- encoder sampling-noise floor ------------------------------------------
    noise = {}
    if args.noise_probe:
        probe_rows = np.sort(rng.choice(ds.count_rows(), size=min(args.noise_probe, ds.count_rows()), replace=False))
        tbl = ds.take(probe_rows.tolist(), columns=["lidar"])
        lidar = _to_2d(tbl.column("lidar")).reshape(len(probe_rows), -1, 3)
        z1 = encode_clouds(model, lidar, args.batch_size, args.invalid_value, args.device)
        z2 = encode_clouds(model, lidar, args.batch_size, args.invalid_value, args.device)
        # two independent draws -> Var(diff) = 2 Var(noise); halve for one draw
        noise = {
            "noise_n": np.int64(len(probe_rows)),
            "noise_mse_per_dim": np.float64(((z1 - z2) ** 2).mean() / 2.0),
            "noise_latent_var": np.float64(z1.var(axis=0).mean()),
        }
        print(f"[dump] encoder sampling noise: {noise['noise_mse_per_dim']:.3e} per dim "
              f"({noise['noise_mse_per_dim'] / max(noise['noise_latent_var'], 1e-12):.2%} of latent variance)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        z=z, ep=ep, step=step, act=act, ep_ptr=ep_ptr,
        roll_pred=roll_pred, roll_true=roll_true, roll_last=roll_last,
        roll_ep=np.asarray(roll_ep, dtype=np.int32), roll_t0=np.asarray(roll_t0, dtype=np.int32),
        act_mean=act_mean, act_std=act_std,
        **{f"state_{c}": v for c, v in state.items()},
        **noise,
    )
    payload["meta"] = np.array(json.dumps({
        "name": name, "run": args.run, "policy": args.policy, "table": args.table,
        "episodes": int(len(pick)), "frameskip": fs, "history": history,
        "predictor_num_frames": num_frames, "horizon": args.horizon,
        "state_cols": state_cols, "seed": args.seed,
    }))
    np.savez_compressed(out, **payload)
    print(f"[dump] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
