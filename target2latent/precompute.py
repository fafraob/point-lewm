"""Cache the frozen world-model latent + sensor-frame goal pose of every frame.

The goal models regress onto ``z = projector(encoder(cloud))`` -- exactly the
tensor ``jepa.JEPA.encode`` writes to ``info["emb"]`` and that the CEM cost
scores rollouts against, so a predicted goal latent is a drop-in replacement
for ``goal_emb``. Running the PointViT encoder inside the training loop would
re-read ~120 KB of lidar per sample per epoch; instead we encode once, here,
and dump a small per-frame cache the trainer holds entirely in GPU memory.

Cached per frame: ``z`` (the regression target), ``coords`` (+ ``rot`` where
the env has rotation channels -- pusht's sensor-frame heading vector) -- the
sensor-frame goal pose from :meth:`EnvSpec.goal_pose` -- and ``ep``/``step``
for episode-level splits and pair building.

Encoder sampling noise: ``PointViTEncoder._group`` re-draws each ball's points
on every forward, so ``Enc(cloud)`` is stochastic and even a perfect model
cannot predict a single realization exactly. ``--noise-probe N`` encodes N
frames twice and reports the spread -- the irreducible floor every offline
metric downstream is measured against.

Run from the repository root (per env and encoder checkpoint; roughly half an
hour per 2 M frames on a single GPU at stride 5)::

    # a released checkpoint (checkpoints/<env>/<model>/, scripts/download_checkpoints.py)
    python -m target2latent.precompute --env cube \\
        --cache-dir . --policy cube/point-delta-jepa/weights.pt \\
        --out $PLWM_T2L_ROOT/cache/cube_point_delta_jepa_s5.npz

    # a training run of your own (experiment_logs/<run>/checkpoints/<policy>)
    python -m target2latent.precompute --env cube \\
        --run point_delta_jepa_cube --policy point_delta_jepa_cube/weights_final.pt \\
        --out $PLWM_T2L_ROOT/cache/cube_point_delta_jepa_own_s5.npz

Resumable: shards are written to ``<out>.shards/`` as they finish and merged at
the end, so an interrupted run picks up where it stopped.

Image world models (``--modality image``): the same cache can be built from
the table's ``pixels`` column (JPEG blobs) through a LeWM image checkpoint --
``z`` is then ``projector(ViT CLS)`` exactly as ``LeWM.encode`` writes it to
``info["emb"]`` and as eval.py's CEM cost scores rollouts against. Frames are
decoded and ImageNet-normalized like eval.py's ``img_transform``. The folder
checkpoints under ``<repo>/checkpoints/<name>/`` are addressed with
``--cache-dir <repo> --policy <name>`` (the ``cache_dir``/``policy`` pair the
image eval configs use)::

    python -m target2latent.precompute --env cube --modality image \\
        --cache-dir . --policy image_lewm_cube \\
        --out $PLWM_T2L_ROOT/cache/cube_image_lewm_s5.npz

The goal pose side is identical to the point-cloud caches (same
``EnvSpec.goal_pose``, same sensor-frame coordinates), so heads trained on an
image cache take the very same inputs as the point-cloud ones. The released
closed-loop evaluation (``eval_3dtarget.py``) covers the point-cloud world
models; an image cache serves the offline metrics of ``target2latent.train``.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from target2latent.envs import LOGS_ROOT, SPECS


def load_encoder(run, policy, logs_root, device="cuda", cache_dir=None, modality="pointcloud"):
    """Load the frozen world model; ``encode(x) -> (n_frames, D)``.

    Point clouds: ``projector(encoder(packed))`` is the same composition
    ``JEPA.encode`` applies; calling the two parts directly avoids
    reshape-to-(B, T, D) guessing.

    Images: ``LeWM.encode`` itself (ViT CLS token -> projector) on a
    ``(B, 1, C, H, W)`` batch, so the cached ``z`` is bit-for-bit the tensor
    the planner's cost uses.

    ``cache_dir`` names the folder whose ``checkpoints/<policy>`` holds the
    weights (the image eval configs' ``cache_dir``); otherwise the run layout
    ``<logs_root>/<run>/checkpoints/<policy>`` is used.
    """
    import stable_worldmodel as swm

    cache_dir = str(cache_dir) if cache_dir else str(Path(logs_root) / run)
    model = swm.wm.utils.load_pretrained(policy, cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)
    print(f"[precompute] loaded {policy} from {cache_dir}")
    print(f"[precompute] model={type(model).__name__} encoder={type(model.encoder).__name__} "
          f"projector={type(model.projector).__name__}")

    if modality == "image":
        @torch.inference_mode()
        def encode(frames):
            # (B, C, H, W) -> LeWM.encode wants (B, T, C, H, W); T=1 here
            return model.encode({"pixels": frames.unsqueeze(1)})["emb"][:, 0]
    else:
        @torch.inference_mode()
        def encode(packed):
            return model.projector(model.encoder(packed))

    return encode


def image_transform(img_size=224):
    """eval.py's ``img_transform``: uint8 -> float [0,1] -> ImageNet norm -> resize."""
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def decode_frames(blobs, transform, device="cuda"):
    """JPEG blobs (list of bytes) -> ``(B, C, H, W)`` float tensor on ``device``.

    ``decode_jpeg`` returns RGB uint8 CHW; the eval sees the same frame as HWC
    uint8 from the env / dataset and ``ToImage`` brings it to CHW -- identical
    input to the normalization either way.
    """
    from torchvision.io import ImageReadMode, decode_jpeg

    raw = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs]
    imgs = decode_jpeg(raw, mode=ImageReadMode.RGB)
    return torch.stack([transform(im) for im in imgs]).to(device)


def pack(clouds, invalid_value=-1.0, device="cuda"):
    """``(B, N, 3)`` float32 clouds -> packed ``{coord, batch, feat}``.

    Byte-identical to the training ``collate_point_cloud`` / eval
    ``LidarPlanAdapter._pack``: drop points whose coords are all
    ``invalid_value`` (missed rays), concatenate, index by cloud. ``feat`` is
    ``None`` -- these checkpoints are geometry-only.
    """
    pts = torch.as_tensor(clouds, device=device)
    keep = ~(pts == invalid_value).all(dim=-1)
    counts = keep.sum(dim=1)
    batch = torch.repeat_interleave(torch.arange(pts.shape[0], device=device), counts)
    return {"coord": pts[keep].float(), "batch": batch, "feat": None}


def process_batch(tbl, spec, encode, batch_size, invalid_value, device,
                  modality="pointcloud", transform=None):
    """Encode one Lance row batch -> dict of numpy arrays (one row per frame)."""
    zs = []
    if modality == "image":
        blobs = tbl.column("pixels").to_pylist()
        for i in range(0, len(blobs), batch_size):
            frames = decode_frames(blobs[i : i + batch_size], transform, device)
            zs.append(encode(frames).float().cpu())
    else:
        lidar = np.stack(tbl.column("lidar").to_numpy(zero_copy_only=False))
        lidar = lidar.reshape(len(lidar), -1, 3).astype(np.float32)
        for i in range(0, len(lidar), batch_size):
            zs.append(encode(pack(lidar[i : i + batch_size], invalid_value, device)).float().cpu())
    z = torch.cat(zs).numpy().astype(np.float32)

    raw = {c: np.stack(tbl.column(c).to_numpy(zero_copy_only=False)) for c in spec.columns}
    coords, rot = spec.goal_pose(raw)
    out = {
        "z": z,
        "coords": coords,
        "ep": np.asarray(tbl.column("episode_idx").to_numpy(zero_copy_only=False),
                         dtype=np.int32),
        "step": np.asarray(tbl.column("step_idx").to_numpy(zero_copy_only=False),
                           dtype=np.int32),
    }
    if rot is not None:
        out["rot"] = rot
    return out


def noise_probe(ds, encode, rows, batch_size, invalid_value, device):
    """Encode the same clouds twice; report the encoder's own sampling spread."""
    tbl = ds.take(rows, columns=["lidar"])
    lidar = np.stack(tbl.column("lidar").to_numpy(zero_copy_only=False))
    lidar = lidar.reshape(len(lidar), -1, 3).astype(np.float32)
    out = []
    for _ in range(2):
        zs = [
            encode(pack(lidar[i : i + batch_size], invalid_value, device)).float().cpu()
            for i in range(0, len(lidar), batch_size)
        ]
        out.append(torch.cat(zs).numpy())
    z1, z2 = out
    # two independent draws -> Var(diff) = 2 Var(noise); halve to get one draw
    noise_mse = float(((z1 - z2) ** 2).mean() / 2.0)
    across = float(z1.var(axis=0).mean())
    return {
        "n": len(rows),
        "noise_mse_per_dim": noise_mse,
        "latent_var_per_dim": across,
        "noise_fraction_of_variance": noise_mse / max(across, 1e-12),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(SPECS))
    p.add_argument("--dataset", default=None, help="override the spec's lance path")
    p.add_argument("--logs-root", default=LOGS_ROOT,
                   help="root of the world-model run folders (default: $PLWM_LOGS_ROOT, "
                        "see paths.py)")
    p.add_argument("--run", default=None,
                   help="run folder under --logs-root (required unless --cache-dir is given)")
    p.add_argument("--policy", required=True,
                   help="<name>/weights_epoch_N.pt under <run>/checkpoints/, or a folder "
                        "checkpoint name under <cache-dir>/checkpoints/")
    p.add_argument("--cache-dir", default=None,
                   help="folder whose checkpoints/<policy> holds a folder checkpoint "
                        "(weights.pt + config.json), e.g. the repo root for the released "
                        "the released checkpoints/<env>/<model>/ folders and the image_lewm_<env> checkpoints; "
                        "overrides --logs-root/--run")
    p.add_argument("--modality", default="pointcloud", choices=["pointcloud", "image"],
                   help="which observation column the frozen model encodes: the "
                        "packed `lidar` cloud or the JPEG `pixels` frame (LeWM image)")
    p.add_argument("--img-size", type=int, default=224, help="image modality: resize target")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=5, help="keep every Nth dataset row")
    p.add_argument("--batch-size", type=int, default=64, help="frames per encoder forward")
    p.add_argument("--rows-per-shard", type=int, default=20000)
    p.add_argument("--invalid-value", type=float, default=-1.0)
    p.add_argument("--noise-probe", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="debug: stop after N kept rows")
    args = p.parse_args()
    if not args.cache_dir and not args.run:
        p.error("one of --run (run layout) or --cache-dir (folder checkpoint) is required")
    obs_col = "pixels" if args.modality == "image" else "lidar"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    import lance

    spec = SPECS[args.env]
    dataset = args.dataset or spec.dataset
    ds = lance.dataset(dataset)
    n_rows = ds.count_rows()
    keep = np.arange(0, n_rows, args.stride)
    if args.limit:
        keep = keep[: args.limit]
    print(f"[precompute] {args.env}: {n_rows} rows, stride {args.stride} -> {len(keep)} frames")

    encode = load_encoder(args.run, args.policy, args.logs_root,
                          cache_dir=args.cache_dir, modality=args.modality)
    transform = image_transform(args.img_size) if args.modality == "image" else None

    # The ViT is deterministic; the ball-query resampling the probe measures
    # only exists in the point-cloud encoder.
    if args.noise_probe and args.modality == "image":
        print("[precompute] image modality: encoder is deterministic, skipping the noise probe")
    elif args.noise_probe:
        rng = np.random.default_rng(args.seed)
        probe_rows = np.sort(rng.choice(n_rows, size=args.noise_probe, replace=False)).tolist()
        stats = noise_probe(ds, encode, probe_rows, args.batch_size, args.invalid_value, "cuda")
        print("[precompute] encoder sampling-noise probe:", stats)

    all_cols = [obs_col, "episode_idx", "step_idx", *spec.columns]
    out = Path(args.out)
    shard_dir = Path(str(out) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards = [keep[i : i + args.rows_per_shard] for i in range(0, len(keep), args.rows_per_shard)]
    t0 = time.time()
    done = 0
    for si, rows in enumerate(shards):
        path = shard_dir / f"shard_{si:05d}.npz"
        if path.exists():
            done += len(rows)
            continue
        tbl = ds.take(rows.tolist(), columns=all_cols)
        d = process_batch(tbl, spec, encode, args.batch_size, args.invalid_value, "cuda",
                          modality=args.modality, transform=transform)
        np.savez(path, **d)
        done += len(rows)
        el = time.time() - t0
        rate = done / max(el, 1e-9)
        print(
            f"[precompute] shard {si + 1}/{len(shards)}  {done}/{len(keep)} frames  "
            f"{rate:.0f} frames/s  eta {(len(keep) - done) / max(rate, 1e-9) / 60:.1f} min",
            flush=True,
        )

    print("[precompute] merging shards ...")
    merged = {}
    for si in range(len(shards)):
        with np.load(shard_dir / f"shard_{si:05d}.npz") as f:
            for k in f.files:
                merged.setdefault(k, []).append(f[k])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    merged["_meta_env"] = np.array(args.env)
    merged["_meta_run"] = np.array(args.run or "")
    merged["_meta_policy"] = np.array(args.policy)
    merged["_meta_cache_dir"] = np.array(str(args.cache_dir or ""))
    merged["_meta_modality"] = np.array(args.modality)
    merged["_meta_stride"] = np.array(args.stride)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **merged)
    print(f"[precompute] wrote {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    for k, v in merged.items():
        if isinstance(v, np.ndarray) and v.ndim:
            print(f"    {k:10s} {v.shape} {v.dtype}")


if __name__ == "__main__":
    main()
