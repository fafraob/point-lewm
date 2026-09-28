"""Precompute frozen Utonia embeddings for every frame of a lidar dataset.

The Utonia-WM baseline trains only the predictor on FROZEN Utonia features
(``pc_encoders.utonia_encoder.UtoniaEncoder``). Encoding on the fly would put
a 137M-parameter backbone forward (~110 ms/cloud, unbatchable) inside every training
step; since frozen features are constant, this script computes them ONCE per
(dataset, encoder config) and training reads them back through the data
pipeline (``emb_cache`` in the utoniawm data config — see train.py).

Output (next to each other, default: the config's data.obs.emb_cache, i.e.
$PLWM_DATA_ROOT/utonia_features_<env>.npy, or data/utonia_features_<env>.npy
when PLWM_DATA_ROOT is unset):
  <out>.npy        float16 memmap (total_rows, num_tokens * embed_dim), row i =
                   the flattened token grid of the dataset's global row i
                   (episode-major, the same indexing LanceDataset._cache
                   columns use; tokens in the encoder's flat cell order, so
                   the data pipeline's reshape recovers (num_tokens, embed_dim)).
                   fp16 because the token grid is ~100x a pooled vector:
                   256 tokens x 192 dims x 2.01M frames is ~198 GB at fp16 and
                   would be ~396 GB at fp32. The encoder rounds its LIVE
                   features through fp16 identically, so cached training
                   features and closed-loop eval features share one
                   quantization grid and agree to within a single fp16 step
                   (see "How close is cache to live" in utonia_encoder.py).
  <out>.meta.json  the RESOLVED encoder config + dataset path + row count +
                   num_tokens. train.py refuses a cache whose encoder config
                   does not match the run's (a silently stale cache would
                   invalidate the experiment).

Resumable: progress is tracked per episode in the meta file; rerun after an
interruption and it continues where it stopped. Change the encoder config ->
delete the cache (the meta mismatch will refuse it anyway).

Sharding (one process per GPU; ~halves the wall time per extra GPU). Rows are
independent and 96 KB each -- an exact multiple of the page size -- so shards
writing DISJOINT episode ranges of the same .npy never touch a shared page.
Each shard keeps its own progress sidecar and stays independently resumable, and
the canonical .meta.json (the only thing training accepts) appears only once
--finalize has confirmed the shards cover every episode exactly once:

    python precompute_utonia.py --allocate --out /data/tok.npy          # once
    python precompute_utonia.py --out /data/tok.npy --ep-start 0    --ep-end 5000 &
    python precompute_utonia.py --out /data/tok.npy --ep-start 5000 --ep-end 10000 &
    wait
    python precompute_utonia.py --out /data/tok.npy --finalize
    python precompute_utonia.py --out /data/tok.npy --verify

Keep all shards of one cache ON THE SAME MACHINE: they coordinate through
nothing but the shared mmap, and mmap write coherence across machines is not
guaranteed on network filesystems.

Run (one RTX 4090 in fp32: ~15 frames/s end to end, so ~37 h for the 2M-frame
cube table; raise model.encoder.max_clouds_per_forward on a bigger card).
Point --out at a disk with room for the full cache (~198 GB at the defaults):
    python precompute_utonia.py                                # utoniawm_cube config
    python precompute_utonia.py --dataset /path/to.lance --out /path/x.npy
    python precompute_utonia.py --verify                       # check it afterwards
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

# Same environment pinning as train.py (set before stable_worldmodel import):
# keep the swm cache next to the runs (<repo>/experiment_logs or $PLWM_LOGS_ROOT,
# see paths.py); avoid cuBLASLt's unsupported fused fp16 epilogue on huge
# point batches.
from paths import LOGS_ROOT  # noqa: E402

os.environ.setdefault("STABLEWM_HOME", LOGS_ROOT)
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--config-name", default="utoniawm_cube",
        help="train config whose model.encoder + data.obs.emb_cache to use",
    )
    p.add_argument("--clobber", action="store_true",
                   help="delete an existing cache whose encoder config differs "
                        "instead of refusing (rebuilds from scratch)")
    p.add_argument("--dataset", default=None,
                   help="lance dataset path (default: the config's data.dataset.name)")
    p.add_argument("--out", default=None,
                   help=".npy output (default: the config's data.obs.emb_cache)")
    p.add_argument("--clouds-per-batch", type=int, default=64,
                   help="frames per outer loop iteration (host->device transfer "
                        "and cache write granularity only -- the encoder always "
                        "forwards ONE cloud at a time through the backbone, so "
                        "this cannot change the features)")
    p.add_argument("--invalid-value", type=float, default=None,
                   help="override data.obs.invalid_value for dropping missed rays")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="stop after this many episodes (smoke tests only -- an "
                        "incomplete cache is refused by training)")
    p.add_argument("--verify", type=int, nargs="?", const=8, default=None,
                   metavar="N",
                   help="encode nothing; instead re-encode 3 frames from each of "
                        "N already-cached episodes (default 8) and compare them "
                        "with the cache. Run this after a build finishes.")
    # -- sharded builds (see "Sharding" in the module docstring) --------------
    p.add_argument("--allocate", action="store_true",
                   help="create the (sparse) .npy and exit. Run ONCE before "
                        "launching shards, which never create the file.")
    p.add_argument("--ep-start", type=int, default=None, metavar="A",
                   help="shard mode: first episode (inclusive). Requires --ep-end.")
    p.add_argument("--ep-end", type=int, default=None, metavar="B",
                   help="shard mode: last episode (EXCLUSIVE). Requires --ep-start.")
    p.add_argument("--finalize", action="store_true",
                   help="merge the per-shard progress files into the cache's "
                        "canonical .meta.json (which is what training accepts). "
                        "Fails unless the shards cover every episode exactly once.")
    args = p.parse_args()
    if (args.ep_start is None) != (args.ep_end is None):
        p.error("--ep-start and --ep-end must be given together")
    if args.ep_start is not None:
        if args.ep_start < 0 or args.ep_end <= args.ep_start:
            p.error(f"empty or negative shard range [{args.ep_start}, {args.ep_end})")
        if args.max_episodes is not None:
            p.error("--max-episodes cannot be combined with a shard range")
    return args


def verify_cache(args, emb, lengths, offsets, meta, load_episode, encode_block, K, D):
    """Re-encode sampled cached frames live and compare (``--verify``).

    Worth running after a multi-hour build: it is the one check that exercises
    the whole chain end to end, and it caught two real defects (batch-dependent
    attention patch sizing, and Lightning's autocast reaching the projection).
    Cache and live are expected to agree to within ONE fp16 quantization step,
    not bitwise -- see "How close is cache to live" in
    ``pc_encoders/utonia_encoder.py``. Exits non-zero on a larger deviation.
    """
    done = int(meta["episodes_done"])
    assert done > 0, "nothing to verify yet -- no episode has been encoded"
    rng = np.random.default_rng(0)
    eps = rng.choice(done, size=min(args.verify, done), replace=False)
    worst, worst_where, n_diff, n_tot, scale = 0.0, None, 0, 0, 0.0
    for ep in sorted(int(e) for e in eps):
        n = int(lengths[ep])
        clouds = load_episode(ep)
        # first, middle and last frame of the episode: the ends are where an
        # off-by-one in the row offsets would show up
        for f in sorted({0, n // 2, n - 1}):
            live = encode_block(clouds[f : f + 1], ep)[0]
            cached = torch.from_numpy(
                np.asarray(emb[offsets[ep] + f]).astype(np.float32)
            ).view(K, D).to(live.device)
            d = (cached - live).abs()
            n_diff += int((d > 0).sum())
            n_tot += d.numel()
            scale = max(scale, float(live.abs().max()))
            if float(d.max()) > worst:
                worst, worst_where = float(d.max()), (ep, f)
    h = torch.tensor(scale, dtype=torch.float16)
    ulp = float(torch.nextafter(h, torch.tensor(1e4, dtype=torch.float16)) - h)
    print(f"[verify] {len(eps)} episodes x 3 frames: {n_diff}/{n_tot} entries differ "
          f"({100 * n_diff / max(n_tot, 1):.4f}%), max |diff| {worst:.3e} at "
          f"episode/frame {worst_where}; fp16 step at |token|<={scale:.3f} is {ulp:.3e}")
    if worst > ulp * 1.001:
        raise SystemExit(
            f"[verify] FAILED: {worst:.3e} exceeds one fp16 step ({ulp:.3e}). That is a "
            "pipeline mismatch (encoder config / drop rule / row offsets), not "
            "kernel-selection noise -- do NOT train on this cache."
        )
    print("[verify] OK: within one fp16 quantization step of a live encode.")


def finalize_shards(out, meta_path, meta, keys, n_episodes):
    """Merge per-shard progress files into the canonical ``.meta.json``.

    Until this runs, a sharded cache has no canonical meta and training refuses
    it -- so a half-built cache cannot be trained on by accident. The merge only
    succeeds if the shards were built with the SAME encoder/dataset/obs config as
    the current one and together cover every episode exactly once, each finished.
    """
    shards = sorted(out.parent.glob(f"{out.name}.shard*-*.meta.json"))
    if not shards:
        raise SystemExit(
            f"[precompute] no shard progress files next to {out} "
            f"(expected {out.name}.shard<A>-<B>.meta.json)"
        )
    spans, bad = [], []
    for p in shards:
        m = json.loads(p.read_text())
        if {k: m.get(k) for k in keys} != {k: meta[k] for k in keys}:
            bad.append(f"{p.name}: built with a different encoder/dataset/obs config")
            continue
        a, b, done = int(m["ep_start"]), int(m["ep_end"]), int(m["episodes_done"])
        if done < b:
            bad.append(f"{p.name}: unfinished ({done}/{b} episodes)")
            continue
        spans.append((a, b, p.name))
    spans.sort()
    cursor = 0
    for a, b, name in spans:
        if a != cursor:
            bad.append(
                f"{name}: starts at {a} but episodes [{cursor}, {a}) are "
                + ("covered twice" if a < cursor else "covered by no shard")
            )
        cursor = max(cursor, b)
    if cursor != n_episodes:
        bad.append(f"shards stop at episode {cursor}, dataset has {n_episodes}")
    if bad:
        raise SystemExit("[precompute] cannot finalize:\n  " + "\n  ".join(bad))
    meta["episodes_done"] = n_episodes
    meta.pop("ep_start", None)
    meta.pop("ep_end", None)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[precompute] finalized {len(spans)} shards covering "
          f"{n_episodes} episodes -> {meta_path}")
    print("[precompute] now check it: rerun with --verify")


def compose_cfg(config_name):
    import hydra

    with hydra.initialize(config_path="config/train", version_base=None):
        return hydra.compose(config_name=config_name)


def main():
    args = parse_args()
    cfg = compose_cfg(args.config_name)
    assert "utonia" in str(cfg.model.encoder._target_).lower(), (
        f"--config-name {args.config_name} does not use the Utonia encoder"
    )

    import hydra
    import stable_worldmodel as swm

    from utils import encoder_cache_signature

    encoder = hydra.utils.instantiate(cfg.model.encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()

    dataset_path = args.dataset or cfg.data.dataset.name
    out = Path(args.out or cfg.data.obs.emb_cache)
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    # The cache must NOT depend on the training window protocol: read raw
    # frames (num_steps=1, frameskip=1), lidar column only. LOCAL_DATASET_DIR
    # redirects the dataset location exactly like train.py.
    ds = swm.data.load_dataset(
        dataset_path, num_steps=1, frameskip=1,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR", None),
        keys_to_load=[cfg.data.obs.get("live_source_key", "lidar")],
    )
    src = cfg.data.obs.get("live_source_key", "lidar")
    invalid = args.invalid_value
    if invalid is None:
        invalid = cfg.data.obs.get("live_invalid_value", -1.0)
    lengths = np.asarray(ds.lengths, dtype=int)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    total = int(offsets[-1])
    D = int(cfg.model.encoder.embed_dim)
    K = int(encoder.num_tokens)  # fixed token grid (see UtoniaEncoder)
    row_width = K * D

    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta = {
        "dataset": str(dataset_path),
        "rows": total,
        "embed_dim": D,
        "num_tokens": K,
        "encoder": encoder_cache_signature(cfg.model.encoder),
        # The raw-observation rule ACTUALLY USED (CLI override included): which
        # column was read and which points counted as missed rays. It is not in
        # the encoder config but it changes every cached feature, so training
        # cross-checks it too (utils.obs_cache_signature).
        "obs": {"live_source_key": src, "live_invalid_value": invalid},
        "episodes_done": 0,
    }
    keys = ("dataset", "rows", "embed_dim", "num_tokens", "encoder", "obs")
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.finalize:
        return finalize_shards(out, meta_path, meta, keys, len(lengths))

    # A shard tracks its own progress in its own sidecar, so the canonical
    # .meta.json only ever appears once --finalize has checked full coverage --
    # training can therefore never pick up a half-built sharded cache.
    sharded = args.ep_start is not None
    if sharded:
        ep_first, ep_last = args.ep_start, min(args.ep_end, len(lengths))
        meta.update(ep_start=ep_first, ep_end=ep_last, episodes_done=ep_first)
        meta_path = out.with_suffix(f"{out.suffix}.shard{ep_first}-{ep_last}.meta.json")
    else:
        ep_first, ep_last = 0, len(lengths)

    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if {k: old.get(k) for k in keys} != {k: meta[k] for k in keys}:
            if not args.clobber:
                raise SystemExit(
                    f"existing cache {out} was built with a different "
                    f"encoder/dataset config -- rerun with --clobber to rebuild "
                    f"(or delete it and {meta_path})"
                )
            print(f"[precompute] --clobber: discarding stale progress {meta_path}")
            if not sharded:  # a shard must never delete its siblings' rows
                out.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        else:
            meta = old
            print(f"[precompute] resuming: {meta['episodes_done']}/{ep_last} "
                  "episodes done")
    need = total * row_width * 2
    print(f"[precompute] cache: {total} rows x {K} tokens x {D} dims fp16 = "
          f"{need / 1e9:.1f} GB -> {out}")
    # Pre-flight free-space check: the memmap is created SPARSE, so without this
    # a multi-day build would only hit ENOSPC once it had filled the disk.
    anchor = out.parent
    while not anchor.exists():
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free + (out.stat().st_blocks * 512 if out.exists() else 0)
    if free < need * 1.02:
        raise SystemExit(
            f"[precompute] only {free / 1e9:.1f} GB free on the filesystem holding "
            f"{anchor}, but the cache needs {need / 1e9:.1f} GB. Pass --out on a "
            "bigger disk (and point data.obs.emb_cache at it for training), or "
            "shrink the grid / embed_dim."
        )
    if out.exists():
        emb = np.lib.format.open_memmap(out, mode="r+")
        assert emb.shape == (total, row_width), (emb.shape, (total, row_width))
    elif sharded:
        # Shards must never create the file: two of them racing on "w+" would
        # truncate each other. Allocate once up front instead.
        raise SystemExit(
            f"[precompute] {out} does not exist yet. A shard never creates it -- "
            f"run `--allocate` once first (same --out/--config-name)."
        )
    else:
        emb = np.lib.format.open_memmap(
            out, mode="w+", dtype=np.float16, shape=(total, row_width)
        )
    if args.allocate:
        emb.flush()
        print(f"[precompute] allocated {out} ({total} x {row_width} fp16, sparse). "
              "Launch the shards now.")
        return

    in_ch = 3

    def load_episode(ep):
        n = int(lengths[ep])
        return ds.load_chunk([ep], [0], [n])[0][src].reshape(n, -1, in_ch).float()

    def encode_block(pts, ep="?"):
        """Live-encode a (F, N, in_ch) block of raw frames -> (F, K, D)."""
        keep = ~(pts == invalid).all(dim=-1) if invalid is not None else torch.ones(
            pts.shape[:2], dtype=torch.bool
        )
        counts = keep.sum(dim=1)
        assert bool((counts > 0).all()), f"episode {ep} has an empty cloud"
        batch = torch.repeat_interleave(torch.arange(len(pts)), counts)
        packed = {"coord": pts[keep].to(device), "batch": batch.to(device), "feat": None}
        with torch.inference_mode():
            e = encoder(packed)
        assert e.shape[1:] == (K, D), (e.shape, (K, D))
        return e

    if args.verify:
        return verify_cache(args, emb, lengths, offsets, meta, load_episode,
                            encode_block, K, D)

    t0 = time.time()
    done_rows = int(offsets[meta["episodes_done"]])
    last_ep = ep_last if args.max_episodes is None else min(
        ep_last, args.max_episodes
    )
    if sharded:
        print(f"[precompute] shard [{ep_first}, {ep_last}) of {len(lengths)} "
              f"episodes -> rows [{offsets[ep_first]}, {offsets[ep_last]}), "
              f"progress in {meta_path.name}")
    for ep in range(meta["episodes_done"], last_ep):
        n = int(lengths[ep])
        clouds = load_episode(ep)
        for f0 in range(0, n, args.clouds_per_batch):
            f1 = min(f0 + args.clouds_per_batch, n)
            e = encode_block(clouds[f0:f1], ep)  # (F, K, D)
            # flatten the token grid into one cache row; the encoder already
            # rounded through fp16, so this cast is exact.
            emb[offsets[ep] + f0 : offsets[ep] + f1] = (
                e.reshape(f1 - f0, row_width).cpu().numpy().astype(np.float16)
            )
        meta["episodes_done"] = ep + 1
        if (ep + 1) % 20 == 0 or ep + 1 == last_ep:
            emb.flush()
            meta_path.write_text(json.dumps(meta, indent=1))
            rows = int(offsets[ep + 1]) - done_rows
            rate = rows / max(time.time() - t0, 1e-9)
            eta = (int(offsets[last_ep]) - int(offsets[ep + 1])) / max(rate, 1e-9) / 3600
            print(f"[precompute] episode {ep + 1}/{last_ep} "
                  f"({rate:.0f} frames/s, ~{eta:.1f} h left)", flush=True)
    emb.flush()
    meta_path.write_text(json.dumps(meta, indent=1))
    if meta["episodes_done"] < last_ep:
        state = (f"INCOMPLETE ({meta['episodes_done']}/{last_ep} episodes -- "
                 "rerun the same command to continue)")
    elif sharded:
        state = (f"shard [{ep_first}, {ep_last}) done -- run --finalize once "
                 "EVERY shard has finished")
    else:
        state = "done"
    print(f"[precompute] {state}: {out} ({total} x {K} x {D})")


if __name__ == "__main__":
    main()
