#!/usr/bin/env python3
"""Colour a point cloud by what the encoder's CLS token attends to.

    pixi run python scripts/viz_attention.py \
        --logs-root . --run . --policy reacher/point-lewm/weights.pt \
        --table data/reacher.lance -n 4

The checkpoint is ``<logs-root>/<run>/checkpoints/<policy>``: ``--logs-root . --run .``
addresses the released checkpoints under ``checkpoints/<env>/<model>/``
(scripts/download_checkpoints.py); a training run of your own is ``--run <folder>
--policy <config name>/weights_final.pt`` under ``--logs-root`` (default
``paths.LOGS_ROOT``, i.e. ``$PLWM_LOGS_ROOT`` or ``experiment_logs/``).

Writes one PLY per sampled frame (points coloured by attention), a second PLY of
the token centres, and a .npy of the raw per-point weights.

WHAT IS BEING SHOWN. PointViT turns a cloud into ``num_tokens`` group tokens --
an FPS centre plus its radius neighbourhood -- prepends the ViT's CLS token, and
the embedding it hands the world model is the CLS row. So "what the encoder
looks at" is the attention from CLS to those group tokens, and a token's weight
belongs to the points in its group. A point covered by several groups takes the
largest weight covering it; a point in none stays 0 (FPS+radius grouping does
not tile the cloud, which is itself worth seeing).

--mode picks how layers combine:
  last     CLS attention in the final layer -- what the last block reads
  mean     averaged over layers
  rollout  Abnar & Zuidema attention rollout: propagate (A + I)/2 through the
           stack, which accounts for information already mixed into tokens by
           earlier layers and is usually the honest picture for a deep ViT

The absolute numbers matter less than the spread: uniform attention means the
encoder has no preference, and a cloud whose task-relevant part is a few percent
of the points (a thin arm over a large floor) will show it here.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# paths.py lives at the repo root, which is not on sys.path when this script
# is run as `python scripts/viz_attention.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LOGS_ROOT  # noqa: E402


def _ensure_repo_importable():
    """Put the repo root on sys.path.

    A checkpoint's config names `jepa.JEPA` and `pc_encoders...`, which live at
    the repo root -- but running `python scripts/viz_attention.py` puts only
    `scripts/` on the path, so instantiate() cannot find them.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _ensure_swm_importable():
    """Fall back to the stable-worldmodel checkout shipped in the repo root."""
    try:
        import stable_worldmodel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    vendored = Path(__file__).resolve().parents[1] / "stable-worldmodel"
    if (vendored / "stable_worldmodel" / "__init__.py").is_file():
        sys.path.insert(0, str(vendored))


def write_ply(path, xyz, attention):
    """Points plus a per-point `attention` scalar -- no colours.

    Storing the value rather than a colour keeps the file the DATA: the ramp is
    a presentation choice that belongs to whatever renders it (see
    scripts/render_ply.py --cmap), and baking one in means re-running the model
    to change it, and losing the numbers to a lossy 8-bit encoding on the way.
    `attention` is the standard scalar slot most viewers (MeshLab, CloudCompare,
    polyscope) will offer to colour by.
    """
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float attention\n")
        f.write("end_header\n")
        for point, value in zip(xyz, attention):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {value:.8f}\n")


def capture_attention(vit, x):
    """Run the transformer stack, returning (output, [attn per layer]).

    Two ways in, because which one works depends on the transformers version and
    the attention implementation it selected. Patching SDPA recomputes the exact
    weights the kernel would have used (ViT self-attention has no mask and no
    dropout at eval, so softmax(QK^T/sqrt(d)) IS the attention); if the model
    never calls SDPA, fall back to asking the layers for their weights.
    """
    captured = []
    original = torch.nn.functional.scaled_dot_product_attention

    def patched(q, k, v, *args, **kwargs):
        weights = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1]), dim=-1)
        captured.append(weights.detach().float().cpu())
        return original(q, k, v, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched
    try:
        for layer in vit.layers:
            out = layer(x, None)
            x = out[0] if isinstance(out, tuple) else out
    finally:
        torch.nn.functional.scaled_dot_product_attention = original

    if captured:
        return x, captured

    for layer in vit.layers:  # eager path
        out = layer(x, None, output_attentions=True)
        x = out[0] if isinstance(out, tuple) else out
        if isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1]):
            captured.append(out[1].detach().float().cpu())
    if not captured:
        sys.exit("could not capture attention from this transformer implementation")
    return x, captured


def cls_attention(attns, mode):
    """(layers, B, heads, 1+T, 1+T) -> CLS->token weights (B, T), normalized."""
    per_layer = [a.mean(dim=1) for a in attns]  # average heads: (B, 1+T, 1+T)
    if mode == "last":
        w = per_layer[-1][:, 0, 1:]
    elif mode == "mean":
        w = torch.stack([a[:, 0, 1:] for a in per_layer]).mean(dim=0)
    else:  # rollout
        n = per_layer[0].shape[-1]
        eye = torch.eye(n).unsqueeze(0)
        roll = None
        for a in per_layer:
            a = (a + eye) / 2.0
            a = a / a.sum(dim=-1, keepdim=True)
            roll = a if roll is None else a @ roll
        w = roll[:, 0, 1:]
    return (w / w.sum(dim=-1, keepdim=True)).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder under --logs-root")
    ap.add_argument("--policy", required=True, help="e.g. point_lewm_reacher/weights_final.pt")
    ap.add_argument("--logs-root", default=LOGS_ROOT,
                    help="where the training runs live (default: paths.LOGS_ROOT)")
    ap.add_argument("--table", required=True, help="lance table to draw clouds from")
    ap.add_argument("--column", default="lidar")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--invalid-value", type=float, default=-1.0)
    ap.add_argument("-n", "--num", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cell-size", type=float, default=None,
                    help="occlusion cell edge in metres (default: scene extent / 12)")
    ap.add_argument("--occlusion-batch", type=int, default=8,
                    help="ablated clouds per forward pass")
    ap.add_argument("--method", choices=("attention", "grad", "occlusion"), default="attention",
                    help="attention: CLS->token weights, PointViT only. grad: |d||z||/dx| per "
                         "point, which works for ANY encoder -- Utonia (PTv3) has no CLS token "
                         "and attends only within local patches, so there is no equivalent row to "
                         "plot. occlusion: delete a cell of points and measure how far the "
                         "embedding moves -- no gradients, no architectural assumptions, so it "
                         "works on the FROZEN Utonia backbone (which runs under no_grad and "
                         "voxelises coordinates into integer keys, blocking gradients twice over) "
                         "and on PointViT alike. Slower than grad, and the only method that gives "
                         "comparable numbers across both encoders.")
    ap.add_argument("--mode", choices=("last", "mean", "rollout"), default="rollout")
    ap.add_argument("--group-radius", type=float, default=None,
                    help="ABLATION: regroup with this radius instead of the checkpoint's. "
                         "Editing config/train/model/*.yaml does NOT affect an existing "
                         "checkpoint -- load_pretrained rebuilds the encoder from the "
                         "config.json frozen beside the weights -- so this is the only way to "
                         "see a different grouping without retraining. It feeds the model local "
                         "geometry it was never trained on, so read the result as 'what this "
                         "encoder does with a different tokenizer', not as a better map.")
    ap.add_argument("--out", type=Path, default=Path("attention"))
    args = ap.parse_args()

    _ensure_repo_importable()

    # Read the clouds FIRST: a wrong path then fails in a second instead of after
    # the model load, and the table opens before anything else has touched the
    # lance runtime.
    import lancedb

    p = Path(args.table).expanduser()
    ds = lancedb.connect(str(p.parent)).open_table(p.stem).to_lance()
    rng = np.random.default_rng(args.seed)
    rows = sorted(rng.choice(ds.count_rows(), size=args.num, replace=False).tolist())
    flat = np.asarray(ds.take(rows, columns=[args.column]).column(args.column)
                      .combine_chunks().flatten(), dtype=np.float64)
    clouds = flat.reshape(len(rows), -1, args.in_channels)

    _ensure_swm_importable()
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(args.policy, cache_dir=str(Path(args.logs_root) / args.run))
    encoder = model.encoder
    if args.method == "attention":
        for attr in ("_fps_centers", "_group", "_tokenize", "vit"):
            if not hasattr(encoder, attr):
                sys.exit(f"{type(encoder).__name__} has no .{attr}: CLS attention is PointViT-only. "
                         f"Use --method grad, which works for any encoder.")
    if args.group_radius is not None:
        print(f"[eval] ABLATION: group_radius {encoder.group_radius} -> {args.group_radius} "
              f"(the checkpoint was trained at {encoder.group_radius})")
        encoder.group_radius = float(args.group_radius)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    # Describe whatever encoder this is: the tokenizer geometry below exists on
    # PointViT and not on PTv3-style encoders, so ask rather than assume.
    shape = ", ".join(
        f"{name}={getattr(encoder, attr)}"
        for name, attr in (("tokens", "num_tokens"), ("group", "group_size"),
                           ("r", "group_radius"), ("grid", "grid_size"))
        if hasattr(encoder, attr)
    )
    print(f"{p.name}: rows {rows} | encoder {type(encoder).__name__} ({shape}) "
          f"| method {args.method}" + (f"/{args.mode}" if args.method == "attention" else "")
          + f" | {device}")

    for row, cloud in zip(rows, clouds):
        xyz = cloud[~(cloud == args.invalid_value).all(axis=1)]
        coord = torch.as_tensor(xyz, dtype=torch.float32, device=device)
        batch = torch.zeros(len(coord), dtype=torch.long, device=device)

        if args.method == "occlusion":
            # Cells rather than single points: removing one point of 7000 moves
            # the embedding by nothing measurable, and a cell is also what a
            # reader can see. Cells come from a voxel grid over the cloud, so
            # they are the same size everywhere and independent of the encoder.
            with torch.inference_mode():
                base = encoder({"coord": coord, "batch": batch, "feat": None})
                base = base.float().reshape(-1)
                extent = float((coord.max(0).values - coord.min(0).values).max())
                size = args.cell_size or extent / 12.0
                keys = torch.floor((coord - coord.min(0).values) / size).long()
                _, cell_id = torch.unique(keys, dim=0, return_inverse=True)
                n_cells = int(cell_id.max().item()) + 1

                per_point = np.zeros(len(coord))
                for start in range(0, n_cells, args.occlusion_batch):
                    ids = range(start, min(start + args.occlusion_batch, n_cells))
                    coords, batches, kept = [], [], []
                    for slot, cid in enumerate(ids):
                        keep = cell_id != cid
                        if int(keep.sum()) < 32:      # do not hand the encoder an empty cloud
                            continue
                        coords.append(coord[keep])
                        batches.append(torch.full((int(keep.sum()),), len(kept),
                                                  device=coord.device, dtype=torch.long))
                        kept.append(cid)
                    if not kept:
                        continue
                    embs = encoder({"coord": torch.cat(coords), "batch": torch.cat(batches),
                                    "feat": None}).float().reshape(len(kept), -1)
                    moved = (embs - base).norm(dim=1) / base.norm()
                    for cid, value in zip(kept, moved.tolist()):
                        per_point[(cell_id == cid).cpu().numpy()] = value

            coord_out = coord.detach().cpu().numpy()
            tag = f"row{row:07d}_occlusion"
            write_ply(args.out / f"{tag}.ply", coord_out, per_point)
            np.save(args.out / f"{tag}_weights.npy", per_point)
            share = np.sort(per_point)[::-1][: max(1, len(per_point) // 100)].sum() / max(per_point.sum(), 1e-12)
            print(f"  row {row}: {len(coord_out)} pts in {n_cells} cells of {size:.3f} m | "
                  f"top 1% of points carry {share:.0%} of the effect | "
                  f"max embedding shift {per_point.max():.3f} (relative)")
            continue

        if args.method == "grad":
            # Saliency of the embedding w.r.t. the input points: one backward
            # pass, no assumptions about how the encoder routes information.
            # NOT inference_mode -- that disables autograd outright.
            coord.requires_grad_(True)
            with torch.enable_grad():
                emb = encoder({"coord": coord, "batch": batch, "feat": None})
                (emb.float() ** 2).sum().backward()
            per_point = coord.grad.norm(dim=1).detach().cpu().numpy()
            coord_out = coord.detach().cpu().numpy()

            tag = f"row{row:07d}_grad"
            write_ply(args.out / f"{tag}.ply", coord_out, per_point)
            np.save(args.out / f"{tag}_weights.npy", per_point)
            share = np.sort(per_point)[::-1][: max(1, len(per_point) // 100)].sum() / per_point.sum()
            print(f"  row {row}: {len(coord_out)} pts | top 1% of points carry {share:.0%} of the "
                  f"saliency | max/mean {per_point.max() / per_point.mean():.1f}x")
            continue

        with torch.inference_mode():
            data = encoder.remove_ground({"coord": coord, "batch": batch, "feat": None})
            coord_g, batch_g = data["coord"], data["batch"].long()
            centers = coord_g[encoder._fps_centers(coord_g, batch_g, 1).reshape(-1)]
            nbr_idx = encoder._group(coord_g, batch_g, centers, 1)
            tokens = encoder._tokenize(coord_g, data.get("feat"), centers, nbr_idx)
            tokens = tokens.view(1, encoder.num_tokens, encoder.token_dim)
            cls = encoder.vit.embeddings.cls_token.expand(1, -1, -1)
            _, attns = capture_attention(encoder.vit, torch.cat([cls, tokens.to(cls.dtype)], dim=1))

        weight = cls_attention(attns, args.mode)[0]                      # (num_tokens,)
        groups = nbr_idx.reshape(encoder.num_tokens, encoder.group_size).cpu().numpy()

        # A point takes the largest weight among the groups covering it.
        per_point = np.zeros(len(coord_g), dtype=np.float64)
        np.maximum.at(per_point, groups.reshape(-1), np.repeat(weight, encoder.group_size))
        covered = per_point > 0

        pts = coord_g.cpu().numpy()
        tag = f"row{row:07d}_{args.mode}"
        write_ply(args.out / f"{tag}.ply", pts, per_point)
        write_ply(args.out / f"{tag}_centers.ply", centers.cpu().numpy(), weight)
        np.save(args.out / f"{tag}_weights.npy", per_point)

        order = np.sort(weight)[::-1]
        top10 = order[:10].sum()
        uniform = 1.0 / encoder.num_tokens
        print(f"  row {row}: {len(pts)} pts, {covered.mean():.0%} covered by a token | "
              f"top-10 tokens hold {top10:.0%} of attention "
              f"(uniform would be {10 * uniform:.0%}) | "
              f"max/mean token weight {weight.max() / weight.mean():.1f}x")

    print(f"\nwrote {len(rows)} clouds to {args.out} (open the .ply in MeshLab/CloudCompare)")


if __name__ == "__main__":
    main()
