#!/usr/bin/env python3
"""Write out a random sample of point clouds from a lidar table, to look at.

    pixi run python scripts/dump_clouds.py TABLE.lance
    pixi run python scripts/dump_clouds.py TABLE.lance -n 20 --out /tmp/clouds
    pixi run python scripts/dump_clouds.py TABLE.lance --format npy --keep-misses

PLY by default, which opens in MeshLab / CloudCompare / Blender and in the
repo's own polyscope viewer. Colours are written when the table has a row-aligned
`<column>_rgb`, so an RGB table looks the way the model sees it.

Misses are dropped by default: the sensor emits a fixed grid and writes
`miss_value` on all three coords of a ray that hit nothing, and those points are
not part of the cloud -- they are the absence of one. Keeping them (--keep-misses)
piles thousands of points on a single spot and makes a scene unreadable, but it
is the honest picture if you are checking the fill rate.

Each file is named for the row it came from (episode/step where the table records
them), and a manifest.txt lists the exact rows, so a cloud that looks wrong can be
traced back and re-read.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def write_ply(path, xyz, rgb=None):
    """Minimal ASCII PLY -- no dependency, opens everywhere."""
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]:d} {c[1]:d} {c[2]:d}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", help="path to the .lance table")
    ap.add_argument("-n", "--num", type=int, default=10, help="how many clouds (default 10)")
    ap.add_argument("--out", type=Path, default=Path("clouds"), help="output directory")
    ap.add_argument("--column", default="lidar", help="point-cloud column (default: lidar)")
    ap.add_argument("--in-channels", type=int, default=3, help="values per point (default: 3)")
    ap.add_argument("--miss-value", type=float, default=-1.0, help="coord value of a missed ray")
    ap.add_argument("--keep-misses", action="store_true", help="do not drop missed rays")
    ap.add_argument("--format", choices=("ply", "npy", "xyz"), default="ply")
    ap.add_argument("--seed", type=int, default=0, help="which random rows (default 0)")
    args = ap.parse_args()

    import lancedb

    p = Path(args.table)
    ds = lancedb.connect(str(p.parent)).open_table(p.stem).to_lance()
    names = {f.name for f in ds.schema}
    if args.column not in names:
        sys.exit(f"no column '{args.column}' in {p.name}; found: {sorted(names)}")

    total = ds.count_rows()
    rng = np.random.default_rng(args.seed)
    rows = sorted(rng.choice(total, size=min(args.num, total), replace=False).tolist())

    # Labels come from the table when it records them, so a file name points back
    # at a row you can re-read rather than at an opaque index.
    wanted = [args.column] + [c for c in ("episode_idx", "step_idx", f"{args.column}_rgb") if c in names]
    batch = ds.take(rows, columns=wanted)

    def col(name, width):
        flat = np.asarray(batch.column(name).combine_chunks().flatten())
        return flat.reshape(len(rows), -1, width)

    clouds = col(args.column, args.in_channels).astype(np.float64)
    colors = col(f"{args.column}_rgb", 3) if f"{args.column}_rgb" in wanted else None
    episodes = np.asarray(batch.column("episode_idx")).ravel() if "episode_idx" in wanted else None
    steps = np.asarray(batch.column("step_idx")).ravel() if "step_idx" in wanted else None

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = [f"# {p}  rows {total}  column {args.column}  seed {args.seed}"]
    print(f"{p.name}: {total} rows, {clouds.shape[1]} points/frame -> {args.out}")

    for i, row in enumerate(rows):
        xyz = clouds[i]
        rgb = colors[i] if colors is not None else None
        if not args.keep_misses:
            keep = ~(xyz == args.miss_value).all(axis=1)
            xyz, rgb = xyz[keep], (rgb[keep] if rgb is not None else None)
        tag = f"row{row:07d}"
        if episodes is not None:
            tag += f"_ep{int(episodes[i])}"
        if steps is not None:
            tag += f"_step{int(steps[i])}"

        out = args.out / f"{tag}.{args.format}"
        if args.format == "ply":
            c = None
            if rgb is not None:
                c = rgb.astype(np.float64)
                c = np.clip(c * 255 if c.max() <= 1.0 else c, 0, 255).astype(np.uint8)
            write_ply(out, xyz, c)
        elif args.format == "npy":
            np.save(out, xyz.astype(np.float32))
        else:
            np.savetxt(out, xyz, fmt="%.6f")

        extent = xyz.max(axis=0) - xyz.min(axis=0) if len(xyz) else np.zeros(3)
        line = (f"{out.name}: {len(xyz)} points"
                + (f" of {clouds.shape[1]}" if not args.keep_misses else "")
                + f", extent {np.round(extent, 3).tolist()}")
        manifest.append(line)
        print("  " + line)

    (args.out / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nwrote {len(rows)} clouds + manifest.txt to {args.out}")


if __name__ == "__main__":
    main()
