#!/usr/bin/env python3
"""Show what the ablation actually does to a cloud -- is the perturbation too much?

    pixi run python scripts/perturb_preview.py
    pixi run python scripts/perturb_preview.py --env pusht --row 12345
    pixi run python scripts/perturb_preview.py --env reacher tworoom pusht

Renders one frame from the table under every condition in the ablation sweep,
side by side, and prints the numbers that decide whether a level is sane: how
many returns survive, how far a point moves, and -- the one that matters -- that
displacement as a multiple of the cloud's own point spacing. Noise well below
the spacing softens a surface; noise well above it destroys the surface, because
a point no longer lands nearer its true neighbours than to anything else.

It imports CloudPerturbation FROM eval_lidar.py rather than reimplementing it,
so what you see here is exactly what the planner was fed -- a preview that
merely approximates the eval would be worse than none.
"""

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import DATA_ROOT, DATASETS  # noqa: E402
from render_ply import fit_distance, scene_frame  # noqa: E402  (same directory)

# condition label | kwargs, matching config/eval_sweep/ablation_*.conf
CONDITIONS = [
    ("clean",            {}),
    ("noise 0.25%",      dict(noise_frac=0.0025)),
    ("noise 0.5%",       dict(noise_frac=0.005)),
    ("dropout 25%",      dict(dropout_prob=0.25)),
    ("dropout 50%",      dict(dropout_prob=0.5)),
]
TABLES = dict(DATASETS)   # env -> table file name under --data (paths.py)


def load_perturbation():
    """The eval's own class, executed out of eval_lidar.py without its imports."""
    import torch
    src = (Path(__file__).resolve().parents[1] / "eval_lidar.py").read_text()
    start, end = src.index("class CloudPerturbation:"), src.index("class LidarPlanAdapter")
    mod = types.ModuleType("_perturb")
    mod.__dict__["torch"] = torch
    exec(compile(src[start:end], "eval_lidar.py", "exec"), mod.__dict__)
    return mod.CloudPerturbation


def write_ply(path, xyz, scalar):
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float attention\nend_header\n")
        for p, v in zip(xyz, scalar):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {v:.6f}\n")


def nn_spacing(pts, rng, k=1500):
    s = pts[rng.choice(len(pts), min(k, len(pts)), replace=False)]
    d = np.linalg.norm(s[:, None] - s[None], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", nargs="+", default=["reacher"], choices=list(TABLES))
    ap.add_argument("--data", type=Path, default=Path(DATA_ROOT),
                    help="folder holding the .lance tables (default: paths.DATA_ROOT)")
    ap.add_argument("--row", type=int, default=636961)
    ap.add_argument("--invalid-value", type=float, default=-1.0)
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--radius", type=float, default=0.004,
                    help="point radius in METRES. Absolute on purpose: a relative radius "
                         "rescales with the cloud, so a thinned cloud would be drawn with "
                         "bigger dots and dropout would look milder than it is")
    ap.add_argument("--azimuth", type=float, default=8.0)
    ap.add_argument("--elevation", type=float, default=-14.0)
    ap.add_argument("--distance", type=float, default=0.85,
                    help="multiplier on the fitted distance; below 1 crops in")
    ap.add_argument("--cmap", default="Blues")
    ap.add_argument("--cmap-range", type=float, nargs=2, default=(0.38, 0.95))
    ap.add_argument("--width", type=float, default=11.0)
    ap.add_argument("--font", type=float, default=9.0)
    ap.add_argument("--out", type=Path, default=Path("figures/perturb_preview"))
    args = ap.parse_args()

    import lancedb
    import torch
    CloudPerturbation = load_perturbation()
    rng = np.random.default_rng(0)
    work = args.out.parent / "_perturb"
    work.mkdir(parents=True, exist_ok=True)

    panels, stats = {}, []
    for env in args.env:
        p = (args.data / TABLES[env]).expanduser()
        ds = lancedb.connect(str(p.parent)).open_table(p.stem).to_lance()
        row = args.row % ds.count_rows()
        raw = np.asarray(ds.take([row], columns=["lidar"]).column("lidar")
                         .combine_chunks().flatten(), dtype=np.float32).reshape(-1, 3)
        clouds = torch.from_numpy(raw)[None, None]          # (B=1, T=1, N, 3)
        base = raw[~(raw == args.invalid_value).all(1)]
        spacing = nn_spacing(base, rng)

        # ONE camera for all five panels, fitted to the CLEAN cloud. Letting
        # each render fit its own distance is what made the first version
        # useless: dropout shrinks the cloud slightly, the camera moved in, and
        # the thinned panel came back drawn with bigger dots -- so the two
        # things you are trying to compare, density and framing, both changed.
        center, _, axis_u, axis_v, normal = scene_frame(base)
        offset = -center                       # sensor sits at the origin
        sideways = np.cross(normal, offset)
        n = np.linalg.norm(sideways)
        sideways = sideways / n if n > 1e-9 else axis_u
        az, el = np.radians(args.azimuth), np.radians(args.elevation)
        rotated = np.cos(az) * offset + np.sin(az) * np.linalg.norm(offset) * sideways
        direction = rotated / np.linalg.norm(rotated) + np.tan(el) * normal
        direction /= np.linalg.norm(direction)
        eye = center + direction * args.distance * fit_distance(
            base, center, direction, 45.0, 1.0, 1.06)
        cam = work / f"{env}_camera.json"
        cam.write_text(json.dumps({"eye": eye.tolist(), "lookat": center.tolist()}))

        for label, kw in CONDITIONS:
            pert = CloudPerturbation(invalid_value=args.invalid_value, seed=0, **kw)
            out = pert(clouds) if pert.active else clouds
            arr = out[0, 0].numpy()
            keep = ~(arr == args.invalid_value).all(1)
            pts = arr[keep]
            moved = (np.linalg.norm(arr[keep] - raw[keep], axis=1)
                     if kw.get("noise_frac") else np.zeros(len(pts)))
            stats.append((env, label, len(pts), len(pts) / len(base),
                          float(np.median(moved)), float(np.median(moved)) / spacing, spacing))

            tag = f"{env}_{label.replace(' ', '').replace('%', '').replace('.', '')}"
            z = pts[:, 2] - pts[:, 2].min() + 1e-3
            write_ply(work / f"{tag}.ply", pts, z)
            cmd = [sys.executable, str(Path(__file__).with_name("render_ply.py")),
                   str(work / f"{tag}.ply"), "--out", str(work),
                   "--size", f"{args.size}x{args.size}",
                   "--camera-json", str(cam),
                   "--radius", str(args.radius),
                   "--radius-mode", "absolute", "--cmap", args.cmap,
                   "--cmap-range", str(args.cmap_range[0]), str(args.cmap_range[1]),
                   "--clip-percentile", "99.5"]
            if subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL).returncode == 0:
                panels[(env, label)] = work / f"{tag}.png"

    print(f"{'env':9s} {'condition':13s} {'points':>7s} {'kept':>6s} "
          f"{'moved':>8s} {'moved/spacing':>14s}")
    for env, label, n, frac, moved, ratio, spacing in stats:
        print(f"{env:9s} {label:13s} {n:7d} {frac:5.0%} {moved*100:7.2f}cm "
              f"{ratio:13.2f}x   (spacing {spacing*100:.2f}cm)")

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.compression"] = 9
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif"],
                         "font.size": args.font})

    envs = [e for e in args.env if any((e, l) in panels for l, _ in CONDITIONS)]
    if not envs:
        sys.exit("nothing rendered")
    cell = args.width / len(CONDITIONS)
    fig = plt.figure(figsize=(args.width, cell * len(envs) + 0.35))
    gs = fig.add_gridspec(len(envs), len(CONDITIONS), wspace=0.02, hspace=0.02,
                          left=0.035, right=0.998, top=0.90, bottom=0.005)
    for r, env in enumerate(envs):
        for c, (label, _) in enumerate(CONDITIONS):
            ax = fig.add_subplot(gs[r, c]); ax.set_axis_off()
            if (env, label) in panels:
                ax.imshow(plt.imread(panels[(env, label)]), interpolation="antialiased")
            if r == 0:
                ax.set_title(label, fontsize=args.font + 1, pad=4)
            if c == 0:
                ax.text(-0.02, 0.5, env, transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=args.font + 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200)
    print(f"\nwrote {args.out}.png and {args.out}.pdf")


if __name__ == "__main__":
    main()
