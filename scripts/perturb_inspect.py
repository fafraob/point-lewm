#!/usr/bin/env python3
"""Open the ablation's clouds in polyscope and look at them yourself.

    pixi run python scripts/perturb_inspect.py
    pixi run python scripts/perturb_inspect.py --env pusht --row 12345
    pixi run python scripts/perturb_inspect.py --side-by-side

Loads one frame from the table, applies every condition in the ablation sweep,
and registers each as its own point cloud. All of them sit at the SAME
coordinates with only `clean` enabled, so you tick through them in the
structure list and the scene changes under a camera that never moves -- which
is the only way to judge "is this too much" without your eye being fooled by a
reframing.

`--side-by-side` lays them out along x instead, for a single screenshot.

CloudPerturbation is imported FROM eval_lidar.py, not reimplemented, so what
you are looking at is exactly what the planner was fed.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import DATA_ROOT, DATASETS  # noqa: E402

CONDITIONS = [
    ("clean",        {}),
    ("noise_0.25%",  dict(noise_frac=0.0025)),
    ("noise_0.5%",   dict(noise_frac=0.005)),
    ("dropout_25%",  dict(dropout_prob=0.25)),
    ("dropout_50%",  dict(dropout_prob=0.5)),
]
TABLES = dict(DATASETS)   # env -> table file name under --data (paths.py)


def load_perturbation():
    import torch
    src = (Path(__file__).resolve().parents[1] / "eval_lidar.py").read_text()
    start, end = src.index("class CloudPerturbation:"), src.index("class LidarPlanAdapter")
    mod = types.ModuleType("_perturb")
    mod.__dict__["torch"] = torch
    exec(compile(src[start:end], "eval_lidar.py", "exec"), mod.__dict__)
    return mod.CloudPerturbation


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="reacher", choices=list(TABLES))
    ap.add_argument("--data", type=Path, default=Path(DATA_ROOT),
                    help="folder holding the .lance tables (default: paths.DATA_ROOT)")
    ap.add_argument("--row", type=int, default=636961)
    ap.add_argument("--invalid-value", type=float, default=-1.0)
    ap.add_argument("--radius", type=float, default=0.004,
                    help="point radius in METRES (absolute, so a thinned cloud is not "
                         "quietly drawn with fatter dots)")
    ap.add_argument("--side-by-side", action="store_true",
                    help="lay the conditions out along x instead of stacking them")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lancedb
    import torch
    CloudPerturbation = load_perturbation()

    p = (args.data / TABLES[args.env]).expanduser()
    ds = lancedb.connect(str(p.parent)).open_table(p.stem).to_lance()
    row = args.row % ds.count_rows()
    raw = np.asarray(ds.take([row], columns=["lidar"]).column("lidar")
                     .combine_chunks().flatten(), dtype=np.float32).reshape(-1, 3)
    clouds = torch.from_numpy(raw)[None, None]
    base = raw[~(raw == args.invalid_value).all(1)]

    rng = np.random.default_rng(0)
    s = base[rng.choice(len(base), min(1500, len(base)), replace=False)]
    d = np.linalg.norm(s[:, None] - s[None], axis=2)
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(1)))
    span = float(np.linalg.norm(base.max(0) - base.min(0)))
    step = 1.15 * float(base[:, 1].max() - base[:, 1].min())

    import polyscope as ps
    ps.init()
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    ps.set_SSAA_factor(2)

    print(f"{args.env}  row {row}   scan extent {span:.2f} m   "
          f"median point spacing {spacing*100:.2f} cm\n")
    print(f"{'condition':13s} {'points':>7s} {'kept':>6s} {'moved':>9s} {'vs spacing':>11s}")
    for i, (label, kw) in enumerate(CONDITIONS):
        pert = CloudPerturbation(invalid_value=args.invalid_value, seed=args.seed, **kw)
        arr = (pert(clouds) if pert.active else clouds)[0, 0].numpy()
        keep = ~(arr == args.invalid_value).all(1)
        pts = arr[keep].copy()
        moved = float(np.median(np.linalg.norm(arr[keep] - raw[keep], axis=1))) \
            if kw.get("noise_frac") else 0.0
        print(f"{label:13s} {len(pts):7d} {len(pts)/len(base):5.0%} "
              f"{moved*100:8.2f}cm {moved/spacing:10.2f}x")

        if args.side_by_side:
            pts[:, 1] += i * step
        c = ps.register_point_cloud(label, pts)
        c.set_radius(args.radius, relative=False)
        c.add_scalar_quantity("height", pts[:, 2] - pts[:, 2].min(), enabled=True, cmap="blues")
        # Stacked: only `clean` on, so the structure list is a toggle between
        # conditions under a fixed camera. Side-by-side: all on at once.
        c.set_enabled(args.side_by_side or label == "clean")

    print("\nTick the conditions on and off in the structure list on the left.")
    ps.show()


if __name__ == "__main__":
    main()
