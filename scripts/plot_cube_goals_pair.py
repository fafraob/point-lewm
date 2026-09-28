#!/usr/bin/env python3
"""Figure 5: the commanded-goal grid, target-to-latent beside the rebuilt goal cloud.

From the repository root, after the sweep
``config/eval_sweep/3dtarget_cube_grid.conf`` has run::

    pixi run python scripts/plot_cube_goals_pair.py

Identical camera, renderer, colour scale and style for both arms (the helpers of
plot_cube_start_and_goals.py), so the two panels differ only in the errors.

The sweep runs every commanded goal from five grasped starts, all commanding the
same goal set. Each sphere is one commanded goal, coloured by the MEDIAN closest
approach over the five starts; the scene is rendered at the centre start (s0).
The five start cube poses are drawn as blue boxes of the cube's own size -- a
hue the green->red error ramp never uses -- slightly translucent so the real
cube at the rendered start stays visible inside its marker.

Writes figures/goal_coverage/cube_goals_{t2l_mlp_z,goalcloud}.{png,pdf}.
Needs EGL for headless rendering; MUJOCO_GL is set below.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # scripts/ helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import RESULTS_ROOT, T2L_ROOT, dataset_path  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

ARMS = (("t2l_mlp_z", "cube_goals_t2l_mlp_z"), ("goalcloud", "cube_goals_goalcloud"))
#: start markers: blue, outside the error ramp; about goal-sphere size, so the
#: real (larger) cube marks the rendered start itself
START_RGBA = (0.16, 0.42, 0.92, 0.70)
START_HALF_M = 0.013


def per_goal_median(results, arm):
    """``(goals (n, 3) m, median error (n,) cm, n_starts (n,))`` over the starts."""
    errs = defaultdict(list)
    for f in sorted(Path(results).glob(f"{arm}_z*_s*/seed*/grid_results.json")):
        for c in json.loads(f.read_text())["cases"]:
            errs[tuple(np.round(c["goal"], 4))].append(c["closest_distance"] * 100.0)
    if not errs:
        raise SystemExit(f"no {arm} results under {results}")
    keys = sorted(errs)
    return (np.array(keys), np.array([np.median(errs[k]) for k in keys]),
            np.array([len(errs[k]) for k in keys]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="default: $PLWM_RESULTS_ROOT/3dtarget_cube_grid")
    ap.add_argument("--table", type=Path, default=Path(dataset_path("cube")))
    ap.add_argument("--grid", type=Path, default=None,
                    help="default: $PLWM_T2L_ROOT/grid, where cube_grid_goals.py writes")
    ap.add_argument("--start", type=int, default=0, help="which start's scene to render")
    ap.add_argument("--vmax", type=float, default=4.0, help="cm: the 4 cm success threshold")
    ap.add_argument("--render-size", type=int, default=1800)
    ap.add_argument("--out", type=Path, default=Path("figures/goal_coverage"))
    args = ap.parse_args()
    results = args.results or Path(RESULTS_ROOT) / "3dtarget_cube_grid"

    import plot_cube_start_and_goals as M
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/eval/3dtarget_cube_grid.yaml")
    grid = args.grid or Path(T2L_ROOT) / "grid"
    start = json.loads((grid / f"cube_grid_goals_s{args.start}.json").read_text())["meta"]["start"]
    boxes = []
    for f in sorted(grid.glob("cube_grid_goals_s[0-9].json")):
        boxes.append((json.loads(f.read_text())["meta"]["start"]["cube_pos"], START_RGBA,
                      START_HALF_M))
    print(f"[starts] {len(boxes)} start poses drawn")
    args.out.mkdir(parents=True, exist_ok=True)
    data = {arm: per_goal_median(results, arm) for arm, _ in ARMS}
    # one colour bar for both panels: overflow arrow on both or on neither
    extend = "max" if max(d[1].max() for d in data.values()) > args.vmax else "neither"
    for arm, name in ARMS:
        goals, err, n = data[arm]
        ok = err < args.vmax
        print(f"[{arm}] {len(goals)} goals x {n.min()}-{n.max()} starts: median {np.median(err):.2f} cm, "
              f"{(~ok).sum()} goals with median error over {args.vmax:.0f} cm")
        markers = M.build_markers(goals, err, ok, args.vmax)
        img = M.render_start(cfg, args.table, start["episode"],
                             start["step"], args.render_size, markers=markers, style="solid",
                             neutral_cube=False, boxes=boxes)
        M.plot(args.out / name, img, err, ok, args.vmax, arm, extend=extend)
        print(f"  wrote {args.out / name}.png/.pdf")


if __name__ == "__main__":
    main()
