#!/usr/bin/env python3
"""Commanded 3-D cube goals on height layers, from five designed grasped starts.

App. D.2.1 of the paper (Figure 5, Table 11) asks whether planning toward a
typed 3-D target reaches goals the expert never aimed at. This script builds
that goal set. From the repository root::

    pixi run python scripts/cube_grid_goals.py
    pixi run python scripts/cube_grid_goals.py --heights 0.02 0.08 0.16 0.24 0.32

WHY OGB-CUBE. The cube TRAVELS through 3-D while carried, but every episode's
GOAL is a placement on the table -- ``privileged/target_block_pos`` is z = 0.02 m
in all 10,000 episodes, inside x [0.30, 0.55], y [-0.30, 0.30]. "Hold the cube
16 cm above the table" is therefore a goal the expert never once aimed at, while
remaining reachable and inside the world model's experience. The other three
environments have no such gap: their experts drove the agent everywhere, so a
near-exact goal observation exists for every commandable pose and the comparison
could only show parity.

WHAT IS *NOT* TESTABLE. The arm cannot lift the cube above 0.35 m: the
environment clips the effector target to ``_workspace_bounds`` =
[[0.25, -0.35, 0.02], [0.60, 0.35, 0.35]] on every step, which is exactly where
the data stops (max 0.348). Goals above that are unreachable for every method,
oracle included, so they measure impossibility rather than generalization and
are deliberately not part of this grid.

THE SET
  start    frames with the cube already in the gripper, from the pinned
           evaluation episodes of ``eval_starts/`` (never in target2latent's
           training). Within one grid file only the goal varies, so the error
           map is attributable to the goal. ``--n-starts`` writes one file per
           start (``_s0`` ... ``_s4``) laid out as a 2x2 FACTORIAL in (x, y)
           plus a centre point: the corners of the placement box and its middle,
           each from a different episode, with the cube's start height held
           inside ``--start-height`` so height belongs to the goal layer and not
           to the start. Repeating the whole grid over the five is the
           replication axis available without retraining a head -- it shows the
           ranking does not hinge on one start pose, and a start effect reads as
           a direction across the table rather than as five unrelated numbers.
           The goal set is IDENTICAL across the files (goals within
           ``--min-goal-dist`` of ANY start are dropped), so every arm and every
           start is paired goal by goal.
  goals    a lattice over the expert's placement box at each ``--heights``
           layer. Layer z = 0.02 is the in-distribution control; the rest are
           the never-commanded regime.
  retrieval  per goal, the training frame whose cube is nearest in 3-D -- the
           goal cloud a retrieval baseline would encode, and the distance is how
           far off the commanded position that observation is (it also drags in
           whatever the arm happened to be doing).

OUTPUT (one ``cube_grid_goals_s<k>.json`` per start, under
``$PLWM_T2L_ROOT/grid`` by default): one block per height layer, each with
``goals`` (3-D, metres), that file's start frame, per-goal ``retrieval``
{episode, step, pos, dist_m} and the start->goal distance. Distances are METRES
here and reported in centimetres everywhere a human reads them.

Then rebuild the goal-cloud baseline with ``scripts/cube_goal_scans.py``, check
both with ``pytest tests/test_cube_grid.py``, and run the sweep
``config/eval_sweep/3dtarget_cube_grid.conf``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import REPO_ROOT, T2L_ROOT, dataset_path  # noqa: E402

#: the environment clips the effector target to this box on every step
WORKSPACE = np.array([[0.25, -0.35, 0.02], [0.60, 0.35, 0.35]])
#: the environment scores a cube placement as solved at 4 cm
SUCCESS_M = 0.04


def load(table, eval_dir):
    import lance

    ds = lance.dataset(str(table))
    tb = ds.to_table(columns=["episode_idx", "step_idx", "privileged/block_0_pos",
                              "proprio/gripper_contact", "proprio/effector_pos"])
    ep = tb.column("episode_idx").to_numpy()
    step = tb.column("step_idx").to_numpy()
    pos = np.stack(tb.column("privileged/block_0_pos").to_numpy(zero_copy_only=False)).astype(float)
    grip = np.stack(tb.column("proprio/gripper_contact").to_numpy(zero_copy_only=False)).reshape(-1)
    eff = np.stack(tb.column("proprio/effector_pos").to_numpy(zero_copy_only=False)).astype(float)
    files = sorted(Path(eval_dir).glob("starts_cube_seed*.json"))
    if not files:
        raise SystemExit(f"no starts_cube_seed*.json in {eval_dir}: the split would be wrong")
    eval_eps = sorted({e for p in files for e in json.load(open(p))["episodes"]})
    train = ~np.isin(ep, eval_eps) & (ep % 10 != 9)
    return ep, step, pos, grip, eff, train, eval_eps, len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=Path(dataset_path("cube")),
                    help="the OGB-Cube LiDAR table (docs/data.md)")
    ap.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "eval_starts",
                    help="the pinned evaluation start lists; their episodes are the "
                         "split this script draws its starts from")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.02, 0.08, 0.16, 0.24, 0.32],
                    help="metres above the table; 0.02 is the expert's placement height")
    ap.add_argument("--x", type=float, nargs=3, default=[0.30, 0.55, 0.05],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--y", type=float, nargs=3, default=[-0.30, 0.30, 0.10],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--goal-offset", type=int, default=25,
                    help="the eval slices start..start+offset, so the start needs that much room")
    ap.add_argument("--start-height", type=float, nargs=2, default=[0.09, 0.14],
                    metavar=("MIN", "MAX"), help="cube height at the start, while grasped")
    ap.add_argument("--min-goal-dist", type=float, default=0.05,
                    help="metres: drop goals closer than this to ANY start (success is 4 cm)")
    ap.add_argument("--n-starts", type=int, default=5,
                    help="grid files to write, one per designed start; start 0 is the centre one")
    ap.add_argument("--start-x", type=float, nargs=2, default=[0.33, 0.52], metavar=("LO", "HI"),
                    help="the factorial's x levels for the start (inside the placement box)")
    ap.add_argument("--start-y", type=float, nargs=2, default=[-0.22, 0.22], metavar=("LO", "HI"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: $PLWM_T2L_ROOT/grid")
    args = ap.parse_args()
    eval_dir = args.eval_dir
    out_dir = args.out or Path(T2L_ROOT) / "grid"

    from scipy.spatial import cKDTree

    ep, step, pos, grip, eff, train, eval_eps, n_files = load(args.table, eval_dir)
    centre = np.array([(WORKSPACE[0, 0] + WORKSPACE[1, 0]) / 2, 0.0])

    # ---- the starts: a 2x2 factorial in (x, y) plus a centre point
    #
    # Not farthest-point sampling and not "whatever the data offers": the five
    # starts are a DESIGN. Four sit at the corners of the placement box and one
    # at its centre, which spans the horizontal extent of the workspace with the
    # fewest runs and lets a start effect be read as a direction (does commanding
    # across the table behave differently from commanding within a corner?)
    # rather than as five unrelated numbers. The cube's start HEIGHT is held
    # inside --start-height for all five, so height stays a property of the GOAL
    # layer and is not confounded with the start. Each start comes from a
    # different eval episode.
    ep_len = np.bincount(ep)
    held = (grip > 0.5) & (np.linalg.norm(pos - eff, axis=1) < 0.05)
    ok = (held & np.isin(ep, eval_eps) & (step + args.goal_offset < ep_len[ep])
          & (pos[:, 2] >= args.start_height[0]) & (pos[:, 2] <= args.start_height[1]))
    if not ok.any():
        raise SystemExit("no grasped eval frame matches --start-height; widen it")
    cand = np.flatnonzero(ok)
    design = [("centre", centre)]
    for sx in (args.start_x[0], args.start_x[1]):
        for sy in (args.start_y[0], args.start_y[1]):
            design.append((f"x{'lo' if sx == args.start_x[0] else 'hi'}"
                           f"_y{'lo' if sy == args.start_y[0] else 'hi'}", np.array([sx, sy])))
    design = design[: args.n_starts]
    rows, starts = [], []
    for name, tgt in design:
        free = cand[~np.isin(ep[cand], ep[rows])] if rows else cand
        if not len(free):
            raise SystemExit(f"ran out of distinct grasped eval episodes at design point {name}")
        r = int(free[np.argmin(np.linalg.norm(pos[free, :2] - tgt, axis=1))])
        rows.append(r)
        starts.append(dict(design_point=name, design_target_xy=[round(float(v), 3) for v in tgt],
                           episode=int(ep[r]), step=int(step[r]),
                           cube_pos=[round(float(v), 4) for v in pos[r]],
                           effector_pos=[round(float(v), 4) for v in eff[r]],
                           offset_from_design_cm=round(
                               float(np.linalg.norm(pos[r, :2] - tgt) * 100), 2),
                           grasped=True))

    # ---- goals: a lattice per height layer
    gx = np.arange(args.x[0], args.x[1] + 1e-9, args.x[2])
    gy = np.arange(args.y[0], args.y[1] + 1e-9, args.y[2])
    tr_rows = np.flatnonzero(train)
    tree = cKDTree(pos[train])

    # one lattice, shared by every start: a goal too close to ANY start would be
    # trivial for that start and the files would no longer be goal-by-goal paired
    layers = []
    start_pos = np.stack([pos[r] for r in rows])
    for z in args.heights:
        pts = np.array([[x, y, z] for x in gx for y in gy])
        keep = (np.linalg.norm(pts[:, None] - start_pos[None], axis=-1)
                >= args.min_goal_dist).all(1)
        pts = pts[keep]
        inside = ((pts >= WORKSPACE[0]).all(1) & (pts <= WORKSPACE[1]).all(1))
        assert inside.all(), "a lattice point left the arm's workspace"
        d_ret, j = tree.query(pts)
        layers.append((z, pts, d_ret, tr_rows[j]))

    out_dir.mkdir(parents=True, exist_ok=True)
    for k, (start, r) in enumerate(zip(starts, rows)):
        blocks = []
        for z, pts, d_ret, ret_rows in layers:
            blocks.append(dict(
                height_m=round(float(z), 3),
                regime=("in-distribution: the expert's placement height"
                        if abs(z - 0.02) < 1e-6 else
                        "never commanded: the expert never aimed here"),
                n_goals=len(pts),
                goals=[[round(float(v), 4) for v in p] for p in pts],
                episodes=[start["episode"]] * len(pts),
                start_steps=[start["step"]] * len(pts),
                start_to_goal_m=[round(float(v), 4) for v in np.linalg.norm(pts - pos[r], axis=1)],
                retrieval=[dict(episode=int(ep[q]), step=int(step[q]),
                                pos=[round(float(v), 4) for v in pos[q]], dist_m=round(float(d), 4))
                           for q, d in zip(ret_rows, d_ret)],
                retrieval_dist_m=dict(median=round(float(np.median(d_ret)), 4),
                                      max=round(float(d_ret.max()), 4)),
            ))
        meta = dict(
            env="cube", table=str(args.table), units="metres",
            success_m=SUCCESS_M, workspace=WORKSPACE.tolist(),
            heights_m=[b["height_m"] for b in blocks],
            n_goals_per_layer=[b["n_goals"] for b in blocks],
            goal_offset_steps=args.goal_offset, seed=args.seed,
            start_index=k, n_starts=len(starts),
            all_starts=[s["cube_pos"] for s in starts],
            split=(f"start from the {len(eval_eps)} eval episodes of {n_files} "
                   f"starts_cube_seed*.json "
                   f"files; retrieval searches the train split only"),
            why=("every dataset target is z=0.02 in x[0.30,0.55] y[-0.30,0.30], so every layer "
                 "above that is a goal regime the expert never aimed at; the arm cannot exceed "
                 "z=0.35 (workspace clamp), so nothing above it is testable"),
            start=start,
            episodes_per_arm=int(sum(b["n_goals"] for b in blocks)),
        )
        stem = out_dir / f"cube_grid_goals_s{k}"
        stem.with_suffix(".json").write_text(json.dumps(dict(meta=meta, blocks=blocks), indent=2)
                                             + "\n")
        print(f"start {k} {start['design_point']:>9s}: cube {start['cube_pos']} "
              f"({start['offset_from_design_cm']:.1f} cm off the design point, "
              f"episode {start['episode']}, step {start['step']}) -> {stem.name}.json")
    for i, b in enumerate(blocks):
        print(f"  layer {i} z={b['height_m'] * 100:.0f} cm: {b['n_goals']:3d} goals, "
              f"nearest recorded cube pose {b['retrieval_dist_m']['median'] * 100:.1f} cm "
              f"(max {b['retrieval_dist_m']['max'] * 100:.1f} cm)")
    print(f"{sum(b['n_goals'] for b in blocks)} goals per arm per start, "
          f"{len(starts)} starts -> {out_dir}")


if __name__ == "__main__":
    main()
