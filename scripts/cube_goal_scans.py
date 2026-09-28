#!/usr/bin/env python3
"""Rebuild the goal observation for a commanded cube position -- and check it.

The goal-cloud arm of the commanded-goal grid (App. D.2.1, Table 11) needs a
goal observation for a pose no recorded frame contains. This script renders one
and, before using it, measures whether it is a fair one. From the repository
root::

    pixi run python scripts/cube_goal_scans.py --validate-only   # just the checks
    pixi run python scripts/cube_goal_scans.py                   # checks, then write

WHY THIS IS HARDER THAN IT LOOKS. A cube commanded to 16 cm above the table
cannot float -- the gripper must hold it -- and the commanded goal says nothing
about HOW. The grasp yaw is free (measured over the training split: frames whose
cube sits within 1 cm of each other carry gripper yaws spanning the full
-pi..pi), so "the goal cloud for a cube at p" is a FAMILY of observations, not
one. A rebuilt baseline has to pick a member of that family, and the pick is
ours, not the task's.

That makes fidelity the first question, before any comparison is run: is a
rebuilt member a plausible member? Two tests, both ray by ray against recorded
frames (fixed ray grid, same sensor, so ray i corresponds to ray i -- exact, no
matching):

  A. SCANNER PARITY. Restore the frame's own ``qpos``/``qvel`` and re-scan. This
     isolates the scan path from the rebuild; it must come back at ~0.

  B. REBUILD FIDELITY. Throw away everything except what a commanded goal
     carries -- the cube position -- plus the one free choice the rebuild makes
     (the grasp yaw), re-derive the arm by IK, close the gripper on the cube,
     and scan. Compare to the recorded cloud of that frame.

B can never be 0: the recorded arm has a particular elbow branch and the cube a
particular orientation, and a rebuild knows neither. The criterion is therefore
RELATIVE, and the script measures both sides of it:

    is the rebuild further from a recorded observation of this pose than two
    recorded observations of the same pose are from each other?

If it lands inside that natural spread, the rebuild is just another member of
the family and a baseline built on it is fair. If it lands well outside, the
rebuild is an outlier -- the encoder would be seeing a scene the data never
contains -- and a baseline built on it would lose for reasons that have nothing
to do with goal representation. The script then REFUSES to write.

OUTPUT (``--validate-only``): the numbers above, and
``cube_goal_scans_validation.json`` next to the grid files. Without the flag it
also writes ``cube_goal_scans.npz`` -- ``clouds`` (n, n_rays*3) flat like the
dataset column, ``goals`` (n, 3), ``block``, ``yaw``, ``cube_returns`` -- one
rebuilt cloud per commanded goal, which ``eval.grid_scans`` reads.

Run ``scripts/cube_grid_goals.py`` first; check both with
``pytest tests/test_cube_grid.py``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import REPO_ROOT, T2L_ROOT, dataset_path  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")


def _spec():
    from target2latent.envs import SPECS

    return SPECS["cube"]


def build_env(cfg):
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from stable_worldmodel.wrapper import make_lidar_pre_wrappers

    sensor = OmegaConf.to_container(cfg.lidar.sensor, resolve=True)
    world = swm.World(env_name=cfg.world.env_name, num_envs=1, max_episode_steps=10,
                      image_shape=(int(cfg.eval.img_size), int(cfg.eval.img_size)),
                      pre_wrappers=make_lidar_pre_wrappers(sensor))
    world.reset(seed=0)
    env = world.envs.envs[0]
    lidar = env
    while lidar is not None and not hasattr(lidar, "_get_point_cloud"):
        lidar = getattr(lidar, "env", None)
    if lidar is None:
        raise SystemExit("no lidar wrapper in the env stack")
    return env.unwrapped, lidar


def scan(lidar):
    return np.asarray(lidar._get_point_cloud()[0], dtype=np.float32)


def restore(base, qpos, qvel):
    """Test A: the frame's own full state."""
    import mujoco

    base._data.qpos[:] = np.asarray(qpos, dtype=float)
    base._data.qvel[:] = np.asarray(qvel, dtype=float)
    mujoco.mj_forward(base._model, base._data)


def gripper_slice(base):
    """The gripper's qpos block: everything between the arm and the cube joint.

    The Robotiq 2F-85 is an 8-joint linkage (driver, coupler, spring link and
    follower per side) and ``mj_forward`` does NOT project qpos onto the
    equality constraints that couple them, so setting the driver joint alone
    leaves the other seven wherever they happened to be. A rebuild must write
    the whole block or it inherits the previous scene's fingers.
    """
    lo = int(np.max(base._arm_joint_ids)) + 1
    hi = int(base._model.jnt_qposadr[base._model.joint("object_joint_0").id])
    return slice(lo, hi)


def neutral(base):
    """Wipe the scene to the home pose: nothing may leak into a rebuild."""
    import mujoco

    base._data.qpos[:] = 0.0
    base._data.qvel[:] = 0.0
    base._data.qpos[base._arm_joint_ids] = base._home_qpos
    base._data.joint("object_joint_0").qpos[3:] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(base._model, base._data)


def rebuild(base, cube_pos, yaw, grip_offset, grip_qpos):
    """Test B: the scene a commanded goal + one grasp yaw implies.

    The pinch site is driven to the cube centre plus the grasp offset measured
    on held frames, the effector spun to ``yaw`` about the down-rotation, the
    arm solved by IK from the home pose, the cube placed at the commanded point
    with its yaw aligned to the gripper, and the whole gripper linkage set to
    the configuration it holds the cube in throughout the data.
    """
    import mujoco
    from ogbench.manipspace import lie

    neutral(base)
    eff_ori = lie.SO3.from_z_radians(float(yaw)) @ base._effector_down_rotation
    eff_pos = np.asarray(cube_pos, dtype=float) + grip_offset
    T_wa = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos) @ base._T_pa
    qpos_arm = base._ik.solve(pos=T_wa.translation(), quat=T_wa.rotation().wxyz,
                              curr_qpos=base._home_qpos)
    base._data.qpos[base._arm_joint_ids] = qpos_arm
    base._data.qpos[gripper_slice(base)] = grip_qpos
    q = base._data.joint("object_joint_0").qpos
    q[:3] = np.asarray(cube_pos, dtype=float)
    q[3:] = lie.SO3.from_z_radians(float(yaw)).wxyz
    mujoco.mj_forward(base._model, base._data)
    return qpos_arm


def ray_stats(a, b, miss_value=-1.0):
    """Ray-by-ray comparison of two clouds on the same ray grid."""
    ma = np.all(a == miss_value, axis=-1)
    mb = np.all(b == miss_value, axis=-1)
    both = ~ma & ~mb
    d = np.linalg.norm(a[both] - b[both], axis=-1)
    return dict(median_cm=float(np.median(d) * 100), mean_cm=float(np.mean(d) * 100),
                p95_cm=float(np.percentile(d, 95) * 100),
                rays_moved_2cm=int((d > 0.02).sum()),
                frac_moved_2cm=float((d > 0.02).mean()),
                hit_miss_disagree=float((ma != mb).mean()))


def load_frames(table, eval_dir, n, height_min, rng):
    """Held eval frames, plus for each one a sibling frame at the same cube pose."""
    import lance
    from scipy.spatial import cKDTree

    ds = lance.dataset(str(table))
    tb = ds.to_table(columns=["episode_idx", "step_idx", "privileged/block_0_pos",
                              "proprio/gripper_contact", "proprio/effector_pos",
                              "proprio/effector_yaw", "proprio/gripper_opening"])
    ep = tb.column("episode_idx").to_numpy()
    step = tb.column("step_idx").to_numpy()
    col = lambda k: np.stack(tb.column(k).to_numpy(zero_copy_only=False)).astype(float)
    pos, eff = col("privileged/block_0_pos"), col("proprio/effector_pos")
    grip = col("proprio/gripper_contact").reshape(-1)
    yaw = col("proprio/effector_yaw").reshape(-1)
    opening = col("proprio/gripper_opening").reshape(-1)
    files = sorted(Path(eval_dir).glob("starts_cube_seed*.json"))
    if not files:
        raise SystemExit(f"no starts_cube_seed*.json in {eval_dir}")
    eval_eps = sorted({e for p in files for e in json.load(open(p))["episodes"]})

    held = (grip > 0.5) & (np.linalg.norm(pos - eff, axis=1) < 0.05)
    # the rebuild's constants, measured on held frames (a rebuild knows no more):
    # where the effector sits relative to the cube it holds, and the pose the
    # gripper linkage settles into around it
    grip_offset = np.median(eff[held] - pos[held], axis=0)
    med_opening = float(np.median(opening[held]))
    sample = rng.choice(np.flatnonzero(held), size=min(512, int(held.sum())), replace=False)
    grip_qpos = np.median(np.stack(ds.take(sorted(sample.tolist()), columns=["qpos"])
                                   .column("qpos").to_numpy(zero_copy_only=False)), axis=0)

    ok = held & (pos[:, 2] > height_min)
    rows = np.flatnonzero(ok)
    # pick eval frames, each with a same-pose sibling from another episode: the
    # sibling gives the natural spread the rebuild is judged against
    tree = cKDTree(pos[rows])
    picks = []
    for r in rng.permutation(rows[np.isin(ep[rows], eval_eps)]):
        nb = rows[tree.query_ball_point(pos[r], r=0.01)]
        other = nb[ep[nb] != ep[r]]
        if len(other):
            picks.append((int(r), int(other[0])))
        if len(picks) >= n:
            break
    if len(picks) < n:
        raise SystemExit(f"only {len(picks)} held eval frames have a same-pose sibling")
    return dict(picks=picks, pos=pos, yaw=yaw, opening=opening, ep=ep, step=step,
                held=held, grip_offset=grip_offset, med_opening=med_opening,
                grip_qpos_full=grip_qpos, table=ds)


def grasp_yaws(F, goals, spec_train_only=True):
    """Per commanded goal, the grasp yaw the data actually uses nearest to it.

    The one free parameter of a rebuilt cube goal is the yaw of the hand holding
    it. Sampling it uniformly would put some goal clouds in poses the encoder
    never saw; borrowing the yaw of the nearest held frame keeps every rebuilt
    observation inside the distribution, which is the choice most favourable to
    the goal-cloud baseline and therefore the honest one to compare against.
    """
    from scipy.spatial import cKDTree

    rows = np.flatnonzero(F["held"])
    _, j = cKDTree(F["pos"][rows]).query(np.asarray(goals, dtype=float))
    src = rows[j]
    return F["yaw"][src], np.linalg.norm(F["pos"][src] - goals, axis=1)


def build_all(base, lidar, F, grip_qpos, args, out_dir, miss, validation):
    """One rebuilt goal cloud per commanded goal, keyed by (block, goal order).

    ``eval_3dtarget.load_grid_block`` selects a layer's clouds with
    ``flatnonzero(block == b)``, so rows must be written layer by layer in the
    grid file's own goal order, and the goals are re-checked against that file
    on load.
    """
    spec = json.loads(Path(args.goals).read_text())
    all_goals = np.concatenate([np.asarray(b["goals"], dtype=float) for b in spec["blocks"]])
    yaw, yaw_src_cm = grasp_yaws(F, all_goals)
    print(f"\n[build] grasp yaw per goal taken from the nearest held frame "
          f"({np.median(yaw_src_cm) * 100:.1f} cm away, max {yaw_src_cm.max() * 100:.1f} cm)")

    clouds, block_id, goals_out, yaw_out, seen = [], [], [], [], []
    k = 0
    for bi, b in enumerate(spec["blocks"]):
        for g in b["goals"]:
            rebuild(base, np.asarray(g, dtype=float), yaw[k], F["grip_offset"], grip_qpos)
            c = scan(lidar)
            # the goal cloud must actually SHOW the commanded cube, or the
            # baseline is being handed an observation of nothing
            d = np.linalg.norm(c - _spec().world_to_sensor(np.asarray([g]))[0], axis=-1)
            d[np.all(c == miss, axis=-1)] = np.inf
            seen.append(int((d < 0.045).sum()))
            clouds.append(c.reshape(-1))
            block_id.append(bi)
            goals_out.append(g)
            yaw_out.append(float(yaw[k]))
            k += 1
        print(f"  layer {bi} z={b['height_m'] * 100:.0f} cm: {b['n_goals']} goals, "
              f"cube returns median {np.median(seen[-b['n_goals']:]):.0f}")
    clouds = np.stack(clouds).astype(np.float32)
    seen = np.asarray(seen)
    # Goals the sensor cannot depict. NOT an error: the lidar has a finite field
    # of view, so a cube commanded to the far corners at height falls outside it
    # and the rebuilt observation -- the best any goal-cloud method could ever
    # obtain, oracle included -- shows the scene WITHOUT the thing that was
    # asked for. A typed goal names the same pose regardless. The analysis
    # stratifies on this, so it is recorded per goal rather than rejected.
    blind = np.flatnonzero(seen == 0)
    if len(blind):
        g = np.asarray(goals_out)[blind]
        print(f"\n[build] {len(blind)} of {len(seen)} goals lie outside the sensor's view: "
              f"|y| >= {np.abs(g[:, 1]).min() * 100:.0f} cm, z >= {g[:, 2].min() * 100:.0f} cm. "
              "Their rebuilt goal cloud cannot show the cube at all.")
        for gg in g:
            print(f"    {np.round(gg, 3).tolist()}")

    meta = dict(env="cube", eval_config=args.eval_config, goals_file=str(args.goals),
                miss_value=miss, n=len(clouds), validation=validation,
                grip_offset_m=F["grip_offset"].tolist(), gripper_qpos=grip_qpos.tolist(),
                yaw_source="nearest held frame in cube-position space",
                cube_returns=dict(median=int(np.median(seen)), min=int(seen.min()),
                                  max=int(seen.max())),
                goals_outside_sensor_view=dict(
                    n=int(len(blind)), idx=blind.tolist(),
                    goals=np.asarray(goals_out)[blind].tolist(),
                    note=("the lidar's field of view does not reach these commanded poses, so "
                          "no goal observation of them exists -- rebuilt, retrieved or recorded")),
                note=("clouds[i] is the scan of goals[i]; flat (n_rays*3) like the lidar column. "
                      "Rows are ordered block by block in the grid file's goal order, which is "
                      "what eval_3dtarget.load_grid_block assumes."))
    stem = out_dir / "cube_goal_scans"
    np.savez_compressed(stem.with_suffix(".npz"), clouds=clouds,
                        block=np.asarray(block_id, dtype=int),
                        goals=np.asarray(goals_out, dtype=float),
                        yaw=np.asarray(yaw_out, dtype=float),
                        cube_returns=seen, meta=json.dumps(meta))
    print(f"[build] {len(clouds)} goal clouds, {clouds.shape[1] // 3} rays each, "
          f"cube returns median {np.median(seen):.0f} (min {seen.min()}); "
          f"{np.mean(clouds.reshape(len(clouds), -1, 3) == miss):.1%} miss components")
    print(f"wrote {stem}.npz ({stem.with_suffix('.npz').stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=Path(dataset_path("cube")),
                    help="the OGB-Cube LiDAR table (docs/data.md)")
    ap.add_argument("--eval-dir", type=Path, default=REPO_ROOT / "eval_starts")
    ap.add_argument("--eval-config", default="3dtarget_cube_grid")
    ap.add_argument("--validate-frames", type=int, default=24)
    ap.add_argument("--height-min", type=float, default=0.10,
                    help="m: validate on frames where the cube is carried, the regime under test")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--goals", type=Path, default=None,
                    help="grid file whose goals to rebuild (default: cube_grid_goals_s0.json; "
                         "every start file carries the same goal set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: $PLWM_T2L_ROOT/grid, where cube_grid_goals.py writes")
    args = ap.parse_args()
    eval_dir = args.eval_dir
    out_dir = args.out or Path(T2L_ROOT) / "grid"
    args.goals = args.goals or out_dir / "cube_grid_goals_s0.json"
    rng = np.random.default_rng(args.seed)

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path("config/eval") / f"{args.eval_config}.yaml")
    miss = float(cfg.lidar.invalid_value)
    F = load_frames(args.table, eval_dir, args.validate_frames,
                    args.height_min, rng)
    print(f"[rebuild] grasp offset (effector - cube), median over held frames: "
          f"{np.round(F['grip_offset'] * 100, 2)} cm; gripper opening {F['med_opening']:.3f}")

    base, lidar = build_env(cfg)
    n_rays = len(scan(lidar))
    gsl = gripper_slice(base)
    grip_qpos = F["grip_qpos_full"][gsl]
    print(f"[rebuild] gripper qpos block {gsl.start}:{gsl.stop} "
          f"({gsl.stop - gsl.start} joints), median over held frames "
          f"{np.round(grip_qpos, 3)}")

    rows = sorted({r for pair in F["picks"] for r in pair})
    t = F["table"].take(rows, columns=["lidar", "qpos", "qvel"])
    rec = np.stack(t.column("lidar").to_numpy(zero_copy_only=False)).reshape(len(rows), -1, 3)
    qpos = np.stack(t.column("qpos").to_numpy(zero_copy_only=False)).astype(float)
    qvel = np.stack(t.column("qvel").to_numpy(zero_copy_only=False)).astype(float)
    at = {r: i for i, r in enumerate(rows)}
    if rec.shape[1] != n_rays:
        raise SystemExit(f"sensor mismatch: env casts {n_rays} rays, table stores {rec.shape[1]}")

    A, B, S, sane = [], [], [], []
    for r, sib in F["picks"]:
        i, j = at[r], at[sib]
        restore(base, qpos[i], qvel[i])
        A.append(ray_stats(rec[i], scan(lidar), miss))
        q_true = base._data.qpos[base._arm_joint_ids].copy()
        q_arm = rebuild(base, F["pos"][r], F["yaw"][r], F["grip_offset"], grip_qpos)
        B.append(ray_stats(rec[i], scan(lidar), miss))
        S.append(ray_stats(rec[i], rec[j], miss))
        # the rebuild is only meaningful if the IK actually placed the arm: the
        # solved joints must differ from home and the cube must end up where
        # it was commanded
        sane.append(dict(
            joint_err_to_recorded=float(np.abs(q_arm - q_true).max()),
            joint_move_from_home=float(np.abs(q_arm - base._home_qpos).max()),
            cube_placement_cm=float(np.linalg.norm(
                base._data.joint("object_joint_0").qpos[:3] - F["pos"][r]) * 100)))

    def agg(rows_, key):
        return float(np.median([x[key] for x in rows_]))

    print(f"\n{len(F['picks'])} carried eval frames (cube above {args.height_min * 100:.0f} cm), "
          f"{n_rays} rays each. Medians over frames:\n")
    print(f"{'':34s}{'per-ray':>10s}{'p95':>9s}{'rays > 2 cm':>13s}{'hit/miss':>10s}")
    for name, rows_ in (("A  re-scan of the frame's own state", A),
                        ("B  IK rebuild from (cube pos, yaw)", B),
                        ("   two recorded frames, same pose", S)):
        print(f"{name:34s}{agg(rows_, 'median_cm'):9.2f}cm{agg(rows_, 'p95_cm'):8.1f}cm"
              f"{agg(rows_, 'rays_moved_2cm'):13.0f}{agg(rows_, 'hit_miss_disagree'):9.1%}")

    print(f"\n[sanity] IK solved joints move {agg(sane, 'joint_move_from_home'):.2f} rad from home "
          f"and differ from the recorded arm by {agg(sane, 'joint_err_to_recorded'):.2f} rad; "
          f"cube lands {agg(sane, 'cube_placement_cm'):.3f} cm off the commanded point")

    ratio = agg(B, "rays_moved_2cm") / max(agg(S, "rays_moved_2cm"), 1)
    verdict = ("FAIR: the rebuild sits inside the spread of recorded observations of the "
               "same pose" if ratio <= 1.5 else
               "NOT FAIR: the rebuild is an outlier -- a baseline built on it would be "
               "handicapped by the rendering, not by the representation")
    print(f"\nB/spread ratio on rays moved > 2 cm: {ratio:.2f}  -> {verdict}")

    out = dict(env="cube", eval_config=args.eval_config, n_frames=len(F["picks"]),
               n_rays=n_rays, height_min_m=args.height_min,
               grip_offset_m=F["grip_offset"].tolist(), gripper_opening=F["med_opening"],
               gripper_qpos=grip_qpos.tolist(),
               scanner_parity=A, rebuild_vs_recorded=B, recorded_vs_recorded=S, sanity=sane,
               ratio_rays_moved=ratio, verdict=verdict)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "cube_goal_scans_validation.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")

    if args.validate_only:
        return
    if ratio > 1.5:
        raise SystemExit("refusing to build: the rebuild is an outlier against recorded "
                         "observations of the same pose, so the baseline would be handicapped "
                         "by the rendering. Fix the rebuild, do not ship this.")
    build_all(base, lidar, F, grip_qpos, args, out_dir, miss, out)


if __name__ == "__main__":
    main()
