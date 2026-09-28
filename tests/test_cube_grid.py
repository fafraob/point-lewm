"""The commanded-goal grid: its five starts, its goals, its rebuilt clouds.

The sweep these tests guard (App. D.2.1, Figure 5, Table 11) compares two goal
sources on identical commanded goals -- the target-to-latent head (MLP + z_now)
against a REBUILT goal cloud (the environment set to the commanded pose and
re-scanned, scripts/cube_goal_scans.py). Both arms are run from five designed
grasped starts. That design only means anything if the artefacts on disk
actually hold to it, which is what these tests check:

  * the grid config stays a copy of the standard OGB-Cube 3D-target config, so
    grid numbers remain comparable to the ordinary sweep;
  * the five start files carry ONE identical goal set, five distinct episodes
    and a start height inside the band, so a start effect is a start effect and
    not a goal-set or height difference;
  * the rebuilt clouds line up with the goals layer by layer in the order
    ``eval_3dtarget.load_grid_block`` selects them -- a silent misalignment here
    would hand every goal the wrong observation and still run.

Everything that needs the generated artefacts skips when they are absent, so
the config guards run in a fresh clone. Generate them with
scripts/cube_grid_goals.py and scripts/cube_goal_scans.py.

Run with:  pixi run python -m pytest tests/test_cube_grid.py -q
"""

import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "eval"
BASE, GRID = "3dtarget_cube", "3dtarget_cube_grid"
#: blocks that MUST be identical: they define what is being evaluated and how
SHARED_BLOCKS = ("world", "lidar", "plan_config", "dataset", "solver", "target_goal")
#: eval keys the grid deliberately changes; everything else in `eval` is shared
GRID_ONLY = {"num_eval", "eval_budget", "grid_file", "grid_scans", "grid_block",
             "grid_limit", "grid_no_terminate", "grid_success", "grid_retrieval",
             "grid_trace"}

N_STARTS = 5
#: scripts/cube_grid_goals.py --start-height
START_HEIGHT_M = (0.09, 0.14)
#: scripts/cube_grid_goals.py --min-goal-dist
MIN_GOAL_DIST_M = 0.05


def load(name, **overrides):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=name, overrides=[f"{k}={v}" for k, v in overrides.items()])


@pytest.fixture(scope="module")
def cfgs():
    common = dict(policy="x", run="y")
    return load(BASE, **common), load(GRID, **common)


@pytest.fixture(scope="module")
def grid_dir(cfgs):
    _, grid = cfgs
    d = Path(grid.eval.grid_file).parent
    if not (d / "cube_grid_goals_s0.json").is_file():
        pytest.skip(f"{d} holds no grid files (run scripts/cube_grid_goals.py)")
    return d


@pytest.fixture(scope="module")
def starts(grid_dir):
    out = []
    for k in range(N_STARTS):
        f = grid_dir / f"cube_grid_goals_s{k}.json"
        if not f.is_file():
            pytest.skip(f"{f} missing")
        out.append(json.loads(f.read_text()))
    return out


# --------------------------------------------------------------- the config

@pytest.mark.parametrize("block", SHARED_BLOCKS)
def test_shared_blocks_identical(cfgs, block):
    base, grid = cfgs
    assert OmegaConf.to_container(base[block], resolve=False) == \
        OmegaConf.to_container(grid[block], resolve=False), (
        f"config/eval/{GRID}.yaml diverged from {BASE}.yaml in '{block}': the grid must "
        f"change only the goal source and the budget, or its numbers are not comparable"
    )


def test_eval_block_differs_only_where_intended(cfgs):
    base, grid = cfgs
    b = OmegaConf.to_container(base.eval, resolve=False)
    g = OmegaConf.to_container(grid.eval, resolve=False)
    assert set(g) - set(b) <= GRID_ONLY
    differing = {k for k in set(b) & set(g) if b[k] != g[k]}
    assert differing <= GRID_ONLY, f"unexpected eval differences: {sorted(differing - GRID_ONLY)}"


def test_success_threshold_is_the_envs_own(cfgs):
    _, grid = cfgs
    assert grid.eval.grid_success == pytest.approx(0.04), (
        "the environment scores a cube placement as solved at 4 cm; the grid must use the "
        "threshold or its success rate means something else"
    )


# --------------------------------------------------------------- the starts

def test_five_starts_share_one_goal_set(starts):
    ref = [np.asarray(b["goals"], dtype=float) for b in starts[0]["blocks"]]
    for k, spec in enumerate(starts[1:], start=1):
        got = [np.asarray(b["goals"], dtype=float) for b in spec["blocks"]]
        assert len(got) == len(ref), f"start {k} has {len(got)} layers, start 0 has {len(ref)}"
        for bi, (a, b) in enumerate(zip(ref, got)):
            assert a.shape == b.shape and np.allclose(a, b), (
                f"start {k} layer {bi} commands different goals than start 0 -- the arms and "
                f"starts would no longer be paired goal by goal"
            )


def test_starts_are_distinct_episodes_and_grasped(starts):
    eps = [s["meta"]["start"]["episode"] for s in starts]
    assert len(set(eps)) == len(eps), f"starts repeat an episode: {eps}"
    for k, s in enumerate(starts):
        assert s["meta"]["start"]["grasped"] is True, f"start {k} does not hold the cube"


def test_start_height_is_held_constant(starts):
    z = np.array([s["meta"]["start"]["cube_pos"][2] for s in starts])
    lo, hi = START_HEIGHT_M
    assert ((z >= lo) & (z <= hi)).all(), (
        f"start heights {np.round(z, 3).tolist()} leave the band {START_HEIGHT_M}; height would "
        f"be confounded with the start"
    )


def test_starts_span_the_workspace(starts):
    xy = np.array([s["meta"]["start"]["cube_pos"][:2] for s in starts])
    # a 2x2 factorial plus a centre point: four corners, each in its own quadrant
    # relative to the centre start, or the design has collapsed
    quad = {(bool(p[0] > xy[0, 0]), bool(p[1] > xy[0, 1])) for p in xy[1:]}
    assert len(quad) == 4, f"the four corner starts do not occupy four quadrants: {xy.tolist()}"


def test_no_goal_is_trivial_from_any_start(starts):
    pos = np.array([s["meta"]["start"]["cube_pos"] for s in starts])
    goals = np.concatenate([np.asarray(b["goals"], dtype=float) for b in starts[0]["blocks"]])
    d = np.linalg.norm(goals[:, None] - pos[None], axis=-1)
    assert d.min() >= MIN_GOAL_DIST_M - 1e-9, (
        f"a goal sits {d.min() * 100:.1f} cm from a start, inside the {MIN_GOAL_DIST_M * 100:.0f} "
        f"cm floor -- it would be solved by standing still"
    )


def test_goals_stay_inside_the_reachable_workspace(starts):
    # the environment clamps the effector to this box every step; a goal outside it is
    # unreachable for every method, oracle included, and would measure nothing
    lo = np.array([0.25, -0.35, 0.02])
    hi = np.array([0.60, 0.35, 0.35])
    goals = np.concatenate([np.asarray(b["goals"], dtype=float) for b in starts[0]["blocks"]])
    assert (goals >= lo).all() and (goals <= hi).all()


# --------------------------------------------------- the rebuilt goal clouds

@pytest.fixture(scope="module")
def scans(grid_dir):
    f = grid_dir / "cube_goal_scans.npz"
    if not f.is_file():
        pytest.skip(f"{f} missing (run scripts/cube_goal_scans.py)")
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def test_scans_align_with_the_goals_layer_by_layer(scans, starts):
    """Mirror eval_3dtarget.load_grid_block's selection, exactly."""
    goals = np.asarray(scans["goals"])
    for bi, b in enumerate(starts[0]["blocks"]):
        want = np.asarray(b["goals"], dtype=float)
        idx = np.flatnonzero(scans["block"] == bi)[: len(want)]
        assert len(idx) == len(want), (
            f"layer {bi}: {len(idx)} rebuilt clouds for {len(want)} goals"
        )
        assert np.allclose(goals[idx][:, : want.shape[1]], want, atol=1e-2), (
            f"layer {bi}: rebuilt clouds are not in the grid file's goal order -- every goal "
            f"would silently get another goal's observation"
        )


def test_scan_rays_match_the_sensor(scans, cfgs):
    _, grid = cfgs
    n_rays = int(grid.lidar.sensor.h_res) * int(grid.lidar.sensor.v_res)
    assert scans["clouds"].shape[1] == n_rays * 3, (
        f"clouds carry {scans['clouds'].shape[1] // 3} rays, the configured sensor casts {n_rays}"
    )


def test_rebuild_was_validated_and_passed(scans):
    meta = json.loads(str(scans["meta"]))
    v = meta["validation"]
    assert v["scanner_parity"], "no scanner-parity record"
    # A: re-scanning a frame's own state must reproduce the stored cloud exactly
    assert max(x["median_cm"] for x in v["scanner_parity"]) == pytest.approx(0.0, abs=1e-6)
    # B: the rebuild must sit inside the spread of recorded observations of the
    # same pose, or the baseline is handicapped by the rendering
    assert v["ratio_rays_moved"] <= 1.5, v["verdict"]


def test_goals_outside_the_sensor_view_are_recorded(scans):
    """Not an error: the sensor cannot see the far corners at height.

    Those goals are the sharpest case in the sweep -- no goal observation of
    them exists, rebuilt or otherwise -- so they must be labelled rather than
    quietly averaged into the layer.
    """
    meta = json.loads(str(scans["meta"]))
    blind = meta["goals_outside_sensor_view"]
    assert set(blind["idx"]) == set(np.flatnonzero(scans["cube_returns"] == 0).tolist())
    for g in blind["goals"]:
        assert abs(g[1]) >= 0.30 - 1e-9 and g[2] >= 0.16 - 1e-9, (
            f"a goal outside the sensor's view sits at {g}, not in the far-corner-at-height "
            f"region -- the cause may not be the field of view"
        )


def test_visible_goals_show_a_real_cube(scans):
    seen = np.asarray(scans["cube_returns"])
    vis = seen[seen > 0]
    assert np.median(vis) >= 20, (
        f"visible goal clouds carry a median of {np.median(vis):.0f} cube returns; too few for "
        f"the encoder to locate the commanded cube"
    )
