"""Scanview guarantees: the sensor's camera renders the scene the scan saw.

Dataset-free checks on the three LiDAR-ready scenes (TwoRoom / PushT mirrors,
Reacher3D):

* restoring a recorded state is exact and order-independent (the pymunk
  centre-of-gravity trap that displaced the T in the first Push-T tables);
* a restored state recasts the very scan the live env produced;
* a depth image from the ``iso`` camera agrees with the scan's forward range
  at every ray, so the render and the scan share pose and field of view;
* style changes colours only, never the scan;
* ``render_camera`` turns the plain ``render()`` into the sensor's view.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import pytest

import stable_worldmodel  # noqa: F401  (registers swm/ envs)
from stable_worldmodel.envs import replay
from stable_worldmodel.envs.scene_render import SceneStyle
from stable_worldmodel.wrapper import AddRaycastLidarWrapper


ISO = dict(iso_azimuth=45.0, iso_elevation=45.0)
# (env id, max_range) as in the generating configs
# (stable-worldmodel/scripts/data/config/lidar_from_h5_*.yaml).
SCENES = {
    'swm/TwoRoomLidar-v1': 8.0,
    'swm/PushTLidar-v1': 12.0,
    'swm/Reacher3D-v0': 4.0,
}
RES = 100


def make_env(env_id, **kwargs):
    env = gym.make(env_id, render_mode='rgb_array', **ISO, **kwargs)
    env.reset(seed=1)
    return env


@pytest.fixture
def render_context():
    """Skip when no offscreen GL context can be made (same convention as
    stable-worldmodel/tests/test_lidar.py: e.g. after polyscope has used and
    torn down EGL in this process, MuJoCo cannot get one)."""
    import mujoco

    try:
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><visual><global offwidth="16" offheight="16"/></visual>'
            '<worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
        )
        r = mujoco.Renderer(model, 16, 16)
        r.close()
    except Exception as exc:  # no offscreen GL on this machine / process
        pytest.skip(f'cannot create a mujoco render context: {exc}')


def make_sensor(env, max_range):
    w = AddRaycastLidarWrapper(
        env,
        h_res=RES,
        v_res=RES,
        match_camera_fov=True,
        mount_camera='iso',
        frame='sensor',
        max_range=max_range,
        miss_value=-1.0,
        backend='mujoco',
        add_goal_lidar=False,
        include_static=True,
    )
    w._detect_backend()
    return w


def live_episode(env, n_steps=7):
    """Step the live env and record its state the way the tables store it."""
    base = env.unwrapped
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(n_steps):
        env.step(rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32))
        if hasattr(base, 'agent_position'):  # TwoRoom
            rows.append(
                dict(
                    pos_agent=np.asarray(base.agent_position, dtype=np.float64),
                    pos_target=np.asarray(base.target_position, dtype=np.float64),
                )
            )
        elif hasattr(base, 'block'):  # PushT: the 7-D state column
            rows.append(
                dict(
                    state=np.array(
                        [
                            *base.agent.position,
                            *base.block.position,
                            base.block.angle,
                            *base.agent.velocity,
                        ],
                        dtype=np.float64,
                    )
                )
            )
        else:  # Reacher3D
            rows.append(
                dict(qpos=base.data.qpos.copy(), qvel=base.data.qvel.copy())
            )
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def cast(env, sensor):
    """The scan of the env's current scene, as the wrapper takes it after step()."""
    base = env.unwrapped
    pts, _ = sensor._cast(base.model, base.data)
    return pts


def test_pusht_restore_is_independent_of_previous_pose():
    env = make_env('swm/PushTLidar-v1')
    base = env.unwrapped
    ep = {
        'state': np.array(
            [[100.0, 120.0, 250.0, 260.0, 0.3, 0.0, 0.0],
             [110.0, 125.0, 300.0, 200.0, 2.9, 0.0, 0.0]]
        )
    }
    replay.begin_episode(env, ep)
    replay.set_step(env, ep, 1)
    direct = np.array(base.block.position)
    replay.set_step(env, ep, 0)  # a very different angle first ...
    replay.set_step(env, ep, 1)  # ... must not leak into this restore
    after = np.array(base.block.position)
    np.testing.assert_allclose(direct, (300.0, 200.0), atol=1e-9)
    np.testing.assert_allclose(after, (300.0, 200.0), atol=1e-9)
    assert abs(base.block.angle - 2.9) < 1e-12
    env.close()


@pytest.mark.parametrize('env_id,max_range', SCENES.items())
def test_restored_state_recasts_the_live_scan(env_id, max_range):
    live = make_env(env_id)
    live_sensor = make_sensor(live, max_range)
    ep = live_episode(live)
    # the live scan of the last recorded step
    expected = cast(live, live_sensor)
    live.close()

    fresh = make_env(env_id)
    sensor = make_sensor(fresh, max_range)
    replay.begin_episode(fresh, ep)
    replay.set_step(fresh, ep, len(next(iter(ep.values()))) - 1)
    got = cast(fresh, sensor)
    fresh.close()
    np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize('env_id,max_range', SCENES.items())
def test_depth_from_iso_matches_scan_range(render_context, env_id, max_range):
    env = make_env(env_id)
    sensor = make_sensor(env, max_range)
    pts = cast(env, sensor)
    depth = env.unwrapped.render_depth('iso', RES, RES).reshape(-1)
    hit = pts[:, 0] > 0
    assert hit.mean() > 0.5
    rel = np.abs(depth[hit] - pts[hit, 0]) / pts[hit, 0]
    # The rasteriser samples a hair off the ray centres (a 1/res effect), so
    # exact equality is not expected; a wrong camera, fov or aspect would be
    # off by whole percent at the median.
    assert np.median(rel) < 0.01
    assert np.percentile(rel, 90) < 0.03
    env.close()


@pytest.mark.parametrize('env_id,max_range', SCENES.items())
def test_style_does_not_change_the_scan(env_id, max_range):
    scans = []
    for style in (None, SceneStyle(ground='checker', light='spot', shadows=True)):
        env = make_env(env_id, style=style)
        sensor = make_sensor(env, max_range)
        scans.append(cast(env, sensor))
        env.close()
    np.testing.assert_array_equal(scans[0], scans[1])


@pytest.mark.parametrize('env_id', SCENES)
def test_render_camera_is_the_sensor_view(render_context, env_id):
    env = make_env(env_id)
    base = env.unwrapped
    plain = np.asarray(base.render())
    iso = np.asarray(base.render(camera='iso'))
    assert iso.shape == (224, 224, 3) and iso.dtype == np.uint8
    assert iso.shape == plain.shape
    assert not np.array_equal(iso, plain)
    env.close()

    env = make_env(env_id, render_camera='iso')
    base = env.unwrapped
    np.testing.assert_array_equal(
        np.asarray(base.render()), np.asarray(base.render(camera='iso'))
    )
    env.close()
