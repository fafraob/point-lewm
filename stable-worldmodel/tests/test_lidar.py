"""Tests for the LiDAR wrapper's RGB/color pipeline.

Pure-math paths (ray grids, pinhole color sampling, color resolution,
output-path handling) run against stub models so they need neither mujoco nor
a GPU. An end-to-end raycast+render test runs when mujoco and an offscreen GL
context are available and is skipped otherwise.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from stable_worldmodel.data.utils import resolve_output_dataset
from stable_worldmodel.viz.lidar import resolve_colors, sampled_mask
from stable_worldmodel.wrapper import AddRaycastLidarWrapper


@pytest.fixture
def base_env():
    env = MagicMock(spec=gym.Env)
    env.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    env.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
    env.unwrapped = env
    return env


def cam_model(fovy: float):
    """Stub MjModel exposing just what the camera-math paths read."""
    model = MagicMock()
    model.camera.return_value = SimpleNamespace(id=0)
    model.cam_fovy = np.array([fovy])
    return model


##########################
## constructor contract ##
##########################


def test_add_rgb_requires_mount_camera(base_env):
    with pytest.raises(ValueError, match='mount_camera'):
        AddRaycastLidarWrapper(base_env, add_rgb=True, mount_camera=None)


def test_match_camera_fov_requires_mount_camera(base_env):
    with pytest.raises(ValueError, match='mount_camera'):
        AddRaycastLidarWrapper(
            base_env, match_camera_fov=True, mount_camera=None
        )


def test_match_camera_fov_conflicts_with_azel(base_env):
    with pytest.raises(ValueError, match='az/el'):
        AddRaycastLidarWrapper(
            base_env,
            match_camera_fov=True,
            mount_camera='cam',
            az=[0.0],
            el=[0.0],
        )


###################
## scan patterns ##
###################


def test_azel_grid_shape_and_unit_norm(base_env):
    w = AddRaycastLidarWrapper(base_env, h_res=16, v_res=4)
    assert w._dirs_sensor.shape == (64, 3)
    assert np.allclose(np.linalg.norm(w._dirs_sensor, axis=1), 1.0)


def test_camera_grid_hits_each_pixel_exactly_once(base_env):
    """The camera-matched pattern is the inverse of the color projection.

    Projecting its rays back through the pinhole model must land on every
    pixel center of the h_res x v_res image exactly once — the property that
    guarantees ``add_rgb`` colors every hit.
    """
    h_res, v_res, fovy = 8, 6, 60.0
    w = AddRaycastLidarWrapper(
        base_env,
        h_res=h_res,
        v_res=v_res,
        match_camera_fov=True,
        mount_camera='cam',
    )
    assert w._dirs_sensor is None  # deferred until a model is available
    dirs = w._camera_grid(cam_model(fovy))
    assert dirs.shape == (h_res * v_res, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)

    # Sensor -> camera frame, then the projection _sample_rgb uses.
    d_cam = dirs @ w._CAM_FRAME_FIX.T
    z = -d_cam[:, 2]
    assert np.all(z > 0)
    focal = 0.5 * v_res / np.tan(0.5 * np.deg2rad(fovy))
    u = np.floor(0.5 * h_res + focal * d_cam[:, 0] / z).astype(int)
    v = np.floor(0.5 * v_res - focal * d_cam[:, 1] / z).astype(int)
    assert sorted(zip(v, u)) == [
        (r, c) for r in range(v_res) for c in range(h_res)
    ]


####################
## color sampling ##
####################


def test_sample_rgb_quadrants_and_fills(base_env):
    img = np.zeros((8, 8, 3), np.uint8)
    img[:4, :4] = (255, 0, 0)  # top-left red
    img[:4, 4:] = (0, 255, 0)  # top-right green
    img[4:, :4] = (0, 0, 255)  # bottom-left blue
    img[4:, 4:] = (255, 255, 255)  # bottom-right white
    base_env.render = MagicMock(return_value=img)

    w = AddRaycastLidarWrapper(base_env, mount_camera='cam', add_rgb=True)
    model = cam_model(fovy=90.0)
    # Identity camera pose: world dirs are camera-frame dirs (x right, y up,
    # looking down -z). Directions need not be normalized to project.
    data = SimpleNamespace(cam_xmat=np.eye(3).reshape(1, 9))
    dirs = np.array(
        [
            [-0.5, 0.5, -1.0],  # up-left -> red
            [0.5, 0.5, -1.0],  # up-right -> green
            [0.5, -0.5, -1.0],  # down-right -> white
            [2.0, 0.0, -1.0],  # outside the frustum -> fill
            [0.0, 0.0, 1.0],  # behind the camera -> fill
            [0.0, 0.0, -1.0],  # no-hit ray -> fill
        ]
    )
    hit = np.array([1, 1, 1, 1, 1, 0], dtype=bool)

    rgb = w._sample_rgb(model, data, dirs, hit)
    assert rgb.shape == (6, 3) and rgb.dtype == np.float32
    assert np.allclose(rgb[0], [1, 0, 0])
    assert np.allclose(rgb[1], [0, 1, 0])
    assert np.allclose(rgb[2], [1, 1, 1])
    assert np.all(rgb[3:] == -1.0)
    base_env.render.assert_called_once_with(camera='cam')


######################
## color resolution ##
######################


def test_sampled_mask_flags_fill_rows():
    rgb = np.array([[0.2, 0.4, 0.6], [-1, -1, -1], [1.0, 0.0, 1.0]])
    assert sampled_mask(rgb).tolist() == [True, False, True]


def test_resolve_colors_mixes_rgb_and_depth():
    pts = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    rgb = np.array([[0.2, 0.4, 0.6], [-1, -1, -1], [0.9, 0.1, 0.1]])
    out = resolve_colors(pts, rgb, 0.0, 3.0)
    assert np.allclose(out[0], rgb[0]) and np.allclose(out[2], rgb[2])
    assert not np.allclose(out[1], rgb[1])  # fill row got a depth color
    assert np.all((out >= 0.0) & (out <= 1.0))
    # Without rgb every point is depth-colored.
    assert resolve_colors(pts, None, 0.0, 3.0).shape == (3, 3)


def test_panel_renderer_uses_point_colors():
    ps = pytest.importorskip('polyscope')
    try:  # a GL context (headless EGL or a display) is required to render
        ps.set_allow_headless_backends(True)
        ps.init()
    except Exception as exc:  # pragma: no cover - no GL on this machine
        pytest.skip(f'no GL context for polyscope: {exc}')
    from stable_worldmodel.plot import LidarPanelRenderer

    pts = np.random.default_rng(0).uniform(-1, 1, size=(64, 3))
    rgb = np.full((64, 3), 0.5, dtype=np.float32)
    rgb[32:] = -1.0  # half the rays carry no camera color
    r = LidarPanelRenderer(size=64)
    try:
        plain = r.render(pts)
        colored = r.render(pts, rgb)
    finally:
        r.close()
    assert plain.shape == colored.shape == (64, 64, 3)
    assert not np.array_equal(plain, colored)


def test_panel_renderer_eye_through_sensor_origin():
    """Framing must survive the eye landing on the world origin.

    Sensor-frame clouds put the sensor at (0,0,0); ``_set_camera`` walks the
    eye from the scene center back toward it, so ``distance * lim == |target|``
    parks the eye exactly there -- and polyscope's ``look_at_dir`` renders a
    blank white frame from a world-origin eye (regression: every lidar panel
    of an eval video rendered white).
    """
    ps = pytest.importorskip('polyscope')
    try:
        ps.set_allow_headless_backends(True)
        ps.init()
    except Exception as exc:  # pragma: no cover - no GL on this machine
        pytest.skip(f'no GL context for polyscope: {exc}')
    from stable_worldmodel.plot import LidarPanelRenderer

    # Spherical shell of radius r around a center at |c| = distance * r: the
    # 95th-percentile half-extent is ~r, so the eye walks distance*r back
    # from the center and lands on the origin.
    rng = np.random.default_rng(0)
    d = rng.normal(size=(5000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r_shell = 0.75
    center = np.array([1.6 * r_shell, 0.0, 0.0])
    pts = center + r_shell * d

    r = LidarPanelRenderer(size=64, distance=1.6)
    try:
        img = r.render(pts)
    finally:
        r.close()
    assert float(img.std()) > 1.0, 'blank frame: camera degenerated at origin'


#################
## output path ##
#################


def test_resolve_output_dataset_default_layout(tmp_path):
    out = resolve_output_dataset(None, 'ogbench/foo.lance', tmp_path)
    assert out == tmp_path / 'datasets' / 'ogbench/foo.lance'


def test_resolve_output_dataset_explicit_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = resolve_output_dataset('foo.lance', 'ignored.lance', None)
    assert out == tmp_path / 'foo.lance'


def test_resolve_output_dataset_explicit_absolute(tmp_path):
    out = resolve_output_dataset(tmp_path / 'bar.lance', 'ignored.lance', None)
    assert out == tmp_path / 'bar.lance'


################################
## end-to-end (needs mujoco)  ##
################################

_SCENE = """
<mujoco>
  <visual><global offwidth="160" offheight="120"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" specular="0 0 0"
           castshadow="false"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0 0 1 1"/>
    <geom name="red" type="box" pos="2 0.8 0.5" size="0.4 0.4 0.5"
          rgba="1 0 0 1"/>
    <geom name="green" type="box" pos="2 -0.8 0.5" size="0.4 0.4 0.5"
          rgba="0 1 0 1"/>
    <camera name="cam" pos="0 0 0.5" xyaxes="0 -1 0 0 0 1"/>
  </worldbody>
</mujoco>
"""


class _BoxEnv(gym.Env):
    """Raw-MuJoCo backend env: two colored boxes in front of a camera."""

    action_space = gym.spaces.Box(-1, 1, (1,), np.float64)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float64)

    def __init__(self, mujoco):
        self.model = mujoco.MjModel.from_xml_string(_SCENE)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._renderer = mujoco.Renderer(self.model, height=120, width=160)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._mujoco.mj_forward(self.model, self.data)
        return np.zeros(1), {}

    def step(self, action):
        self._mujoco.mj_step(self.model, self.data)
        return np.zeros(1), 0.0, False, False, {}

    def render(self, camera='cam'):
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        self._renderer.close()


@pytest.fixture
def box_env():
    mujoco = pytest.importorskip('mujoco')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    try:
        env = _BoxEnv(mujoco)
    except Exception as exc:  # no offscreen GL on this machine
        pytest.skip(f'cannot create a mujoco render context: {exc}')
    yield env
    env.close()


def test_end_to_end_colors_match_hit_geoms(box_env):
    env = AddRaycastLidarWrapper(
        box_env,
        h_res=40,
        v_res=30,
        mount_camera='cam',
        match_camera_fov=True,
        add_goal_lidar=False,
        add_rgb=True,
    )
    _, info = env.reset()
    pts, rgb = info['lidar'], info['lidar_rgb']
    assert pts.shape == rgb.shape == (40 * 30, 3)

    hit = ~np.all(pts == -1.0, axis=1)
    colored = sampled_mask(rgb)
    # Camera-matched pattern: exactly the hits are colored.
    assert np.array_equal(hit, colored)
    assert hit.sum() > 100

    # Sensor frame is x=fwd, y=left, z=up; boxes sit left (+y, red) and
    # right (-y, green) of center. Sampled colors must match the geoms.
    on_red = colored & (pts[:, 1] > 0.3) & (pts[:, 2] > -0.2)
    on_green = colored & (pts[:, 1] < -0.3) & (pts[:, 2] > -0.2)
    assert on_red.sum() > 5 and on_green.sum() > 5
    assert np.all(rgb[on_red].argmax(axis=1) == 0)
    assert np.all(rgb[on_green].argmax(axis=1) == 1)

    _, _, _, _, info = env.step(np.zeros(1))
    assert 'lidar_rgb' in info
