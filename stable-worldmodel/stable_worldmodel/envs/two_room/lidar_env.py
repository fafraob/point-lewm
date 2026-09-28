"""TwoRoom with a MuJoCo *mirror* scene for top-down LiDAR.

The original :class:`TwoRoomEnv` is a pure-2D kinematic navigator (torch masks,
no physics engine), so the raycast LiDAR -- which needs MuJoCo geometry -- can't
attach to it. Rather than re-implement the dynamics (which would change the
behavior), :class:`TwoRoomLidarEnv` keeps the 2D env **exactly as-is** and, each
reset/step, mirrors its state into a lightweight 3D MuJoCo "shadow" scene:

* ground plane,
* raised border frame + central dividing wall (split around the door gaps),
* a mocap cylinder for the agent, moved to the agent's pixel position,
* optionally a taller marker at the target/goal position
  (``goal_lidar_height``). **Off by default**: the goal is a task instruction,
  not something a sensor should see, so leaving it out of the scan forces a
  policy/world-model to infer it from elsewhere. Pass a height (e.g. 0.30 m) to
  put it back in the cloud, where its height separates it by a z-threshold from
  the agent (0.12 m) and walls (0.20 m). Either way the 2D env still draws the
  goal in the RGB image.

Two cameras are defined for the sensor to ride:

* ``topdown`` — straight down from ``_CAM_Z``, fovy matched to the 2D image's
  extent so a ``match_camera_fov`` scan covers exactly the rendered arena and
  ray -> pixel color lookups line up with the image.
* ``iso`` — an oblique view from ``iso_azimuth`` / ``iso_elevation`` (default
  45/45), framed on the arena's bounding sphere. Harder than top-down: walls
  occlude, the floor is foreshortened, and range varies across the scan.

The scene is exposed as ``self.model`` / ``self.data`` so
:class:`~stable_worldmodel.wrapper.AddRaycastLidarWrapper` (raw-MuJoCo backend)
raycasts it from either camera. The 2D behavior is 100% unchanged; the MuJoCo
model is only a geometry proxy for the sensor, and ``render()`` still returns
the original 2D RGB image.

Pixel -> world mapping: ``world = (px - 112) * S`` for x, ``(112 - py) * S`` for
y (image y-down -> world y-up), z in ``[0, wall_h]``. Walls are taller than the
agent so a top-down scan separates them by height.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ..camera_xml import iso_camera_xml, topdown_camera_xml
from .env import TwoRoomEnv


_S = 0.01  # metres per pixel
_WALL_H = 0.20  # wall / border height (m)
_AGENT_H = 0.12  # agent cylinder height (m)
_GOAL_H = 0.30  # goal marker height (m) -- taller than everything else so the
#                 goal is separable in the cloud by a simple z-threshold
#                 (opt-in: see goal_lidar_height)
_CAM_Z = 2.5  # top-down camera height (m)
_ISO_AZ = 45.0  # oblique camera azimuth (deg, CCW from +x)
_ISO_EL = 45.0  # oblique camera elevation (deg above the ground plane)
_ISO_FILL = 2.5  # iso camera distance in units of the scene's bounding radius


def _px2wx(px: float) -> float:
    return (float(px) - 112.0) * _S


def _px2wy(py: float) -> float:
    return (112.0 - float(py)) * _S


def _solid_intervals(lo: float, hi: float, gaps: list[tuple[float, float]]):
    """Complement of ``gaps`` within ``[lo, hi]`` -> list of solid intervals."""
    gaps = sorted((max(lo, a), min(hi, b)) for a, b in gaps if b > lo and a < hi)
    out, cur = [], lo
    for a, b in gaps:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


class TwoRoomLidarEnv(TwoRoomEnv):
    """TwoRoom + a synced MuJoCo mirror scene for top-down LiDAR."""

    def __init__(
        self,
        goal_lidar_height: float | None = None,
        iso_azimuth: float = _ISO_AZ,
        iso_elevation: float = _ISO_EL,
        iso_distance: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.goal_lidar_height = goal_lidar_height
        self.iso_azimuth = float(iso_azimuth)
        self.iso_elevation = float(iso_elevation)
        # None -> derived in _camera_geoms() from the arena's bounding sphere.
        self.iso_distance = (
            None if iso_distance is None else float(iso_distance)
        )
        self.model = None
        self.data = None
        self._agent_mocap_id = -1
        self._goal_mocap_id = -1

    # -- gym API: run the 2D env, then mirror its state --------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._build_mirror()
        return obs, info

    def step(self, action):
        out = super().step(action)
        self._sync_agent()
        return out

    def _set_state(self, state):
        # Eval restores recorded states through this hook (config/eval/*.yaml
        # callables). Mirror the new pose immediately: without this the scan
        # would still see the previous agent position until the next step().
        super()._set_state(state)
        self._sync_agent()

    # -- mirror construction / sync ---------------------------------------
    def _box(self, name, x_lo, x_hi, y_lo, y_hi, h, rgba):
        cx = (_px2wx(x_lo) + _px2wx(x_hi)) / 2.0
        cy = (_px2wy(y_lo) + _px2wy(y_hi)) / 2.0
        hx = abs(_px2wx(x_hi) - _px2wx(x_lo)) / 2.0
        hy = abs(_px2wy(y_hi) - _px2wy(y_lo)) / 2.0
        return (
            f'<geom name="{name}" type="box" pos="{cx:.4f} {cy:.4f} {h / 2:.4f}"'
            f' size="{max(hx, 1e-3):.4f} {max(hy, 1e-3):.4f} {h / 2:.4f}"'
            f' rgba="{rgba}"/>'
        )

    def _wall_geoms(self) -> list[str]:
        bs = self.BORDER_SIZE
        hi = self.IMG_SIZE - bs
        half = self.wall_thickness // 2
        g = []
        # Border frame (four bars around the play area).
        c = '0.25 0.25 0.28 1'
        t = bs
        g.append(self._box('border_l', bs - t, bs, bs, hi, _WALL_H, c))
        g.append(self._box('border_r', hi, hi + t, bs, hi, _WALL_H, c))
        g.append(self._box('border_t', bs, hi, bs - t, bs, _WALL_H, c))
        g.append(self._box('border_b', bs, hi, hi, hi + t, _WALL_H, c))
        # Central dividing wall, split around the door gap(s).
        wc = self.WALL_CENTER
        gaps = [
            (
                float(self.door_positions[i]) - float(self.door_sizes[i]),
                float(self.door_positions[i]) + float(self.door_sizes[i]),
            )
            for i in range(self.num_doors)
        ]
        wcol = '0.15 0.15 0.18 1'
        for k, (a, b) in enumerate(_solid_intervals(bs, hi, gaps)):
            if self.wall_axis == 1:  # vertical wall at x=center, doors along y
                g.append(
                    self._box(
                        f'wall_{k}', wc - half, wc + half, a, b, _WALL_H, wcol
                    )
                )
            else:  # horizontal wall at y=center, doors along x
                g.append(
                    self._box(
                        f'wall_{k}', a, b, wc - half, wc + half, _WALL_H, wcol
                    )
                )
        return g

    def _camera_xml(self) -> str:
        """The ``topdown`` + ``iso`` cameras, framed on the arena.

        ``topdown``'s fovy is derived from the image extent rather than left at
        MuJoCo's 45 deg default: the 2D render spans +/-1.12 m, so a default
        camera at ``_CAM_Z`` would see only the inner ~92% of it and every
        ray -> pixel color lookup would land ~8% too far out.

        ``iso`` sits on the ``iso_azimuth`` / ``iso_elevation`` bearing at a
        distance that fits the arena's bounding sphere (corner-to-corner, walls
        included) into its fov, so the whole scene stays in the scan whatever
        angle is chosen.
        """
        half = self.IMG_SIZE * _S / 2.0
        return (
            topdown_camera_xml(height=_CAM_Z, half_extent=half)
            + '\n    '
            + iso_camera_xml(
                radius=float(np.hypot(half * np.sqrt(2.0), _WALL_H)),
                azimuth=self.iso_azimuth,
                elevation=self.iso_elevation,
                distance=self.iso_distance,
                fill=_ISO_FILL,
            )
        )

    def _goal_body_xml(self) -> str:
        if self.goal_lidar_height is None:
            return ''
        r = float(self.variation_space['target']['radius'].value.item()) * _S
        zh = self.goal_lidar_height / 2
        return (
            f'<body name="goal" mocap="true" pos="0 0 {zh:.4f}">'
            f'<geom name="goal" type="cylinder" size="{r:.4f} {zh:.4f}"'
            f' rgba="0.2 0.75 0.3 1"/></body>'
        )

    def _build_mirror(self) -> None:
        arena = self.IMG_SIZE * _S
        agent_r = float(self.variation_space['agent']['radius'].value.item())
        walls = '\n    '.join(self._wall_geoms())
        goal_body = self._goal_body_xml()
        cameras = self._camera_xml()
        xml = f"""
<mujoco model="two_room_mirror">
  <option timestep="0.02"><flag contact="disable"/></option>
  <visual><global offwidth="224" offheight="224"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9"
             rgb2="0.8 0.8 0.82" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="{arena:.3f} {arena:.3f} 0.1"
          material="grid"/>
    {cameras}
    {walls}
    <body name="agent" mocap="true" pos="0 0 {_AGENT_H / 2:.4f}">
      <geom name="agent" type="cylinder" size="{agent_r * _S:.4f} {_AGENT_H / 2:.4f}"
            rgba="0.85 0.2 0.2 1"/>
    </body>
    {goal_body}
  </worldbody>
</mujoco>
"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._agent_mocap_id = int(self.model.body('agent').mocapid[0])
        self._goal_mocap_id = (
            int(self.model.body('goal').mocapid[0])
            if self.goal_lidar_height is not None
            else -1
        )
        # Goal is fixed per episode -> place it once here.
        if self._goal_mocap_id >= 0:
            gx, gy = (
                float(self.target_position[0]),
                float(self.target_position[1]),
            )
            self.data.mocap_pos[self._goal_mocap_id] = [
                _px2wx(gx),
                _px2wy(gy),
                self.goal_lidar_height / 2,
            ]
        self._sync_agent()

    def _sync_agent(self) -> None:
        if self.data is None:
            return
        ax, ay = float(self.agent_position[0]), float(self.agent_position[1])
        self.data.mocap_pos[self._agent_mocap_id] = [
            _px2wx(ax),
            _px2wy(ay),
            _AGENT_H / 2,
        ]
        mujoco.mj_forward(self.model, self.data)
