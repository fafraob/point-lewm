"""PushT with a MuJoCo *mirror* scene for top-down LiDAR.

:class:`PushT` runs on the 2D ``pymunk`` engine, so the raycast LiDAR can't
attach to it. :class:`PushTLidarEnv` keeps the pymunk dynamics **exactly as-is**
and mirrors the state each reset/step into a lightweight 3D MuJoCo shadow scene:

* ground plane,
* a mocap cylinder for the pusher (read from the agent's pymunk circle),
* a mocap body for the block, with one raised box per pymunk polygon (so the
  default T -- and L / Z / + / ... -- all mirror generically), placed/oriented
  from ``block.position`` / ``block.angle``,
* optionally a copy of the block at the **goal pose**, taller
  (``goal_lidar_height``, e.g. 0.30 m vs the live block's 0.15 m). **Off by
  default**: the goal is a task instruction, not something a sensor should see,
  so leaving it out forces a policy/world-model to infer it from elsewhere.
  Pass a height to put it in the cloud, where it stays separable from the live
  block by a z-threshold. Either way the 2D env still draws it in the RGB image.

Two cameras are defined for the sensor to ride:

* ``topdown`` -- straight down, fovy matched to the pygame image's extent so a
  ``match_camera_fov`` scan covers exactly the rendered arena.
* ``iso`` -- an oblique view from ``iso_azimuth`` / ``iso_elevation`` (default
  45/45), framed on the arena's bounding sphere. Harder than top-down: the
  block and pusher occlude, and depth varies across the scan.

Exposed as ``self.model`` / ``self.data`` for
:class:`~stable_worldmodel.wrapper.AddRaycastLidarWrapper` (raw-MuJoCo backend).
The 2D behavior is untouched; ``render()`` still returns the original pygame RGB.

The 512-px pymunk arena maps to MuJoCo at ``S`` m/px with a y-flip (pymunk is
y-up, the RGB image is y-down) so the LiDAR view lines up with the RGB; block
polygons are reflected locally and the angle negated to stay consistent.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pymunk

from ..camera_xml import iso_camera_xml, topdown_camera_xml
from .env import PushT


_S = 0.005  # metres per pixel (512 px -> 2.56 m)
_C = 256.0  # arena centre (px)
_H_BLOCK = 0.15  # block height (m)
_H_AGENT = 0.15  # pusher height (m)
_GOAL_H = 0.30  # goal-T height (m) -- taller than the live block so the goal is
#                 separable in the cloud by a simple z-threshold
#                 (opt-in: see goal_lidar_height)
_CAM_Z = 2.5  # top-down camera height (m)
_ISO_AZ = 45.0  # oblique camera azimuth (deg, CCW from +x)
_ISO_EL = 45.0  # oblique camera elevation (deg above the ground plane)
_ISO_FILL = 2.5  # iso camera distance in units of the scene's bounding radius


def _wx(x: float) -> float:
    return (float(x) - _C) * _S


def _wy(y: float) -> float:
    return (_C - float(y)) * _S  # y-flip: pymunk y-up -> image y-down


class PushTLidarEnv(PushT):
    """PushT + a synced MuJoCo mirror scene for top-down LiDAR."""

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
        # None -> derived in _camera_xml() from the arena's bounding sphere.
        self.iso_distance = (
            None if iso_distance is None else float(iso_distance)
        )
        self.model = None
        self.data = None
        self._agent_mid = -1
        self._block_mid = -1
        self._goal_mid = -1

    # -- gym API: run the 2D env, then mirror its state --------------------
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._build_mirror()
        return obs, info

    def step(self, action):
        out = super().step(action)
        self._sync()
        return out

    def _set_state(self, state):
        # Eval restores recorded states through this hook (config/eval/*.yaml
        # callables). Mirror the new pose immediately: without this the scan
        # would still see the previous block/pusher pose until the next step().
        super()._set_state(state)
        self._sync()

    # -- read pymunk shapes -----------------------------------------------
    @staticmethod
    def _poly_aabb(shape):
        v = np.asarray(shape.get_vertices(), dtype=np.float64)
        cx, cy = (v[:, 0].min() + v[:, 0].max()) / 2, (
            v[:, 1].min() + v[:, 1].max()
        ) / 2
        hx, hy = (v[:, 0].max() - v[:, 0].min()) / 2, (
            v[:, 1].max() - v[:, 1].min()
        ) / 2
        return cx, cy, hx, hy

    def _block_geoms(self, prefix: str, height: float, rgba: str) -> str:
        # One raised box per polygon; reflect local y (cy) to match the y-flip.
        zh = height / 2
        g = []
        for i, s in enumerate(self.block.shapes):
            if isinstance(s, pymunk.Circle):
                r = s.radius * _S
                g.append(
                    f'<geom name="{prefix}_{i}" type="cylinder" '
                    f'pos="0 0 {zh:.4f}" size="{r:.4f} {zh:.4f}" rgba="{rgba}"/>'
                )
                continue
            cx, cy, hx, hy = self._poly_aabb(s)
            g.append(
                f'<geom name="{prefix}_{i}" type="box" '
                f'pos="{cx * _S:.4f} {-cy * _S:.4f} {zh:.4f}" '
                f'size="{max(hx * _S, 1e-3):.4f} {max(hy * _S, 1e-3):.4f} '
                f'{zh:.4f}" rgba="{rgba}"/>'
            )
        return '\n      '.join(g)

    def _agent_geom(self) -> str:
        r = _H_AGENT  # fallback
        for s in self.agent.shapes:
            if isinstance(s, pymunk.Circle):
                r = s.radius * _S
                break
        return (
            f'<geom name="agent" type="cylinder" pos="0 0 {_H_AGENT / 2:.4f}" '
            f'size="{r:.4f} {_H_AGENT / 2:.4f}" rgba="0.2 0.4 0.85 1"/>'
        )

    # -- mirror construction / sync ---------------------------------------
    def _camera_xml(self) -> str:
        """The ``topdown`` + ``iso`` cameras, framed on the arena.

        ``topdown``'s fovy follows the pygame image's half-extent
        (``window_size / 2`` px) rather than MuJoCo's 45 deg default, which at
        ``_CAM_Z`` would cover only the inner ~81% of the render and skew every
        ray -> pixel color lookup. ``iso`` frames the arena's bounding sphere
        (corner-to-corner, block height included) from the configured bearing.
        """
        half = self.window_size * _S / 2.0
        return (
            topdown_camera_xml(height=_CAM_Z, half_extent=half)
            + '\n    '
            + iso_camera_xml(
                radius=float(np.hypot(half * np.sqrt(2.0), _H_BLOCK)),
                azimuth=self.iso_azimuth,
                elevation=self.iso_elevation,
                distance=self.iso_distance,
                fill=_ISO_FILL,
            )
        )

    def _goal_body_xml(self) -> str:
        if self.goal_lidar_height is None:
            return ''
        geoms = self._block_geoms('goal', self.goal_lidar_height, '0.2 0.75 0.3 1')
        return f'<body name="goal" mocap="true" pos="0 0 0">\n      {geoms}\n    </body>'

    def _build_mirror(self) -> None:
        arena = self.window_size * _S
        block_geoms = self._block_geoms('block', _H_BLOCK, '0.45 0.5 0.6 1')
        goal_body = self._goal_body_xml()
        cameras = self._camera_xml()
        xml = f"""
<mujoco model="pusht_mirror">
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
    <body name="block" mocap="true" pos="0 0 0">
      {block_geoms}
    </body>
    <body name="agent" mocap="true" pos="0 0 0">
      {self._agent_geom()}
    </body>
    {goal_body}
  </worldbody>
</mujoco>
"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._agent_mid = int(self.model.body('agent').mocapid[0])
        self._block_mid = int(self.model.body('block').mocapid[0])
        self._goal_mid = (
            int(self.model.body('goal').mocapid[0])
            if self.goal_lidar_height is not None
            else -1
        )
        # Goal T is fixed per episode -> place it once here.
        if self._goal_mid >= 0:
            gx, gy, ga = (
                float(self.goal_pose[0]),
                float(self.goal_pose[1]),
                -float(self.goal_pose[2]),
            )
            self.data.mocap_pos[self._goal_mid] = [_wx(gx), _wy(gy), 0.0]
            self.data.mocap_quat[self._goal_mid] = [
                np.cos(ga / 2),
                0.0,
                0.0,
                np.sin(ga / 2),
            ]
        self._sync()

    def _sync(self) -> None:
        if self.data is None:
            return
        ax, ay = self.agent.position
        self.data.mocap_pos[self._agent_mid] = [_wx(ax), _wy(ay), 0.0]
        bx, by = self.block.position
        self.data.mocap_pos[self._block_mid] = [_wx(bx), _wy(by), 0.0]
        # y-flip reflects the frame, so negate the angle (rotation about z).
        a = -float(self.block.angle)
        self.data.mocap_quat[self._block_mid] = [
            np.cos(a / 2),
            0.0,
            0.0,
            np.sin(a / 2),
        ]
        mujoco.mj_forward(self.model, self.data)
