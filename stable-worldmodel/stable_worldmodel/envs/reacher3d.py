"""A minimal 3D two-link reacher for top-down LiDAR.

Unlike the dm_control reacher (a flat, planar scene where the arm is a thin
overlay on the floor), this environment is genuinely 3D: two **raised cuboid**
links rotate over a ground plane, so a top-down LiDAR returns real height
structure -- ground at z~0 and link/base tops at z~0.05-0.10.

It is a plain ``gymnasium.Env`` backed by raw MuJoCo (exposes ``.model`` /
``.data``), so :class:`~stable_worldmodel.wrapper.AddRaycastLidarWrapper` works
out of the box with ``mount_camera='topdown'``.

The task is the dm_control reacher's ``qpos_match``: call
:meth:`set_target_qpos` and the episode terminates once every joint is within
``qpos_threshold`` of that pose. There is deliberately no target OBJECT -- a
marker would be visible to the LiDAR and hand a goal-conditioned model the
answer its goal observation is supposed to carry -- so the goal lives as joint
angles, which no sensor can see. The wrapper orients the scan to
the camera's view direction, so a normal grid FOV becomes a downward cone.

A second camera, ``iso``, views the same workspace disc obliquely from
``iso_azimuth`` / ``iso_elevation`` (default 45/45) -- mount the sensor there
for a harder scan (the links occlude each other and the ground is
foreshortened). ``topdown`` is left exactly as it was, since it also defines
what ``render()`` returns. There is no goal/target object in this scene, so
unlike the TwoRoom / PushT mirrors there is nothing to exclude from the cloud.

The arm is reacher-like: two hinge joints about the vertical axis, driven by two
motors. Links are cuboids (swap ``type="box"`` for ``type="capsule"`` in the XML
for capsules).
"""

from __future__ import annotations

import threading

import gymnasium as gym
import mujoco
import numpy as np

from .camera_xml import iso_camera_xml


# A 3D re-skin of the dm_control "two-link planar reacher": same dynamics
# parameters (timestep 0.02, joint damping 0.01, motor gear 0.05, wrist limited
# to +-160 deg, contact disabled, light comparable link masses) so it *behaves*
# like the original -- both joints move, neither free-spins. The only changes
# are visual/for-LiDAR: the links are raised cuboids on a 'post' above the
# ground and a 'topdown' camera looks straight down, so a top-down scan gets
# real height (ground z~0, link tops z~0.08). Swap type="box" for
# type="capsule" on the links if you want capsules.
#
# INERTIALS ARE THE ORIGINAL'S TOO, declared explicitly on each body. The
# cuboid links look nothing like the source capsules, and MuJoCo would otherwise
# derive mass properties from those boxes: measured against dm_control, the same
# torque for two substeps moved the wrist 13% less and reversed the sign of the
# shoulder's reaction. That matters because datasets replayed into this scene
# (scripts/data/lidar_from_h5.py) keep the SOURCE's actions, so a world model
# trained on them learns the source's action->motion relation and then
# mispredicts its own control authority here -- an error of the same order as
# the reacher eval's 0.05 rad success tolerance. An explicit <inertial> keeps
# the visual/LiDAR geometry as cuboids while making the DYNAMICS identical:
# values are dm_control reacher's own body_ipos / body_iquat / body_mass /
# body_inertia, and with them this scene reproduces dm_control's transitions to
# machine precision (verified over random states/torques).
#
# THE INERTIALS DEFINE THE DYNAMICS OF THIS ENV. Data collected with the
# geometry-derived cuboid inertias instead encodes different dynamics, and a
# model trained on it mispredicts against this env by the same ~0.013 rad/step
# the explicit inertials remove for replayed data, so such tables must be
# regenerated with this scene.
#
# LINK LENGTHS ARE THE ORIGINAL'S: shoulder->wrist 0.12 m, wrist->fingertip
# 0.12 m. Recovered exactly (lstsq residual 0.00000 m over 20k frames) from the
# LeWM reacher dataset's own qpos/finger_pos, so replaying that dataset's joint
# angles here reproduces its fingertip, not one ~1 cm / 10 deg off. Keep the
# link2 geom spanning the full 0.12 (pos 0.06, half-size 0.06) if you retune.
#
# WRIST_RANGE (radians) mirrors the XML wrist limit; reset() samples within it.
WRIST_RANGE = (-160.0 * np.pi / 180.0, 160.0 * np.pi / 180.0)

_CAM_Z = 0.75  # top-down camera height (m); its fovy stays MuJoCo's 45 deg
_LINK_TOP = 0.08  # tallest geometry above the ground (m)
# Workspace disc the top-down camera sees at ground level -- the iso camera is
# framed on the same disc so both views cover the same scene.
_VIEW_R = _CAM_Z * np.tan(np.deg2rad(45.0) / 2)
# Per-joint tolerance (radians) of the goal test, and the reason it is 0.05:
# that is dm_control-based ReacherQPosMatchTask's _DEFAULT_QPOS_THRESHOLD, and
# the link lengths here are the original's, so the tolerance means the same
# fingertip precision. Keeping the number identical is what lets a LiDAR arm
# evaluated in this scene be compared with a pixel arm evaluated in the
# dm_control reacher.
_QPOS_THRESHOLD = 0.05

_ISO_AZ = 45.0  # oblique camera azimuth (deg, CCW from +x)
_ISO_EL = 45.0  # oblique camera elevation (deg above the ground plane)
_ISO_FILL = 2.5  # iso camera distance in units of the scene's bounding radius

_XML_TEMPLATE = """
<mujoco model="reacher3d">
  <option timestep="0.02"><flag contact="disable"/></option>
  <visual><global offwidth="640" offheight="640"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.35 0.28"
             rgb2="0.22 0.28 0.22" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <default>
    <joint type="hinge" axis="0 0 1" damping="0.01"/>
    <motor gear="0.05" ctrlrange="-1 1" ctrllimited="true"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="1 1 0.1" material="grid"/>
    <camera name="topdown" pos="0 0 0.75" xyaxes="1 0 0 0 1 0"/>
    {iso_camera}
    <geom name="post" type="cylinder" fromto="0 0 0 0 0 0.05" size="0.03"
          rgba="0.30 0.30 0.32 1"/>
    <body name="arm" pos="0 0 0.05">
      <joint name="shoulder"/>
      <inertial pos="0.06 0.0 0.0" quat="0.7071067811865476 0.0 -0.7071067811865475 0.0" mass="0.04188790204786391" diaginertia="6.33135639453463e-05 6.33135639453463e-05 2.0525072003453316e-06"/>
      <geom name="link1" type="box" pos="0.06 0 0" size="0.06 0.02 0.03"
            mass="0.042" rgba="0.85 0.25 0.20 1"/>
      <body name="hand" pos="0.12 0 0">
        <joint name="wrist" limited="true" range="-160 160"/>
        <inertial pos="0.05 0.0 0.0" quat="0.7071067811865476 0.0 -0.7071067811865475 0.0" mass="0.03560471674068432" diaginertia="3.9175660390264734e-05 3.9175660390264734e-05 1.7383479349863525e-06"/>
        <geom name="link2" type="box" pos="0.06 0 0" size="0.06 0.018 0.03"
              mass="0.036" rgba="0.20 0.35 0.85 1"/>
        <body name="finger" pos="0.12 0 0">
          <inertial pos="0.0 0.0 0.0" quat="1.0 0.0 0.0 0.0" mass="0.004188790204786391" diaginertia="1.6755160819145565e-07 1.6755160819145565e-07 1.6755160819145565e-07"/>
          <geom name="finger" type="box" size="0.022 0.022 0.03" mass="0.01"
                rgba="0.15 0.70 0.35 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder" joint="shoulder"/>
    <motor name="wrist" joint="wrist"/>
  </actuator>
</mujoco>
"""


class Reacher3DEnv(gym.Env):
    """3D two-link reacher (cuboid links on a ground plane), top-down camera."""

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(
        self,
        render_mode: str | None = 'rgb_array',
        width: int = 224,
        height: int = 224,
        frame_skip: int = 1,
        iso_azimuth: float = _ISO_AZ,
        iso_elevation: float = _ISO_EL,
        iso_distance: float | None = None,
        qpos_threshold: float = _QPOS_THRESHOLD,
        **kwargs,
    ) -> None:
        xml = _XML_TEMPLATE.format(
            iso_camera=iso_camera_xml(
                radius=float(np.hypot(_VIEW_R, _LINK_TOP)),
                azimuth=iso_azimuth,
                elevation=iso_elevation,
                distance=iso_distance,
                fill=_ISO_FILL,
            )
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.width = int(width)
        self.height = int(height)
        self.frame_skip = int(frame_skip)
        self.camera_id = int(self.model.camera('topdown').id)
        self._renderer: mujoco.Renderer | None = None
        self._renderer_thread: int | None = None
        self._rng = np.random.default_rng()
        # Goal state for the qpos-match test. None until set_target_qpos() is
        # called, and an episode without a goal never terminates early -- the
        # same contract as ReacherQPosMatchTask, so data collection (which never
        # sets one) is unaffected.
        self.target_qpos: np.ndarray | None = None
        self.qpos_threshold = float(qpos_threshold)
        self._closest_goal = float('inf')

        self.action_space = gym.spaces.Box(
            -1.0, 1.0, (self.model.nu,), dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), dtype=np.float32
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # Random arm pose so datasets cover the workspace. Shoulder is free;
        # the wrist is limited (+-160 deg), so sample it within WRIST_RANGE.
        self.data.qpos[0] = self._rng.uniform(-np.pi, np.pi)
        self.data.qpos[1] = self._rng.uniform(*WRIST_RANGE)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._closest_goal = float('inf')
        return self._obs(), {}

    def step(self, action):
        """Step the physics, ending the episode once the goal pose is reached.

        The goal test runs after EVERY substep rather than once per env step,
        matching how the dm_control arm checks inside its action_repeat loop --
        an arm swinging through the target must be caught the same way in both,
        or the two are not scored under the same rule.
        """
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        terminated = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            distance = self.goal_distance()
            self._closest_goal = min(self._closest_goal, distance)
            if distance < self.qpos_threshold:
                terminated = True
                break
        return self._obs(), 0.0, terminated, False, {}

    def set_state(self, qpos, qvel=None):
        """Restore a recorded MuJoCo state (the eval configs' ``set_state``).

        Kinematics are refreshed with ``mj_forward`` so anything reading the
        scene right after -- a render, or the LiDAR wrapper's raycast -- sees
        the restored pose rather than the previous step's.
        """
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qvel is not None:
            self.data.qvel[:] = np.asarray(qvel, dtype=np.float64).reshape(-1)
        mujoco.mj_forward(self.model, self.data)

    def set_target_qpos(self, target_qpos):
        """Set the goal pose the episode is scored against.

        Same name and signature as ReacherDMControlWrapper.set_target_qpos, so a
        dataset-driven eval config drives this scene with the identical callable
        it uses for the dm_control reacher:

            - method: set_target_qpos
              args:
                target_qpos:
                  value: goal_qpos

        Not cleared by reset(), matching the dm_control task: the goal is set
        once around the reset that starts a replayed window.
        """
        q = np.asarray(target_qpos, dtype=np.float64).reshape(-1)
        if q.shape != (self.model.nq,):
            raise ValueError(
                f'target_qpos must have {self.model.nq} entries, got {q.shape[0]}'
            )
        self.target_qpos = q
        self._closest_goal = self.goal_distance()

    def goal_distance(self) -> float:
        """Largest per-joint distance to the goal pose, or inf without a goal.

        Raw subtraction, exactly as ReacherQPosMatchTask compares -- not the
        shortest angle on the circle. The shoulder is an unlimited hinge, so a
        pose one full turn from the goal reads ~6.28 rad away and does not
        count; that is the original's behaviour, and matching it matters more
        than the corner case.
        """
        if self.target_qpos is None:
            return float('inf')
        return float(np.max(np.abs(self.data.qpos - self.target_qpos)))

    def closest_goal_distance(self) -> float:
        """Smallest goal_distance() reached since the goal was set.

        Reported by evaluation harnesses: with a tolerance as tight as
        _QPOS_THRESHOLD, "how near did it get" separates a policy that never
        approached from one that arrived and missed the band, which a success
        rate alone cannot.
        """
        return self._closest_goal

    def render(self, width=None, height=None, camera_id=None):
        h = int(height or self.height)
        w = int(width or self.width)
        # A MuJoCo GL context belongs to the thread that made it current:
        # reusing this renderer from another thread dies with EGL_BAD_ACCESS
        # (GLX raises its own X error). That happens for real -- World.collect
        # streams episodes into LanceDB, which consumes the generator (and so
        # runs the env) on its own background thread. Rebuild the renderer
        # whenever the calling thread changes; closing the old one releases
        # its context so the new thread can take one.
        tid = threading.get_ident()
        if (
            self._renderer is None
            or self._renderer_thread != tid
            or (self._renderer.height, self._renderer.width) != (h, w)
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
            self._renderer_thread = tid
        self._renderer.update_scene(
            self.data, camera=self.camera_id if camera_id is None else camera_id
        )
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
            self._renderer_thread = None
