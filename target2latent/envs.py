"""Per-environment specs: goal parameterization, frames, dataset columns.

One :class:`EnvSpec` per environment describes everything env-specific the rest
of the package needs, so ``models.py`` / ``train.py`` / ``eval_3dtarget.py``
stay env-agnostic:

* the **conditioning layout** -- how many 3-D position channels and how many
  rotation channels are part of the goal,
* the **frame transforms** -- privileged dataset state (world / pixel space)
  -> 3-D poses in the **sensor frame** (the frame the stored clouds and the
  encoder's normalization affine live in),
* the **encoder affine** (``norm_center`` / ``norm_scale``) copied from the
  matching ``config/train/model/*.yaml``,
* the dataset **columns** the precompute reads and the ``goal_*`` info keys the
  closed-loop eval reads.

Goal parameterization (the user-facing "place the objects in the scan" spec)
----------------------------------------------------------------------------
``tworoom``   agent position                      -> coords (3,)
``pusht``     T position + T heading + ball       -> coords (6,) = [T, ball],
              position                               rot (3,) heading vector
``cube``      cube position                       -> coords (3,)
``reacher``   wrist joint position + fingertip    -> coords (6,) = [wrist, finger]
              position

Positions are 3-D points in the sensor frame. The only rotation that enters is
the pusht T's planar yaw, and it enters as a **sensor-frame heading vector**:
the T's in-plane heading axis ``(cos psi, sin psi, 0)`` (world frame) rotated
into the sensor frame -- exactly the direction the placed T mesh's local axis
reads as in the 3-D scan. Unique per yaw, continuous, no double cover, so no
rotation augmentation is needed. There is deliberately NO orientation channel
for the cube: a cube's orientation is only defined up to its 24-fold octahedral
symmetry (many distinct quaternions label identical point clouds), so a
quaternion is an ill-posed conditioning signal -- the goal is *where* the cube
is, and the heads average/sample over the orientation.

The reacher has no target object at all (the env scores a joint-angle match,
see ``stable_worldmodel/envs/reacher3d.py``), so its goal is the ARM's pose:
the 3-D positions of the wrist joint (end of link 1) and of the fingertip (end
of link 2), both at the links' mid-height. Two points because one is not
enough -- a fingertip position alone has two inverse-kinematics branches
(elbow left / elbow right) that produce different scans, and a head conditioned
on it would average them. Wrist + fingertip is forward kinematics of the goal
``qpos`` (shoulder angle from the wrist point, wrist angle from the fingertip),
so the conditioning carries exactly the information the success test reads,
laid out as two placeable points like the pusht T + ball.

Frames
------
The stored ``lidar`` clouds are in the **sensor frame**: ``p_sensor =
rot^T (p_world - origin)`` with ``rot = cam_xmat @ CAM_FRAME_FIX`` -- the exact
transform swm's ``AddRaycastLidarWrapper`` applies (``frame: sensor``). The
sensor rides a *static* camera in all four envs (cube: ``front_pixels`` on the
world body; pusht/tworoom/reacher: the ``iso`` camera of the MuJoCo scene, fixed
by the arena geometry), so ``origin``/``rot`` are constants. The cached values
below were read off live envs (verified invariant across resets/seeds) and are
re-checkable on demand: ``python -m target2latent.envs <env>`` transforms the
privileged object positions of a few dataset rows and measures the distance to
the nearest recorded lidar return -- a wrong transform shows up as
centimetres-to-metres of offset instead of the object's own surface distance.

pusht / tworoom pixel -> world mapping (copied verbatim from the mirror envs
``stable_worldmodel/envs/{pusht,two_room}/lidar_env.py``):

* pusht:   ``wx = (px - 256) * 0.005``, ``wy = (256 - py) * 0.005``; the y-flip
  reflects the frame, so the block's mirror yaw is ``-state[4]``. The T body
  origin and the pusher center sit at ``z = 0`` (the mirror places both mocap
  bodies on the ground plane; their geoms extrude upward).
* tworoom: ``wx = (px - 112) * 0.01``, ``wy = (112 - py) * 0.01``; the agent
  mocap body sits at ``z = 0.06`` (half the 0.12 m cylinder).

reacher forward kinematics (``stable_worldmodel/envs/reacher3d.py``): both
hinges are vertical, link lengths 0.12 m / 0.12 m (the dm_control original's,
recovered from the source dataset; ``qpos`` -> ``finger_pos`` reproduces the
table's own column to 1e-8), the arm body and hence both joint origins sit at
``z = 0.05`` (link boxes span z 0.02..0.08).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------- paths
#
# Datasets, world-model runs and this package's artifacts (latent caches, goal
# heads) are located through the repository-wide convention in paths.py
# (environment variables PLWM_DATA_ROOT / PLWM_LOGS_ROOT / PLWM_T2L_ROOT); the
# fixed evaluation start lists ship with the repository under eval_starts/.
# Every script also exposes an explicit override (--dataset / --logs-root /
# --eval-dir), these are only the defaults.
import sys as _sys

_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATASETS as _DATASETS  # noqa: E402
from paths import DATA_ROOT as _DATA_ROOT  # noqa: E402
from paths import LOGS_ROOT, REPO_ROOT, T2L_ROOT  # noqa: E402,F401

#: the shipped ``starts_<env>_seed<k>.json`` window files (scripts/make_eval_starts.py)
EVAL_STARTS_DIR = os.path.join(str(REPO_ROOT), "eval_starts")

#: swm ``AddRaycastLidarWrapper._CAM_FRAME_FIX``: sensor (x fwd, y left, z up)
#: <- MuJoCo camera (x right, y up, looking down -z). A copy rather than an
#: import so this module works without swm on the path.
CAM_FRAME_FIX = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=np.float64
)


# ----------------------------------------------------------------- env specs


@dataclass(frozen=True)
class EnvSpec:
    name: str
    #: sensor pose: ``p_sensor = (p_world - origin) @ rot``  (rot = world<-sensor)
    sensor_origin: np.ndarray
    sensor_rot: np.ndarray
    #: encoder coordinate affine (config/train/model/point_delta_jepa_<env>.yaml)
    norm_center: tuple
    norm_scale: float
    #: number of 3-D position channels in the conditioning (3 or 6)
    coord_dim: int
    #: rotation channels following the coords (0, or 3 for pusht's sensor-frame
    #: heading vector)
    rot_dim: int
    #: dataset columns the precompute reads besides ``lidar``/``episode_idx``/``step_idx``
    columns: tuple
    #: ``goal_*`` info keys the closed-loop eval reads
    goal_keys: tuple
    #: default lance dataset path
    dataset: str
    #: hydra eval config name (config/eval/) whose world/lidar blocks match the dataset
    eval_config: str = ""

    @property
    def cond_dim_base(self):
        return self.coord_dim + self.rot_dim

    def world_to_sensor(self, p_world):
        p = np.asarray(p_world, dtype=np.float64)
        return (p - self.sensor_origin) @ self.sensor_rot

    def dir_world_to_sensor(self, d_world):
        """Direction (no origin shift), world axes -> sensor axes."""
        return np.asarray(d_world, dtype=np.float64) @ self.sensor_rot

    # -- privileged state -> (coords (N, coord_dim), rot (N, rot_dim) | None) --
    def goal_pose(self, raw):
        """Dict of raw dataset columns -> sensor-frame conditioning arrays."""
        raise NotImplementedError


class CubeSpec(EnvSpec):
    # position only: a cube's orientation is ill-posed conditioning (24-fold
    # octahedral symmetry), see the module docstring
    def goal_pose(self, raw):
        pos = np.asarray(raw["privileged/block_0_pos"], dtype=np.float64).reshape(-1, 3)
        return self.world_to_sensor(pos).astype(np.float32), None


class PushTSpec(EnvSpec):
    #: metres per pixel / arena centre -- pusht/lidar_env.py's _S / _C
    PX_SCALE = 0.005
    PX_CENTER = 256.0

    def _px_to_world(self, xy_px):
        xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
        w = np.zeros((len(xy), 3))
        w[:, 0] = (xy[:, 0] - self.PX_CENTER) * self.PX_SCALE
        w[:, 1] = (self.PX_CENTER - xy[:, 1]) * self.PX_SCALE
        return w  # z = 0: both mocap bodies sit on the ground plane

    def goal_pose(self, raw):
        # state = [agent_x, agent_y, block_x, block_y, block_angle, vx, vy] (px)
        st = np.asarray(raw["state"], dtype=np.float64).reshape(-1, 7)
        t_pos = self.world_to_sensor(self._px_to_world(st[:, 2:4]))
        ball_pos = self.world_to_sensor(self._px_to_world(st[:, 0:2]))
        # the mirror's y-flip reflects the frame -> the block's world yaw is -angle;
        # the yaw enters as the T's in-plane heading axis, in the sensor frame
        psi = -st[:, 4]
        d_world = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], axis=1)
        rot = self.dir_world_to_sensor(d_world)
        coords = np.concatenate([t_pos, ball_pos], axis=1)
        return coords.astype(np.float32), rot.astype(np.float32)


class TwoRoomSpec(EnvSpec):
    PX_SCALE = 0.01
    PX_CENTER = 112.0
    AGENT_Z = 0.06  # agent mocap body height: half the 0.12 m cylinder

    def _px_to_world(self, xy_px):
        xy = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
        w = np.full((len(xy), 3), self.AGENT_Z)
        w[:, 0] = (xy[:, 0] - self.PX_CENTER) * self.PX_SCALE
        w[:, 1] = (self.PX_CENTER - xy[:, 1]) * self.PX_SCALE
        return w

    def goal_pose(self, raw):
        pos = self.world_to_sensor(self._px_to_world(raw["pos_agent"]))
        return pos.astype(np.float32), None


class ReacherSpec(EnvSpec):
    """Arm pose as two sensor-frame points: wrist joint and fingertip."""

    LINK1 = 0.12  # shoulder -> wrist (m), reacher3d.py
    LINK2 = 0.12  # wrist -> fingertip (m)
    ARM_Z = 0.05  # height of the arm body / joint origins (m)

    def _fk_world(self, qpos):
        """``(N, 2)`` joint angles -> world-frame wrist ``(N, 3)``, fingertip ``(N, 3)``."""
        q = np.asarray(qpos, dtype=np.float64).reshape(-1, 2)
        a0 = q[:, 0]
        a1 = q[:, 0] + q[:, 1]
        wrist = np.stack(
            [self.LINK1 * np.cos(a0), self.LINK1 * np.sin(a0), np.full_like(a0, self.ARM_Z)],
            axis=1,
        )
        finger = wrist + np.stack(
            [self.LINK2 * np.cos(a1), self.LINK2 * np.sin(a1), np.zeros_like(a1)], axis=1
        )
        return wrist, finger

    def goal_pose(self, raw):
        wrist, finger = self._fk_world(raw["qpos"])
        coords = np.concatenate(
            [self.world_to_sensor(wrist), self.world_to_sensor(finger)], axis=1
        )
        return coords.astype(np.float32), None


#: Cached ``front_pixels`` sensor pose in ``swm/OGBCube-v0``. Verified: the
#: camera sits on the world body, invariant across mj_forward with randomized
#: qpos and across reset(seed=...); transform validated against the stored
#: clouds (nearest return = the cube's 2 cm half-extent on every sampled frame).
_CUBE_ORIGIN = np.array([1.053, -0.014, 0.639])
_CUBE_ROT = np.array(
    [
        [-0.7781291764852688, 0.0, -0.6281042666710385],
        [0.0, -1.0, 0.0],
        [-0.6281042666710385, 0.0, 0.7781291764852688],
    ]
)

#: Cached ``iso`` camera poses of the MuJoCo mirror scenes (45/45 bearing, the
#: setting that generated the tables -- config/eval/3dtarget_{pusht,tworoom}.yaml).
#: Read off live PushTLidarEnv / TwoRoomLidarEnv instances, verified invariant
#: across resets/seeds and identical in rotation (same bearing).
_ISO_ROT = np.array(
    [
        [-0.5000000773622201, 0.7071067811865475, -0.4999999226377678],
        [-0.5000000773622201, -0.7071067811865475, -0.4999999226377679],
        [-0.7071066717798296, 0.0, 0.7071068905932484],
    ]
)
_PUSHT_ORIGIN = np.array([2.2705, 2.2705, 3.211])
_TWOROOM_ORIGIN = np.array([1.9956, 1.9956, 2.8222])
#: ``iso`` camera of ``swm/Reacher3D-v0`` at the same 45/45 bearing (identical
#: rotation, 0.80 m out: ``iso_camera_xml(radius=hypot(_VIEW_R, _LINK_TOP),
#: fill=2.5)``). Read off a live env (cam_xpos / cam_xmat, invariant across
#: resets); the XML stores the position at 4 decimals, so this is exact.
_REACHER_ORIGIN = np.array([0.4010, 0.4010, 0.5671])

SPECS = {
    "cube": CubeSpec(
        name="cube",
        sensor_origin=_CUBE_ORIGIN,
        sensor_rot=_CUBE_ROT,
        norm_center=(1.27, 0.0, 0.25),
        norm_scale=0.75,
        coord_dim=3,
        rot_dim=0,
        columns=("privileged/block_0_pos",),
        goal_keys=("goal_privileged_block_0_pos",),
        dataset=f"{_DATA_ROOT}/{_DATASETS['cube']}",
        eval_config="3dtarget_cube",
    ),
    "pusht": PushTSpec(
        name="pusht",
        sensor_origin=_PUSHT_ORIGIN,
        sensor_rot=_ISO_ROT,
        norm_center=(4.88, 0.0, 0.33),
        norm_scale=2.01,
        coord_dim=6,
        rot_dim=3,
        columns=("state",),
        goal_keys=("goal_state",),
        dataset=f"{_DATA_ROOT}/{_DATASETS['pusht']}",
        eval_config="3dtarget_pusht",
    ),
    "tworoom": TwoRoomSpec(
        name="tworoom",
        sensor_origin=_TWOROOM_ORIGIN,
        sensor_rot=_ISO_ROT,
        norm_center=(4.29, 0.0, 0.29),
        norm_scale=1.77,
        coord_dim=3,
        rot_dim=0,
        columns=("pos_agent",),
        goal_keys=("goal_pos_agent",),
        dataset=f"{_DATA_ROOT}/{_DATASETS['tworoom']}",
        eval_config="3dtarget_tworoom",
    ),
    "reacher": ReacherSpec(
        name="reacher",
        sensor_origin=_REACHER_ORIGIN,
        sensor_rot=_ISO_ROT,
        norm_center=(0.98, 0.0, 0.17),
        norm_scale=0.48,
        coord_dim=6,
        rot_dim=0,
        columns=("qpos",),
        goal_keys=("goal_qpos",),
        dataset=f"{_DATA_ROOT}/{_DATASETS['reacher']}",
        eval_config="3dtarget_reacher",
    ),
}


# ------------------------------------------------------------- verification


def verify_against_dataset(env, n_rows=16, invalid_value=-1.0, stride=9973):
    """Self-check: do the transformed object positions land on their own points?

    Transforms the privileged object position(s) of a few dataset rows into the
    sensor frame and reports the distance to the nearest valid lidar return per
    object. A correct transform gives roughly the object's own surface offset
    (cube: its 2 cm half-extent; pusht T / ball and the tworoom agent: within
    their footprint, but the reference point sits at the body origin so up to
    ~10 cm is geometric, not an error; reacher wrist / fingertip: inside the
    link boxes, 2-3 cm to the nearest face); a wrong one gives tens of
    centimetres to metres.
    """
    import lance

    spec = SPECS[env]
    ds = lance.dataset(spec.dataset)
    rows = [(i * stride) % ds.count_rows() for i in range(n_rows)]
    cols = ["lidar", *spec.columns]
    tbl = ds.take(rows, columns=cols)
    clouds = np.stack(tbl.column("lidar").to_numpy(zero_copy_only=False)).reshape(
        len(rows), -1, 3
    )
    raw = {
        c: np.stack(tbl.column(c).to_numpy(zero_copy_only=False)) for c in spec.columns
    }
    coords, _ = spec.goal_pose(raw)
    n_obj = spec.coord_dim // 3
    mins = np.zeros((len(rows), n_obj))
    for i, cloud in enumerate(clouds):
        valid = cloud[~np.all(cloud == invalid_value, axis=-1)]
        for j in range(n_obj):
            p = coords[i, 3 * j : 3 * j + 3]
            mins[i, j] = np.linalg.norm(valid - p[None], axis=1).min()
    return mins


if __name__ == "__main__":  # python -m target2latent.envs [env ...]
    import sys

    failed = False
    for env in sys.argv[1:] or list(SPECS):
        mins = verify_against_dataset(env)
        print(f"[{env}] nearest lidar return per transformed object position (m):")
        for j in range(mins.shape[1]):
            m = mins[:, j]
            print(f"    object {j}: min {m.min():.4f}  median {np.median(m):.4f}  "
                  f"max {m.max():.4f}")
        ok = mins.max() < 0.15
        failed |= not ok
        print(f"[{env}] {'PASS' if ok else 'FAIL'} (all offsets < 0.15 m)")
    sys.exit(int(failed))
