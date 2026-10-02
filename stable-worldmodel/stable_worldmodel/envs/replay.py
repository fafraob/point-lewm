"""Restore recorded states into the LiDAR-ready scenes, without stepping physics.

Every offline pass over a recorded table -- re-sensing it with the raycast
LiDAR, rendering it from the sensor's camera, checking one against the other --
needs the same thing: put the env into the exact state of row ``t`` of an
episode, then read the scene. Two restore modes, chosen by the env:

* **mirror** envs (``swm/TwoRoomLidar-v1`` / ``swm/PushTLidar-v1``, detected by
  ``_build_mirror``): the table stores the 2D state (``pos_agent`` /
  ``pos_target``, ``state`` / ``goal_pose``); it is written straight onto the
  2D env and its MuJoCo mirror syncs. The mirror is rebuilt once per episode,
  so callers that cache ``env.model`` / ``env.data`` must re-read them after
  :func:`begin_episode`.
* **MuJoCo-native** envs (``swm/Reacher3D-v0``, OGBench): ``qpos`` / ``qvel``
  go onto ``data`` followed by ``mj_forward``.

``ep`` is a mapping from column name to an array indexed by step, e.g. the
columns of one episode read from a Lance table or an h5 file.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np


def is_mirror_env(env) -> bool:
    """A mirror env keeps 2D dynamics + a MuJoCo shadow scene."""
    return hasattr(env.unwrapped, '_build_mirror')


def begin_episode(env, ep: Mapping[str, np.ndarray]) -> None:
    """Per-episode setup: the goal (fixed within an episode) and the mirror."""
    base = env.unwrapped
    if not is_mirror_env(env):
        return
    import torch  # the 2D TwoRoom env keeps its state as tensors

    if hasattr(base, 'target_position') and 'pos_target' in ep:  # TwoRoom
        base.target_position = torch.as_tensor(
            np.asarray(ep['pos_target'][0]), dtype=torch.float32
        )
    if hasattr(base, 'goal_pose') and 'goal_pose' in ep:  # PushT
        base.goal_pose = np.asarray(ep['goal_pose'][0], dtype=np.float64)
    base._build_mirror()


def set_step(env, ep: Mapping[str, np.ndarray], t: int) -> None:
    """Put ``env`` into the state recorded at step ``t`` of ``ep``.

    PushT tables come in two flavours: swm's own collections write
    ``pos_agent`` / ``block_pose``, the LeWM expert set stores only the
    canonical 7-D ``state`` (``[agent_xy, block_xy, block_angle, agent_vel]``).
    Positions are written onto the pymunk bodies directly -- no ``space.step``
    -- so the pose is the recorded one, not one physics tick later.

    The block's ``angle`` is written BEFORE its ``position``. pymunk's position
    setter places the body so that its origin lands at the given point *using
    the rotation the body has at that moment* (the T's centre of gravity is
    45 px from its origin); setting the angle afterwards rotates about the
    centre of gravity and drags the origin by ``(R_old - R_new) . cog``, up to
    a few centimetres per frame while the block turns. Angle first makes the
    restore exact and independent of whatever pose the body had before.
    """
    base = env.unwrapped
    if is_mirror_env(env):
        import torch

        if hasattr(base, 'agent_position'):  # TwoRoom
            base.agent_position = torch.as_tensor(
                np.asarray(ep['pos_agent'][t]), dtype=torch.float32
            )
            base._sync_agent()
            return
        if 'pos_agent' in ep:  # PushT with explicit pose columns
            ax, ay = np.asarray(ep['pos_agent'][t], dtype=np.float64)
            bx, by, ba = np.asarray(ep['block_pose'][t], dtype=np.float64)[:3]
        else:  # PushT from the 7-D `state` column
            st = np.asarray(ep['state'][t], dtype=np.float64)
            if st.shape[0] < 5:
                raise RuntimeError(
                    f'PushT state column has {st.shape[0]} dims; expected at '
                    'least 5 ([agent_xy, block_xy, block_angle]).'
                )
            ax, ay, bx, by, ba = st[:5]
        base.agent.position = (float(ax), float(ay))
        base.block.angle = float(ba)
        base.block.position = (float(bx), float(by))
        base._sync()
        return

    qpos = np.asarray(ep['qpos'][t], dtype=np.float64)
    qvel = np.asarray(ep['qvel'][t], dtype=np.float64) if 'qvel' in ep else None
    if hasattr(base, 'set_state'):
        base.set_state(qpos, qvel)
        return
    model, data = base.model, base.data
    data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
