"""Re-render LiDAR point clouds for an existing image dataset.

Reads an swm-format ``.h5`` table, restores each recorded state in the
environment (no physics stepping) and casts the same raycast scan the online
collection uses. Point colors are sampled from the *stored* images through the
wrapper's pinhole projection — no re-render, no parallax.

Two state-restore modes, auto-detected:

* **MuJoCo-native** (default): the h5 stores the full MuJoCo state per step
  (``qpos``/``qvel``); we write it onto the env's ``data`` and ``mj_forward``.
  Used by OGBench / dm_control envs (including ``swm/Reacher3D-v0``).
* **mirror** envs (``swm/TwoRoomLidar-v1`` / ``swm/PushTLidar-v1``): the h5
  stores 2D state (``pos_agent`` / ``pos_target`` / ``block_pose`` /
  ``goal_pose``); the base 2D env's dynamics are untouched — we set its native
  state each step and let its MuJoCo *mirror* scene sync, then cast against it.
  Detected by the env exposing ``_build_mirror``.

Output is a ``.lance`` table with the source columns (images JPEG-encoded by the
Lance writer) plus the computed ``lidar`` / ``lidar_rgb`` columns. The source
``ep_idx``/``step_idx`` are dropped in favor of the writer-managed index columns.
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

from itertools import islice

import gymnasium as gym
import h5py
import hydra
import mujoco
import numpy as np
import torch
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import stable_worldmodel  # noqa: F401  (registers swm/ envs)
from stable_worldmodel.data.formats.lance import LanceWriter
from stable_worldmodel.wrapper import AddRaycastLidarWrapper


# Preferred leading column order (OGBench schema); any other per-step source
# column is appended after these so nothing is dropped (e.g. TwoRoom's
# pos_agent / pos_target / proprio).
COLUMN_ORDER = [
    'proprio/joint_pos',
    'proprio/joint_vel',
    'proprio/effector_pos',
    'proprio/effector_yaw',
    'proprio/gripper_opening',
    'proprio/gripper_vel',
    'proprio/gripper_contact',
    'privileged/block_0_pos',
    'privileged/block_0_quat',
    'privileged/block_0_yaw',
    'privileged/target_block',
    'privileged/target_block_pos',
    'privileged/target_block_yaw',
    'prev_qpos',
    'prev_qvel',
    'qpos',
    'qvel',
    'control',
    'time',
    'target',
    'success',
    'lidar',
    'lidar_rgb',
    'render_time',
    'pixels',
    'observation',
    'reward',
    'terminated',
    'truncated',
    'action',
    'id',
]
COMPUTED = {'lidar', 'lidar_rgb'}
# Not per-step data columns (episode metadata / writer-managed indices).
# Both index spellings appear in the wild: swm/OGBench tables use `ep_idx`,
# the LeWM PushT set uses `episode_idx`. The Lance writer owns those names and
# drops incoming copies with a warning, so keep them out of the column list.
_NON_STEP = {'ep_idx', 'episode_idx', 'step_idx', 'ep_len', 'ep_offset'}


class OfflineImageLidar(AddRaycastLidarWrapper):
    """Lidar wrapper whose ``add_rgb`` colors come from a supplied image.

    ``_render_rgb`` normally grabs the mount-camera frame; overriding it to
    return the stored dataset image keeps the exact collection-time projection
    while skipping the offscreen render.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self._offline_img = None

    def set_image(self, img: np.ndarray | None) -> None:
        self._offline_img = img

    def _render_rgb(self) -> np.ndarray:
        if self._offline_img is not None:
            return self._offline_img
        return super()._render_rgb()


# ---------------------------------------------------------------------------
# State restore
# ---------------------------------------------------------------------------


def is_mirror_env(env) -> bool:
    """A mirror env keeps 2D dynamics + a MuJoCo shadow scene (``_build_mirror``)."""
    return hasattr(env.unwrapped, '_build_mirror')


def mirror_begin_episode(base, src, start):
    """Set the per-episode goal and (re)build the mirror; return (model, data)."""
    if hasattr(base, 'target_position') and 'pos_target' in src:  # TwoRoom
        base.target_position = torch.as_tensor(
            np.asarray(src['pos_target'][start]), dtype=torch.float32
        )
    if hasattr(base, 'goal_pose') and 'goal_pose' in src:  # PushT
        base.goal_pose = np.asarray(src['goal_pose'][start], dtype=np.float64)
    base._build_mirror()
    return base.model, base.data


def mirror_set_state(base, src, off) -> None:
    """Set the mirror env's native 2D state for step ``off`` and sync mocaps.

    PushT tables come in two flavours: swm's own collections write
    ``pos_agent`` / ``block_pose`` columns, while the LeWM expert set stores
    only the canonical 7-D ``state``
    (``[agent_xy, block_xy, block_angle, agent_vel]``, the vector the env's
    ``_set_state`` consumes). Positions are written straight onto the pymunk
    bodies -- no ``space.step`` -- so the replay reproduces the recorded pose
    exactly instead of advancing physics a tick.
    """
    if hasattr(base, 'agent_position'):  # TwoRoom
        base.agent_position = torch.as_tensor(
            np.asarray(src['pos_agent'][off]), dtype=torch.float32
        )
        base._sync_agent()
        return

    if 'pos_agent' in src:  # PushT with explicit pose columns
        ax, ay = np.asarray(src['pos_agent'][off], dtype=np.float64)
        bp = np.asarray(src['block_pose'][off], dtype=np.float64)
        bx, by, ba = float(bp[0]), float(bp[1]), float(bp[2])
    else:  # PushT from a 7-D `state` column
        st = np.asarray(src['state'][off], dtype=np.float64)
        if st.shape[0] < 5:
            raise RuntimeError(
                f'PushT state column has {st.shape[0]} dims; expected at '
                'least 5 ([agent_xy, block_xy, block_angle]).'
            )
        ax, ay, bx, by, ba = st[0], st[1], st[2], st[3], st[4]
    base.agent.position = (float(ax), float(ay))
    base.block.position = (bx, by)
    base.block.angle = ba
    base._sync()


def check_replay_fidelity(wrapper, model, data, src, offset, tol):
    """Compare a re-render of the restored state against the stored pixels."""
    data.qpos[:] = src['qpos'][offset]
    data.qvel[:] = src['qvel'][offset]
    mujoco.mj_forward(model, data)
    wrapper.set_image(None)
    rendered = np.asarray(wrapper._render_rgb(), dtype=np.float32)
    stored = src['pixels'][offset].astype(np.float32)
    if rendered.shape != stored.shape:
        raise RuntimeError(
            f'render check: shape mismatch — rendered {rendered.shape} vs '
            f'stored pixels {stored.shape}. Env/camera config does not match '
            'the source dataset.'
        )
    diff = np.abs(rendered - stored).mean()
    if diff > tol:
        raise RuntimeError(
            f'render check: mean abs pixel difference {diff:.2f} > {tol} — '
            'the env model likely changed since the dataset was collected. '
            'Set check_render=false only if you know why.'
        )
    logging.info(f'render check passed (mean abs pixel diff {diff:.2f})')


def episode_stream(src, episodes, wrapper, model, data, columns, mirror):
    """Yield one lance-ready episode dict per source episode."""
    ep_len, ep_offset = src['ep_len'][:], src['ep_offset'][:]
    base = wrapper.env.unwrapped
    for ep in episodes:
        start, length = int(ep_offset[ep]), int(ep_len[ep])
        pixels = src['pixels'][start : start + length]
        # In mirror mode the model is rebuilt per episode (goal moves); the
        # CPU raycaster re-reads it each cast, so grab the fresh handles here.
        if mirror:
            model, data = mirror_begin_episode(base, src, start)

        clouds, colors = [], []
        for t in range(length):
            off = start + t
            if mirror:
                mirror_set_state(base, src, off)
            else:
                data.qpos[:] = src['qpos'][off]
                data.qvel[:] = src['qvel'][off]
                mujoco.mj_forward(model, data)
            if wrapper.add_rgb:
                wrapper.set_image(pixels[t])
            pts, rgb = wrapper._cast(model, data)
            clouds.append(pts.reshape(-1))
            if rgb is not None:
                colors.append(rgb.reshape(-1).astype(np.float32))

        computed = {
            'lidar': clouds,
            'lidar_rgb': colors,
            'pixels': list(pixels),
        }
        yield {
            col: computed[col]
            if col in computed
            else list(src[col.replace('/', '_')][start : start + length])
            for col in columns
        }


def select_columns(src, add_rgb):
    """Computed cols + every per-step source column (COLUMN_ORDER first)."""
    n_total = int(src['pixels'].shape[0])
    known = [
        c
        for c in COLUMN_ORDER
        if c in COMPUTED or c.replace('/', '_') in src
    ]
    known_flat = {c.replace('/', '_') for c in COLUMN_ORDER}
    extra = [
        k
        for k in src.keys()
        if k not in _NON_STEP
        and k not in known_flat
        and getattr(src[k], 'ndim', 0) >= 1
        and src[k].shape[0] == n_total
    ]
    columns = known + extra
    if not add_rgb and 'lidar_rgb' in columns:
        columns.remove('lidar_rgb')
    return columns


@hydra.main(
    version_base=None, config_path='./config', config_name='lidar_from_h5'
)
def run(cfg: DictConfig):
    src = h5py.File(cfg.input, 'r')
    n_episodes = int(src['ep_len'].shape[0])

    ep_start = int(cfg.ep_start or 0)
    ep_end = int(cfg.ep_end) if cfg.ep_end is not None else n_episodes

    env = gym.make(cfg.env_name, render_mode='rgb_array', **cfg.env)
    env.reset(seed=0)
    mirror = is_mirror_env(env)

    lidar_kwargs = OmegaConf.to_container(cfg.lidar, resolve=True)
    if mirror and lidar_kwargs.get('backend') == 'warp':
        # The mirror is rebuilt per episode; warp caches its mesh on the first
        # model and would go stale. The CPU backend re-reads the model.
        logging.warning('mirror env: forcing lidar backend mujoco (was warp).')
        lidar_kwargs['backend'] = 'mujoco'
    wrapper = OfflineImageLidar(env, add_goal_lidar=False, **lidar_kwargs)
    wrapper._detect_backend()

    if mirror:
        logging.info(
            f'{cfg.env_name}: mirror env — restoring 2D state (no qpos/qvel).'
        )
        model, data = env.unwrapped.model, env.unwrapped.data
    else:
        model, data = wrapper._get_model_data()
        if (
            model.nq != src['qpos'].shape[1]
            or model.nv != src['qvel'].shape[1]
        ):
            raise RuntimeError(
                f'state dim mismatch: env has nq={model.nq}/nv={model.nv}, '
                f'dataset has qpos={src["qpos"].shape[1]}/'
                f'qvel={src["qvel"].shape[1]}.'
            )
        if cfg.check_render:
            check_replay_fidelity(
                wrapper,
                model,
                data,
                src,
                src['ep_offset'][ep_start],
                cfg.check_render_tol,
            )

    columns = select_columns(src, wrapper.add_rgb)

    with LanceWriter(cfg.output, mode=cfg.write_mode) as writer:
        done = writer._ep_idx  # episodes already in the table (append mode)
        if done:
            logging.info(f'resuming: {done} episodes already in {cfg.output}')
        todo = range(ep_start + done, ep_end)
        if not todo:
            logging.success('nothing to do — all episodes already converted')
            return

        stream = episode_stream(
            src,
            tqdm(todo, desc='episodes'),
            wrapper,
            model,
            data,
            columns,
            mirror,
        )
        # Commit every chunk_size episodes (one Lance version each) so an
        # interrupted run resumes from the last committed chunk.
        while True:
            before = writer._ep_idx
            writer.write_episodes(islice(stream, int(cfg.chunk_size)))
            if writer._ep_idx == before:
                break

    logging.success(
        f'wrote lidar for episodes [{ep_start + done}, {ep_end}) '
        f'to {cfg.output}'
    )


if __name__ == '__main__':
    run()
