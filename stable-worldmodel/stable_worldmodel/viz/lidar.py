"""Visualize LiDAR point clouds collected by ``AddRaycastLidarWrapper``.

Installed as the ``swm-lidar-viz`` console command (needs the ``viz`` extra:
``pip install 'stable-worldmodel[viz]'``). Two modes:

* **dataset** (default) — load a collected ``.lance`` episode, write an RGB
  rollout video plus an RGB|point-cloud panel video, and open an interactive
  polyscope viewer with a time slider (RGB frame + point cloud kept in sync).
  Misses are filtered out; points are colored with the recorded ``lidar_rgb``
  when the dataset has it. On headless machines the videos still get written
  (pass ``--no-show`` to skip the window attempt entirely).

    swm-lidar-viz --dataset data/cube.lance --episode 0

* **live** — step a single env with a policy and stream the LiDAR cloud into
  polyscope in real time (watch behavior in the point cloud). Needs a display;
  set ``MUJOCO_GL=egl`` for offscreen env rendering.

    swm-lidar-viz --live
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.data import drop_lidar_misses, reshape_lidar
from stable_worldmodel.plot import LidarPanelRenderer, ensure_polyscope_init
from stable_worldmodel.plot.video_utils import save_panel_video, save_video


def jet_colors(values: np.ndarray) -> np.ndarray:
    """Map values in ``[0, 1]`` to ``(N, 3)`` RGB via a self-contained jet."""
    v = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=1)


def depth_color(pts: np.ndarray, zmin: float, zspan: float) -> np.ndarray:
    """Color a cloud by distance from the sensor origin (near=blue, far=red)."""
    dist = np.linalg.norm(pts, axis=1)
    return jet_colors((dist - zmin) / zspan)


def sampled_mask(rgb: np.ndarray) -> np.ndarray:
    """Rows of ``lidar_rgb`` that carry a real camera color.

    Fill rows (rays outside the camera frustum, written as the wrapper's
    ``rgb_fill_value``) lie outside ``[0, 1]``.
    """
    return np.all((rgb >= 0.0) & (rgb <= 1.0), axis=1)


def resolve_colors(
    pts: np.ndarray, rgb: np.ndarray | None, zmin: float, zspan: float
) -> np.ndarray:
    """Per-point display colors: recorded RGB where sampled, depth elsewhere.

    ``rgb`` rows are the wrapper's ``lidar_rgb`` values for the same points;
    rows at the fill value (no camera pixel for that ray) fall back to depth
    coloring so out-of-frustum points stay visible.
    """
    fallback = depth_color(pts, zmin, zspan)
    if rgb is None or not len(pts):
        return fallback
    return np.where(sampled_mask(rgb)[:, None], rgb, fallback).astype(
        np.float32
    )


def _user_x_displays() -> list[str]:
    """X displays (``:N``) whose socket in ``/tmp/.X11-unix`` we own.

    When ``DISPLAY`` is unset over SSH there may still be a usable local X
    server (xpra, VNC, a desktop session); owned sockets are the ones we can
    actually connect to, unlike e.g. gdm's login-screen display.
    """
    displays = []
    try:
        uid = os.getuid()
        for sock in sorted(Path('/tmp/.X11-unix').iterdir()):
            if sock.name.startswith('X') and sock.stat().st_uid == uid:
                displays.append(':' + sock.name[1:])
    except OSError:
        pass
    return displays


def _require_polyscope():
    try:
        import polyscope as ps  # noqa: F401
        import polyscope.imgui as psim  # noqa: F401

        return ps, psim
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            'polyscope is required for LiDAR visualization; install the viz '
            "extra: pip install 'stable-worldmodel[viz]'"
        ) from exc


# ---------------------------------------------------------------------------
# Dataset mode
# ---------------------------------------------------------------------------

# Preferred RGB column order: plain single-view first, then multiview cameras.
_PIXEL_KEYS = (
    'pixels',
    'pixels_front',
    'pixels_front_pixels',
    'pixels_side',
    'pixels_side_pixels',
)


def _select_frames(ep: dict) -> list[np.ndarray]:
    """Return ``[(H, W, 3) uint8, ...]`` from whichever pixel column exists.

    Handles both single-view ``pixels`` (channels-first ``(T, 3, H, W)``) and
    multiview camera columns (channels-last ``(T, H, W, 3)``) by detecting
    which axis holds the 3 channels.
    """
    key = next((k for k in _PIXEL_KEYS if k in ep), None)
    if key is None:
        avail = [k for k in ep if str(k).startswith('pixels')]
        raise KeyError(
            f'no RGB column found; looked for {_PIXEL_KEYS}, '
            f'available pixel-like columns: {avail}'
        )
    arr = np.asarray(ep[key])
    if arr.ndim == 4 and arr.shape[1] == 3:  # (T, 3, H, W) -> (T, H, W, 3)
        arr = arr.transpose(0, 2, 3, 1)
    return [f for f in arr.astype(np.uint8)]


def view_dataset(
    dataset: str,
    episode: int,
    out_dir: Path,
    miss_value: float,
    point_radius: float,
    max_view: float | None,
    show: bool = True,
) -> None:
    ds = swm.data.load_dataset(dataset, num_steps=1)
    ep = ds.load_episode(episode)

    frames = _select_frames(ep)
    lidar = reshape_lidar(ep['lidar'])  # (T, n_rays, 3)
    lidar_np = np.asarray(lidar)
    n_steps = lidar_np.shape[0]
    print(
        f'episode {episode}: {n_steps} steps, '
        f'lidar {lidar_np.shape}, frame {frames[0].shape}'
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f'episode{episode}_rgb.mp4'
    save_video(video_path, frames, fps=15)
    print(f'wrote {video_path}')

    rgb_np = None
    if 'lidar_rgb' in ep:
        rgb_np = np.asarray(reshape_lidar(ep['lidar_rgb']))  # (T, n_rays, 3)

    # Pre-filter misses (and optional range crop) per step, keeping the
    # recorded per-point colors aligned with the surviving points.
    clouds, cloud_rgbs = [], []
    for t in range(n_steps):
        pts = lidar_np[t]
        keep = ~np.all(pts == miss_value, axis=-1)
        pts = pts[keep]
        rgb = rgb_np[t][keep] if rgb_np is not None else None
        if max_view is not None and len(pts):
            near = np.linalg.norm(pts, axis=1) <= max_view
            pts = pts[near]
            rgb = rgb[near] if rgb is not None else None
        clouds.append(np.ascontiguousarray(pts, dtype=np.float32))
        cloud_rgbs.append(rgb)

    # Global depth range for stable colors across the episode.
    alld = np.concatenate(
        [np.linalg.norm(c, axis=1) for c in clouds if len(c)]
    )
    zmin = float(alld.min()) if alld.size else 0.0
    zspan = (
        float(alld.max() - zmin) if alld.size and alld.max() > zmin else 1.0
    )

    # The interactive viewer below needs a WINDOWED polyscope backend, and
    # polyscope initializes once per process -- claim it before the panel
    # renderer defaults to headless EGL.
    headless_linux = sys.platform == 'linux' and not (
        os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    )
    if show and not headless_linux:
        ensure_polyscope_init(prefer_headless=False)

    # RGB + point-cloud side-by-side video: the deliverable on headless
    # machines, and a shareable artifact everywhere else.
    renderer = LidarPanelRenderer(miss_value=miss_value, max_view=max_view)
    lidar_frames = [
        renderer.render(clouds[t], cloud_rgbs[t]) for t in range(n_steps)
    ]
    renderer.close()
    panel_path = out_dir / f'episode{episode}_panel.mp4'
    save_panel_video(
        panel_path,
        {'rgb': np.stack(frames), 'lidar': np.stack(lidar_frames)},
        panel_size=renderer.size,
    )
    print(f'wrote {panel_path}')

    if not show:
        return

    # No window server: the panel videos above are the deliverable.
    if headless_linux:
        displays = _user_x_displays()
        hint = (
            f'this machine has X display(s) you own: {", ".join(displays)} '
            f'(e.g. an xpra/VNC session) — retry with DISPLAY={displays[0]}'
            if displays
            else 're-run from a session with a display'
        )
        print(
            f'no display for the interactive viewer; {hint}, or rely on the '
            'episode videos above (--no-show skips this attempt).'
        )
        return

    ps, psim = _require_polyscope()
    ensure_polyscope_init(prefer_headless=False)
    ps.set_up_dir('z_up')

    # Default to just the camera-colored points when the dataset has them;
    # the checkbox brings the depth-colored out-of-frustum points back.
    has_rgb = rgb_np is not None
    state = {'t': 0, 'play': False, 'rgb_only': has_rgb}

    def show_step(t: int) -> None:
        pts, rgb = clouds[t], cloud_rgbs[t]
        if rgb is not None and state['rgb_only']:
            keep = sampled_mask(rgb)
            pts, rgb = pts[keep], rgb[keep]
        pc = ps.register_point_cloud('lidar', pts, radius=point_radius)
        if len(pts):
            pc.add_color_quantity(
                'color', resolve_colors(pts, rgb, zmin, zspan), enabled=True
            )
        img = np.asarray(frames[t], dtype=np.float32) / 255.0
        ps.add_color_image_quantity('rgb', img, enabled=True)

    show_step(0)

    def callback():
        changed, state['t'] = psim.SliderInt(
            'step', state['t'], 0, n_steps - 1
        )
        _, state['play'] = psim.Checkbox('play', state['play'])
        if has_rgb:
            toggled, state['rgb_only'] = psim.Checkbox(
                'camera-colored points only', state['rgb_only']
            )
            changed = changed or toggled
        if state['play']:
            state['t'] = (state['t'] + 1) % n_steps
            changed = True
        if changed:
            show_step(state['t'])

    ps.set_user_callback(callback)
    ps.show()


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------


def view_live(
    env_name: str,
    env_type: str,
    policy: str,
    miss_value: float,
    point_radius: float,
    max_view: float | None,
    seed: int,
) -> None:
    import gymnasium as gym

    from stable_worldmodel.wrapper import AddRaycastLidarWrapper

    os.environ.setdefault('MUJOCO_GL', 'egl')
    env = gym.make(
        env_name,
        env_type=env_type,
        render_mode='rgb_array',
        width=224,
        height=224,
    )
    env = AddRaycastLidarWrapper(
        env,
        mount_camera='front_pixels',
        min_alpha=0.5,
        miss_value=miss_value,
        add_rgb=True,
    )
    _, info = env.reset(seed=seed)

    def act():
        # Random actions by default; swap in a trained policy here.
        return env.action_space.sample()

    ps, psim = _require_polyscope()
    ensure_polyscope_init(prefer_headless=False)
    ps.set_up_dir('z_up')

    state = {'rgb_only': True}

    def refresh(info):
        pts = np.asarray(info['lidar'])
        keep = ~np.all(pts == miss_value, axis=-1)
        pts = pts[keep]
        rgb = info.get('lidar_rgb')
        rgb = np.asarray(rgb)[keep] if rgb is not None else None
        if max_view is not None and len(pts):
            near = np.linalg.norm(pts, axis=1) <= max_view
            pts = pts[near]
            rgb = rgb[near] if rgb is not None else None
        if rgb is not None and state['rgb_only']:
            sampled = sampled_mask(rgb)
            pts, rgb = pts[sampled], rgb[sampled]
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        pc = ps.register_point_cloud('lidar', pts, radius=point_radius)
        if len(pts):
            d = np.linalg.norm(pts, axis=1)
            span = float(d.max()) if d.max() > 0 else 1.0
            pc.add_color_quantity(
                'color', resolve_colors(pts, rgb, 0.0, span), enabled=True
            )
        if 'goal_lidar' in info:
            # Live wrapper output is already (n_rays, 3); no reshape needed.
            g = drop_lidar_misses(np.asarray(info['goal_lidar']), miss_value)
            ps.register_point_cloud(
                'goal',
                np.ascontiguousarray(g, dtype=np.float32),
                radius=point_radius,
                color=(0.1, 0.9, 0.1),
            )

    refresh(info)

    def callback():
        _, state['rgb_only'] = psim.Checkbox(
            'camera-colored points only', state['rgb_only']
        )
        _, _, term, trunc, info = env.step(act())
        if term or trunc:
            _, info = env.reset()
        refresh(info)

    ps.set_user_callback(callback)
    ps.show()
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='data/lidar_demo.lance')
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--out-dir', default='viz')
    p.add_argument('--live', action='store_true', help='stream a live rollout')
    p.add_argument(
        '--no-show',
        action='store_true',
        help='dataset mode: only write the episode videos, skip the '
        'interactive polyscope viewer (e.g. on headless machines)',
    )
    p.add_argument('--env-name', default='swm/OGBCube-v0')
    p.add_argument('--env-type', default='single')
    p.add_argument('--policy', default='random', choices=['random'])
    p.add_argument('--miss-value', type=float, default=-1.0)
    p.add_argument('--radius', type=float, default=0.002)
    p.add_argument(
        '--max-view',
        type=float,
        default=None,
        help='drop points beyond this range (m) for clarity',
    )
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if args.live:
        view_live(
            args.env_name,
            args.env_type,
            args.policy,
            args.miss_value,
            args.radius,
            args.max_view,
            args.seed,
        )
    else:
        view_dataset(
            args.dataset,
            args.episode,
            Path(args.out_dir),
            args.miss_value,
            args.radius,
            args.max_view,
            show=not args.no_show,
        )


if __name__ == '__main__':
    main()
