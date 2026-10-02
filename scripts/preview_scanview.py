"""Side-by-side preview of a table's stored image, the scanview render and the scan.

For whole episodes of a Lance table, writes one video per episode with three
panels: the stored ``pixels`` (the original camera), the ``iso`` render of the
restored state (what the sensor's camera sees) and the stored scan drawn as a
depth image from the sensor pose (each ray is one pixel, so it is pixel-aligned
with the middle panel). A misaligned camera shows up as the scan's silhouettes
not sitting on the render's.

    python scripts/preview_scanview.py --env pusht --table data/pusht.lance \
        --episodes 3 4 5 --out previews/

The env is built with its default style, i.e. exactly as the dataset renders.
"""

from __future__ import annotations

import argparse
import io
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import imageio.v2 as imageio
import lance
import numpy as np
from PIL import Image, ImageDraw

import stable_worldmodel  # noqa: F401  (registers swm/ envs)
from stable_worldmodel.envs import replay


# Per environment: env id, the state columns the restore reads, the sensor's
# max_range (depth colour scale) and the ray grid of the stored scans.
ENVS = {
    'tworoom': dict(id='swm/TwoRoomLidar-v1', cols=['pos_agent', 'pos_target'], max_range=8.0),
    'pusht': dict(id='swm/PushTLidar-v1', cols=['state'], max_range=12.0),
    'reacher': dict(id='swm/Reacher3D-v0', cols=['qpos', 'qvel'], max_range=4.0),
}
GRID = 100
SKY = np.array([0.78, 0.86, 0.95]) * 255


def read_episode(table: str, ep: int, cols: list[str]) -> dict:
    ds = lance.dataset(table)
    tbl = ds.scanner(
        columns=cols + ['pixels', 'lidar', 'step_idx'],
        filter=f'episode_idx = {ep}',
    ).to_table()
    out = {c: np.asarray(tbl[c].to_pylist()) for c in cols + ['lidar', 'step_idx']}
    out['pixels'] = [
        np.asarray(Image.open(io.BytesIO(b)).convert('RGB'))
        for b in tbl['pixels'].to_pylist()
    ]
    order = np.argsort(out['step_idx'])
    return {k: (v[order] if isinstance(v, np.ndarray) else [v[i] for i in order]) for k, v in out.items()}


def scan_as_depth_image(lidar: np.ndarray, max_range: float, size: int) -> np.ndarray:
    """The stored scan from the sensor pose: forward range per ray, dark = near."""
    fwd = lidar.reshape(GRID, GRID, 3)[..., 0]
    hit = fwd > 0
    shade = (1.0 - np.clip(fwd / max_range, 0, 1)) * 230 + 20
    img = np.broadcast_to(SKY, (GRID, GRID, 3)).copy()
    img[hit] = shade[hit, None]
    return np.asarray(
        Image.fromarray(img.astype(np.uint8)).resize((size, size), Image.NEAREST)
    )


def label(img: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(img).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 14], fill=(0, 0, 0))
    d.text((3, 1), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--env', choices=ENVS, required=True)
    p.add_argument('--table', required=True, help='Lance table of that env')
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--fps', type=int, default=10)
    args = p.parse_args()
    spec = ENVS[args.env]
    os.makedirs(args.out, exist_ok=True)

    env = gym.make(spec['id'], render_mode='rgb_array', iso_azimuth=45.0, iso_elevation=45.0)
    env.reset(seed=0)
    base = env.unwrapped
    for ep_idx in args.episodes:
        ep = read_episode(args.table, ep_idx, spec['cols'])
        replay.begin_episode(env, ep)
        path = os.path.join(args.out, f'{args.env}_ep{ep_idx:05d}.mp4')
        with imageio.get_writer(path, fps=args.fps, codec='libx264', quality=8, macro_block_size=1) as w:
            for t in range(len(ep['step_idx'])):
                replay.set_step(env, ep, t)
                render = np.asarray(base.render(camera='iso'))
                size = render.shape[0]
                frame = np.concatenate(
                    [
                        label(ep['pixels'][t], f'original  t={t}'),
                        label(render, 'scanview (iso render)'),
                        label(scan_as_depth_image(ep['lidar'][t], spec['max_range'], size), 'stored scan from sensor pose'),
                    ],
                    axis=1,
                )
                w.append_data(frame)
        print('wrote', path)
    env.close()


if __name__ == '__main__':
    main()
