from __future__ import annotations

from pathlib import Path

import numpy as np


def save_video(path: Path, frames: list[np.ndarray], fps: int = 15) -> None:
    if not frames:
        return
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    out = imageio.get_writer(str(path), fps=fps, codec='libx264')
    for f in frames:
        out.append_data(f)
    out.close()


def save_panel_video(
    video_path, env_panels, fps: int = 15, panel_size: int | None = None
) -> None:
    """Compose one env's labeled panels side-by-side into a single mp4.

    ``env_panels`` maps a label to that env's data: a ``(T, H, W, C)``
    sequence or a ``(H, W, C)`` still repeated for every frame. See
    :func:`save_panel_videos` for ``panel_size``.
    """
    from PIL import Image, ImageDraw, ImageFont

    labels = list(env_panels)
    panels = [np.asarray(env_panels[label]) for label in labels]

    def _fit(frame):
        if panel_size is None:
            return np.asarray(frame)
        img = Image.fromarray(np.asarray(frame).astype(np.uint8))
        return np.asarray(img.resize((panel_size, panel_size), Image.BILINEAR))

    if panel_size is not None:
        h = w = panel_size
    else:
        s = panels[0]
        h, w = s.shape[1:3] if s.ndim == 4 else s.shape[:2]
    n = len(labels)
    pad, gap, lh = max(12, w // 14), max(10, w // 16), max(22, w // 9)
    cw = (2 * pad + n * w + (n - 1) * gap + 15) // 16 * 16
    ch = (2 * pad + h + lh + 15) // 16 * 16
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', max(12, w // 14))
    except OSError:
        font = ImageFont.load_default()
    y_text = pad + h + max(8, lh // 4)

    T = max((len(p) for p in panels if p.ndim == 4), default=1)
    composed = []
    for t in range(T):
        c = np.full((ch, cw, 3), 250, dtype=np.uint8)
        for j, p in enumerate(panels):
            frame = p[min(t, len(p) - 1)] if p.ndim == 4 else p
            frame = _fit(frame)
            x = pad + j * (w + gap)
            c[pad : pad + h, x : x + w] = frame
        img = Image.fromarray(c)
        draw = ImageDraw.Draw(img)
        for j, label in enumerate(labels):
            b = draw.textbbox((0, 0), label, font=font)
            x = pad + j * (w + gap) + w // 2 - (b[2] - b[0]) // 2
            draw.text((x, y_text), label, fill=(130, 130, 130), font=font)
        composed.append(np.array(img))
    save_video(Path(video_path), composed, fps=fps)


def save_panel_videos(
    video_dir, panels, fps: int = 15, panel_size: int | None = None
) -> None:
    """Save one ``env_{i}.mp4`` per env with labeled panels side-by-side.

    ``panels`` maps a label to per-env data indexable by env index. Each
    per-env entry is either a ``(T, H, W, C)`` sequence or a ``(H, W, C)``
    still that is repeated for every frame.

    ``panel_size`` resizes every panel frame to ``(panel_size, panel_size)``
    before compositing. Use it to (a) mix panels of different native
    resolutions (e.g. a low-res RGB view next to a high-res LiDAR render) and
    (b) raise the overall video resolution. ``None`` keeps each panel's native
    size (all panels must then already share one size).
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    labels = list(panels)
    for i in range(len(panels[labels[0]])):
        env_panels = {label: panels[label][i] for label in labels}
        save_panel_video(
            video_dir / f'env_{i}.mp4', env_panels, fps, panel_size
        )
