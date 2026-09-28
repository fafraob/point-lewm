"""DataLoader collation for packed point-cloud batches.

Turns a list of per-sample dicts -- each holding a dense per-frame point cloud
under ``point_key`` with shape ``(T, N, C)`` -- into the packed representation
the :class:`pc_encoders.base.PointCloudEncoder` interface expects, while
collating every other key normally via :func:`torch.utils.data.default_collate`.

Each ``(sample, timestep)`` frame becomes one cloud, so the packed ``batch``
index runs over ``B * T`` clouds in sample-major, frame-minor order -- which is
exactly what JEPA's ``rearrange("(b t) d -> b t d", b=B)`` expects after
encoding. Point counts may differ per frame; nothing is padded.

The collate does NO subsampling: every surviving point is packed, and any
per-cloud point budget is applied by the encoder itself with GPU FPS
(:func:`pc_encoders.sampling.fps_index`). Keeping
the workers to a cheap reshape-and-concat is what lets several of them run
without pegging the CPUs; the float32 cast (upstream in the transform, enforced
here) keeps the buffered batches at half the lance column width.
"""

import torch
from torch.utils.data import default_collate


def collate_point_cloud(batch, point_key="points", feat_key=None, drop_value=None):
    """Collate a minibatch, packing ``point_key`` into ``{coord, batch, feat}``.

    Args:
        batch: list of per-sample dicts. ``sample[point_key]`` has shape
            ``(T, N, C)`` (``N`` may vary across frames/samples). When
            ``feat_key`` is given, ``sample[feat_key]`` has shape ``(T, N, F)``.
        point_key: dict key holding the per-frame point coordinates. The packed
            result is written back under this same key.
        feat_key: optional dict key holding per-point features; when ``None`` the
            packed batch has ``feat=None``.
        drop_value: if not ``None``, points whose coords are ALL equal to this
            value are removed per frame before packing -- e.g. ``-1.0`` sentinels
            marking missing lidar returns. This makes clouds ragged, which the
            packed layout handles natively. Assumes each frame keeps >= 1 point.

    Returns:
        dict where ``point_key`` maps to the packed batch
        ``{"coord": (M, C), "batch": (M,), "feat": (M, F) | None}`` and every
        other key is ``default_collate``d to a leading batch dimension. ``coord``
        and ``feat`` are cast to ``float32`` (the lance columns are ``float64``;
        the encoders consume ``float32``, so packing wider only wasted RAM).
    """
    coords, feats, sizes = [], [], []
    for sample in batch:
        frames = sample[point_key]  # (T, N, C)
        feat_frames = sample[feat_key] if feat_key is not None else None
        for t in range(frames.shape[0]):
            frame = frames[t]  # (N, C)
            feat_t = feat_frames[t] if feat_frames is not None else None
            if drop_value is not None:  # drop all-sentinel points (e.g. (-1,-1,-1))
                keep = ~(frame == drop_value).all(dim=-1)
                frame = frame[keep]
                if feat_t is not None:
                    feat_t = feat_t[keep]
            # float32: a no-op after the transform's early cast, but enforced
            # here so a float64 column can never double the pinned batches.
            coords.append(frame.to(torch.float32))
            sizes.append(frame.shape[0])
            if feat_t is not None:
                feats.append(feat_t.to(torch.float32))

    counts = torch.tensor(sizes, dtype=torch.long)
    packed = {
        "coord": torch.cat(coords, dim=0),
        # cloud index in [0, B*T) for every point, in sample-major/frame-minor order
        "batch": torch.repeat_interleave(torch.arange(len(sizes)), counts),
        "feat": torch.cat(feats, dim=0) if feats else None,
    }

    drop = {point_key} if feat_key is None else {point_key, feat_key}
    rest = default_collate([{k: v for k, v in s.items() if k not in drop} for s in batch])
    rest[point_key] = packed
    return rest
