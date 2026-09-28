"""Point-cloud encoder interface for the JEPA world models (Point-LeWM / Point-Delta-JEPA).

Encoders in this package plug into :class:`jepa.JEPA` as the observation encoder
for the point-cloud (lidar) modality, playing the same role the ViT plays for
pixels. They consume a *packed* batch of point clouds (the Pointcept /
PyTorch-Geometric convention) and return one embedding per cloud.

Packed batch format
-------------------
A batch of ``num_clouds`` point clouds, each with a possibly different number of
points, is represented as flat tensors plus an index mapping every point back to
its cloud::

    data = {
        "coord": FloatTensor (M, C),          # point coords (C = in_channels; 3 = xyz for lidar)
        "batch": LongTensor  (M,),            # cloud index in [0, num_clouds) per point
        "feat":  FloatTensor (M, F) | None,   # optional per-point features (may be None)
    }

where ``M = sum_i N_i`` is the total number of points in the batch. Because
nothing is padded, clouds may have different sizes -- which is exactly why this
representation is preferred over a dense ``(num_clouds, N, 3)`` tensor.

Contract
--------
``forward(data) -> Tensor (num_clouds, embed_dim)``: exactly one pooled
embedding per cloud, ordered by ascending batch index. JEPA reshapes this into
``(B, T, D)`` where ``num_clouds == B * T`` -- one cloud per (sample, timestep)
frame. Normalization / voxelization / sampling strategies are the encoder's
responsibility, not the data pipeline's, so different backbones can preprocess
the raw ``coord`` however they need.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

# A packed batch of point clouds. Kept as a plain ``dict`` for zero-friction
# interop with Hydra, DataLoader collation and point-cloud libraries.
PackedPointCloud = dict


class PointCloudEncoder(nn.Module, ABC):
    """Abstract base class for point-cloud encoders consumed by JEPA.

    Subclasses only need to implement :meth:`forward`. Keep constructor
    arguments plain (Hydra-friendly) so encoders can be swapped entirely from
    config via ``model.encoder._target_``.

    Args:
        embed_dim: dimensionality of the per-cloud embedding returned by
            :meth:`forward`. Must equal the JEPA ``embed_dim`` so the projector
            and predictor line up.
        in_channels: number of coordinate channels per point (``3`` for xyz).
        input_key: overrides the observation key this encoder reads from the
            JEPA info dict (defaults to the class attribute ``"points"``). JEPA
            derives its ``obs_key`` from this, so it must match the key the
            collate_fn writes.
        ground_plane: optional ``(nx, ny, nz, d)`` plane in the RAW sensor
            frame. When set, points with ``|n . xyz + d| < ground_thresh`` are
            removed from every cloud before encoding (see
            :meth:`remove_ground`) -- on this task's lidar the table plane is
            ~80% of the points, so removal spends the encoder's point budget on
            the actual objects. The plane is fixed (sensor and table are static
            across the dataset): fit it once offline (RANSAC) and store it in
            config. ``None`` (default) disables removal entirely.
        ground_thresh: inlier distance (meters, raw frame) for ``ground_plane``.
            Also eats the bottom sliver of objects resting on the plane, so
            keep it near the sensor noise floor.
        ground_min_points: safety floor -- a cloud that removal would shrink
            below this many points is kept WHOLE instead (a near-empty cloud
            would starve downstream sampling/grouping; an all-plane cloud is a
            data problem removal shouldn't silently amplify).
    """

    #: Key under which the packed point-cloud dict lives in the JEPA info dict.
    input_key: str = "points"

    def __init__(
        self,
        embed_dim: int,
        in_channels: int = 3,
        input_key: str | None = None,
        ground_plane: tuple[float, float, float, float] | None = None,
        ground_thresh: float = 0.008,
        ground_min_points: int = 64,
    ):
        super().__init__()
        self._embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)
        if input_key is not None:
            self.input_key = input_key

        if ground_plane is not None:
            assert len(ground_plane) == 4, ground_plane
            plane = torch.tensor(ground_plane, dtype=torch.float32)
            norm = plane[:3].norm()
            assert norm > 0, ground_plane
            plane = plane / norm  # unit normal; d scales with it
        else:
            plane = None
        # Non-persistent buffer: moves with .to(device), never trained, kept
        # OUT of the state dict so the config is the single source of truth
        # (same rationale as the encoders' norm_center).
        self.register_buffer("ground_plane", plane, persistent=False)
        self.ground_thresh = float(ground_thresh)
        self.ground_min_points = int(ground_min_points)

    @property
    def embed_dim(self) -> int:
        """Output embedding dimension (one vector per point cloud)."""
        return self._embed_dim

    @abstractmethod
    def forward(self, data: PackedPointCloud) -> torch.Tensor:
        """Encode a packed batch of point clouds.

        Args:
            data: packed batch with keys ``coord`` (M, in_channels), ``batch``
                (M,) and optional ``feat`` (M, F) which may be ``None``.

        Returns:
            Tensor of shape ``(num_clouds, embed_dim)`` -- one embedding per
            cloud, ordered by ascending batch index.
        """
        raise NotImplementedError

    # -- small shared helpers for subclasses --------------------------------

    def remove_ground(self, data: PackedPointCloud) -> PackedPointCloud:
        """Drop ``ground_plane`` inliers from a packed batch (no-op when unset).

        Removal happens on the RAW coords, before any encoder-side
        normalization, mirroring how the plane was fitted. ``feat`` rows stay
        aligned with the surviving coords. Clouds that would fall below
        ``ground_min_points`` are kept whole (see the class docstring). Called
        by every encoder at the top of ``forward`` so it applies identically in
        training and closed-loop eval.
        """
        if self.ground_plane is None:
            return data
        coord = data["coord"]
        batch = data["batch"].long()
        dist = (coord[:, :3].float() @ self.ground_plane[:3] + self.ground_plane[3]).abs()
        keep = dist >= self.ground_thresh
        if keep.all():
            return data
        n_clouds = int(batch.max().item()) + 1 if batch.numel() else 0
        starved = torch.bincount(batch[keep], minlength=n_clouds) < self.ground_min_points
        if starved.any():
            keep = keep | starved[batch]
        feat = data.get("feat")
        out = dict(data)
        out["coord"], out["batch"] = coord[keep], batch[keep]
        out["feat"] = feat[keep] if feat is not None else None
        return out

    @staticmethod
    def num_clouds(data: PackedPointCloud) -> int:
        """Number of point clouds in a packed batch (max batch index + 1)."""
        batch = data["batch"]
        return int(batch.max().item()) + 1 if batch.numel() else 0

    @staticmethod
    def batch_to_offset(batch: torch.Tensor) -> torch.Tensor:
        """Convert a PyG-style ``batch`` index to a Pointcept-style ``offset``.

        ``offset[i]`` is the exclusive end index of cloud ``i`` in the packed
        tensors (``offset = cumsum(counts)``). Provided because several point
        backbones (Pointcept, Point Transformer v3) expect ``offset`` rather
        than ``batch``; encoders can call this instead of reimplementing it.
        """
        counts = torch.bincount(batch)
        return torch.cumsum(counts, dim=0)
