"""Hand-crafted per-cell point statistics as a JEPA observation encoder --
the Vox-WM arm (voxstats configs), the geometric baseline for Utonia-WM.

Same recipe as the Utonia-WM arm (pc_encoders/utonia_encoder.py) with the
frozen 137M-parameter foundation backbone swapped for plain per-cell
statistics of the 3D points: identical preprocessing (canonical table frame,
``coord_scale``, 1 cm voxel dedup), identical fixed token grid (``grid_dims``
cells spanning ``grid_bounds``, flattened x-major), identical TokenPredictor
and prediction-only loss. The ONLY change is what a token holds, so the
comparison answers: does the frozen Utonia encoder buy anything over raw
geometry at identical tokenization?

    raw sensor-frame cloud
      -> optional fixed-plane ground removal (base class)
      -> canonical frame + coord_scale + 1 cm voxel dedup
         (kept in LOCKSTEP with UtoniaEncoder -- the helpers are copied
         verbatim and tests/test_voxstats.py cross-checks the two)
      -> per-cell statistics over the cell's surviving points (below)

      -> optional FIXED orthonormal lift FEATURE_DIM -> ``embed_dim`` per token
         (see "Token width" below)

    Output: ``(num_clouds, num_tokens, embed_dim)`` -- one token per grid
    cell, cells flattened x-major (x * Dy * Dz + y * Dz + z).

Token width. The statistics vector is FEATURE_DIM = 25 wide. With
``embed_dim == FEATURE_DIM`` tokens are the raw statistics (the original
variant, whose predictor ran 25 wide). With ``embed_dim > FEATURE_DIM``
(the default configs use 192, the Utonia-WM token width) every token is
mapped through a FIXED matrix ``Q`` (``embed_dim x FEATURE_DIM``) with
orthonormal columns, drawn once from ``lift_seed`` (QR of a seeded Gaussian,
the same construction as UtoniaEncoder's fixed JL projection, only up instead
of down). The lift is training-free and isometric -- ``|Q a - Q b| = |a - b|``
-- so the latent space keeps every property below (deterministic, collapse-
proof, exact train == eval features) and the planning cost is unchanged up to
the predictor; what changes is that the TokenPredictor now has the SAME width,
parameter count and per-token capacity as the Utonia-WM arm, so the two arms
differ only in what a token holds. The matrix is exported with the weights
(persistent buffer, like UtoniaEncoder.frozen_proj) and re-derived from
``lift_seed`` at build time, so a checkpoint can never be paired with a
different lift.

Per-cell features (FEATURE_DIM = 25). Every channel is O(1) BY CONSTRUCTION
from fixed config constants -- never data-derived normalization -- so, like
the Utonia arm, the latent space is a deterministic function of the input:
training targets are fixed, collapse is impossible, and live eval reproduces
training features exactly. Offsets ``o = (coord - cell_center) / cell_size``
are per-axis normalized to [-0.5, 0.5] (out-of-bounds points are clamped onto
the grid box first -- the stats analogue of UtoniaEncoder clamping them into
edge cells); statistics of ABSOLUTE positions would be dominated by the
cell-center constant the predictor's per-token position embedding already
carries.

    0      occupancy      1.0 if the cell holds any (deduped) point
    1      log-count      log1p(n) / log1p(count_norm), n AFTER dedup -- the
                          dedup normalizes ray density, so n measures occupied
                          surface, not distance to the sensor
    2:5    mean           per-axis mean of o
    5:8    std            per-axis std of o (population; 0 for a single point)
    8:11   cov            off-diagonal covariances of o -- (xy, xz, yz)
    11:14  min            per-axis min of o
    14:17  max            per-axis max of o
    17:25  sub-occupancy  2x2x2 sub-cell occupancy bits, x-major
                          (sx * 4 + sy * 2 + sz, where s_i = [o_i >= 0])

Empty cells are all-zero tokens; the occupancy channel disambiguates "empty"
from "one point exactly at the cell center". An empty CLOUD (every ray
missed) keeps an all-zero token grid, exactly like the Utonia arm.

Determinism: reductions run through ``segment_csr`` after a stable sort by
(cloud, cell) key -- no atomics, and a cloud's packed rows stay contiguous
through the sort, so its features are BIT-IDENTICAL whatever else shares the
batch. ``forward`` disables any ambient (Lightning bf16) autocast around the
whole encode, so trainer precision never leaks into the features and live
closed-loop eval (fp32, no trainer) sees the training latents exactly.

No cache, no checkpoint, no fp16 round-trip: the statistics cost microseconds
per cloud, so training consumes RAW clouds live through the ordinary lidar
data configs (config/train/data/lidar_*.yaml) -- none of the Utonia arm's
precompute machinery applies. The encoder has no trainable parameters; its
only state is the fixed lift matrix above (a function of ``lift_seed``), so
the config remains the single source of truth.
"""

import math

import torch

from .base import PackedPointCloud, PointCloudEncoder

#: Width of the per-cell statistics vector documented above. ``embed_dim``
#: (the token width the predictor sees) is taken from config and must be
#: >= FEATURE_DIM: equal means raw statistics tokens, larger means the fixed
#: orthonormal lift documented in the module docstring.
FEATURE_DIM = 25


class VoxelStatsEncoder(PointCloudEncoder):
    """Per-cell point statistics on the Utonia-WM token grid. See module
    docstring. Output: ``(num_clouds, num_tokens, embed_dim)``.

    Args:
        embed_dim: token width handed to the predictor. ``FEATURE_DIM`` (= 25)
            outputs the raw statistics; a larger value applies the fixed
            orthonormal lift (192 = the Utonia-WM arm's width). Smaller is
            rejected (it would have to discard statistics).
        lift_seed: seed of the fixed lift matrix (only used when
            ``embed_dim > FEATURE_DIM``). Same seed -> bit-identical lift.
        grid_dims / grid_bounds / coord_scale / grid_size / canonical_plane /
            canonical_center: identical semantics (and defaults) to
            :class:`~pc_encoders.utonia_encoder.UtoniaEncoder` -- keep them
            equal to the environment's utoniawm model config so the two arms
            tokenize the same cells.
        count_norm: fixed scale of the log-count channel:
            ``log1p(n) / log1p(count_norm)`` is ~1 for a cell holding
            ``count_norm`` deduped points. A config constant, never a dataset
            statistic (see module docstring).
        in_channels / input_key / ground_*: see :class:`PointCloudEncoder`.
    """

    def __init__(
        self,
        embed_dim=FEATURE_DIM,
        in_channels=3,
        grid_dims=(8, 8, 4),
        grid_bounds=((-0.80, -0.82, -0.01), (0.90, 0.82, 0.50)),
        coord_scale=4.0,
        grid_size=0.01,
        canonical_plane=(0.628104, 0.0, -0.778129, -0.638995),
        canonical_center=(1.27, 0.0, 0.25),
        count_norm=256,
        lift_seed=0,
        input_key="points",
        ground_plane=None,
        ground_thresh=0.008,
        ground_min_points=64,
    ):
        super().__init__(
            embed_dim=embed_dim, in_channels=in_channels, input_key=input_key,
            ground_plane=ground_plane, ground_thresh=ground_thresh,
            ground_min_points=ground_min_points,
        )
        assert embed_dim >= FEATURE_DIM, (
            "embed_dim must be >= the statistics width (raw tokens at "
            f"{FEATURE_DIM}, or a fixed orthonormal lift above it)", embed_dim
        )
        self.lift_seed = int(lift_seed)
        if embed_dim > FEATURE_DIM:
            # Fixed orthonormal lift FEATURE_DIM -> embed_dim (module docstring):
            # QR of a seeded Gaussian, computed in float64 on the CPU so it is
            # identical on every machine; persistent so the weight export
            # pins it (UtoniaEncoder.frozen_proj does the same).
            g = torch.Generator().manual_seed(self.lift_seed)
            A = torch.randn(embed_dim, FEATURE_DIM, generator=g, dtype=torch.float64)
            Q, _ = torch.linalg.qr(A)  # (embed_dim, FEATURE_DIM), orthonormal columns
            self.register_buffer("lift", Q.float(), persistent=True)
        else:
            self.lift = None
        self.coord_scale = float(coord_scale)
        self.grid_size = float(grid_size)
        assert count_norm >= 1, count_norm
        self.count_norm = int(count_norm)
        self._log_count_norm = math.log1p(self.count_norm)

        # Token grid: cell edges are config constants in the SCALED canonical
        # frame -- copied verbatim from UtoniaEncoder (non-persistent buffers,
        # config is the single source of truth).
        dims = torch.tensor([int(v) for v in grid_dims], dtype=torch.long)
        lo = torch.tensor([float(v) for v in grid_bounds[0]]) * self.coord_scale
        hi = torch.tensor([float(v) for v in grid_bounds[1]]) * self.coord_scale
        assert dims.shape == (3,) and bool((dims > 0).all()), grid_dims
        assert lo.shape == hi.shape == (3,) and bool((hi > lo).all()), grid_bounds
        self.num_tokens = int(dims.prod())
        self.register_buffer("grid_dims", dims, persistent=False)
        self.register_buffer("grid_lo", lo, persistent=False)
        self.register_buffer("grid_hi", hi, persistent=False)
        self.register_buffer("grid_cell", (hi - lo) / dims, persistent=False)

        # Canonical frame: identical construction to UtoniaEncoder (Rodrigues
        # rotation of the fitted plane normal onto +z, plane -> z=0, workspace
        # center -> xy origin) -- cross-checked by tests/test_voxstats.py.
        plane = torch.tensor(canonical_plane, dtype=torch.float64)
        n, d = plane[:3] / plane[:3].norm(), plane[3] / plane[:3].norm()
        center = torch.tensor(canonical_center, dtype=torch.float64)
        if torch.dot(n, center) + d < 0:  # orient: workspace center above table
            n, d = -n, -d
        z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        v = torch.linalg.cross(n, z)
        c = torch.dot(n, z)
        assert 1 + c > 1e-6, "plane normal is anti-parallel to z; flip canonical_plane"
        V = torch.tensor(
            [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]],
            dtype=torch.float64,
        )
        R = torch.eye(3, dtype=torch.float64) + V + V @ V / (1 + c)  # Rodrigues
        t = torch.cat([-(R @ center)[:2], d.reshape(1)])
        self.register_buffer("canon_R", R.float(), persistent=False)
        self.register_buffer("canon_t", t.float(), persistent=False)

    # -- preprocessing (copied verbatim from UtoniaEncoder) ------------------

    def _canonicalize(self, coord):
        """Raw sensor coords -> scaled canonical frame (see module docstring)."""
        return (coord @ self.canon_R.T + self.canon_t) * self.coord_scale

    def _voxel_dedup(self, coord, batch, n_clouds):
        """Deterministic 1-point-per-voxel subsample of the packed batch.

        Identical to UtoniaEncoder._voxel_dedup: voxel key per point
        (per-cloud min-rebased integer grid coords), keep the FIRST point of
        each (cloud, voxel) in packed-row order. Returns kept row indices.
        """
        gc = torch.floor(coord / self.grid_size).long()
        gmin = torch.full((n_clouds, 3), torch.iinfo(torch.long).max,
                          device=coord.device, dtype=torch.long)
        gmin.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, 3), gc,
                             reduce="amin", include_self=False)
        gc = gc - gmin[batch]
        assert int(gc.max()) < (1 << 15), (
            "grid coords exceed the 15-bit voxel key budget -- cloud extent "
            f"{int(gc.max())} voxels; lower coord_scale or raise grid_size"
        )
        key = (batch << 45) | (gc[:, 0] << 30) | (gc[:, 1] << 15) | gc[:, 2]
        order = torch.argsort(key, stable=True)
        ks = key[order]
        head = torch.ones_like(ks, dtype=torch.bool)
        head[1:] = ks[1:] != ks[:-1]
        return order[head].sort().values  # back to packed-row order

    # -- forward ------------------------------------------------------------

    def forward(self, data: PackedPointCloud) -> torch.Tensor:
        coord = data["coord"].float()
        batch = data["batch"].long()
        n_clouds = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_clouds == 0:
            return coord.new_zeros((0, self.num_tokens, self.embed_dim))
        assert coord.shape[1] == self.in_channels, (coord.shape, self.in_channels)
        assert bool((batch.diff() >= 0).all()), "packed batch index is not sorted"

        # Ambient autocast OFF for the whole encode (same rationale as the
        # Utonia arm: the trainer runs precision bf16 and _canonicalize is an
        # autocast-eligible matmul; the features must be identical between
        # bf16 training and fp32 closed-loop eval).
        with torch.autocast(coord.device.type, enabled=False):
            data = self.remove_ground({"coord": coord, "batch": batch, "feat": None})
            coord, batch = data["coord"].float(), data["batch"].long()
            coord = self._canonicalize(coord)
            keep = self._voxel_dedup(coord, batch, n_clouds)
            feats = self._featurize(coord[keep], batch[keep], n_clouds)
            if self.lift is not None:  # fixed isometric lift to embed_dim
                feats = feats @ self.lift.T
            return feats

    def _featurize(self, coord, batch, n_clouds):
        """Per-cell statistics -> ``(n_clouds, num_tokens, FEATURE_DIM)``.

        Stable sort by (cloud, cell) key then ``segment_csr`` -- no atomics,
        bit-reproducible, and independent of batch composition (a cloud's
        rows stay contiguous and in packed order through the sort).
        """
        import torch_scatter

        K = self.num_tokens
        # Out-of-bounds points land on the grid-box face of their edge cell,
        # keeping every offset inside [-0.5, 0.5] (UtoniaEncoder clamps the
        # cell INDEX only; features here are positions, so the coordinate
        # itself must be boxed).
        coord = coord.clamp(min=self.grid_lo, max=self.grid_hi)
        rel = (coord - self.grid_lo) / self.grid_cell
        idx = rel.long().clamp(min=torch.zeros_like(self.grid_dims),
                               max=self.grid_dims - 1)
        cell = (idx[:, 0] * self.grid_dims[1] + idx[:, 1]) * self.grid_dims[2] + idx[:, 2]
        o = (rel - idx.float() - 0.5).clamp(-0.5, 0.5)  # cell-relative offsets
        # 2x2x2 sub-cell bit per axis: which half of the cell (x-major flat).
        sub = (o >= 0).long()
        sub_flat = (sub[:, 0] * 2 + sub[:, 1]) * 2 + sub[:, 2]

        key = batch * K + cell
        order = torch.argsort(key, stable=True)
        ks, os_ = key[order], o[order]
        head = torch.ones_like(ks, dtype=torch.bool)
        head[1:] = ks[1:] != ks[:-1]
        uniq = ks[head]
        idx_of_head = head.nonzero(as_tuple=True)[0]
        ptr = torch.cat([idx_of_head, idx_of_head.new_tensor([len(ks)])])

        mean = torch_scatter.segment_csr(os_, ptr, reduce="mean")
        mean_sq = torch_scatter.segment_csr(os_ * os_, ptr, reduce="mean")
        std = (mean_sq - mean * mean).clamp_min(0).sqrt()
        prods = torch.stack(
            [os_[:, 0] * os_[:, 1], os_[:, 0] * os_[:, 2], os_[:, 1] * os_[:, 2]], dim=1
        )
        cov = torch_scatter.segment_csr(prods, ptr, reduce="mean") - torch.stack(
            [mean[:, 0] * mean[:, 1], mean[:, 0] * mean[:, 2], mean[:, 1] * mean[:, 2]],
            dim=1,
        )
        omin = torch_scatter.segment_csr(os_, ptr, reduce="min")
        omax = torch_scatter.segment_csr(os_, ptr, reduce="max")
        counts = (ptr[1:] - ptr[:-1]).float()
        logn = torch.log1p(counts) / self._log_count_norm
        occ = torch.ones_like(logn)

        dense = coord.new_zeros(n_clouds * K, FEATURE_DIM)
        dense[uniq, :17] = torch.cat(
            [occ.unsqueeze(1), logn.unsqueeze(1), mean, std, cov, omin, omax], dim=1
        )
        # Sub-occupancy: plain index assignment of the constant 1.0 -- write
        # collisions all carry the same value, so this stays deterministic.
        subocc = coord.new_zeros(n_clouds * K * 8)
        subocc[key * 8 + sub_flat] = 1.0
        dense[:, 17:] = subocc.view(-1, 8)
        return dense.view(n_clouds, K, FEATURE_DIM)
