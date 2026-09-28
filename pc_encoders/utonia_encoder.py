"""Frozen Utonia backbone as a JEPA observation encoder — the Utonia-WM arm.

This arm reproduces DINO-WM's recipe on point clouds. That recipe is: a
FROZEN pretrained foundation encoder supplies the
latent space, only the dynamics predictor is trained on top, and planning
minimizes distance in that frozen space. This encoder is the point-cloud
instantiation of that recipe: the pretrained model is Utonia (Pointcept's
cross-domain Point Transformer V3, ICML'26 — vendored under ``utonia/``, see
its README), the analogue of DINOv2 for point clouds. Everything here is
frozen and deterministic; the JEPA around it trains the predictor with the
prediction loss ONLY (no SIGReg / LDAD — the targets are fixed frozen
features, so latent collapse is impossible by construction).

Preprocessing per cloud (all constants from config, never per-frame
statistics — per-frame centering would erase the absolute cube position):

    raw sensor-frame cloud
      -> optional fixed-plane ground removal (base class; off by default)
      -> CANONICAL FRAME: rotate the fitted table plane onto the xy-plane
         (z up), table at z = 0, workspace center at the xy origin. Utonia
         was trained on gravity-aligned scenes; its README says exactly this
         for non-indoor data ("ensure ... the road plane aligned with the
         xy-plane").
      -> x ``coord_scale``: Utonia rescales every domain to a shared
         perceptual granularity; its tabletop-manipulation demo uses
         ``transform.default(4.0)``, so 4.0 is the default here.
      -> 1 cm voxel dedup (GridSample equivalent), DETERMINISTIC: first
         point per voxel in packed-row order (upstream's mode="train" picks
         a random one; the encoder is frozen, so augmentation buys nothing
         and determinism keeps closed-loop eval reproducible).
      -> feat = [coord, 0, 0, 0, 0, 0, 0]: the checkpoint's unified modality
         interface is (coord, color, normal) with "default zeros for missing
         modalities" (paper sec. on Causal Modality Blinding — geometry-only
         input is IN-distribution, the model was trained with modality
         dropout). NOTE colors would enter as color/255, so zeros are the
         correct raw fill.
      -> frozen PTv3 encoder (enc_mode: hierarchical, output = coarsest
         stage, 576-d per super-point, ~1k super-points per cloud here)
      -> TOKEN GRID: super-points are binned by their centroid into a fixed
         ``grid_dims`` (default 8x8x4 = 256) grid of cells spanning
         ``grid_bounds`` (canonical-frame constants from config — the table
         bbox is bit-stable across frames, so the cells are spatially
         corresponded across frames/episodes, the point-cloud analogue of
         DINO-WM's DINOv2 patch grid; DINO-WM's default is likewise 256
         patches). Cells are pooled deterministically ("meanmax": mean || max
         per cell, 1152-d; ``segment_csr`` after a stable sort — no atomics,
         no learned readout: a trained tokenizer could collapse under a pure
         prediction loss). Empty cells are all-zero tokens: occupancy is
         implicit in the feature magnitude, and predicting a cell going
         empty is part of the dynamics. Out-of-bounds points clamp into the
         edge cells.
      -> FIXED random orthogonal projection to ``embed_dim`` PER TOKEN
         (default 192 = the other arms' latent width). Seeded and frozen — a
         Johnson-Lindenstrauss-style compression that keeps relative
         distances while staying training-free, so the collapse-impossible
         property of the frozen space is preserved (a TRAINED projection
         could zero everything out under the pure prediction loss). Identity
         when ``embed_dim`` equals the pooled width (set ``embed_dim: 1152``
         to plan in the raw per-cell pooled space).
      -> fp16 ROUND-TRIP: features are quantized through float16 and back.
         The embedding cache stores fp16 (256 tokens x 2M frames does not fit
         disk at fp32); rounding the live path identically holds cached
         training features and closed-loop eval features to the SAME
         quantization grid (see "How close is cache to live" below).

    Output: ``(num_clouds, num_tokens, embed_dim)`` — one token per grid
    cell, cells flattened x-major (x * Dy * Dz + y * Dz + z).

Determinism (within a process; see the cache-vs-live note below for across):
the checkpoint is loaded with ``enable_flash=False`` (so results
never depend on whether flash-attn is installed — the vendored SDPA patch
keeps memory sane, see utonia/README.md) and ``shuffle_orders`` is forced off
on the model AND on every GridPooling stage (upstream hardcodes it True
there — the 13% run-to-run feature variance we measured came from exactly
that). ``traceable`` is forced off too: we never unpool, and parents retained
per stage are pure memory overhead. Every ``SubMConv3d`` is pinned to spconv's
NATIVE algorithm: spconv otherwise chooses an implicit-GEMM variant by
benchmarking it once per process, which made the features reproducible within
a run but differ by up to 2 fp16 ULPs BETWEEN runs — enough to break the
cache-equals-live-encode guarantee this arm depends on. The conv BIASES are
additionally stripped from spconv and added back explicitly (``_backbone``):
Native's inference path applies a bias only when the GEMM kernel its runtime
tuner picked happens to fuse it, and silently drops it otherwise, keyed on
the exact row count — so the same conv could apply the bias for a 16-cloud
batch and drop it for a single cloud, which put a batch-built cache ~1.1
absolute away from a live single-cloud encode of the very same frame.

Batching is safe — but only because of a fix in the vendored backbone. Upstream's
non-flash attention set, per forward and per stage, ``patch_size = min(smallest
cloud in the batch, patch_size_max)``, because its padding refused to pad a
cloud smaller than one patch. That coupled clouds: measured here, the same cloud
encoded beside 63 same-size neighbours moved **1.4% (rel L2)**, and beside a
300-point cloud **56%**. ``utonia/model.py`` now pads every cloud up to a whole
number of patches and masks the padding of a sub-patch cloud (vendored patch #2
— which is what upstream's flash path does via per-cloud ``cu_seqlens``), so
``patch_size`` stays at the configured 1024 no matter the batch. Verified: the
fix leaves unbatched features BIT-IDENTICAL, and after it a cloud's features
vary by at most half an fp16 cache step (rel L2 <= 0.003%) across batch sizes 1
to 64, without growing. ``max_clouds_per_forward`` is therefore a pure
throughput knob (~9 clouds/s unbatched vs ~19 at 16, i.e. 62 h vs 29 h for the
2M-frame cache) and is excluded from the cache signature.

One caveat worth knowing before chasing a small discrepancy: this backbone
AMPLIFIES bit-level noise. Traced on a 300-point cloud, changing the batch
perturbs the first block's output by 2e-8 (identical inputs, different GEMM
tiling) and 24 blocks later that is 0.16 -- roughly 2x growth per block. On this
dataset it stays harmless because every cloud is the same size (~9850 points, so
1069-1109 coarse tokens) and the measured spread across batch sizes is <= half
an fp16 step; but a cloud far smaller than its batch-mates can move by ~1%.
``precompute_utonia.py --verify`` is what checks this empirically on the real
data, comparing single-cloud encodes against a batch-built cache.

Empty clouds (a frame whose every ray missed) are skipped rather than batched
-- the backbone cannot run on zero points -- and keep an all-zero token grid,
which is exactly what "no returns anywhere" should encode.

How close is cache to live: within ONE fp16 quantization step, not bit-exact.
Measured on real frames (precompute process vs. a fresh live encode of the
same clouds): 0.08% of entries differ, every one of them by a single fp16
rounding step (max 9.8e-4 on tokens of magnitude ~3.4). Exact equality is not
attainable because spconv and cuBLAS select kernels by benchmarking them at
runtime, so the choice — and the last bits of the result — can differ between
processes; the fp16 cache grid then turns a sub-ULP difference into an
occasional single-step flip. What matters is that every SYSTEMATIC divergence
is gone (batch-composition patch sizing was up to 56%, ambient autocast 7%),
leaving
a residual ~4 orders of magnitude below the difference between two different
scenes, which is what the planning cost actually measures. Do not "fix" a
1-ULP mismatch by rebuilding the cache; a LARGER one is a real pipeline bug.

Cached-embedding passthrough: frozen features are constant, so training runs
on embeddings precomputed once by ``precompute_utonia.py`` (a 137M-parameter
backbone at ~110 ms/cloud would otherwise dominate every epoch). The cache is
served through the data pipeline as "``num_tokens``-point clouds" whose
"points" ARE the tokens: a packed batch whose coord width equals
``embed_dim`` (and differs from ``in_channels``) is reshaped to
``(num_clouds, num_tokens, embed_dim)`` and returned. Live clouds (width
``in_channels``) always take the full path — closed-loop eval needs no
special casing. The backbone is lazy-loaded on first LIVE forward, so cached
training and unit tests never read the 550 MB checkpoint.

The backbone is deliberately HIDDEN from the nn.Module tree (kept in a plain
dict slot, not a submodule): it never trains, so this keeps its 137M
parameters out of the optimizer, out of every exported ``weights_epoch_*.pt``
(which would otherwise grow by ~550 MB per epoch), and out of ``.to()`` —
the device move happens lazily per forward instead.
"""

from pathlib import Path

import torch

from .base import PackedPointCloud, PointCloudEncoder

# Repo root (this file lives in <root>/pc_encoders/): relative checkpoint
# paths resolve against it so the same config works from any working
# directory -- Hydra runs train/eval from per-run output dirs.
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Stamp of the FEATURE-PRODUCING CODE in this module, carried in every
#: embedding cache's signature (see utils.encoder_cache_signature). The config
#: signature alone cannot notice that the implementation changed, so a cache
#: built before such a change would be silently reused with features the
#: current code would never produce. BUMP THIS whenever a code edit changes the
#: returned features -- preprocessing, tokenization, pooling, the projection or
#: its numerics -- and existing caches will be refused instead of trusted.
#: The current stamp (5) covers: a fixed token grid with a per-cloud
#: projection and an fp16 round-trip, the whole per-cloud pipeline
#: (canonicalization and dedup included) run per cloud so no step's numerics
#: depend on batch composition, ambient autocast disabled around the encode, spconv pinned to
#: its Native algorithm so features are reproducible across processes, the
#: attention patch size fixed at its configured value so clouds can be
#: batched, and the conv biases applied explicitly (see ``_backbone``) because
#: spconv's Native inference path drops a fused bias for some tuned kernels.
#: ``precompute_utonia.py --verify`` checks an existing cache against the
#: live encoder.
FEATURE_VERSION = 5


def _add_stripped_bias(module, inputs, output):
    """Forward hook restoring the conv bias stripped in ``_backbone``.

    spconv's Native inference path fuses the bias into the GEMM epilogue
    only for SOME tuned kernels and silently drops it for the rest (see the
    comment at the strip site). A plain elementwise add cannot be skipped by
    any kernel choice, and matches the fused epilogue bitwise (same two fp32
    operands).
    """
    return output.replace_feature(
        output.features + module.subm_bias.to(output.features.dtype)
    )


class UtoniaEncoder(PointCloudEncoder):
    """Frozen Utonia (PTv3) features, tokenized on a fixed grid. See module
    docstring. Output: ``(num_clouds, num_tokens, embed_dim)``.

    Args:
        embed_dim: output dim PER TOKEN (192 by default = the other arms'
            latent width). When it differs from the per-cell pooled width
            (``feature_dim`` x 2 for ``"meanmax"``), a fixed seeded random
            orthogonal projection maps pooled -> ``embed_dim`` (see module
            docstring); when equal, the pooled features pass unprojected.
        grid_dims: token grid resolution ``(Dx, Dy, Dz)``; ``num_tokens`` is
            their product (default 8x8x4 = 256 = DINO-WM's patch count).
            Adjustable per environment together with ``grid_bounds``.
        grid_bounds: ``((lx, ly, lz), (hx, hy, hz))`` grid extent in CANONICAL
            METERS (after the canonical transform, before ``coord_scale``) —
            fixed config constants, never per-frame statistics, so cells keep
            their spatial identity across frames. Defaults span the
            cube table's bbox (measured bit-stable) plus a
            small margin; out-of-bounds points clamp into edge cells.
        in_channels: coordinate channels of LIVE clouds (3). Must differ from
            ``embed_dim`` (the cached-passthrough discriminator).
        feature_dim: the backbone's coarsest-stage channel width (576 for the
            released Utonia checkpoint); asserted against the actual forward.
        proj_seed: seed of the fixed random projection. Part of the feature
            signature: the embedding cache, training and live eval all
            regenerate the SAME matrix from it (it is also persisted in the
            weight exports as a buffer).
        ckpt: Utonia checkpoint ``.pth``; relative paths resolve against the
            repo root. Pre-download with
            ``huggingface_hub.hf_hub_download("Pointcept/Utonia", "utonia.pth",
            local_dir="checkpoints/utonia")`` or ``precompute_utonia.py``
            (offline machines: copy the file, no network needed at run time).
        pool: per-CELL readout over the coarsest-stage super-point features
            binned into the cell: ``"mean"`` | ``"max"`` | ``"meanmax"``
            (concat, default).
        coord_scale: perceptual-granularity rescale applied after the
            canonical transform (4.0 = Utonia's own tabletop-manipulation
            demo setting).
        grid_size: voxel size of the dedup grid in SCALED canonical units
            (upstream default 0.01).
        canonical_plane: table plane ``(nx, ny, nz, d)`` in the RAW sensor
            frame (defaults to the plane fitted on the cube table — the same
            values ground removal uses when enabled). Defines
            the canonical frame: plane -> z=0, normal -> +z. The normal sign
            is auto-oriented so ``canonical_center`` lands at positive z.
        canonical_center: workspace center (raw sensor frame) mapped to the
            canonical xy origin; sits above the table, which also fixes the
            plane orientation.
        max_clouds_per_forward: how many clouds share one backbone forward.
            A pure THROUGHPUT knob since vendored patch #2 (see above): ~9
            clouds/s at 1, ~19 at 16, and a cloud's features shift by at most
            half an fp16 cache step either way. 16 peaks at ~2 GB on this
            dataset's density; raise it on a bigger card.
        bf16: run the frozen backbone under bf16 autocast (~0.5% relative
            feature error; only ~15% faster in the mandatory one-cloud loop,
            which is launch-bound — it was ~2x back when the backbone was
            batched). MUST match between the embedding cache and live eval —
            precompute_utonia.py records it in the cache meta and train.py
            cross-checks. Default off: exact fp32 everywhere. ``forward``
            disables any ambient (Lightning) autocast around the whole encode,
            so trainer precision never leaks in and this flag is the only
            thing that sets the backbone's precision.
        input_key / ground_*: see :class:`PointCloudEncoder`. Ground removal
            (when enabled) runs BEFORE the canonical transform, on raw
            coords, exactly like the PointViT arms.
    """

    def __init__(
        self,
        embed_dim=192,
        in_channels=3,
        ckpt="checkpoints/utonia/utonia.pth",
        pool="meanmax",
        feature_dim=576,
        proj_seed=0,
        grid_dims=(8, 8, 4),
        grid_bounds=((-0.80, -0.82, -0.01), (0.90, 0.82, 0.50)),
        coord_scale=4.0,
        grid_size=0.01,
        canonical_plane=(0.628104, 0.0, -0.778129, -0.638995),
        canonical_center=(1.27, 0.0, 0.25),
        max_clouds_per_forward=16,
        bf16=False,
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
        assert embed_dim != in_channels, (
            "embed_dim == in_channels would make cached embeddings "
            "indistinguishable from live clouds", embed_dim, in_channels
        )
        assert pool in ("mean", "max", "meanmax"), pool
        self.ckpt = str(ckpt)
        self.pool = pool
        self.feature_dim = int(feature_dim)
        self.pooled_dim = self.feature_dim * (2 if pool == "meanmax" else 1)
        self.coord_scale = float(coord_scale)
        self.grid_size = float(grid_size)
        self.max_clouds_per_forward = int(max_clouds_per_forward)
        assert self.max_clouds_per_forward >= 1, max_clouds_per_forward
        self.bf16 = bool(bf16)

        # Token grid: cell edges are config constants in the SCALED canonical
        # frame (grid_bounds are canonical meters; the backbone sees scaled
        # coords, so scale once here). Non-persistent buffers, config is the
        # single source of truth (same rationale as canon_R/canon_t).
        # element-wise float()/int() so nested OmegaConf ListConfigs (from the
        # YAML) convert as reliably as plain tuples
        dims = torch.tensor([int(v) for v in grid_dims], dtype=torch.long)
        lo = torch.tensor([float(v) for v in grid_bounds[0]]) * self.coord_scale
        hi = torch.tensor([float(v) for v in grid_bounds[1]]) * self.coord_scale
        assert dims.shape == (3,) and bool((dims > 0).all()), grid_dims
        assert lo.shape == hi.shape == (3,) and bool((hi > lo).all()), grid_bounds
        self.num_tokens = int(dims.prod())
        self.register_buffer("grid_dims", dims, persistent=False)
        self.register_buffer("grid_lo", lo, persistent=False)
        self.register_buffer("grid_cell", (hi - lo) / dims, persistent=False)

        # Fixed random ORTHOGONAL projection pooled -> embed_dim (see module
        # docstring). Regenerated identically from proj_seed at every
        # instantiation (cache build, training, eval), and additionally
        # persisted in the state dict so weight exports carry the exact
        # matrix. Identity (no buffer) when the dims already match.
        if embed_dim != self.pooled_dim:
            assert embed_dim < self.pooled_dim, (embed_dim, self.pooled_dim)
            g = torch.Generator().manual_seed(int(proj_seed))
            A = torch.randn(self.pooled_dim, embed_dim, generator=g, dtype=torch.float64)
            Q, _ = torch.linalg.qr(A)  # orthonormal columns: distance-preserving
            self.register_buffer("frozen_proj", Q.float(), persistent=True)
        else:
            self.frozen_proj = None

        # Canonical frame: R maps the (auto-oriented) plane normal to +z,
        # t puts the plane at z=0 and the workspace center at the xy origin.
        # Constants derived from config -> non-persistent buffers (same
        # rationale as norm_center in the PointViT encoder).
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

        # Frozen backbone: lazy-loaded, and hidden from the module tree (a
        # dict slot is invisible to nn.Module registration) -- see module
        # docstring for why. _backbone_slot["model"] is None until the first
        # live forward.
        self._backbone_slot = {"model": None}

    # -- backbone -----------------------------------------------------------

    def _resolve_ckpt(self) -> str:
        p = Path(self.ckpt)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(
                f"Utonia checkpoint not found at {p}. Download it once with:\n"
                "  pixi run python -c \"from huggingface_hub import hf_hub_download; "
                "hf_hub_download('Pointcept/Utonia', 'utonia.pth', "
                "local_dir='checkpoints/utonia')\"\n"
                "(or copy it there on offline machines)"
            )
        return str(p)

    def _backbone(self, device):
        model = self._backbone_slot["model"]
        if model is None:
            import spconv.pytorch as spconv
            import utonia  # vendored, repo root
            from spconv.core import ConvAlgo
            from utonia.model import GridPooling

            model = utonia.load(
                self._resolve_ckpt(),
                # deterministic + flash-free everywhere (see module docstring)
                custom_config=dict(enable_flash=False, shuffle_orders=False),
            )
            # Upstream hardcodes shuffle_orders=True / traceable=True inside
            # every GridPooling stage (the model-level flags don't reach
            # them); flip the attributes post-hoc -- both are read at forward
            # time.
            for m in model.modules():
                if isinstance(m, GridPooling):
                    m.shuffle_orders = False
                    m.traceable = False
            # Pin every position-encoding sparse conv to spconv's NATIVE
            # algorithm. spconv's default picks the implicit-GEMM variant by
            # BENCHMARKING it at first use and caching the winner per process,
            # so the features were reproducible within a run but NOT across
            # runs: two precompute processes on identical input differed by up
            # to 2 fp16 cache ULPs, which is exactly the cache-vs-eval
            # mismatch this arm cannot tolerate. Native is timing-independent.
            #
            # ...for the CONVOLUTION. Its BIAS is not: spconv 2.3.8's Native
            # inference path (SPCONV_CPP_GEMM ConvGemmOps.indice_conv) applies
            # the bias only when the GEMM kernel its runtime tuner picked
            # happens to fuse it, and SILENTLY DROPS it otherwise -- and the
            # tuner keys on the exact row count, so the same conv applies the
            # bias for one batch composition and drops it for another (found
            # via precompute_utonia.py --verify: a batch-built cache row and a
            # single-cloud live encode of the same frame disagreed by exactly
            # -bias at every cpe conv, amplified to ~1.1 absolute after 24
            # blocks; see tests/test_utoniawm.py::test_spconv_bias_stripped).
            # So take the bias away from spconv entirely: run every conv
            # bias-free (bit-equal to the reference algo, measured 2.6e-8) and
            # add the bias back OURSELVES as a plain elementwise add, which no
            # kernel choice can skip.
            for m in model.modules():
                if isinstance(m, spconv.SubMConv3d):
                    m.algo = ConvAlgo.Native
                    if m.bias is not None:
                        m.register_buffer("subm_bias", m.bias.detach().clone())
                        m.register_parameter("bias", None)
                        m.register_forward_hook(_add_stripped_bias)
            model.eval()
            model.requires_grad_(False)
            self._backbone_slot["model"] = model
        if next(model.parameters()).device != device:
            model.to(device)
        return model

    # -- preprocessing ------------------------------------------------------

    def _canonicalize(self, coord):
        """Raw sensor coords -> scaled canonical frame (see module docstring)."""
        return (coord @ self.canon_R.T + self.canon_t) * self.coord_scale

    def _voxel_dedup(self, coord, batch, n_clouds):
        """Deterministic 1-point-per-voxel subsample of the packed batch.

        Equivalent of upstream GridSample: voxel key per point (per-cloud
        min-rebased integer grid coords), keep the FIRST point of each
        (cloud, voxel) in packed-row order -- a stable sort by key keeps
        row order within equal keys, so the group heads are the
        deterministic representatives. Returns (kept row indices,
        rebased grid_coord of the kept rows).
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
        keep = order[head].sort().values  # back to packed-row order
        return keep, gc[keep]

    # -- forward ------------------------------------------------------------

    def forward(self, data: PackedPointCloud) -> torch.Tensor:
        coord = data["coord"].float()
        batch = data["batch"].long()
        n_clouds = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_clouds == 0:
            return coord.new_zeros((0, self.num_tokens, self.embed_dim))

        # Cached-embedding passthrough: precomputed tokens arrive as
        # "num_tokens-point clouds" whose "points" ARE the tokens (see
        # module docstring). Checked BEFORE ground removal -- embeddings
        # are not geometry. Rows per cloud keep cache row order == flat
        # cell order (the collate packs frames in order and never permutes
        # points within a frame).
        if coord.shape[1] == self.embed_dim:
            counts = torch.bincount(batch, minlength=n_clouds)
            assert bool((counts == self.num_tokens).all()), (
                "cached-token batch must hold exactly num_tokens rows per "
                f"cloud, got counts {counts.unique().tolist()} != {self.num_tokens}"
            )
            return coord.view(n_clouds, self.num_tokens, self.embed_dim)

        assert coord.shape[1] == self.in_channels, (coord.shape, self.in_channels)

        model = self._backbone(coord.device)
        # Clouds are sliced by cumulative counts, which assumes the packed rows
        # are grouped by ascending cloud index (what collate_point_cloud emits).
        # Asserted because an unsorted batch would silently mix clouds together.
        assert bool((batch.diff() >= 0).all()), "packed batch index is not sorted"
        counts = torch.bincount(batch, minlength=n_clouds)
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        out = coord.new_zeros(n_clouds, self.num_tokens, self.embed_dim)
        # Empty clouds are dropped from the backbone work (their token grid stays
        # zero). Since they own no rows, the surviving clouds remain contiguous,
        # so a chunk of them is still one row slice.
        alive = (counts > 0).nonzero(as_tuple=True)[0]
        # Ambient autocast OFF for the whole encode (Lightning trains under
        # precision: bf16, and both _canonicalize and the token projection are
        # autocast-eligible matmuls): precision must be pinned by self.bf16
        # alone, which _encode_chunk re-enables for the backbone only. Without
        # this, a live encode under a bf16 trainer produced features ~7% (rel L2)
        # away from the fp32 ones the cache and closed-loop eval hold.
        with torch.autocast(coord.device.type, enabled=False):
            for i0 in range(0, len(alive), self.max_clouds_per_forward):
                ids = alive[i0 : i0 + self.max_clouds_per_forward]
                r0, r1 = int(starts[ids[0]]), int(starts[ids[-1] + 1])
                sub = torch.repeat_interleave(
                    torch.arange(len(ids), device=coord.device), counts[ids]
                )
                out[ids] = self._encode_chunk(model, coord[r0:r1], sub, len(ids))
        # fp16 round-trip: the embedding cache stores fp16 (disk budget);
        # quantizing the live path identically puts both on the same grid
        # (see "How close is cache to live" in the module docstring).
        return out.half().float()

    def _encode_chunk(self, model, coord, batch, n_clouds):
        """Full pipeline for a chunk -> ``(n_clouds, num_tokens, embed_dim)``."""
        data = self.remove_ground({"coord": coord, "batch": batch, "feat": None})
        coord, batch = data["coord"].float(), data["batch"].long()

        coord = self._canonicalize(coord)
        keep, grid_coord = self._voxel_dedup(coord, batch, n_clouds)
        coord, batch = coord[keep], batch[keep]

        # Unified modality interface: [coord | color=0 | normal=0].
        feat = torch.cat([coord, coord.new_zeros(len(coord), 6)], dim=1)
        point = dict(
            coord=coord,
            grid_coord=grid_coord.int(),
            feat=feat,
            batch=batch,
        )
        # Frozen forward: never any grad, and the ONLY place bf16 may apply --
        # forward() has already disabled any ambient (Lightning) autocast, so
        # precision is pinned by self.bf16 alone and cached, live-train and eval
        # features all match.
        with torch.no_grad(), torch.autocast(
            coord.device.type, dtype=torch.bfloat16, enabled=self.bf16
        ):
            enc = model(point)
        f = enc["feat"].float()
        assert f.shape[-1] == self.feature_dim, (
            f"backbone stage width {f.shape[-1]} != configured "
            f"feature_dim {self.feature_dim}"
        )
        # Bin super-points by centroid into the fixed token grid. The coarsest
        # stage's coords are the mean positions of the pooled points
        # (GridPooling reduces coord with "mean"), already scaled canonical.
        cell = self._cell_index(enc["coord"].float())
        tok = self._tokenize(f, cell, enc["batch"].long(), n_clouds)
        if self.frozen_proj is not None:  # fixed orthogonal pooled -> embed_dim
            tok = tok @ self.frozen_proj
        return tok

    def _cell_index(self, coord):
        """Flat token-cell index per point (scaled canonical coords).

        Out-of-bounds points clamp into the edge cells; flat order is x-major
        (``x * Dy * Dz + y * Dz + z``), matching the cache layout.
        """
        idx = torch.floor((coord - self.grid_lo) / self.grid_cell).long()
        idx = idx.clamp(min=torch.zeros_like(self.grid_dims), max=self.grid_dims - 1)
        return (idx[:, 0] * self.grid_dims[1] + idx[:, 1]) * self.grid_dims[2] + idx[:, 2]

    def _tokenize(self, f, cell, ob, n_clouds):
        """Deterministic per-cell pooling -> ``(n_clouds, num_tokens, pooled_dim)``.

        Stable sort by (cloud, cell) key then ``segment_csr`` -- no atomics,
        so the result is bit-reproducible (fp16 rounding downstream must
        never see run-to-run summation-order noise). Empty cells stay zero.
        """
        import torch_scatter

        K = self.num_tokens
        key = ob * K + cell
        order = torch.argsort(key, stable=True)
        ks = key[order]
        head = torch.ones_like(ks, dtype=torch.bool)
        head[1:] = ks[1:] != ks[:-1]
        uniq = ks[head]
        idx_of_head = head.nonzero(as_tuple=True)[0]
        ptr = torch.cat([idx_of_head, idx_of_head.new_tensor([len(ks)])])
        fs = f[order]
        pooled = []
        if self.pool in ("mean", "meanmax"):
            pooled.append(torch_scatter.segment_csr(fs, ptr, reduce="mean"))
        if self.pool in ("max", "meanmax"):
            pooled.append(torch_scatter.segment_csr(fs, ptr, reduce="max"))
        pooled = torch.cat(pooled, dim=-1)
        dense = f.new_zeros(n_clouds * K, self.pooled_dim)
        dense[uniq] = pooled
        return dense.view(n_clouds, K, self.pooled_dim)

