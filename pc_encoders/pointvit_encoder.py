"""PointViT — a Point-BERT/Uni3D-style tokenizer feeding the *same* ViT the image
model uses, with its CLS token read out as the per-cloud latent.

This is the point-cloud analogue of the image ViT in this repo
(``stable_pretraining.backbone.utils.vit_hf``, wired in
``config/train/model/point_lewm_<env>.yaml`` at ``vit_size: tiny``): that model
splits an image into patches, prepends a learned
CLS token, runs a bidirectional transformer, and JEPA reads
``last_hidden_state[:, 0]`` (the CLS) as the frame embedding (see
``jepa.JEPA.encode``). Here the "patches" are local point neighborhoods:

    packed {coord, batch, feat}
        -> centers: ``num_tokens`` FPS centers per cloud
        -> ball query: ``group_size`` points sampled uniformly at random from
           the ``group_radius`` ball around each center
        -> tiny PointNet (LN): one token embedding per group  (num_tokens tokens)
        -> + center positional embedding (3DETR-style sinusoidal + Linear)
        -> prepend the ViT's learned CLS token                (num_tokens + 1)
        -> the SAME HuggingFace ViT (vit_hf) transformer + final LayerNorm
        -> read out the CLS token -> Linear -> (num_clouds, embed_dim)

Rather than re-implement a transformer, this **builds around the exact ViT the
image model uses**: it instantiates ``vit_hf(size=...)`` and reuses that model's
own ``cls_token``, transformer ``layers`` and final ``layernorm`` (HF's
``ViTModel.forward`` is just ``embeddings -> layers -> layernorm``; we swap the
image patch-embedding for the point tokenizer and drive the rest ourselves). So
the transformer is byte-for-byte the image backbone -- same size presets (tiny =
192-dim / 12-layer / 3-head, small, base, ...), same init, same code path, and
it would pick up any future pretrained ViT the image side gains. The ViT's own
``patch_embeddings``/``position_embeddings`` are left unused (a handful of dead
params); points get a coordinate-based positional embedding instead.

This is the tokenizer of Point-BERT / Uni3D (FPS + local grouping + a small
PointNet per patch, then a plain ViT), which is also the encoder R3D (2026)
builds on -- with their kNN grouping swapped for PointNet++'s ball query
(radius + random subsample): each token's receptive field is a FIXED physical
scale instead of a density-dependent one, so tokens on the dense table plane
and on the sparse cube silhouette describe neighborhoods of the same size.
Two further choices follow that line of work:

* **LayerNorm only, no BatchNorm.** The tokenizer and pos-embed here are LN, and
  the HF ViT is LN throughout. R3D traces the "big 3D encoders underperform for
  control" *scaling paradox* to BatchNorm's batch statistics behaving
  differently between train and eval and under the live-vs-dataset shift of
  closed-loop planning; LN removes that gap. It is also what the rest of this
  repo already does (``module.Block``).
* **A single grouping stage, not a hierarchy.** Uni3D/R3D use ONE FPS+kNN
  tokenizer and let the transformer's global self-attention do the rest, rather
  than stacking PointNet++ set-abstraction layers. Stacking would re-coarsen the
  cloud twice before the transformer ever sees it, which on this lidar (the cube
  is ~0.3% of points) risks averaging the small object away. A hierarchical
  set-abstraction encoder would need multi-scale pooling to avoid this.

Note on the JEPA contract: JEPA wants one vector per cloud, so we read out the
CLS token (R3D instead keeps the whole token set and feeds a diffusion
transformer -- not an option here without changing JEPA). The CLS is the
learned, permutation-invariant readout the transformer fills in, which is a
strictly more expressive pool than mean/max pooling.

Coordinate handling: a *fixed* affine
``(coord - norm_center) / norm_scale`` (bounds measured on the dataset once,
stored in config) maps raw sensor-frame coords to ~[-1, 1]. It is constant, not
per-cloud -- per-cloud standardization would erase the absolute cube position,
which is the task signal. Neighbor offsets fed to the PointNet are divided by
``norm_scale`` too, so local geometry is on the same scale.
"""

import torch
from torch import nn

from .base import PackedPointCloud, PointCloudEncoder
from .sampling import fps_index


class _SinusoidalPosEmbed(nn.Module):
    """3DETR-style per-axis sinusoidal position code, mixed by one Linear.

    Deterministic sin/cos of each (normalized) coordinate axis at ``num_bands``
    log-spaced wavelengths in ``[lambda_min, lambda_max]``, followed by a single
    learned Linear. 3DETR's ``PositionEmbeddingCoordsSine('sine')`` feeds the
    raw code with temperature-spaced frequencies over per-scene [0,1] coords;
    here the coords come from the encoder's FIXED workspace affine instead
    (per-scene normalization would erase the absolute cube position), the
    wavelength range is capped explicitly, and the Linear lets the model
    re-weight/mix bands.

    Wavelengths are in normalized units: physical wavelength = lambda *
    norm_scale. The defaults (2.0 .. 0.013 at norm_scale 0.75 m) span the full
    workspace down to ~1 cm -- fine enough to localize the cube to sub-cm, but
    capped above the sensor's noise scale on curved silhouettes so the top
    bands don't amplify jitter into the latent.
    """

    def __init__(self, in_channels, out_dim, num_bands=32, lambda_min=0.013, lambda_max=2.0):
        super().__init__()
        assert num_bands >= 2 and 0 < lambda_min < lambda_max, (num_bands, lambda_min, lambda_max)
        lam = lambda_max * (lambda_min / lambda_max) ** (
            torch.arange(num_bands, dtype=torch.float32) / (num_bands - 1)
        )
        # Derived from config args -> non-persistent, like norm_center: the
        # config stays the single source of truth, nothing hides in the ckpt.
        self.register_buffer("omega", 2.0 * torch.pi / lam, persistent=False)
        self.proj = nn.Linear(2 * num_bands * in_channels, out_dim)

    def forward(self, x):
        """(G, in_channels) normalized coords -> (G, out_dim)."""
        phase = x.unsqueeze(-1) * self.omega                      # (G, C, B)
        code = torch.cat([phase.sin(), phase.cos()], dim=-1)      # (G, C, 2B)
        return self.proj(code.flatten(1))                         # (G, out_dim)


class _PointNetTokenizer(nn.Module):
    """Point-BERT mini-PointNet: one token embedding per point group.

    For each group of ``group_size`` points (already centered and scaled),
    a shared per-point MLP lifts them, a max-pool summarizes the group, the
    per-point features are concatenated with that summary (the Point-BERT
    "double max-pool" trick, so each point sees its group context), a second
    MLP projects to ``out_dim`` and a final max-pool yields the group token.
    All norms are LayerNorm.

    Input ``x`` is ``(G, group_size, in_dim)`` -- relative xyz (and optional
    per-point feat) of each group; output is ``(G, out_dim)``.
    """

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        h1, h2 = hidden
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(),
            nn.Linear(h1, h1), nn.LayerNorm(h1), nn.GELU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * h1, h2), nn.LayerNorm(h2), nn.GELU(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x):
        f = self.mlp1(x)                       # (G, k, h1)
        g = f.max(dim=1, keepdim=True).values  # (G, 1, h1)
        f = torch.cat([f, g.expand_as(f)], dim=-1)  # (G, k, 2*h1)
        f = self.mlp2(f)                       # (G, k, out_dim)
        return f.max(dim=1).values             # (G, out_dim)


class PointViTEncoder(PointCloudEncoder):
    """FPS + ball query + PointNet tokenizer -> the image model's ViT -> CLS readout.

    Args:
        embed_dim: output embedding dim (must match JEPA ``embed_dim``). The CLS
            token (of the ViT's hidden width) is linearly projected to this.
        in_channels: coordinate channels per point (``3`` for xyz).
        num_tokens: number of FPS centers = number of ViT patch tokens per cloud
            (Point-BERT/Uni3D use 256-512). A cloud with fewer points than this
            gets its centers repeated (see ``_sample_and_group``).
        group_size: points fed to the PointNet per center, sampled uniformly at
            random from that center's ``group_radius`` ball (see :meth:`_group`).
        group_radius: ball-query radius around each center, in RAW sensor-frame
            meters (grouping happens before the norm affine). Tune with ``viz_radius.py`` per environment.
        group_max_neighbors: cap on candidate points the radius query collects
            per center before the random subsample. torch_cluster truncates a
            fuller ball to the FIRST ``group_max_neighbors`` points in packed
            row order (scan order -- NOT random), so keep this comfortably
            above the typical ball occupancy (viz_radius.py reports it).
        vit_size: which ``vit_hf`` size preset to build the transformer from --
            ``"tiny"`` (192-dim / 12-layer / 3-head; what the image model uses),
            ``"small"``, ``"base"``, ``"large"``, ``"giant"``. The hidden width
            (= token width) is taken from this preset, not set here.
        vit_pretrained: passed to ``vit_hf`` -- load google ViT weights (only the
            transformer layers are reused; the image patch-embed is unused). Off
            by default; there is no point-cloud-pretrained ViT here.
        vit_patch_size, image_size: forwarded to ``vit_hf``. They only size the
            ViT's (unused) image patch-embedding / position-embedding, so the
            values are immaterial; kept to match the image config's defaults.
        tokenizer_hidden: (h1, h2) widths of the mini-PointNet tokenizer.
        use_feat: concatenate the packed batch's per-point ``feat`` (recorded
            colors in [0, 1], raw) to each neighbor's relative xyz. Requires an
            RGB lidar data config (``lidar_contains_rgb: true``).
        feat_channels: channels of that ``feat`` stream (3 for rgb).
        norm_center: constant coordinate offset (workspace center, sensor frame)
            subtracted before tokenizing; ``len == in_channels``.
        norm_scale: constant isotropic scale dividing the centered coords (and the
            neighbor offsets); ~workspace half-extent so inputs land in ~[-1, 1].
        pos_embed_type: center positional embedding flavor. ``"sinusoidal"``
            (default) is :class:`_SinusoidalPosEmbed`; ``"mlp"`` is a
            Point-BERT-style learned coordinate MLP (Linear-GELU-Linear over
            the raw normalized xyz). No shipped config uses ``"mlp"``.
        pos_num_bands: sin/cos frequency bands per coordinate axis in the
            sinusoidal center positional embedding (:class:`_SinusoidalPosEmbed`).
        pos_lambda_min, pos_lambda_max: wavelength range of those bands, in
            normalized coordinate units (physical = lambda * ``norm_scale``).
            Defaults span the workspace (~1.5 m) down to ~1 cm at
            ``norm_scale=0.75`` -- capped above the sensor noise scale.
            All three are ignored for ``pos_embed_type="mlp"``.
        input_key: JEPA info-dict key holding the packed cloud.
        ground_plane, ground_thresh, ground_min_points: optional fixed-plane
            ground removal applied to the raw cloud before sampling/grouping --
            see :class:`pc_encoders.base.PointCloudEncoder`. With the table
            gone, the ``num_tokens`` FPS centers cover the objects instead of
            tiling the plane (~80% of this lidar's points).
    """

    def __init__(
        self,
        embed_dim,
        in_channels=3,
        num_tokens=256,
        group_size=32,
        group_radius=0.05,
        group_max_neighbors=128,
        vit_size="tiny",
        vit_pretrained=False,
        vit_patch_size=16,
        image_size=224,
        tokenizer_hidden=(128, 256),
        use_feat=False,
        feat_channels=3,
        norm_center=(0.0, 0.0, 0.0),
        norm_scale=1.0,
        pos_embed_type="sinusoidal",
        pos_num_bands=32,
        pos_lambda_min=0.013,
        pos_lambda_max=2.0,
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
        from stable_pretraining.backbone.utils import vit_hf  # same ViT the image model uses

        assert len(norm_center) == in_channels, (norm_center, in_channels)
        assert num_tokens > 0 and group_size > 0, (num_tokens, group_size)
        assert group_radius > 0, group_radius
        assert group_max_neighbors >= group_size, (group_max_neighbors, group_size)
        self.num_tokens = int(num_tokens)
        self.group_size = int(group_size)
        self.group_radius = float(group_radius)
        self.group_max_neighbors = int(group_max_neighbors)
        self.use_feat = bool(use_feat)
        self.feat_channels = int(feat_channels) if self.use_feat else 0

        # The transformer IS the image backbone: reuse its cls_token, layers and
        # final layernorm (HF ViTModel.forward is embeddings -> layers -> layernorm;
        # we replace the image patch-embedding with the point tokenizer below).
        self.vit = vit_hf(
            size=vit_size, patch_size=vit_patch_size, image_size=image_size,
            pretrained=vit_pretrained,
        )
        self.token_dim = int(self.vit.config.hidden_size)

        # Fixed affine buffer (never trained, kept OUT of the state dict so the
        # config is the single source of truth).
        self.register_buffer(
            "norm_center",
            torch.tensor(norm_center, dtype=torch.float32).view(1, -1),
            persistent=False,
        )
        self.norm_scale = float(norm_scale)

        # Tokenizer input = relative xyz (+ optional per-point feat) per neighbor.
        tok_in = in_channels + self.feat_channels
        self.tokenizer = _PointNetTokenizer(tok_in, tokenizer_hidden, self.token_dim)
        # Center positional embedding (3DETR-style sinusoidal, see
        # _SinusoidalPosEmbed), added once to each group token before layer 0 --
        # input-only, exactly where 3DETR's encoder keeps position (its
        # per-layer q/k injection is a decoder-side scheme). An MLP over raw
        # xyz is smoothly varying (spectral bias) and localizes the cube only
        # coarsely. Fixed sin/cos bands make sub-cm offsets linearly
        # separable. (The ViT's own image position_embeddings are meaningless
        # for points and are left unused.) "mlp" selects a learned
        # coordinate-MLP embedding instead.
        if pos_embed_type == "sinusoidal":
            self.pos_embed = _SinusoidalPosEmbed(
                in_channels, self.token_dim,
                num_bands=pos_num_bands,
                lambda_min=pos_lambda_min, lambda_max=pos_lambda_max,
            )
        elif pos_embed_type == "mlp":
            self.pos_embed = nn.Sequential(
                nn.Linear(in_channels, self.token_dim), nn.GELU(),
                nn.Linear(self.token_dim, self.token_dim),
            )
        else:
            raise ValueError(
                f"pos_embed_type must be 'sinusoidal' or 'mlp', got {pos_embed_type!r}"
            )
        self.out_proj = nn.Linear(self.token_dim, embed_dim)

    def forward(self, data: PackedPointCloud) -> torch.Tensor:
        data = self.remove_ground(data)  # no-op unless ground_plane is configured
        coord = data["coord"].float()
        batch = data["batch"].long()
        feat = data.get("feat")
        if self.use_feat:
            assert feat is not None, (
                "use_feat=True but the packed batch carries no per-point feat -- "
                "the dataset/collate must supply lidar colors (lidar_contains_rgb)"
            )
            feat = feat.float()

        n_clouds = int(batch.max().item()) + 1 if batch.numel() else 0
        if n_clouds == 0:
            return coord.new_zeros((0, self.embed_dim))
        # Center positions (n_clouds * num_tokens, 3) and their radius groups
        # (num_clouds, num_tokens, group_size). FPS centers are input rows, but
        # grouping/tokenizing work on POSITIONS, not row indices.
        centers = coord[self._fps_centers(coord, batch, n_clouds).reshape(-1)]
        nbr_idx = self._group(coord, batch, centers, n_clouds)

        # Tokenizing touches n_clouds * num_tokens * group_size rows -- MORE
        # rows than the raw cloud -- and autograd would retain every widening
        # intermediate (tens of GB at batch_size 256). Checkpointing recomputes
        # the tokenizer during backward instead: numerically exact (the segment
        # has no dropout/RNG), ~3x lower peak GPU memory, one extra tokenizer
        # forward per step. (`centers` enters as a tensor input to the
        # segment.)
        if self.training and torch.is_grad_enabled():
            tokens = torch.utils.checkpoint.checkpoint(
                self._tokenize, coord, feat, centers, nbr_idx, use_reentrant=False
            )
        else:
            tokens = self._tokenize(coord, feat, centers, nbr_idx)
        tokens = tokens.view(n_clouds, self.num_tokens, self.token_dim)

        # Prepend the ViT's own learned CLS token; run the image backbone's
        # transformer over [CLS, patch tokens]. attention_mask=None -> full
        # bidirectional attention, exactly as the image ViT treats its patches.
        cls = self.vit.embeddings.cls_token.expand(n_clouds, -1, -1)
        x = torch.cat([cls, tokens.to(cls.dtype)], dim=1)
        for layer in self.vit.layers:
            x = layer(x, None)
        x = self.vit.layernorm(x)
        return self.out_proj(x[:, 0])                       # CLS -> (num_clouds, embed_dim)

    def _tokenize(self, coord, feat, center, nbr_idx):
        """PointNet each center's radius group into one token.

        ``center`` is ``(n_clouds * num_tokens, 3)`` POSITIONS (gathered
        input rows). Returns
        ``(n_clouds * num_tokens, token_dim)`` -- flat so the forward can
        reshape once. Kept as a single method because it is the
        gradient-checkpointed segment (see ``forward``): everything gathered
        here (the ``(G, k, ...)`` neighbor tensors and the PointNet
        intermediates) is freed after the forward pass and rebuilt in backward.
        """
        G = center.shape[0]
        nbr = coord[nbr_idx.reshape(G, self.group_size)]    # (G, k, 3)

        # Relative, scale-normalized neighbor coords -> local geometry per group.
        rel = (nbr - center.unsqueeze(1)) / self.norm_scale  # (G, k, 3)
        tok_in = rel
        if self.use_feat:
            nbr_feat = feat[nbr_idx.reshape(G, self.group_size)]  # (G, k, F)
            tok_in = torch.cat([rel, nbr_feat], dim=-1)

        tokens = self.tokenizer(tok_in)                      # (G, token_dim)
        center_n = (center - self.norm_center) / self.norm_scale
        return tokens + self.pos_embed(center_n)            # (G, token_dim)

    def _sample_and_group(self, coord, batch, n_clouds):
        """FPS ``num_tokens`` centers per cloud and radius-group each center.

        Returns ``(center_idx, nbr_idx)``: ``center_idx`` is ``(n_clouds,
        num_tokens)`` and ``nbr_idx`` is ``(n_clouds, num_tokens, group_size)``,
        both row-indices into the packed arrays. Kept as the FPS-path
        convenience wrapper (tests/viz call it); ``forward`` composes
        :meth:`_fps_centers` and :meth:`_group` directly.
        """
        center_idx = self._fps_centers(coord, batch, n_clouds)
        nbr_idx = self._group(coord, batch, coord[center_idx.reshape(-1)], n_clouds)
        return center_idx, nbr_idx

    def _fps_centers(self, coord, batch, n_clouds):
        """FPS ``num_tokens`` center row-indices per cloud, ``(n_clouds, T)``.

        Centers come from :func:`pc_encoders.sampling.fps_index`
        (deterministic torch_cluster FPS, exactly ``num_tokens`` per cloud). A
        cloud with fewer than ``num_tokens`` points has its centers repeated
        cyclically (rare -- this lidar has thousands of points per cloud); the
        duplicated tokens are not masked (the HF ViT path takes no key-padding
        mask, exactly as the image ViT never masks patches).
        """
        T = self.num_tokens
        # <= T FPS centers per cloud, grouped by cloud (== T except tiny clouds).
        flat = fps_index(coord, batch, T, n_clouds=n_clouds)
        per = torch.bincount(batch[flat], minlength=n_clouds)  # min(n_c, T) each
        # (n_clouds, T) center matrix; a cloud short of T repeats cyclically.
        start = per.cumsum(0) - per  # exclusive cumsum: each cloud's first row in `flat`
        local = torch.arange(T, device=coord.device).unsqueeze(0) % per.unsqueeze(1)
        return flat[start.unsqueeze(1) + local]  # (n_clouds, T)

    def _group(self, coord, batch, centers, n_clouds):
        """Ball query: ``group_size`` random in-radius points per center.

        PointNet++-style grouping: for each center,
        ``torch_cluster.radius`` collects up to ``group_max_neighbors``
        candidate points within ``group_radius`` (raw sensor-frame meters,
        like the coords at this stage), and ``group_size`` of them are drawn
        uniformly at random -- without replacement; a center whose ball holds
        fewer than ``group_size`` points has its candidates repeated
        cyclically (every FPS center is an input point, so a ball is never
        empty). Unlike kNN the receptive field is a FIXED physical scale,
        independent of local point density.

        ``centers`` is ``(n_clouds * num_tokens, 3)`` positions grouped by
        cloud. Fully batched: one ``radius`` call + one segmented shuffle over
        the whole packed batch. NOTE the subsample is stochastic -- a fresh
        draw every forward (train and eval alike).
        """
        import torch_cluster

        T, k = self.num_tokens, self.group_size
        G = centers.shape[0]
        center_batch = torch.arange(n_clouds, device=coord.device).repeat_interleave(T)
        # detach: group MEMBERSHIP is discrete (no gradient); center gradients
        # flow through the relative coords / pos-embed in _tokenize instead.
        edges = torch_cluster.radius(
            coord, centers.detach(), self.group_radius, batch, center_batch,
            max_num_neighbors=self.group_max_neighbors,
        )
        # Uniform random order inside each center's candidate set: shuffle the
        # edge list globally, then stable-sort by center. Taking the first k of
        # each segment is then a uniform without-replacement draw.
        perm = torch.randperm(edges.shape[1], device=coord.device)
        row, col = edges[0][perm], edges[1][perm]
        order = row.argsort(stable=True)
        row, col = row[order], col[order]
        counts = torch.bincount(row, minlength=G)
        assert int(counts.min()) >= 1, "radius query returned an empty ball"
        start = counts.cumsum(0) - counts  # each center's first edge
        local = torch.arange(k, device=coord.device).unsqueeze(0) % counts.unsqueeze(1)
        return col[start.unsqueeze(1) + local].view(n_clouds, T, k)
