import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Without an initialized process group the Epps-Pulley statistic is computed
    over the local batch, exactly as always. Under DDP (world_size > 1) it is
    instead computed over the UNION batch of all ranks -- rank 0's random
    projections are broadcast and the characteristic-function sums are reduced
    with autograd-aware collectives -- so a 2 x B/2 run optimizes the same
    objective as a single-GPU batch-B run (see :meth:`_forward_ddp` for why the
    gradients also match). forward() is then a collective: every rank must
    reach it the same number of times per step.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D) -- B is the per-rank batch under DDP
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return self._forward_ddp(proj, A)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

    def _forward_ddp(self, proj, A):
        """Epps-Pulley statistic over the union batch of all DDP ranks.

        Equivalent to gathering all shards onto one GPU and running the
        single-GPU forward: the statistic only sees the batch through the
        cos/sin means, and means over equal shards (Lightning's
        DistributedSampler pads/drops to equal per-rank counts) compose by
        summation.
        """
        import torch.distributed as dist
        import torch.distributed.nn  # autograd-aware collectives

        # Every rank must test the SAME projections; without this each rank
        # draws its own A and the reduced sums mix incompatible sketches.
        dist.broadcast(A, src=0)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        B = proj.size(-2) * dist.get_world_size()
        # SUM-reduce the characteristic-function sums with autograd support:
        # forward hands every rank the union-batch sums; backward all-reduces
        # the incoming gradient, and since every rank computes the identical
        # loss that multiplies it by world_size -- exactly cancelling DDP's
        # mean-reduction of parameter gradients. Net effect: parameter
        # gradients equal the single-GPU union-batch gradients.
        cos_mean = torch.distributed.nn.all_reduce(x_t.cos().sum(-3)) / B
        sin_mean = torch.distributed.nn.all_reduce(x_t.sin().sum(-3)) / B
        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * B
        return statistic.mean()
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True, attn_mask=None):
        """
        x : (B, T, D)
        attn_mask : optional (T, T) bool mask (True = may attend); overrides
            ``causal`` -- used for block-causal patterns (TokenPredictor).
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        if attn_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.causal = causal
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), causal=self.causal
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, causal=True):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.causal = causal

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=self.causal)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        causal=True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, causal=causal)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c=None):
        """
        x: (B, T, d)
        c: (B, T, act_dim) action embeddings
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class TokenPredictor(nn.Module):
    """DINO-WM-style token dynamics predictor (arXiv:2411.04983), for
    TOKEN-shaped latents ``(B, T, K, D)`` -- K fixed grid-cell tokens per
    frame (the point-cloud analogue of DINO-WM's DINOv2 patch grid).

    Faithful to DINO-WM's predictor, and deliberately NOT ARPredictor:

    * the action embedding is tiled over the frame's K tokens and
      CONCATENATED onto each token's features (DINO-WM's action injection;
      ARPredictor instead conditions via AdaLN);
    * the backbone is a ViT over all ``T*K`` tokens with a FRAME-block-causal
      mask -- a token attends to every token of its own and earlier frames
      (plain ``is_causal`` over T*K would forbid within-frame attention);
    * the head regresses per-token features: output ``(t, k)`` is the
      prediction for token ``k`` of frame ``t+1`` (same shift semantics as
      ARPredictor, so ``train.lejepa_forward`` needs no changes).

    Positions are factored: a learned per-frame embedding plus a learned
    per-token (grid-cell) embedding -- cells have fixed spatial identity, so
    the token embedding is the patch position embedding of the image case.
    """

    def __init__(
        self,
        *,
        num_frames,
        num_tokens,
        input_dim,
        action_emb_dim,
        hidden_dim,
        depth,
        heads,
        dim_head=64,
        mlp_dim=2048,
        output_dim=None,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_proj = nn.Linear(input_dim + action_emb_dim, hidden_dim)
        self.pos_frame = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim) * 0.02)
        self.pos_token = nn.Parameter(torch.randn(1, 1, num_tokens, hidden_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            nn.ModuleList([
                Attention(hidden_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(hidden_dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim or input_dim)
        # Frame-block-causal mask over T*K tokens (frame-major layout, so a
        # leading square slice stays valid for any T <= num_frames).
        mask = torch.tril(torch.ones(num_frames, num_frames, dtype=torch.bool))
        mask = mask.repeat_interleave(num_tokens, 0).repeat_interleave(num_tokens, 1)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x, c):
        """
        x: (B, T, K, D) token latents
        c: (B, T, A) action embeddings
        """
        B, T, K, _ = x.shape
        assert K == self.num_tokens, (K, self.num_tokens)
        c = c.unsqueeze(2).expand(-1, -1, K, -1)  # tile per token
        h = self.in_proj(torch.cat([x, c], dim=-1))
        h = h + self.pos_frame[:, :T] + self.pos_token
        h = self.dropout(h).reshape(B, T * K, -1)
        m = self.attn_mask[: T * K, : T * K]
        for attn, ff in self.layers:  # pre-norm residual (norms live inside)
            h = h + attn(h, attn_mask=m)
            h = h + ff(h)
        h = self.norm(h)
        return self.head(h).view(B, T, K, -1)


class LatentDifferenceActionDecoder(nn.Module):
    """Delta-JEPA's Latent Difference Action Decoder (LDAD, from the Delta-JEPA paper).

    Reconstructs the ``horizon`` actions executed between two observations from
    their latent displacement alone: {a_t .. a_{t+N-1}} = D(z_{t+N} - z_t).
    ``horizon`` learnable action queries pass through a non-causal Transformer
    whose AdaLN conditioning injects the displacement into every block. The
    decoder never sees an absolute state, so action recovery must be carried by
    the transition geometry -- trained jointly with the prediction loss this
    prevents latent collapse and separates action-conditioned transitions for
    planning. rollout/get_cost never call it; only the optional ldad_cem / ldad
    planners of eval_lidar.py do.
    """

    def __init__(
        self,
        *,
        latent_dim,
        action_dim,
        horizon=5,
        depth=3,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        dropout=0.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_queries = nn.Parameter(torch.randn(1, horizon, latent_dim) * 0.02)
        self.transformer = Transformer(
            latent_dim,
            latent_dim,
            action_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            causal=False,
        )

    def forward(self, delta):
        """
        delta: (B, D) latent displacements z_{t+N} - z_t
        returns: (B, horizon, action_dim) reconstructed action sequences
        """
        queries = self.action_queries.expand(delta.size(0), -1, -1)
        return self.transformer(queries, delta.unsqueeze(1))

    def loss(self, emb, actions):
        """Action reconstruction loss over every horizon-N span of a window.

        emb: (B, T, D) per-frame latents; actions: (B, T, A). ``actions[:, t]``
        drives the t -> t+1 transition, so the span starting at s targets
        ``actions[:, s : s+N]``. Requires T >= N + 1; with the default window
        of exactly N + 1 frames there is a single span (the paper's Eq. 6).
        """
        N = self.horizon
        starts = range(emb.size(1) - N)
        delta = torch.cat([emb[:, s + N] - emb[:, s] for s in starts], dim=0)
        target = torch.cat([actions[:, s : s + N] for s in starts], dim=0)
        return F.mse_loss(self(delta), target)
