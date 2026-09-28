"""The four goal-encoder architectures: {MLP, shortcut diffusion} x {-, +z_now}.

All four map a typed 3-D goal specification -- object position(s), optionally
rotation channels (pusht's sensor-frame heading vector), optionally the latent
of the current observation -- to the frozen world-model latent
``z = projector(encoder(cloud))`` that the planner then uses as its CEM target
instead of encoding a recorded goal cloud.

``mlp``       Fourier-featurized coords + Linear-embedded extras into a
              residual pre-norm MLP, one 192-d output. Fits ``E[z | goal]``.
``shortcut``  a shortcut model (Frans et al. 2024, One-Step Diffusion via
              Shortcut Models) over latents: rectified-flow matching whose
              network also takes the step size ``d``, trained jointly with a
              self-consistency loss whose bootstrap targets come from an EMA
              copy of the weights -- exactly the paper's recipe (its official
              implementation defaults to ``bootstrap_ema: 1``, decay 0.999,
              and we measured that live-weight targets collapse mid-training).
              The same weights sample in 1, 2, 4, ... steps; at 1 step the cost
              is one forward pass, same as the MLP.

Each takes the goal spec alone or additionally conditioned on ``z_now``, the
latent of the current observation (``--use-z``): "the objects at this pose, the
rest of the scene continuing from now". At eval ``z_now`` is recomputed at
every replan.

Conditioning encoding
---------------------
* positions go through :class:`FourierFeatures` -- the same log-spaced sin/cos
  bands as the PointViT encoder's ``_SinusoidalPosEmbed`` -- after the
  *encoder's own* affine ``(p - norm_center) / norm_scale``, so the goal
  coordinates live on exactly the scale the tokenizer sees.
* rotation channels (only pusht has any: the T's heading axis as a unit vector
  in the sensor frame) go through a plain Linear. The representation is unique
  and continuous per yaw, so it needs no augmentation.
* ``z_now`` goes through a plain Linear as well; feeding 192 channels through
  the Fourier code would swamp the handful of coordinates carrying the goal.

Augmentation (``pos_jitter``)
-----------------------------
Object positions are near-continuous and effectively unique per frame, and the
Fourier bands resolve ~1 cm -- so a head can use the position as a lookup key
and recall that single frame's latent (train loss collapses, held-out R^2
drops). 5 mm Gaussian jitter on the position channels during training destroys
the key while staying well under the envs' success tolerances: the model is
forced to answer "what do scenes with the objects *around here* look like",
which is exactly what a goal is.

Latents are centered and divided by a single **scalar** RMS (not per-dim):
per-dim whitening would silently re-weight the planner's isotropic
``||rollout - z_goal||^2`` cost.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------- conditioning


class FourierFeatures(nn.Module):
    """Per-axis sin/cos code at log-spaced wavelengths, then one Linear.

    Same construction as ``pc_encoders.pointvit_encoder._SinusoidalPosEmbed``
    (3DETR-style): fixed bands make sub-centimetre offsets linearly separable
    instead of fighting an MLP's spectral bias. Defaults match the encoder's --
    32 bands over normalized wavelengths [0.013, 2.0].
    """

    def __init__(self, in_dim, out_dim, num_bands=32, lambda_min=0.013, lambda_max=2.0):
        super().__init__()
        lam = lambda_max * (lambda_min / lambda_max) ** (
            torch.arange(num_bands, dtype=torch.float32) / (num_bands - 1)
        )
        self.register_buffer("omega", 2.0 * math.pi / lam, persistent=False)
        self.proj = nn.Linear(2 * num_bands * in_dim + in_dim, out_dim)

    def forward(self, x):
        phase = x.unsqueeze(-1) * self.omega  # (B, C, bands)
        code = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(1)
        return self.proj(torch.cat([code, x], dim=-1))  # keep the raw coords too


class ResBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


# --------------------------------------------------------------------- base


class GoalModelBase(nn.Module):
    """Shared plumbing: input normalization/augmentation + the ``goals()`` API.

    The conditioning vector is laid out as
    ``[coords (coord_dim,), rot (rot_dim,), z_now (latent_dim if use_z)]``.
    ``goals(cond, n)`` returns ``(B, n, D)`` goal latents in raw latent space --
    the MLP repeats its single prediction, the shortcut head draws n samples.
    """

    def __init__(self, latent_dim, z_mean, z_scale, norm_center, norm_scale,
                 coord_dim, rot_dim, use_z):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.coord_dim = int(coord_dim)
        self.rot_dim = int(rot_dim)
        self.use_z = bool(use_z)
        self.extra_dim = self.rot_dim + (self.latent_dim if self.use_z else 0)
        self.register_buffer("z_mean", torch.as_tensor(z_mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("z_scale", torch.as_tensor(float(z_scale), dtype=torch.float32))
        # tiled to coord_dim: pusht carries two 3-D points, same affine on each
        center = np.tile(np.asarray(norm_center, dtype=np.float32), coord_dim // 3)
        self.register_buffer("norm_center", torch.as_tensor(center).view(1, -1))
        self.register_buffer("norm_scale", torch.as_tensor(float(norm_scale), dtype=torch.float32))

    #: std (m) of the Gaussian jitter on the position channels during training
    pos_jitter: float = 0.0

    def norm_cond(self, cond):
        """Encoder affine on coords; train-time jitter on the positions."""
        coord = cond[:, : self.coord_dim]
        rest = cond[:, self.coord_dim :]
        if self.training and self.pos_jitter > 0:
            coord = coord + torch.randn_like(coord) * self.pos_jitter
        coord = (coord - self.norm_center) / self.norm_scale
        return torch.cat([coord, rest], dim=1)

    def norm_z(self, z):
        return (z - self.z_mean) / self.z_scale

    def denorm_z(self, zn):
        return zn * self.z_scale + self.z_mean

    def loss(self, cond, z):
        raise NotImplementedError

    def goals(self, cond, n=1, generator=None):
        raise NotImplementedError

    @torch.no_grad()
    def point_estimate(self, cond):
        return self.goals(cond, n=1)[:, 0]


# ---------------------------------------------------------------------- MLP


class MLPHead(GoalModelBase):
    """Deterministic regression: goal spec -> one latent. Fits ``E[z | goal]``."""

    def __init__(self, width=1024, hidden=2048, depth=4, num_bands=32, dropout=0.0, **kw):
        super().__init__(**kw)
        self.coord_embed = FourierFeatures(self.coord_dim, width, num_bands=num_bands)
        self.extra_embed = nn.Linear(self.extra_dim, width) if self.extra_dim else None
        self.blocks = nn.ModuleList(
            [ResBlock(width, hidden, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, self.latent_dim)

    def forward(self, cond):
        cond_n = self.norm_cond(cond)
        h = self.coord_embed(cond_n[:, : self.coord_dim])
        if self.extra_embed is not None:
            h = h + self.extra_embed(cond_n[:, self.coord_dim :])
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.norm(h))  # normalized latent

    def loss(self, cond, z):
        mse = F.mse_loss(self(cond), self.norm_z(z))
        return mse, {"mse_n": mse.detach()}

    @torch.no_grad()
    def goals(self, cond, n=1, generator=None):
        mean = self.denorm_z(self(cond))
        return mean.unsqueeze(1).expand(-1, n, -1)


# ----------------------------------------------------------------- shortcut


class AdaLNBlock(nn.Module):
    """Residual MLP block modulated by a conditioning vector (AdaLN-Zero).

    DiT's conditioning result ported to an MLP: every block receives
    (scale, shift, gate) computed from the condition. The modulation layer is
    zero-initialized, so at init every block is the identity and the condition
    literally gates how much computation turns on.
    """

    def __init__(self, dim, hidden, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.mod = nn.Linear(cond_dim, 3 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        scale, shift, gate = self.mod(c).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + self.fc2(F.gelu(self.fc1(h))) * gate


class ShortcutHead(GoalModelBase):
    """Shortcut model (Frans et al. 2024) over frozen world-model latents.

    Flow matching whose network takes the step size ``d`` as an extra input and
    is trained, in one run with no teacher, so that its ``d``-sized jump equals
    the composition of two ``d/2``-sized jumps. Losses per batch (the paper's
    ~3:1 split, ``consistency_frac = 0.25``):

    * flow matching at the **finest** ``d`` row (``d = 2^-MAX``, standing in for
      ``d -> 0`` exactly as in the paper's implementation): predict ``z1 - z0``
      on ``z_t = (1-t) z0 + t z1``. This row is the base case the whole
      self-consistency chain bootstraps from. A separate ``d = 0`` row would
      leave the finest row untrained and the chain grounded on its random init.
    * self-consistency: for a random power-of-two ``d``, the ``2d`` prediction is
      regressed on the average of two chained ``d`` steps computed by the **EMA
      target network** (paper default ``bootstrap_ema``; the trainer installs it
      as the plain attribute ``_target_net`` so checkpoints stay clean).

    At inference the same weights sample in 1, 2, 4, ... steps.
    """

    #: finest step size is 1/2**MAX_LOG_STEPS
    MAX_LOG_STEPS = 7

    def __init__(self, n_steps=1, t_dim=128, cond_dim=256, width=1024, hidden=2048,
                 depth=4, num_bands=32, consistency_frac=0.25, **kw):
        super().__init__(**kw)
        self.n_steps = int(n_steps)
        self.consistency_frac = float(consistency_frac)
        self.pos_embed = FourierFeatures(self.coord_dim, cond_dim, num_bands=num_bands)
        self.pos_mlp = nn.Sequential(nn.GELU(), nn.Linear(cond_dim, cond_dim))
        self.extra_embed = nn.Linear(self.extra_dim, cond_dim) if self.extra_dim else None
        # t in [0,1] sinusoidal -> MLP; log2 step-size d via an embedding table
        half = t_dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), half))
        self.register_buffer("t_freqs", freqs, persistent=False)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, cond_dim), nn.GELU(),
                                   nn.Linear(cond_dim, cond_dim))
        # row k <-> step size 2^-k, k in [0, MAX]; the finest row doubles as
        # the flow-matching (d ~ 0) row, grounding the bootstrap chain
        self.d_embed = nn.Embedding(self.MAX_LOG_STEPS + 1, cond_dim)
        nn.init.normal_(self.d_embed.weight, std=0.02)

        self.z_in = nn.Linear(self.latent_dim, width)
        self.blocks = nn.ModuleList(
            [AdaLNBlock(width, hidden, cond_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out_mod = nn.Linear(cond_dim, width)
        self.out = nn.Linear(width, self.latent_dim)
        nn.init.zeros_(self.out_mod.weight)
        nn.init.zeros_(self.out_mod.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    # -- embeddings ---------------------------------------------------------
    def _t_emb(self, t):
        ang = t.view(-1, 1) * self.t_freqs.view(1, -1)
        return self.t_mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

    def cond_vec(self, cond_n, t, d_idx):
        c = self.pos_mlp(self.pos_embed(cond_n[:, : self.coord_dim]))
        if self.extra_embed is not None:
            c = c + self.extra_embed(cond_n[:, self.coord_dim :])
        return c + self._t_emb(t) + self.d_embed(d_idx)

    def velocity(self, zn, cond_n, t, d_idx):
        c = self.cond_vec(cond_n, t, d_idx)
        h = self.z_in(zn)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.norm(h) * (1 + self.out_mod(c)))

    # -- training -----------------------------------------------------------
    def loss(self, cond, z):
        cond_n = self.norm_cond(cond)
        z1 = self.norm_z(z)
        b = z1.shape[0]
        dev = z1.device
        n_sc = int(b * self.consistency_frac)
        d_fm = torch.full((b,), self.MAX_LOG_STEPS, device=dev, dtype=torch.long)

        # --- flow-matching part (finest-d row, standing in for d = 0) -------
        z0 = torch.randn_like(z1)
        t = torch.rand(b, device=dev)
        zt = (1 - t).view(-1, 1) * z0 + t.view(-1, 1) * z1
        v = self.velocity(zt, cond_n, t, d_fm)
        fm = F.mse_loss(v[n_sc:], (z1 - z0)[n_sc:])

        # --- self-consistency part on the first n_sc rows -------------------
        sc = fm.new_zeros(())
        if n_sc:
            cn, zn0, zn1 = cond_n[:n_sc], z0[:n_sc], z1[:n_sc]
            # jump size 2d with d = 2^-k, k in [1, MAX]; t on the 2d grid
            k = torch.randint(1, self.MAX_LOG_STEPS + 1, (n_sc,), device=dev)
            d = 2.0 ** (-k.float())
            n_grid = 0.5 / d  # number of 2d-slots in [0, 1)
            slot = (torch.rand(n_sc, device=dev) * n_grid).floor()
            ts = slot * 2 * d
            zts = (1 - ts).view(-1, 1) * zn0 + ts.view(-1, 1) * zn1
            # Bootstrap targets from the EMA target network (the paper's
            # bootstrap_ema; installed by the trainer). Falls back to the live
            # weights only if none was installed.
            tgt = getattr(self, "_target_net", self)
            with torch.no_grad():
                s1 = tgt.velocity(zts, cn, ts, k)
                z_mid = zts + d.view(-1, 1) * s1
                s2 = tgt.velocity(z_mid, cn, ts + d, k)
                target = 0.5 * (s1 + s2)
            pred = self.velocity(zts, cn, ts, k - 1)  # the 2d jump
            sc = F.mse_loss(pred, target)

        total = fm + sc
        return total, {"fm": fm.detach(), "sc": sc.detach()}

    # -- sampling -----------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond, n=1, n_steps=None, generator=None):
        n_steps = n_steps or self.n_steps
        assert n_steps & (n_steps - 1) == 0 and n_steps <= 2 ** self.MAX_LOG_STEPS, \
            f"n_steps must be a power of two <= {2 ** self.MAX_LOG_STEPS}, got {n_steps}"
        cond_n = self.norm_cond(cond).repeat_interleave(n, dim=0)
        b = cond_n.shape[0]
        zn = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        k = int(math.log2(n_steps))
        d_idx = torch.full((b,), k, device=cond.device, dtype=torch.long)
        d = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((b,), i * d, device=cond.device)
            zn = zn + d * self.velocity(zn, cond_n, t, d_idx)
        return self.denorm_z(zn).view(cond.shape[0], n, self.latent_dim)

    def goals(self, cond, n=1, generator=None):
        return self.sample(cond, n=n, generator=generator)


# ------------------------------------------------------------------ factory


def build_model(head, latent_dim, z_mean, z_scale, norm_center, norm_scale,
                coord_dim, rot_dim, use_z, width=1024, hidden=2048, depth=4,
                num_bands=32, dropout=0.0, flow_steps=1, pos_jitter=0.0):
    kw = dict(latent_dim=latent_dim, z_mean=z_mean, z_scale=z_scale,
              norm_center=norm_center, norm_scale=norm_scale,
              coord_dim=coord_dim, rot_dim=rot_dim, use_z=use_z)
    if head == "mlp":
        model = MLPHead(width=width, hidden=hidden, depth=depth,
                        num_bands=num_bands, dropout=dropout, **kw)
    elif head == "shortcut":
        model = ShortcutHead(n_steps=flow_steps, width=width, hidden=hidden,
                             depth=depth, num_bands=num_bands, **kw)
    else:
        raise ValueError(f"unknown head {head!r} (mlp | shortcut)")
    model.pos_jitter = float(pos_jitter)
    return model


def load_goal_model(model_dir, device="cuda"):
    """Load a trained goal model from ``<model_dir>/model.pt`` -> (model, cfg)."""
    from pathlib import Path

    ckpt = torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(
        cfg["head"], latent_dim=cfg["latent_dim"],
        z_mean=np.asarray(cfg["z_mean"]), z_scale=cfg["z_scale"],
        norm_center=cfg["norm_center"], norm_scale=cfg["norm_scale"],
        coord_dim=cfg["coord_dim"], rot_dim=cfg["rot_dim"], use_z=cfg["use_z"],
        width=cfg["width"], hidden=cfg["hidden"], depth=cfg["depth"],
        num_bands=cfg["num_bands"], dropout=cfg["dropout"],
        flow_steps=cfg["flow_steps"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, cfg
