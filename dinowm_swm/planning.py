"""stable-worldmodel ``Costable`` wrapper around a trained DINO-WM run, so
``eval.py`` plans with it through the SAME CEM / sequence-matched protocol as
the LeWM image checkpoints (config/eval/image_*.yaml).

    model = load_dinowm_cost_model("experiment_logs/dinowm_pusht", "model_latest.pth")  # <run_dir>, checkpoints/<ckpt>
    cost  = model.get_cost(info_dict, action_candidates)   # (B, S)

Loading follows upstream ``plan.py``: the pickled ``predictor`` /
``action_encoder`` / ``proprio_encoder`` come from ``checkpoints/<ckpt>``,
the frozen DINOv2 encoder is re-instantiated from ``hydra.yaml``, and the
pieces are assembled into upstream's ``VWorldModel``. The cost is upstream's
planning objective (``planning.objectives.create_objective_fn`` mode
``last``): MSE between the LAST predicted frame's patch tokens and the goal
frame's tokens, averaged over patches x dims (the proprio term is identically
zero for the noprop arm and is left out). The rollout is upstream's
``VWorldModel.rollout`` step for step, re-implemented only to (a) cache the
history / goal encodings across the CEM iterations that reuse one info dict
(encoding 300 x 50 frames per iteration would dominate the eval otherwise)
and (b) chunk the candidates so the explicit attention scores of the
predictor stay within a few GB.

Conventions shared with eval.py: pixels arrive as (B, S, T, 3, H, W) float
tensors already put through ``self.img_transform`` (upstream's
``default_transform``: [0,1] -> Normalize(0.5, 0.5); VWorldModel resizes to
196 px itself); action candidates are (B, S, horizon, frameskip * action_dim),
z-scored by eval.py's StandardScaler over the Lance table -- the training
normalization used the same statistics (dinowm_swm/h5_dset.py).
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms

from . import add_upstream_to_path
from .h5_dset import STATS_FILENAME

add_upstream_to_path()
import hydra  # noqa: E402
from models.visual_world_model import VWorldModel  # noqa: E402  (vendored dino_wm)


def _autocast(enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(enabled))


class DinoWMCostModel(nn.Module):
    """See module docstring. ``wm`` is upstream's VWorldModel (decoder None)."""

    def __init__(self, wm: VWorldModel, cfg, stats=None, img_size=224, mixed_precision="no", chunk=100):
        super().__init__()
        self.wm = wm
        self.cfg = cfg
        self.stats = stats or {}
        self.num_hist = int(cfg.num_hist)
        self.frameskip = int(cfg.frameskip)
        self.proprio_in = int(wm.proprio_encoder.in_chans)
        self.visual_dim = int(wm.encoder.emb_dim)
        self.bf16 = str(mixed_precision) == "bf16"
        self.chunk = int(chunk)
        # upstream datasets.img_transforms.default_transform on [0, 1] tensors,
        # preceded by uint8 -> float scaling (eval.py hands raw HWC uint8 frames)
        self.img_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # -- device plumbing ----------------------------------------------------
    def _move_plain_tensors(self, device):
        # upstream models.vit.Attention keeps its causal mask as a plain
        # attribute (created with .to('cuda')), which nn.Module.to() ignores
        for m in self.wm.modules():
            b = getattr(m, "bias", None)
            if torch.is_tensor(b) and not isinstance(b, nn.Parameter):
                m.bias = b.to(device)

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        dev = next(self.parameters()).device
        self._move_plain_tensors(dev)
        return out

    def cuda(self, device=None):
        return self.to(torch.device("cuda" if device is None else device))

    # -- encoding -------------------------------------------------------------
    @property
    def device(self):
        return next(self.wm.predictor.parameters()).device

    def _zeros_proprio(self, n, t):
        return torch.zeros(n, t, self.proprio_in, device=self.device)

    def encode_visual(self, pixels):
        """(N, T, 3, H, W) normalized -> patch tokens (N, T, P, D)."""
        with _autocast(self.bf16):
            z = self.wm.encode_obs({"visual": pixels, "proprio": self._zeros_proprio(*pixels.shape[:2])})
        return z["visual"].float()

    def _tile(self, emb, n_patches, repeat):
        # VWorldModel.encode, concat_dim == 1: per-frame vector tiled over the
        # patches and repeated `repeat` times along the channel axis
        return emb.unsqueeze(2).expand(-1, -1, n_patches, -1).repeat(1, 1, 1, repeat)

    def _assemble(self, z_vis, prop_emb, act_emb):
        p = z_vis.shape[2]
        return torch.cat([z_vis,
                          self._tile(prop_emb, p, self.wm.num_proprio_repeat),
                          self._tile(act_emb, p, self.wm.num_action_repeat)], dim=3)

    def rollout_last_visual(self, z_vis0, actions):
        """z_vis0: (N, T, P, D) history tokens; actions: (N, H, A) z-scored,
        H >= T. Returns the visual tokens of the LAST predicted frame (N, P, D)
        after applying all H action chunks (upstream VWorldModel.rollout)."""
        N, T = z_vis0.shape[:2]
        with _autocast(self.bf16):
            act_emb = self.wm.encode_act(actions)                     # (N, H, a)
            prop_emb = self.wm.encode_proprio(self._zeros_proprio(N, 1))  # (N, 1, p) constant
            z = self._assemble(z_vis0, prop_emb.expand(N, T, -1), act_emb[:, :T])
            for t in range(T, actions.shape[1]):
                z_pred = self.wm.predict(z[:, -self.num_hist:])
                z_new = z_pred[:, -1:]
                z_new = torch.cat([z_new[..., : -self.wm.action_dim],
                                   self._tile(act_emb[:, t:t + 1], z.shape[2], self.wm.num_action_repeat)], dim=3)
                z = torch.cat([z, z_new], dim=1)
            z_pred = self.wm.predict(z[:, -self.num_hist:])
        return z_pred[:, -1, :, : self.visual_dim].float()

    # -- Costable ---------------------------------------------------------------
    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        assert "goal" in info_dict and "pixels" in info_dict, info_dict.keys()
        B, S, H, A = action_candidates.shape
        if "_dinowm_goal_z" not in info_dict:
            goal = info_dict["goal"][:, 0]                    # (B, Tg, 3, h, w) -- sample dim is a pure expand
            info_dict["_dinowm_goal_z"] = self.encode_visual(goal[:, -1:].to(self.device))[:, 0]  # (B, P, D)
        if "_dinowm_hist_z" not in info_dict:
            pix = info_dict["pixels"][:, 0]                   # (B, T, 3, h, w)
            info_dict["_dinowm_hist_z"] = self.encode_visual(pix.to(self.device))  # (B, T, P, D)
        z_goal, z_hist = info_dict["_dinowm_goal_z"], info_dict["_dinowm_hist_z"]
        T = z_hist.shape[1]
        assert H >= T, f"horizon {H} shorter than the {T}-frame history"

        costs = torch.empty(B, S, device=self.device)
        for b in range(B):
            for s0 in range(0, S, self.chunk):
                s1 = min(S, s0 + self.chunk)
                n = s1 - s0
                z0 = z_hist[b].unsqueeze(0).expand(n, -1, -1, -1)
                acts = action_candidates[b, s0:s1].to(self.device, torch.float32)
                z_last = self.rollout_last_visual(z0, acts)                    # (n, P, D)
                costs[b, s0:s1] = (z_last - z_goal[b].unsqueeze(0)).pow(2).mean(dim=(1, 2))
        return costs


def load_dinowm_cost_model(run_dir, ckpt_name="model_latest.pth", device="cuda", chunk=100):
    """Assemble a :class:`DinoWMCostModel` from a dinowm_swm run folder
    (``hydra.yaml`` + ``checkpoints/<ckpt_name>``), the way upstream plan.py
    rebuilds its world model."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "hydra.yaml")
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    # full pickled nn.Modules (upstream format): weights_only must be off, and
    # map_location puts the pickled cuda mask attributes on the target device
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    predictor, action_encoder, proprio_encoder = (payload[k] for k in ("predictor", "action_encoder", "proprio_encoder"))
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)   # frozen DINOv2, never saved upstream
    for p in encoder.parameters():
        p.requires_grad_(False)
    wm = VWorldModel(
        image_size=cfg.img_size, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
        encoder=encoder, proprio_encoder=proprio_encoder, action_encoder=action_encoder,
        decoder=None, predictor=predictor,
        proprio_dim=proprio_encoder.emb_dim, action_dim=action_encoder.emb_dim,
        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
        train_encoder=False, train_predictor=False, train_decoder=False,
    ).to(device)
    wm.eval()
    stats_file = run_dir / STATS_FILENAME
    stats = json.loads(stats_file.read_text()) if stats_file.exists() else None
    model = DinoWMCostModel(wm, cfg, stats=stats, img_size=int(cfg.img_size),
                            mixed_precision=cfg.get("mixed_precision", "no"), chunk=chunk)
    model.to(device)
    print(f"[dinowm] loaded {ckpt_path} (epoch {payload.get('epoch')}), encoder {cfg.encoder.name}, "
          f"num_hist {cfg.num_hist}, frameskip {cfg.frameskip}, mixed_precision {model.bf16 and 'bf16' or 'no'}")
    return model
