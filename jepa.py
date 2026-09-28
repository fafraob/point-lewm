"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from pc_encoders import PointCloudEncoder

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        action_decoder=None,
        obs_key=None,
        goal_key="goal",
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Delta-JEPA's LDAD (module.LatentDifferenceActionDecoder): inverse head
        # decoding actions from latent displacements. Trained with the LDAD
        # loss. The default CEM planner does not use it, but the optional
        # planner=ldad_cem / ldad modes of eval_lidar.py do. None for plain
        # LeWM checkpoints.
        self.action_decoder = action_decoder

        # The observation modality is driven by the encoder: a PointCloudEncoder
        # declares its own input_key and consumes a packed {coord, batch, feat}
        # dict, while the image ViT consumes the "pixels" tensor. obs_key/goal_key
        # can be overridden explicitly, but default to the encoder's declared key.
        self.is_point_cloud = isinstance(encoder, PointCloudEncoder)
        self.obs_key = obs_key or getattr(encoder, "input_key", "pixels")
        self.goal_key = goal_key

    def encode(self, info, batch_size=None):
        """Encode observations and actions into per-frame embeddings.

        Reads the observation under ``self.obs_key`` and writes ``info["emb"]``
        of shape (B, T, D) -- or (B, T, K, D) when the encoder returns a token
        set per frame (the Utonia-WM arm's fixed grid). Two modalities are
        supported:

        * image (ViT): ``info[obs_key]`` is a (B, T, C, H, W) tensor; frames are
          flattened, encoded, and the CLS token is projected.
        * point cloud: ``info[obs_key]`` is a packed dict ``{coord, batch, feat}``
          spanning ``B * T`` clouds; the encoder returns one latent per cloud
          (ordered by batch index) -- either a vector (D,) or a token set
          (K, D) -- and we reshape to (B, T, D) / (B, T, K, D).

        ``batch_size`` pins B for the point-cloud path when the info dict carries
        no ``action`` tensor to read it from (e.g. the goal encode in
        :meth:`get_cost`); without it B falls back to the cloud count, which is
        only correct for a single frame per batch element (T == 1).
        """
        obs = info[self.obs_key]

        if self.is_point_cloud:
            feat = self.encoder(obs)  # (B*T, D) or (B*T, K, D), one latent per cloud
            emb = self.projector(feat)
            # B*T clouds are ordered sample-major/frame-minor (see collate_fn);
            # recover B from the action tensor (train/rollout), else the caller's
            # batch_size, else assume one cloud per batch element (T == 1).
            if torch.is_tensor(info.get("action")):
                b = info["action"].size(0)
            elif batch_size is not None:
                b = batch_size
            else:
                b = emb.size(0)
            info["emb"] = rearrange(emb, "(b t) ... -> b t ...", b=b)
        else:
            pixels = obs.float()
            b = pixels.size(0)
            pixels = rearrange(pixels, "b t ... -> (b t) ...")  # flatten for encoding
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]  # cls token
            emb = self.projector(pixels_emb)
            info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D), or (B, T, K, D) for token-shaped latents (Utonia-WM arm)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        # pred_proj (when present) maps one latent vector at a time, so flatten
        # every leading axis -- batch, time, and the tokens of a token set.
        flat = self.pred_proj(preds.reshape(-1, preds.size(-1)))
        return flat.view(*preds.shape[:-1], flat.size(-1))

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        B, S, T = action_sequence.shape[:3]
        # Context length: an image obs is (B, S, T_ctx, C, H, W) so T_ctx = size(2).
        # A packed point-cloud dict has no such axis (the swm solver does not yet
        # expand dict observations over the sample dim S), but it holds B * T_ctx
        # clouds, so T_ctx = num_clouds / B. With a cached "emb" (see below) the
        # obs may be absent entirely; the emb carries T_ctx instead.
        if "emb" in info:
            H = info["emb"].size(2)
        else:
            assert self.obs_key in info, f"{self.obs_key} not in info_dict"
            obs = info[self.obs_key]
            if torch.is_tensor(obs):
                H = obs.size(2)
            else:
                num_clouds = int(obs["batch"].max().item()) + 1
                assert num_clouds % B == 0, (num_clouds, B)
                H = num_clouds // B
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # encode initial state, or reuse the embedding cached by a prior call:
        # the CEM solver passes the same info dict for all its n_steps
        # iterations, so the (constant) observation is encoded once per solve
        # (mirrors stable_worldmodel.wm.lewm.LeWM.rollout)
        if "emb" not in info:
            # copy and encode initial info dict (take the first action sample, S=0)
            _init = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    _init[k] = v[:, 0]
                elif k == self.obs_key:
                    _init[k] = v  # packed point-cloud dict: pass through (see note above)
            _init = self.encode(_init)
            # (B, T_ctx, ...) -> (B, S, T_ctx, ...); the trailing axes are the
            # latent's own (D, or K x D for token latents).
            e = _init["emb"].detach().unsqueeze(1)
            info["emb"] = e.expand(B, S, *e.shape[2:])
        emb = info["emb"]

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T, D) | (B, S, T, K, D)
        goal_emb = info_dict["goal_emb"]  # (B, 1, T_goal, D) | (B, 1, T_goal, K, D)

        # score only the last predicted step against the last goal frame.
        # Expand just that (B, 1, 1, ...) slice to the pred's (B, S, 1, ...) so
        # the sizes match exactly -- a stride-0 view (no full-T tensor), and it
        # keeps mse_loss from warning about mismatched input/target sizes. The
        # time axis is indexed EXPLICITLY (axis 2): token-shaped latents carry a
        # trailing token axis, which a [..., -1:, :] slice would truncate.
        pred_last = pred_emb[:, :, -1:]  # (B, S, 1, ...)
        goal_last = goal_emb[:, :, -1:].expand_as(pred_last)  # (B, S, 1, ...)
        err = F.mse_loss(pred_last, goal_last.detach(), reduction="none")

        # sum over every latent axis (time, tokens, features) -> (B, S). DINO-WM
        # means over patches instead; CEM ranks candidates, so the constant
        # factor K is irrelevant and this keeps the cost definition (and the
        # cost_stream_weights tooling) identical across arms.
        cost = err.sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            v = info_dict[k]
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)
            elif isinstance(v, dict):  # packed point-cloud dict {coord, batch, feat}
                info_dict[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}

        # encode the goal, or reuse the embedding cached by a prior call (the
        # CEM solver reuses the info dict across iterations -- mirrors
        # stable_worldmodel.wm.lewm.LeWM.get_cost)
        if "goal_emb" not in info_dict:
            assert self.goal_key in info_dict, f"{self.goal_key} not in info_dict"

            # Build the goal observation to encode (take the first action sample, S=0).
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            for k in (self.obs_key, self.goal_key):  # carry packed-dict obs/goal through unsliced
                if isinstance(info_dict.get(k), dict):
                    goal[k] = info_dict[k]
            goal[self.obs_key] = goal[self.goal_key]  # feed the goal observation to the encoder

            for k in list(info_dict.keys()):
                if k.startswith("goal_") and k in goal:
                    goal[k[len("goal_") :]] = goal.pop(k)

            goal.pop("action", None)
            # pin B from the candidate batch: the goal carries no action tensor, so
            # encode would otherwise infer B from the goal cloud count.
            goal = self.encode(goal, batch_size=action_candidates.size(0))

            # goal emb is (B, T_goal, D); insert the sample dim so criterion can
            # broadcast it against the (B, S, T, D) rollout.
            info_dict["goal_emb"] = goal["emb"].unsqueeze(1)
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
