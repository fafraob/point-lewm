"""Closed-loop planning evaluation for the lidar (point-cloud) world model.

Lidar counterpart of ``eval.py``: same structure, but the observation is a
raycast point cloud instead of pixels. It replays start states from the lidar
cube dataset into ``swm/OGBCube-v0``, produces live lidar via swm's
``AddRaycastLidarWrapper`` (configured to the exact sensor that generated the
dataset -- see ``lidar.sensor`` in the config), and plans with CEM in the JEPA
latent space toward the goal cloud taken from the dataset. Reports the success
rate.

The dataset-driven rollout, per-episode video, goal re-injection and lidar panel
rendering are all handled by ``swm.World.evaluate`` (``lidar_render_size``); this
script only adds the three lidar/Lance-specific pieces that ``eval.py`` has no
equivalent for: the point-cloud model adapter, positional episode sampling, and
run-dir resolution under ``experiment_logs/``.

Run: ``python eval_lidar.py`` (config: ``config/eval/lidar_cube.yaml``);
``python eval_lidar.py policy=random`` for the random baseline.
"""

import os

# See train.py: avoid CUBLAS_STATUS_NOT_SUPPORTED from cuBLASLt's fused fp16
# Linear+bias epilogue on huge point batches (e.g. on sm80/A100). Must be set
# before torch's first CUDA addmm; pixi.toml sets it too for pixi-run tasks.
os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.wrapper import make_lidar_pre_wrappers


class CloudPerturbation:
    """Eval-time sensor degradation: range noise and dropped returns.

    THIS IS NOT PART OF THE SENSOR. ``lidar.sensor`` in the eval configs is a
    contract -- it reproduces the exact scan that generated the training table,
    and changing a value there silently feeds the model clouds it never saw in
    training. This is the opposite: a deliberate, labelled degradation of a
    faithful scan, applied after the sensor and only at eval, to measure how
    far a world model survives a sensor that is worse than the one it learned
    on. Hence a separate ``lidar.perturb`` block, and hence the defaults below
    are no-ops -- an eval config that does not mention it behaves exactly as
    before.

    NOISE is along-ray, not isotropic: a LiDAR's error is in the RANGE it
    reports, so a point moves along its own line of sight, not in a random
    direction. The clouds are stored in the sensor frame (sensor at the origin),
    so that direction is just the point's own unit vector.

    Its scale is RELATIVE, sigma = ``noise_frac`` x the scan's bounding-box
    diagonal, recomputed per frame. An absolute sigma is not comparable across
    these environments: 2 cm is a light haze on the 4.9 m Push-T arena and a
    catastrophe on the 1.4 m reacher workspace, so a fixed number would report
    four different experiments under one name.

    DROPOUT marks returns as misses rather than deleting them, which is what a
    real lost return is -- and it means the points leave through the same
    ``invalid_value`` filter that ``_pack`` already applies to genuine misses,
    so nothing downstream needs to know this happened.

    ``scope`` decides what gets degraded. The goal cloud comes from the dataset,
    not the sensor, so it is clean unless asked for:
      ``obs``   -- degrade the live observation only (a bad sensor, a stored
                   target: what a deployment actually looks like)
      ``both``  -- degrade the goal the same way (tests the representation
                   rather than the perception, but a failure no longer
                   distinguishes "cannot see" from "aiming at the wrong thing")
    """

    def __init__(self, noise_frac=0.0, dropout_prob=0.0, scope="obs",
                 invalid_value=-1.0, in_channels=3, seed=0):
        if scope not in ("obs", "both"):
            raise ValueError(f"lidar.perturb.scope must be 'obs' or 'both', got {scope!r}")
        if not 0.0 <= float(dropout_prob) < 1.0:
            raise ValueError(f"lidar.perturb.dropout_prob must be in [0, 1), got {dropout_prob}")
        if float(noise_frac) < 0.0:
            raise ValueError(f"lidar.perturb.noise_frac must be >= 0, got {noise_frac}")
        self.noise_frac = float(noise_frac)
        self.dropout_prob = float(dropout_prob)
        self.scope = scope
        self.invalid_value = float(invalid_value)
        self.in_channels = int(in_channels)
        self.seed = int(seed)
        self._generators = {}

    @property
    def active(self):
        return self.noise_frac > 0.0 or self.dropout_prob > 0.0

    def describe(self):
        if not self.active:
            return "none"
        parts = []
        if self.noise_frac > 0.0:
            parts.append(f"range noise sigma={self.noise_frac:.3%} of scan extent")
        if self.dropout_prob > 0.0:
            parts.append(f"dropout {self.dropout_prob:.0%} of returns")
        return f"{' + '.join(parts)}  (scope: {self.scope})"

    def _generator(self, device):
        # One generator per device, seeded once: successive frames draw
        # different realisations, but the whole run replays from `seed`.
        key = str(device)
        if key not in self._generators:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._generators[key] = g
        return self._generators[key]

    def __call__(self, clouds, is_goal=False):
        """Degrade ``(B, T, ...)`` clouds. Returns them unchanged when inactive."""
        if not self.active or (is_goal and self.scope != "both"):
            return clouds

        # Same reshape as _pack: the trailing dims may be flat (N * C, as the
        # dataset goal is stored) or (N, C) (as the live sensor stream is
        # shaped), and (B*T, N, C) handles both.
        shape = clouds.shape
        b, t = shape[:2]
        pts = clouds.reshape(b * t, -1, self.in_channels).float().clone()
        valid = ~(pts == self.invalid_value).all(dim=-1)              # (B*T, N)
        gen = self._generator(pts.device)

        if self.noise_frac > 0.0:
            # Per-frame extent over the VALID points only: misses sit at the
            # sentinel and would otherwise define the bounding box themselves.
            big = torch.finfo(pts.dtype).max
            masked = torch.where(valid.unsqueeze(-1), pts, torch.full_like(pts, big))
            lo = masked.min(dim=1).values
            masked = torch.where(valid.unsqueeze(-1), pts, torch.full_like(pts, -big))
            hi = masked.max(dim=1).values
            extent = torch.linalg.norm(hi - lo, dim=-1)               # (F,)
            extent = torch.nan_to_num(extent, nan=0.0, posinf=0.0, neginf=0.0)
            sigma = (self.noise_frac * extent).unsqueeze(-1)          # (F, 1)

            ranges = torch.linalg.norm(pts, dim=-1)                   # (F, N)
            direction = pts / ranges.clamp_min(1e-9).unsqueeze(-1)    # unit line of sight
            draw = torch.randn(ranges.shape, generator=gen, device=pts.device, dtype=pts.dtype)
            shift = (draw * sigma).unsqueeze(-1) * direction
            pts = torch.where(valid.unsqueeze(-1), pts + shift, pts)

        if self.dropout_prob > 0.0:
            u = torch.rand(valid.shape, generator=gen, device=pts.device, dtype=torch.float32)
            drop = valid & (u < self.dropout_prob)
            pts = torch.where(drop.unsqueeze(-1), torch.full_like(pts, self.invalid_value), pts)

        return pts.reshape(shape).to(clouds.dtype)


class LidarPlanAdapter(nn.Module):
    """Bridge between swm's tensor-only planning stack and the packed-dict JEPA.

    The solver pipeline (WorldModelPolicy -> CEMSolver) only understands flat
    tensors: the env observation arrives as ``lidar`` and the dataset goal as
    ``goal_lidar``. JEPA instead consumes packed point-cloud dicts
    ``{coord, batch, feat}`` under its ``obs_key``/``goal_key``. This adapter
    converts on the fly inside ``get_cost``, dropping invalid points (all coords
    == ``invalid_value``) exactly like the training collate_fn.

    With ``use_rgb`` (RGB lidar ckpts, ``lidar.contains_rgb`` in the config) the
    row-aligned color streams ``<key>_rgb`` / ``goal_<key>_rgb`` -- live from the
    wrapper's ``add_rgb``, goal from the dataset column -- are packed into the
    dict's ``feat``, matching the training collate_fn's ``feat_key``.
    """

    def __init__(self, model, key="lidar", in_channels=3, invalid_value=0.0, use_rgb=False,
                 perturb=None):
        super().__init__()
        self.model = model
        # Eval-time degradation, or None for a faithful scan. Applied in _pack
        # because that is the one place EVERY cloud passes through -- obs and
        # goal, cem and ldad -- so no future call site can quietly bypass it.
        self.perturb = perturb
        self.key = key
        self.goal_key = f"goal_{key}"
        self.rgb_key = f"{key}_rgb"
        self.goal_rgb_key = f"goal_{key}_rgb"
        self.in_channels = in_channels
        self.invalid_value = invalid_value
        self.use_rgb = use_rgb

    def _pack(self, clouds, rgb=None, is_goal=False):
        """Pack clouds ``(B, T, ...)`` into ``{coord, batch, feat}``.

        Mirrors the training collate_fn (``pc_encoders.collate_point_cloud``):
        same sample-major/frame-minor cloud order, same drop of all-invalid
        points (applied to the row-aligned ``rgb`` too), same ``feat`` semantics
        (``None`` without colors) -- the encoder must see the identical
        representation at train and eval. The trailing dims may be flat
        (``N * C``, as the dataset goal is stored) or ``(N, C)`` (as the live
        sensor stream is shaped); reshaping to ``(B*T, N, C)`` handles both.
        """
        if self.perturb is not None:
            clouds = self.perturb(clouds, is_goal=is_goal)
        b, t = clouds.shape[:2]
        pts = clouds.reshape(b * t, -1, self.in_channels)  # (B*T, N, C)
        keep = ~(pts == self.invalid_value).all(dim=-1)  # (B*T, N)
        counts = keep.sum(dim=1)
        batch = torch.repeat_interleave(torch.arange(b * t, device=clouds.device), counts)
        feat = None
        if rgb is not None:
            feat = rgb.reshape(b * t, -1, 3)[keep].float()  # row-aligned colors
        return {"coord": pts[keep].float(), "batch": batch, "feat": feat}

    def get_cost(self, info_dict, action_candidates):
        info = dict(info_dict)
        # Embedding cache: CEM passes the same expanded info dict to every one
        # of its n_steps iterations, and obs/goal are constant within a solve.
        # The first call stores "emb"/"goal_emb" back into that solver-owned
        # dict (see below); later calls skip packing + encoding entirely
        # (mirrors the image LeWM's caching in stable_worldmodel.wm.lewm).
        if "emb" in info and "goal_emb" in info:
            for k in (self.key, self.goal_key, self.rgb_key, self.goal_rgb_key, "goal"):
                info.pop(k, None)
            return self.model.get_cost(info, action_candidates)
        # JEPA encodes the observation once and expands over the sample dim S in
        # latent space, so only sample 0 of the solver-expanded clouds is packed.
        obs = info.pop(self.key)[:, 0]  # (B, T, ...)
        goal = info.pop(self.goal_key)[:, 0][:, -1:]  # last goal frame
        obs_rgb = goal_rgb = None
        if self.use_rgb:
            if self.rgb_key not in info or self.goal_rgb_key not in info:
                raise KeyError(
                    f"use_rgb=True but '{self.rgb_key}'/'{self.goal_rgb_key}' missing from the "
                    "info dict -- the sensor needs add_rgb: true and the dataset a lidar_rgb column"
                )
            obs_rgb = info.pop(self.rgb_key)[:, 0]
            goal_rgb = info.pop(self.goal_rgb_key)[:, 0][:, -1:]
        else:  # colors may still be present (rgb dataset, non-rgb ckpt): drop them
            info.pop(self.rgb_key, None)
            info.pop(self.goal_rgb_key, None)
        info.pop("goal", None)  # dataset goal *image*; replaced by the cloud
        info[self.model.obs_key] = self._pack(obs, obs_rgb)
        info[self.model.goal_key] = self._pack(goal, goal_rgb, is_goal=True)
        cost = self.model.get_cost(info, action_candidates)
        # persist the embeddings the JEPA wrote into our copy back into the
        # solver-owned dict; the cache dies with it at the end of the solve
        info_dict["emb"] = info["emb"]
        info_dict["goal_emb"] = info["goal_emb"]
        return cost


class LidarLDADPlanAdapter(LidarPlanAdapter):
    """LidarPlanAdapter that warm-starts CEM with LDAD-decoded actions.

    Adds swm's ``Actionable`` protocol on top of the plain adapter: CEMSolver's
    ``prepare_init_action`` then fills the initial sampling mean with
    ``D(z_goal - z_now)`` -- the action sequence Delta-JEPA's inverse head
    (``jepa.action_decoder``) attributes to the latent displacement toward the
    goal -- instead of zeros. Everything else about the solve (sampling,
    var_scale, iterations, cost) is unchanged, and CEM always scores the
    unperturbed mean as candidate 0, so the decoded plan itself competes.

    Notes on fit: LDAD decodes exactly its training horizon N (macro-actions,
    z-scored like the CEM candidates -- both use the dataset action scaler).
    At eval, goal_offset = N macro-steps, so the goal displacement matches the
    training displacement scale at episode start. The decoder is queried from
    the *current* frame each replan; a requested horizon beyond N is zero-padded
    (never happens with the standard horizon == N config). ``prefix_actions``
    are skipped over in the decoded sequence -- exact only for the
    receding == horizon setup used here, where prefixes never occur.

    Opt-in via ``planner: ldad_cem`` (warm-started CEM) or ``planner: ldad``
    (the decoded plan is executed directly, no CEM -- see :class:`LDADSolver`)
    in the eval config; the plain adapter stays the default for
    ``planner: cem`` (the protocol check is structural, so the method must not
    exist on the default class). Requires a Delta-JEPA checkpoint.
    """

    def get_action(self, info, horizon=1, prefix_actions=None):
        model = self.model
        device = next(model.parameters()).device
        obs = torch.as_tensor(info[self.key]).to(device)  # (B, T_ctx, ...)
        goal = torch.as_tensor(info[self.goal_key]).to(device)[:, -1:]  # last goal frame
        obs_rgb = goal_rgb = None
        if self.use_rgb:
            obs_rgb = torch.as_tensor(info[self.rgb_key]).to(device)
            goal_rgb = torch.as_tensor(info[self.goal_rgb_key]).to(device)[:, -1:]
        b = obs.shape[0]

        # encode on scratch dicts -- the solver-owned info stays untouched, so
        # get_cost's own emb/goal_emb caching behaves exactly as without us
        z = model.encode({model.obs_key: self._pack(obs, obs_rgb)}, batch_size=b)["emb"][:, -1]
        z_goal = model.encode({model.obs_key: self._pack(goal, goal_rgb, is_goal=True)}, batch_size=b)["emb"][:, -1]

        plan = model.action_decoder(z_goal - z)  # (B, N, action_dim), z-scored space

        n_prev = 0 if prefix_actions is None else prefix_actions.shape[1]
        out = plan[:, n_prev : n_prev + horizon]
        if out.shape[1] < horizon:
            pad = torch.zeros(
                b, horizon - out.shape[1], plan.shape[-1], device=plan.device, dtype=plan.dtype
            )
            out = torch.cat([out, pad], dim=1)
        return out.detach().cpu()


class LDADSolver:
    """``planner: ldad`` -- execute the LDAD-decoded plan directly, no CEM.

    A minimal swm ``Solver``: ``solve`` fills the horizon via swm's
    ``prepare_init_action`` (which calls the :class:`LidarLDADPlanAdapter`'s
    ``get_action``, i.e. ``D(z_goal - z_now)``) and returns that plan as the
    actions, unrefined. Against ``planner: ldad_cem`` this isolates what CEM
    refinement adds on top of the decoded plan; against ``planner: cem`` it
    isolates the decoder itself as an amortized policy. Same z-scored action
    space and receding-horizon protocol as the CEM path -- only the
    optimization is gone.
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def configure(self, *, action_space, n_envs, config):
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        from stable_worldmodel.solver.utils import prepare_init_action

        total_envs = len(next(iter(info_dict.values())))
        actions = prepare_init_action(
            self.model, info_dict, init_action,
            self.horizon, n_envs=total_envs, action_dim=self.action_dim,
        )
        return {"actions": actions}


def sample_eval_starts(
    dataset,
    num_eval,
    goal_offset,
    seed,
    max_start_step=None,
    min_goal_displacement=None,
):
    """Sample ``num_eval`` (episode, start_step) pairs uniformly over all valid
    starting points, i.e. steps that leave room for the goal offset.

    Episodes are positional indices into ``dataset.lengths`` (the Lance reader
    keys ``load_chunk`` by episode position, not by an ``episode_idx`` column --
    this is the point-cloud analog of ``eval.py``'s column-based sampling).

    ``max_start_step`` (config ``eval.max_start_step``, ``null`` = no cap) caps
    the latest start so ``start_step`` is drawn from ``[0, max_start_step]``. The
    expert trajectories were recorded without ``terminate_at_goal``, so late
    starts already sit at the goal and make the task trivial -- cap this to keep
    starts far enough from the goal to be meaningful.

    ``min_goal_displacement`` (config ``eval.min_goal_displacement``, ``null`` =
    off) keeps only starts whose cube (``privileged/block_0_pos``) moves at
    least this far (metres) between ``start`` and ``start + goal_offset`` in the
    expert trajectory. Without it the sampled evals are dominated by windows
    where the expert never moves the cube -- the episode succeeds at reset
    regardless of the policy (measured on the cube table: at the defaults, success ==
    "cube displacement < 5cm" on all 50 draws, and a random policy scores the
    exact same 50% as the planner). 0.05 makes every episode require an actual
    manipulation.
    """
    lengths = np.asarray(dataset.lengths, dtype=int)
    starts_per_ep = np.maximum(lengths - goal_offset, 0)  # start in [0, len-offset-1]
    if max_start_step is not None:
        # keep at most max_start_step+1 starts per episode -> start in [0, cap]
        starts_per_ep = np.minimum(starts_per_ep, int(max_start_step) + 1)
    total = int(starts_per_ep.sum())
    print(total, "valid starting points found for evaluation.")
    if total < num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    g = np.random.default_rng(seed)
    if min_goal_displacement is None:
        flat = np.sort(g.choice(total, size=num_eval, replace=False))
        ep_end = np.cumsum(starts_per_ep)
        episodes = np.searchsorted(ep_end, flat, side="right")
        start_steps = flat - (ep_end[episodes] - starts_per_ep[episodes])
        return episodes, start_steps

    # Displacement filter: rows are stored episode-major/step-minor, so episode
    # e's step s lives at row cumsum(lengths)[e-1] + s. One column read gives
    # every candidate's ||cube(start+offset) - cube(start)|| without touching
    # the heavy lidar/pixel columns.
    pos = np.asarray(dataset.get_col_data("privileged/block_0_pos"), dtype=np.float64)
    row0 = np.concatenate([[0], np.cumsum(lengths)])[:-1]  # first row per episode
    cand_ep = np.repeat(np.arange(len(lengths)), starts_per_ep)
    cand_step = np.concatenate([np.arange(n) for n in starts_per_ep]).astype(int)
    rows = row0[cand_ep] + cand_step
    disp = np.linalg.norm(pos[rows + goal_offset] - pos[rows], axis=1)
    keep = disp >= float(min_goal_displacement)
    n_keep = int(keep.sum())
    print(
        f"{n_keep} starting points move the cube >= {min_goal_displacement} m "
        f"({n_keep / total:.1%} of all valid starts)."
    )
    if n_keep < num_eval:
        raise ValueError(
            "Not enough starts satisfy min_goal_displacement; lower it or "
            "raise goal_offset/max_start_step."
        )
    pick = np.sort(g.choice(n_keep, size=num_eval, replace=False))
    idx = np.flatnonzero(keep)[pick]
    return cand_ep[idx], cand_step[idx]


def load_fixed_stats(stats_file):
    """``{col: {"mean": [...], "std": [...]}}`` from a JSON file, or ``{}``."""
    if not stats_file:
        return {}
    import json

    with open(stats_file) as f:
        fixed = json.load(f)
    print(f"[eval] using fixed column statistics from {stats_file}")
    return fixed


def set_fixed_stats(processor, stats):
    """Put pinned mean/std into a StandardScaler instead of fitting it."""
    processor.mean_ = np.asarray(stats["mean"], dtype=np.float64)
    processor.scale_ = np.asarray(stats["std"], dtype=np.float64)
    processor.var_ = processor.scale_**2
    processor.n_features_in_ = processor.mean_.shape[0]


def resolve_run_dir(cfg) -> str:
    """Resolve the training run folder whose ``checkpoints/<policy>`` we load.

    Training writes one timestamped folder per run under ``logs_root`` (default
    ``experiment_logs/``), each containing ``checkpoints/<run_name>/weights_epoch_*.pt``.
    Precedence:
      1. ``logs_root/run`` when ``run`` is given (a training run of your own), else
      2. ``cache_dir`` (an exact folder; the eval configs default it to the
         repository root, where ``checkpoints/<env>/<model>/`` holds the released
         checkpoints, see scripts/download_checkpoints.py), else
      3. the newest run folder under ``logs_root`` -- the lexicographically last
         name with a ``checkpoints/`` dir, which for the ``YYYY-MM-DD_HH-MM-SS_…``
         naming is the latest run.
    """
    logs_root = Path(cfg.get("logs_root") or swm.data.utils.get_cache_dir())
    run = cfg.get("run")
    if run:
        return str(logs_root / run)
    if cfg.get("cache_dir"):
        return str(cfg.cache_dir)
    runs = (
        sorted(d for d in logs_root.iterdir() if d.is_dir() and (d / "checkpoints").is_dir())
        if logs_root.is_dir()
        else []
    )
    if not runs:
        raise FileNotFoundError(
            f"No run folders with a checkpoints/ dir under {logs_root}. Train first, "
            "or pass run=<folder> or cache_dir=<dir>."
        )
    print(f"[eval] auto-selected latest run: {runs[-1].name}")
    return str(runs[-1])


@hydra.main(version_base=None, config_path="./config/eval", config_name="lidar_cube")
def run(cfg: DictConfig):
    """Run evaluation of the lidar world model vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment; the pre-wrapper adds the lidar sensor below
    # MegaWrapper so `lidar` is lifted into the stacked infos each step.
    # WHAT COUNTS AS SUCCESS IS THE ENV'S BUSINESS, here as in eval.py: every
    # env in this stack ends its own episode at its own goal (two-room by agent
    # distance, pusht by coverage, the cube by terminate_at_goal, Reacher3D by
    # the qpos-match test its dm_control counterpart uses), and swm scores from
    # that flag. Nothing about the criterion lives in this script.
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(
        **cfg.world,
        image_shape=(cfg.eval.img_size, cfg.eval.img_size),
        pre_wrappers=make_lidar_pre_wrappers(cfg.lidar.sensor),
    )

    # the lidar cloud is consumed raw -- no transform (cf. eval.py's img_transform)
    transform = {}

    dataset = swm.data.load_dataset(cfg.eval.dataset_name)

    # z-score processors (actions only -- the lidar cloud is consumed raw).
    # dataset.stats_file (JSON {col: {mean, std}}) pins the statistics instead
    # of fitting them on the table: a subset table (e.g. data_sample/) must
    # plan with the full table's normalizer, not its own.
    fixed_stats = load_fixed_stats(cfg.dataset.get("stats_file"))
    process = {}
    for col in cfg.dataset.stats_keys:
        processor = preprocessing.StandardScaler()
        if col in fixed_stats:
            set_fixed_stats(processor, fixed_stats[col])
        else:
            col_data = np.asarray(dataset.get_col_data(col))
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    # Planner mode -- how actions are produced each replan (Delta-JEPA ckpts
    # required for the two ldad modes):
    #   cem       plain CEM from a zero-initialized sampling mean
    #   ldad_cem  CEM warm-started with the LDAD plan D(z_goal - z_now)
    #   ldad      the LDAD plan executed directly, no CEM refinement
    planner = str(cfg.get("planner", "cem"))
    assert planner in ("cem", "ldad_cem", "ldad"), planner

    cache_dir = None
    if policy != "random":
        cache_dir = resolve_run_dir(cfg)
        model = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cache_dir)
        adapter_cls = LidarPlanAdapter
        if planner in ("ldad_cem", "ldad"):
            if getattr(model, "action_decoder", None) is None:
                raise ValueError(
                    f"planner={planner} requires a Delta-JEPA checkpoint with "
                    "an action_decoder; this checkpoint has none"
                )
            adapter_cls = LidarLDADPlanAdapter
            print(f"[eval] planner={planner}")
        # Eval-time sensor degradation (the noise / dropout ablations). Absent
        # or all-zero -> None, and the adapter runs exactly as it always has.
        perturb_cfg = OmegaConf.to_container(cfg.lidar.get("perturb", {}) or {}, resolve=True)
        perturb = CloudPerturbation(
            invalid_value=cfg.lidar.invalid_value,
            in_channels=cfg.lidar.in_channels,
            seed=int(perturb_cfg.pop("seed", None) or cfg.seed),
            **perturb_cfg,
        )
        if perturb.active:
            print(f"[eval] lidar perturbation: {perturb.describe()}")
        else:
            perturb = None
        model = adapter_cls(
            model,
            key=cfg.lidar.key,
            in_channels=cfg.lidar.in_channels,
            invalid_value=cfg.lidar.invalid_value,
            use_rgb=bool(cfg.lidar.get("contains_rgb", False)),
            perturb=perturb,
        )
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        config = swm.PlanConfig(**cfg.plan_config)
        if planner == "ldad":
            solver = LDADSolver(model=model, device="cuda")
        else:
            solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    # results + videos: an explicit output.dir wins, else alongside the
    # checkpoint (ckpt policy) or this script's dir (random policy).
    if cfg.output.get("dir"):
        results_path = Path(cfg.output.dir)
    elif cfg.policy != "random":
        results_path = Path(cache_dir, "checkpoints", cfg.policy).parent
    else:
        results_path = Path(__file__).parent

    # sample the episodes and the starting indices (positional, Lance-keyed);
    # eval.starts_file (JSON {episodes, start_steps}) pins them explicitly
    # instead -- e.g. to restrict eval to held-out validation windows
    starts_file = cfg.eval.get("starts_file")
    if starts_file:
        import json

        with open(starts_file) as f:
            fixed = json.load(f)
        eval_episodes = np.asarray(fixed["episodes"], dtype=int)
        eval_start_idx = np.asarray(fixed["start_steps"], dtype=int)
        assert len(eval_episodes) == cfg.eval.num_eval, (
            f"starts_file has {len(eval_episodes)} starts, eval.num_eval={cfg.eval.num_eval}"
        )
        print(f"[eval] using fixed starts from {starts_file}")
    else:
        eval_episodes, eval_start_idx = sample_eval_starts(
            dataset,
            cfg.eval.num_eval,
            cfg.eval.goal_offset_steps,
            cfg.seed,
            max_start_step=cfg.eval.get("max_start_step"),
            min_goal_displacement=cfg.eval.get("min_goal_displacement"),
        )
    print("episodes:", eval_episodes, "start steps:", eval_start_idx)

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        # one agent|dataset|goal(+lidar) panel per episode. output.video=false
        # turns them off: a sweep (scripts/run_sweep.sh) evaluates every
        # checkpoint at several seeds, so the mp4 count is num_eval * seeds * checkpoints and
        # dwarfs the numbers it is run for. Unset == true, so nothing changes for
        # an interactive `pixi run eval-lidar`.
        video=results_path if cfg.output.get("video", True) else None,
        # Panels must depict the model INPUT: color them only when the model
        # consumes colors (contains_rgb), else depth-color everywhere -- even
        # if the dataset/sensor happens to record colors.
        lidar_render_rgb=bool(cfg.lidar.get("contains_rgb", False)),
    )
    end_time = time.time()

    print(metrics)

    results_file = results_path / cfg.output.filename
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with results_file.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"planner: {planner if cfg.policy != 'random' else 'random'}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
