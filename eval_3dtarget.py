"""Closed-loop planning with an image-free / cloud-free goal: a typed 3-D pose.

``eval_lidar.py`` plans toward ``z_goal = Enc(goal cloud)``, a recorded point
cloud from the dataset. Here the goal cloud is never used: the goal latent is
*predicted* from the goal objects' 3-D pose alone (cube: position; pusht: T
position + sensor-frame heading vector and ball position; tworoom: agent
position; reacher: wrist-joint and fingertip position, the forward kinematics
of the goal joint angles) by a model trained with ``target2latent.train``. Everything else -- env, sensor,
CEM, horizon, start windows -- is identical to ``eval_lidar.py``, so the
success rates are directly comparable.

The goal pose comes from the dataset's goal-frame privileged columns (the same
values the env's goal-setting callables receive), is transformed into the
sensor frame by :mod:`target2latent.envs`, and -- for a ``--use-z`` model --
is joined by the latent of the *current* observation, recomputed at every
replan.

``target_goal.goal_mode=true`` is the parity self-check: encode the recorded
goal cloud instead (the cost path then reproduces ``eval_lidar.py`` exactly, up
to the encoder's own sampling noise).

Run (from the repo root; ``target_goal.model_dir`` is a target-to-latent head:
the released ones sit next to their encoder under
``checkpoints/<env>/<model>/target2latent/<head>/``, your own come from
``python -m target2latent.train``)::

    python eval_3dtarget.py --config-name=3dtarget_cube \\
        policy=cube/point-delta-jepa/weights.pt \\
        target_goal.model_dir=checkpoints/cube/point-delta-jepa/target2latent/mlp \\
        seed=42 output.dir=eval_results/3dtarget_cube_point_delta_jepa_mlp_seed42

COMMANDED-GOAL GRID (``eval.grid_file``; App. D.2.1, Figure 5 and Table 11 of
the paper). Instead of a recorded goal frame, the goal is a COMMANDED 3-D
position from a lattice over the workspace (``scripts/cube_grid_goals.py``).
:class:`GridGoalDataset` overwrites the goal frame's pose column with that
position, which is what both the goal head and the env's goal-setting callable
read, so the env scores the agent against the commanded position too.
``eval.grid_scans`` (``scripts/cube_goal_scans.py``) supplies, for
``goal_mode=true``, the cloud obtained by placing the cube AT the commanded
position and re-scanning: a goal observation for a pose no recorded frame
contains, and the strongest goal-cloud baseline obtainable in simulation. Per
goal outcomes land in ``grid_results.json`` next to the usual output::

    pixi run eval-3dtarget --config-name=3dtarget_cube_grid eval.grid_block=2 \\
        policy=cube/point-delta-jepa/weights.pt \\
        target_goal.model_dir=checkpoints/cube/point-delta-jepa/target2latent/mlp_z

The whole grid is run by ``config/eval_sweep/3dtarget_cube_grid.conf`` and
aggregated by ``scripts/cube_grid_stats.py`` (Table 11) and
``scripts/plot_cube_goals_pair.py`` (Figure 5); see docs/reproduction.md,
"Commanding goals the expert never aimed at".
"""

from __future__ import annotations

import os

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ["MUJOCO_GL"] = "egl"

import json
import time
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.wrapper import make_lidar_pre_wrappers

from eval_lidar import (CloudPerturbation, LidarPlanAdapter, load_fixed_stats, resolve_run_dir,
                        sample_eval_starts, set_fixed_stats)
from target2latent.envs import SPECS
from target2latent.models import ShortcutHead, load_goal_model


class TargetGoalAdapter(LidarPlanAdapter):
    """``LidarPlanAdapter`` whose goal latent comes from a typed 3-D pose.

    Reimplements ``get_cost`` rather than extending it because the goal side
    changes: instead of encoding a goal cloud, the goal latent is predicted
    once per solve from the goal pose (and, for a ``use_z`` model, the current
    observation's latent). The observation side is unchanged -- same packing,
    same ``jepa.rollout`` with its default history, same encode-once-per-solve
    caching (the CEM solver hands the same expanded info dict to all its
    iterations, so both the observation embedding and the goal latent are
    computed once and stashed there).
    """

    def __init__(self, model, goal_model, goal_cfg, spec, goal_mode="pred",
                 sample_steps=None, **kw):
        super().__init__(model, **kw)
        self.goal_model = goal_model
        self.goal_cfg = goal_cfg or {}
        self.spec = spec
        self.goal_mode = goal_mode
        self.sample_steps = sample_steps  # shortcut only; None -> the model's own
        self.use_z = bool(self.goal_cfg.get("use_z", False))
        self._logged = False
        self.stats = {"n_solves": 0, "goal_time": 0.0}

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _last(info, key):
        """``(B, S, T, C)`` (CEM-expanded) or ``(B, T, C)`` -> ``(B, C)`` goal frame."""
        t = torch.as_tensor(info[key])
        while t.ndim > 3:
            t = t[:, 0]
        return t[:, -1] if t.ndim == 3 else t

    def _encode_obs(self, info, n_samples):
        """Pack + encode the observation once -> ``(B, S, T_ctx, D)``."""
        obs = info[self.key][:, 0] if info[self.key].ndim > 3 else info[self.key]
        b = obs.shape[0]
        packed = self._pack(obs)
        emb = self.model.encode({self.model.obs_key: packed}, batch_size=b)["emb"]
        if not self._logged:
            print(f"[target_goal] env={self.spec.name}  T_ctx={emb.shape[1]}  B={b}  "
                  f"S={n_samples}  head={self.goal_cfg.get('head', 'true')}  "
                  f"use_z={self.use_z}", flush=True)
            self._logged = True
        return emb.unsqueeze(1).expand(b, n_samples, *emb.shape[1:])

    def _goal_latent(self, info, emb):
        """``(B, D)`` goal latent."""
        device = emb.device
        if self.goal_mode == "true":
            # Parity self-check: encode the recorded goal cloud, like eval_lidar.
            goal = info[self.goal_key][:, 0][:, -1:] if info[self.goal_key].ndim > 3 \
                else info[self.goal_key][:, -1:]
            b = goal.shape[0]
            return self.model.encode(
                {self.model.obs_key: self._pack(goal, is_goal=True)}, batch_size=b
            )["emb"][:, -1]

        # goal pose from the dataset's goal-frame privileged columns
        raw = {
            col: self._last(info, key).cpu().numpy()
            for col, key in zip(self.spec.columns, self.spec.goal_keys)
        }
        coords, rot = self.spec.goal_pose(raw)
        parts = [torch.as_tensor(coords, device=device)]
        if rot is not None:
            parts.append(torch.as_tensor(rot, device=device))
        if self.use_z:
            parts.append(emb[:, 0, -1])  # current latent, last context frame
        cond = torch.cat(parts, dim=1).float()
        if isinstance(self.goal_model, ShortcutHead):
            return self.goal_model.sample(cond, n=1, n_steps=self.sample_steps)[:, 0]
        return self.goal_model.goals(cond, n=1)[:, 0]

    # -- cost --------------------------------------------------------------
    def get_cost(self, info_dict, action_candidates):
        jepa = self.model
        n_samples = action_candidates.shape[1]
        info = dict(info_dict)
        for k, v in list(info.items()):
            if torch.is_tensor(v):
                info[k] = v.to(action_candidates.device)

        if "emb" not in info_dict:
            t0 = time.time()
            emb = self._encode_obs(info, n_samples)
            info_dict["emb"] = emb
            info_dict["z_goal"] = self._goal_latent(info, emb)
            self.stats["n_solves"] += 1
            self.stats["goal_time"] += time.time() - t0
        # Drop the raw clouds: rollout reads the cached emb and never touches them.
        for k in (self.key, self.goal_key, self.rgb_key, self.goal_rgb_key, "goal"):
            info.pop(k, None)
        info["emb"] = info_dict["emb"]

        info = jepa.rollout(info, action_candidates)
        pred = info["predicted_emb"][:, :, -1]  # (B, S, D)
        z_goal = info_dict["z_goal"]  # (B, D)
        return ((pred - z_goal.unsqueeze(1)) ** 2).sum(-1)  # (B, S)


class GridGoalDataset:
    """Dataset view whose goal frame carries a COMMANDED pose, not a recorded one.

    ``World.evaluate`` builds its goal state by slicing ``start ..
    start + goal_offset`` out of the dataset and keeping the last frame, so
    overwriting that frame here is enough to redirect everything downstream:
    the pose column becomes ``goal_<col>`` in the env info (read by the goal
    head AND by the env's goal-setting callable, hence by its own success
    check), and the cloud column becomes the goal observation the
    ``goal_mode=true`` arm encodes. Everything else delegates to the real
    dataset, so the start frame, the statistics and the video panels are
    unchanged.
    """

    def __init__(self, base, column, poses, clouds=None, cloud_column="lidar",
                 retrieval=None):
        self._base = base
        self._column = column
        self._poses = np.asarray(poses, dtype=np.float32)
        self._cloud_column = cloud_column
        if retrieval is not None:
            # RETRIEVAL baseline: the goal cloud is the recorded frame whose
            # cube pose is nearest the commanded one -- arm pose, clutter and
            # all. Fetched once here so the rollout does no extra dataset IO.
            eps = np.asarray([r["episode"] for r in retrieval], dtype=int)
            steps = np.asarray([r["step"] for r in retrieval], dtype=int)
            frames = base.load_chunk(eps, steps, steps + 1)
            clouds = np.stack([np.asarray(f[cloud_column])[0].reshape(-1) for f in frames])
            off = [r["dist_m"] for r in retrieval if "dist_m" in r]
            print(f"[grid] retrieval goal clouds: {clouds.shape} from "
                  f"{len(set(eps.tolist()))} episodes"
                  + (f", nearest recorded pose {np.median(off) * 100:.1f} cm away (median)"
                     if off else ""))
        self._clouds = None if clouds is None else np.asarray(clouds, dtype=np.float32)

    def __getattr__(self, name):  # everything else is the real dataset
        return getattr(self._base, name)

    @staticmethod
    def _with_last(val, new_last):
        """Copy ``val`` (torch or numpy) with its last frame replaced."""
        if torch.is_tensor(val):
            out = val.clone()
            out[-1] = torch.as_tensor(new_last, dtype=val.dtype).reshape(val.shape[1:])
            return out
        out = np.array(val, copy=True)
        out[-1] = np.asarray(new_last, dtype=out.dtype).reshape(out.shape[1:])
        return out

    def load_chunk(self, episodes_idx, start, end):
        chunk = self._base.load_chunk(episodes_idx, start, end)
        for i, steps in enumerate(chunk):
            steps[self._column] = self._with_last(steps[self._column], self._poses[i])
            if self._clouds is not None and self._cloud_column in steps:
                steps[self._cloud_column] = self._with_last(
                    steps[self._cloud_column], self._clouds[i].reshape(-1)
                )
        return chunk


def _cube_target(env):
    """Commanded cube target: the mocap the env's set_target_pos callable moved."""
    base = env.unwrapped
    return np.asarray(base._data.mocap_pos[base._cube_target_mocap_ids[0]], dtype=float)


#: Per-environment grid wiring. Only OGB-Cube is shipped: it is the one
#: environment whose recorded goals occupy a strict subset of the commandable
#: workspace (every dataset target is a placement on the table at z = 0.02 m),
#: which is what makes "command a height the expert never aimed at" a question
#: at all. ``pos_key`` is the info field holding the tracked object's pose,
#: ``target_fn`` reads the commanded target off the env, and ``success`` /
#: ``unit`` are the environment's own, so nothing is reported in the wrong
#: units (OGB-Cube is METRES, not pixels).
GRID_ENVS = {
    "cube": dict(pos_key="privileged/block_0_pos", target_fn=_cube_target,
                 success=0.04, unit="m", to_cm=100.0),
}


class GridProbeWrapper(gym.Wrapper):
    """Run the whole budget and record how close the agent actually got.

    The environment terminates the moment the cube is within its 4 cm success
    threshold, so every successful episode's FINAL distance would sit just
    under that threshold and the error map would show the threshold rather than
    the method's precision. This wrapper suppresses termination -- the
    evaluation budget still bounds the episode -- and records the closest
    approach, so the grid measures precision. It is applied only in grid mode;
    the standard sweeps are untouched, and because nothing terminates any more,
    ``success_rate`` from ``World.evaluate`` is meaningless here:
    ``grid_results.json`` recomputes success from the recorded distances.
    """

    def __init__(self, env, threshold, wiring):
        super().__init__(env)
        self.threshold = float(threshold)
        self.wiring = wiring
        self.closest = float("inf")
        self.steps_to_threshold = None
        self.final_pos = None
        self.final_distance = float("nan")
        self.trace = []  # per-step distance, for eval.grid_trace diagnostics
        self._steps = 0

    def _track(self, info):
        # "final" = the last FINITE reading: the pool hands out NaN buffers once
        # the rollout is over, which would otherwise overwrite the real end state.
        pos = info.get(self.wiring["pos_key"])
        if pos is None:
            return
        pos = np.asarray(pos, dtype=float).reshape(-1)
        if not np.isfinite(pos).all():
            return
        self.final_pos = pos.tolist()
        try:
            target = self.wiring["target_fn"](self.env)
        except Exception:
            return
        d = float(np.linalg.norm(pos[: len(target)] - target))
        if not np.isfinite(d):
            return
        self.final_distance = d
        self.trace.append(d)
        self.closest = min(self.closest, d)
        if self.steps_to_threshold is None and d < self.threshold:
            self.steps_to_threshold = self._steps

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        # NOT tracked: World applies the start-state / goal-state callables AFTER
        # reset, so this info still describes the pre-callable episode and its
        # distance belongs to a different goal entirely.
        self.closest, self.steps_to_threshold, self._steps = float("inf"), None, 0
        self.final_pos, self.final_distance = None, float("nan")
        self.trace = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._track(info)
        return obs, reward, False, truncated, info  # never terminate: run the budget


def probe_stats(world):
    """Per-env ``(closest, steps to threshold, final pose, final distance, ...)``."""
    closest, steps, finals, final_d, traces, nsteps = [], [], [], [], [], []
    for env in world.envs.envs:
        w = env
        while w is not None and not isinstance(w, GridProbeWrapper):
            w = getattr(w, "env", None)
        if w is None:
            raise RuntimeError(
                "GridProbeWrapper missing: the per-goal distances of grid_results.json come "
                "from it, so eval.grid_no_terminate=false leaves nothing to report"
            )
        closest.append(w.closest)
        steps.append(w.steps_to_threshold)
        traces.append(list(w.trace))
        nsteps.append(w._steps)
        finals.append(w.final_pos if w.final_pos is not None else [float("nan")] * 3)
        final_d.append(w.final_distance)
    return (np.asarray(closest, dtype=float), steps, np.asarray(finals, dtype=float),
            np.asarray(final_d, dtype=float), traces, nsteps)


def load_grid_block(cfg, goal_mode):
    """``(episodes, start_steps, goals, clouds, retrieval, block)`` for one grid layer."""
    with open(cfg.eval.grid_file) as f:
        spec = json.load(f)
    block = dict(spec["blocks"][int(cfg.eval.get("grid_block", 0))])
    limit = cfg.eval.get("grid_limit")
    if limit:  # smoke runs: the first N goals of the layer, nothing else changes
        block = {k: (v[: int(limit)] if isinstance(v, list) else v) for k, v in block.items()}
        print(f"[grid] eval.grid_limit={limit}: running the first {len(block['goals'])} goals")
    goals = np.asarray(block["goals"], dtype=np.float32)
    assert len(goals) == cfg.eval.num_eval, (
        f"grid layer holds {len(goals)} goals but eval.num_eval={cfg.eval.num_eval}; the "
        f"layers differ in size, so the sweep config sets eval.num_eval per item"
    )
    episodes = np.asarray(block["episodes"], dtype=int)
    start_steps = np.asarray(block["start_steps"], dtype=int)
    if goal_mode == "pred":
        return episodes, start_steps, goals, None, None, spec, block
    if cfg.eval.get("grid_retrieval", False):
        # nearest recorded frame per goal -- what a retrieval baseline can get
        return episodes, start_steps, goals, None, block["retrieval"], spec, block
    scans_file = cfg.eval.get("grid_scans")
    assert scans_file, ("goal_mode=true on a grid needs either eval.grid_retrieval=true "
                        "or eval.grid_scans (the rebuilt goal clouds)")
    with np.load(scans_file) as f:
        scan_goals, scan_clouds, scan_block = f["goals"], f["clouds"], f["block"]
    idx = np.flatnonzero(scan_block == int(cfg.eval.get("grid_block", 0)))[: len(goals)]
    assert np.allclose(np.asarray(scan_goals)[idx], goals, atol=1e-2), (
        "eval.grid_scans was built for different goals than eval.grid_file lists -- "
        "rebuild the clouds with scripts/cube_goal_scans.py"
    )
    print(f"[grid] rebuilt goal clouds from {scans_file}: {scan_clouds[idx].shape}")
    return episodes, start_steps, goals, scan_clouds[idx], None, spec, block


@hydra.main(version_base=None, config_path="./config/eval", config_name="3dtarget_cube")
def run(cfg: DictConfig):
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    tg = OmegaConf.to_container(cfg.target_goal, resolve=True)
    # .lower(): Hydra parses the CLI token `true` as a bool -> str() gives "True".
    goal_mode = str(tg.get("goal_mode", "pred")).lower()
    assert goal_mode in ("pred", "true"), goal_mode
    assert cfg.policy != "random", "this script always needs the world-model ckpt"
    spec = SPECS[cfg.target_goal.env]

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    grid_file = cfg.eval.get("grid_file")
    wiring = GRID_ENVS.get(spec.name)
    threshold = None
    pre_wrappers = list(make_lidar_pre_wrappers(cfg.lidar.sensor))
    if grid_file:
        assert wiring, f"no grid wiring for env {spec.name!r} (GRID_ENVS)"
        # The success threshold is the environment's own, and it is also what
        # grid_results.json recomputes success against, so it is set whether or
        # not termination is suppressed.
        threshold = float(cfg.eval.get("grid_success", wiring["success"]))
    if grid_file and cfg.eval.get("grid_no_terminate", True):
        # PRE-wrapper, i.e. INSIDE MegaWrapper: once the base env reports
        # terminated, that layer freezes the env and every later info it emits
        # is NaN, so hiding the flag further out is too late -- the episode
        # still stops dead at the success threshold. Wrapped here the flag
        # never reaches MegaWrapper and the episode runs the full budget.
        pre_wrappers.append(lambda env: GridProbeWrapper(env, threshold, wiring))
        print(f"[grid] termination suppressed: every episode runs the full "
              f"{cfg.eval.eval_budget}-step budget and the closest approach is recorded")
    world = swm.World(
        **cfg.world,
        image_shape=(cfg.eval.img_size, cfg.eval.img_size),
        pre_wrappers=pre_wrappers,
    )
    dataset = swm.data.load_dataset(cfg.eval.dataset_name)

    # z-score processors. dataset.stats_file (JSON {col: {mean, std}}) pins the
    # statistics instead of fitting them on the table, exactly as in
    # eval_lidar.py: a subset table (data_sample/) must plan with the full
    # table's normalizer, not its own.
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

    cache_dir = resolve_run_dir(cfg)
    jepa = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cache_dir)

    goal_model, goal_cfg = None, {}
    if goal_mode == "pred":
        goal_model, goal_cfg = load_goal_model(tg["model_dir"])
        assert goal_cfg["env"] == spec.name, (
            f"goal model was trained for env {goal_cfg['env']!r}, "
            f"config says {spec.name!r}"
        )
        print(f"[target_goal] goal model: {goal_cfg['head']}"
              f"{'+z' if goal_cfg['use_z'] else ''} from {tg['model_dir']}")

    # Eval-time sensor degradation (the robustness ablation), built exactly as
    # eval_lidar.py builds it, so a `lidar.perturb.*` override means the same
    # thing in both scripts.
    perturb_cfg = OmegaConf.to_container(cfg.lidar.get("perturb", {}) or {}, resolve=True)
    perturb = CloudPerturbation(
        invalid_value=cfg.lidar.invalid_value,
        in_channels=cfg.lidar.in_channels,
        seed=int(perturb_cfg.pop("seed", None) or cfg.seed),
        **perturb_cfg,
    )
    if perturb.active:
        if goal_mode == "pred" and perturb.scope != "obs":
            # The goal is a typed pose read from privileged columns; there is no
            # goal cloud to degrade. Refuse rather than silently run an `obs` row
            # under a `both` label.
            raise ValueError(
                "lidar.perturb.scope=both has no meaning with target_goal.goal_mode=pred "
                "(the goal is a pose, not a cloud); use scope=obs or goal_mode=true"
            )
        print(f"[eval] lidar perturbation: {perturb.describe()}")
    else:
        perturb = None

    model = TargetGoalAdapter(
        jepa, goal_model, goal_cfg, spec,
        goal_mode=goal_mode,
        sample_steps=tg.get("sample_steps"),
        key=cfg.lidar.key,
        in_channels=cfg.lidar.in_channels,
        invalid_value=cfg.lidar.invalid_value,
        perturb=perturb,
    ).to("cuda").eval()
    model.requires_grad_(False)

    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_cfg, process=process, transform={}
    )

    results_path = Path(cfg.output.dir)
    starts_file = cfg.eval.get("starts_file")
    grid_spec = grid_block = grid_goals = None
    if grid_file:
        assert len(spec.columns) == 1, (
            f"the commanded-goal grid overwrites one pose column; {spec.name} has "
            f"{spec.columns}"
        )
        (eval_episodes, eval_start_idx, grid_goals, grid_clouds, grid_retrieval,
         grid_spec, grid_block) = load_grid_block(cfg, goal_mode)
        dataset = GridGoalDataset(dataset, spec.columns[0], grid_goals, grid_clouds,
                                  cloud_column=cfg.lidar.key, retrieval=grid_retrieval)
        print(f"[grid] {grid_file} layer {cfg.eval.get('grid_block', 0)}: "
              f"{len(grid_goals)} commanded goals, goal_mode={goal_mode}"
              + (" (retrieval clouds)" if grid_retrieval else ""))
    elif starts_file:
        with open(starts_file) as f:
            fixed = json.load(f)
        eval_episodes = np.asarray(fixed["episodes"], dtype=int)
        eval_start_idx = np.asarray(fixed["start_steps"], dtype=int)
        assert len(eval_episodes) == cfg.eval.num_eval
        print(f"[eval] using fixed starts from {starts_file}")
    else:
        eval_episodes, eval_start_idx = sample_eval_starts(
            dataset, cfg.eval.num_eval, cfg.eval.goal_offset_steps, cfg.seed,
            max_start_step=cfg.eval.get("max_start_step"),
            min_goal_displacement=cfg.eval.get("min_goal_displacement"),
        )
    print("episodes:", eval_episodes, "start steps:", eval_start_idx)

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path if tg.get("video", True) else None,
        lidar_render_rgb=False,
    )
    dt = time.time() - t0
    print(metrics)
    print(f"[target_goal] goal time: {model.stats['goal_time'] * 1e3:.1f} ms over "
          f"{model.stats['n_solves']} solves")

    summary = {
        "success_rate": metrics["success_rate"],
        "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        "env": spec.name,
        "goal_mode": goal_mode,
        "goal_model": str(tg.get("model_dir", "")),
        "goal_head": goal_cfg.get("head", "true"),
        "goal_use_z": bool(goal_cfg.get("use_z", False)),
        "sample_steps": tg.get("sample_steps"),
        "perturb": perturb.describe() if perturb is not None else "none",
        "seed": int(cfg.seed),
        "starts_file": str(starts_file),
        "num_eval": int(cfg.eval.num_eval),
        "policy": str(cfg.policy),
        "run": str(cfg.get("run")),
        "eval_seconds": dt,
        "goal_ms_total": model.stats["goal_time"] * 1e3,
        "n_solves": model.stats["n_solves"],
    }
    if grid_block is not None:
        # The env's own target IS the commanded position (its goal-setting
        # callable read the overwritten goal column), so the live distance to it
        # is the error we want. Taken from the probe wrappers, not from
        # world.infos: the pool's buffers are NaN once the budget has run out.
        closest, steps_to, final, dist, traces, nsteps = probe_stats(world)
        print(f"[grid] env steps taken: {min(nsteps)}..{max(nsteps)} (budget "
              f"{cfg.eval.eval_budget}); finite distance readings per env: "
              f"{min(len(t) for t in traces)}..{max(len(t) for t in traces)}")
        succ = closest < threshold
        start = grid_spec.get("meta", {}).get("start", grid_block.get("start"))
        start_dist = grid_block.get("start_to_goal_m", [float("nan")] * len(grid_goals))
        grid_out = dict(
            grid_file=str(grid_file), block=int(cfg.eval.get("grid_block", 0)),
            start=start, goal_mode=goal_mode,
            goal_source=("retrieval" if cfg.eval.get("grid_retrieval", False)
                         else ("rebuilt" if goal_mode == "true" else "predicted")),
            units=wiring["unit"], to_cm=wiring["to_cm"],
            height_m=grid_block.get("height_m"), regime=grid_block.get("regime"),
            goal_model=str(tg.get("model_dir", "")),
            eval_budget=int(cfg.eval.eval_budget),
            success_threshold=threshold,
            metric=("closest_distance: how close the cube got to the COMMANDED position "
                    "during the episode -- the primary number, since termination is "
                    "suppressed and every episode runs the full budget. final_distance is "
                    "where it ended (drift after arriving shows up as the difference). "
                    "success is recomputed here as closest < threshold; World.evaluate's "
                    "success_rate is meaningless in grid mode because nothing terminates."),
            median_closest_distance=float(np.median(closest)),
            max_closest_distance=float(closest.max()),
            median_final_distance=float(np.median(dist)),
            success_rate=float(100.0 * succ.mean()),
            cases=[dict(goal_idx=int(gi), goal=[float(v) for v in g],
                        start_to_goal=float(gd), final_pos=[float(v) for v in p],
                        closest_distance=float(c), final_distance=float(d),
                        steps_to_threshold=(None if st is None else int(st)), success=bool(sc),
                        **({"distance_trace": [round(float(x), 4) for x in tr]}
                           if cfg.eval.get("grid_trace") else {}))
                   for gi, g, gd, p, c, d, st, sc, tr in zip(
                       grid_block.get("goal_idx", range(len(grid_goals))), grid_goals,
                       start_dist, final, closest, dist, steps_to, succ, traces)],
        )
        with open(results_path / "grid_results.json", "w") as f:
            json.dump(grid_out, f, indent=2)
        cm = wiring["to_cm"]
        print(f"[grid] closest approach to the commanded position: median "
              f"{grid_out['median_closest_distance'] * cm:.1f} cm, max "
              f"{grid_out['max_closest_distance'] * cm:.1f} cm; within "
              f"{threshold * cm:.1f} cm: {grid_out['success_rate']:.0f}%  -> grid_results.json")
        summary["grid_file"] = str(grid_file)
        summary["grid_block"] = int(cfg.eval.get("grid_block", 0))
        summary["grid_median_closest_cm"] = grid_out["median_closest_distance"] * cm
    with open(results_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with (results_path / cfg.output.filename).open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    run()
