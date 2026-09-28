import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

# Lance datasets have no episode_idx/step_idx columns and live in per-run
# folders under experiment_logs/ -- reuse the lidar eval's positional episode
# sampling and run-dir resolution instead of duplicating them here.
from eval_lidar import load_fixed_stats, resolve_run_dir, sample_eval_starts, set_fixed_stats

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    # A .lance/.h5 path goes through the format-detecting swm.data.load_dataset.
    # A bare name falls back to the HDF5 cache lookup, which needs the optional
    # hdf5 extra (swm.data.HDF5Dataset). Every shipped eval config passes a
    # .lance path.
    if str(dataset_name).endswith((".lance", ".h5")):
        return swm.data.load_dataset(dataset_name)
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="image_pusht")
def run(cfg: DictConfig):
    """Run evaluation of a world model vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # dataset.stats == eval.dataset_name in the shipped configs

    # dataset.stats_file (JSON {col: {mean, std}}) pins the statistics instead of
    # fitting them on the table (eval_lidar.py does the same): a subset table
    # such as data_sample/ must plan with the full table's normalizer.
    fixed_stats = load_fixed_stats(cfg.dataset.get("stats_file"))
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        if col in fixed_stats:
            set_fixed_stats(processor, fixed_stats[col])
        else:
            col_data = stats_dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    cache_dir = None
    if policy != "random":
        # Per-run checkpoint layout (experiment_logs/<run>/checkpoints/<name>/)
        # when the config carries run-dir keys; else the flat cache layout.
        if cfg.get("logs_root") or cfg.get("run") or cfg.get("cache_dir"):
            cache_dir = resolve_run_dir(cfg)
        if str(policy).endswith(".pth"):
            # DINO-WM run folder trained with the vendored upstream code
            # (dinowm_swm): <run>/checkpoints/model_<epoch>.pth + hydra.yaml.
            # Wrapped into a Costable so the CEM below is the same as for
            # every other image arm; see dinowm_swm/planning.py.
            from dinowm_swm.planning import load_dinowm_cost_model
            assert cache_dir, "a DINO-WM .pth policy needs run=/cache_dir= (the run folder)"
            model = load_dinowm_cost_model(cache_dir, ckpt_name=str(policy), device="cuda")
        else:
            model = swm.wm.utils.load_pretrained(cfg.policy, cache_dir=cache_dir)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = None  # built below, once the pixel transform is known

    else:
        policy = swm.policy.RandomPolicy()
        model = None

    # pixel transform: ImageNet-normalized 224 px for the LeWM image
    # checkpoints; a model may bring its own (DINO-WM trains on
    # Normalize(0.5, 0.5) frames, dinowm_swm.planning exposes that transform)
    pixel_transform = getattr(model, "img_transform", None) or img_transform(cfg)
    transform = {"pixels": pixel_transform, "goal": pixel_transform}
    if policy is None:
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    if cfg.output.get("dir"):
        results_path = Path(cfg.output.dir)
    elif cfg.policy != "random":
        results_path = Path(
            cache_dir or swm.data.utils.get_cache_dir(), "checkpoints", cfg.policy
        ).parent
    else:
        results_path = Path(__file__).parent

    # sample the episodes and the starting indices
    if "step_idx" in dataset.column_names:
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
        episode_len = get_episodes_length(dataset, ep_indices)
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
        # Map each dataset row’s episode_idx to its max_start_idx
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
        )

        # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
        valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), "valid starting points found for evaluation.")

        g = np.random.default_rng(cfg.seed)
        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
        )

        # sort increasingly to avoid issues with HDF5Dataset indexing
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

        print(random_episode_indices)

        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

        if len(eval_episodes) < cfg.eval.num_eval:
            raise ValueError("Not enough episodes with sufficient length for evaluation.")
    else:
        # Lance reader: no episode_idx/step_idx columns -- episodes are
        # positional indices into dataset.lengths (same as the lidar eval).
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
        # one agent|dataset|goal panel per episode; output.video=false turns
        # them off (the sweeps in config/eval_sweep/ do, see
        # scripts/run_sweep.sh). Unset == true, so nothing changes for an
        # interactive `pixi run eval`.
        video=results_path if cfg.output.get("video", True) else None,
    )
    end_time = time.time()
    
    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"planner: {'random' if cfg.get('policy', 'random') == 'random' else 'cem'}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
