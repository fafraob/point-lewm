# Training

Each world model has one training config per environment, `config/train/<model>_<env>.yaml` with
`<env>` one of `tworoom`, `reacher`, `pusht`, `cube` (the commands below show `tworoom`). A run is
written to `experiment_logs/<run_name>/` (TensorBoard logs, Lightning checkpoints, the resolved
config) and exports its weights after every epoch to
`experiment_logs/<run_name>/checkpoints/<config>/` (`weights_epoch_N.pt` and `weights_final.pt`, a
copy of the newest export). Name each run after its config: the evaluation configs and sweeps
address a run of your own as `run=<run_name> policy=<config>/weights_final.pt`.

The point-cloud models train on the datasets of [`docs/data.md`](data.md)
(`pixi run download-data <env>`); the image baselines on the original `.h5` image datasets.

## Point-LeWM and Point-Delta-JEPA

```bash
pixi run train --config-name=point_lewm_tworoom      run_name=point_lewm_tworoom
pixi run train --config-name=point_delta_jepa_tworoom run_name=point_delta_jepa_tworoom
```

## Utonia-WM

The encoder is the frozen Utonia backbone (`pixi run download-checkpoints --utonia`, CC-BY-NC-4.0),
so its token grid is computed once per environment (resumable; about 15 frames per second on one
RTX 4090 and shardable over GPUs, see the docstring of `precompute_utonia.py`), checked against a
live encode, and then read by training:

```bash
pixi run precompute-utonia --config-name=utoniawm_tworoom
pixi run precompute-utonia --config-name=utoniawm_tworoom --verify
pixi run train --config-name=utoniawm_tworoom run_name=utoniawm_tworoom
```

## Vox-WM

The encoder is parameter-free and runs live, so no cache is needed:

```bash
pixi run train --config-name=voxstats_tworoom run_name=voxstats_tworoom
```

## Delta-JEPA (image baseline)

The same objective as Point-Delta-JEPA on pixels: the image LeWM stack with SIGReg replaced by the
Latent Difference Action Decoder. It trains on the image datasets themselves (the `.h5` files,
`config/train/data/image_<env>.yaml`), not on the point-cloud datasets. One model config serves all
four environments, because the per-environment keys of the point-cloud encoders describe a cloud's
geometry and have no pixel counterpart:

```bash
pixi run train --config-name=image_delta_jepa_tworoom run_name=image_delta_jepa_tworoom
```

These configs follow the Delta-JEPA paper's own protocol of 50 epochs from scratch, not this
repository's 10-epoch point-cloud protocol, so the comparison against the released image LeWM
checkpoint differs in budget as well as in loss. Every epoch is exported, so an earlier export gives
the equal-budget comparison against the point-cloud arms without a second run.

## DINO-WM (image baseline)

`dinowm_swm/` runs the unmodified official DINO-WM code (`third_party/dino_wm/`) on the image
datasets, with frozen DINOv2 ViT-S/14 patch tokens and no proprioception:

```bash
pixi run torchrun --nproc_per_node=2 -m dinowm_swm.train env=swm_tworoom \
    run_dir=$(realpath -m "${PLWM_LOGS_ROOT:-experiment_logs}")/dinowm_tworoom
```

`run_dir` must be absolute. `training.epochs` is a total budget, so relaunching the same command
resumes from `<run_dir>/checkpoints/model_latest.pth`. DINOv2 is fetched through `torch.hub` on
first use; on a machine without internet access, pre-populate a hub cache and set `TORCH_HOME`.
Every deviation from the original DINO-WM recipe is listed at the top of
`dinowm_swm/conf/train_swm.yaml`. To evaluate such a run:
`pixi run python eval.py --config-name=image_tworoom cache_dir=experiment_logs/dinowm_tworoom policy=model_latest.pth`.

## Target-to-latent heads

The heads map a 3-D target (the cube position, the Two-Room agent position, the Push-T block pose
and ball position, or the Reacher wrist and fingertip positions) and optionally the current latent
to a goal latent of a frozen world model. Training takes minutes on cached latents. For one
environment and world model (here the released Push-T Point-Delta-JEPA):

```bash
T2L=${PLWM_T2L_ROOT:-experiment_logs/target2latent}
# 1. cache the frozen latents of every 5th frame (one cache per checkpoint; ~30 min per 2 M frames)
pixi run python -m target2latent.precompute --env pusht \
    --cache-dir . --policy pusht/point-delta-jepa/weights.pt --out $T2L/cache/pusht_point_delta_jepa_s5.npz
# 2. train the four heads: MLP and shortcut, with and without the current latent
for head in mlp shortcut; do
  pixi run python -m target2latent.train --env pusht --cache $T2L/cache/pusht_point_delta_jepa_s5.npz \
      --head $head --out $T2L/models/pusht_point_delta_jepa_$head
  pixi run python -m target2latent.train --env pusht --cache $T2L/cache/pusht_point_delta_jepa_s5.npz \
      --head $head --use-z --out $T2L/models/pusht_point_delta_jepa_${head}_z
done
# 3. plan toward the 3-D targets on the pinned start lists with a head of yours
pixi run python eval_3dtarget.py --config-name=3dtarget_pusht seed=42 \
    eval.starts_file=eval_starts/starts_pusht_seed42.json \
    target_goal.model_dir=$T2L/models/pusht_point_delta_jepa_mlp_z
```

A checkpoint of your own is addressed with `--run <run_name> --policy <config>/weights_final.pt`
instead of `--cache-dir . --policy ...`. The episodes named in `eval_starts/` are held out from
head training. [`target2latent/README.md`](../target2latent/README.md) documents the goal
parameterization, the frames and the four architectures.

## Protocol of the reported runs

The configs encode the setup of the reported runs. `loader.batch_size` is per GPU.

| Model | Environments | GPUs x batch x accumulation | Effective batch | Schedule | Learning rate |
|---|---|---|---|---|---|
| Point-LeWM | Two-Room, Reacher, Push-T | 1 x 128 x 1 | 128 | 10 epochs | 5e-5 |
| Point-LeWM | OGB-Cube (512 tokens, 0.03 m radius) | 2 x 64 x 1 | 128 | cosine over 50 epochs | 5e-5 |
| Point-Delta-JEPA | Two-Room, Reacher, Push-T | 2 x 64 x 1 | 128 | 10 epochs | 5e-5 |
| Point-Delta-JEPA | OGB-Cube (512 tokens, 0.03 m radius) | 2 x 32 x 2 | 128 | cosine over 50 epochs | 5e-5 |
| Utonia-WM | all | 2 x 256 x 1 | 512 | 100 epochs | 1e-3 |
| Vox-WM | all | 2 x 256 x 1 | 512 | 100 epochs | 1e-3 (Push-T 3e-4) |
| Delta-JEPA (image) | all | 1 x 64 x 2 | 128 | 50 epochs | 5e-5 |
| DINO-WM | all | 2 x 64 | 128 | budget of 100 epochs, resumable | 1e-3 |

All models use AdamW, bf16 and gradient clipping at 1.0 (App. A of the paper). SIGReg is
synchronized across GPUs, so Point-LeWM gives the same objective on one or two GPUs at equal
effective batch. On a machine with a different GPU count, keep
`devices x batch_size x accumulate_grad_batches` equal to the effective batch, e.g.
`trainer.devices=1 loader.batch_size=64 accumulate_grad_batches=2` for a two-GPU Point-Delta-JEPA
config. Resume an interrupted run with `+ckpt_path=experiment_logs/<run_name>/checkpoints/last.ckpt`.
Every epoch's export is kept, and an evaluation can address a specific one with
`policy=<config>/weights_epoch_N.pt` instead of `weights_final.pt`.

## Adding an environment

Three encoder values are properties of the environment's geometry, not of the model: the fixed
workspace affine `norm_center` / `norm_scale` and the ball-query radius `group_radius`
(`config/train/model/*.yaml`). They were measured once per dataset from a sample of frames: the
midpoint of the per-axis 1st and 99th percentiles, the largest half-extent between them, and
roughly the 75th percentile of the distance to the 32nd nearest neighbour (tune it visually with
`pixi run viz-radius`). OGB-Cube is the exception: its radius of 0.03 m with 512 centres was chosen
separately (App. A of the paper). The Utonia-WM and Vox-WM token grids were fitted with
`viz_grid.py`. A new environment also needs a sensor config for the conversion
(`stable-worldmodel/scripts/data/config/`) and, for planning toward 3-D targets, an entry in
`target2latent/envs.py`.
