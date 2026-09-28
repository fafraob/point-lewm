# `target2latent` — typed 3-D goals → world-model latents

Learns the map

```
goal objects' 3-D pose  ──►  z_goal ∈ R^D    (the frozen world-model latent)
```

so that closed-loop planning no longer needs a *recorded goal point cloud*: you
specify **where the objects should be** and the model synthesizes the latent the
CEM planner steers toward. The intended use is manual goal placement in a 3-D
scan — put the cube / T + ball / agent where you want them, read off their pose,
and that pose *is* the goal.

## Goal parameterization (per env)

| env | goal input | conditioning layout |
| --- | --- | --- |
| `tworoom` | agent 3-D position | coords (3,) |
| `pusht` | T 3-D position + heading, ball 3-D position | coords (6,) = [T, ball], rot (3,) |
| `cube` | cube 3-D position | coords (3,) |
| `reacher` | wrist-joint 3-D position + fingertip 3-D position | coords (6,) = [wrist, finger] |

Positions are 3-D points in the **sensor frame** (the frame the stored clouds
live in). The only rotation that enters is the pusht T's planar yaw, as a
**sensor-frame heading vector**: the T's in-plane heading axis rotated into the
sensor frame — the direction the placed T mesh's local axis reads as in the
scan. Unique per yaw and continuous, so no rotation augmentation is needed.
The cube deliberately has **no orientation channel**: a cube's orientation is
only defined up to its 24-fold octahedral symmetry (many distinct quaternions
label identical point clouds), so it is an ill-posed conditioning signal — the
goal is *where* the cube is, and the heads average/sample over orientation.
The reacher has no target object (the env scores a joint-angle match), so its
goal is the **arm's pose** as two points: the wrist joint and the fingertip,
i.e. forward kinematics of the goal `qpos`. Two points, not one: a fingertip
position alone has two inverse-kinematics branches (elbow left / right) that
give different scans, and a head conditioned on it would average them.
`target2latent/envs.py` holds the per-env constants (verified sensor poses,
pixel→world mappings, the encoder's normalization affine) and a self-check —
`python -m target2latent.envs` transforms the privileged object positions of
dataset rows and measures the distance to the nearest recorded lidar return
(expected: cube = its 2 cm half-extent, pusht ball = its 7.5 cm radius,
tworoom agent = its 6 cm body-height offset, reacher wrist / fingertip =
2-3 cm, the link and finger box faces).

## The four architectures (`models.py`)

| head | z_now | what it is |
| --- | --- | --- |
| `mlp` | – | Fourier-featurized coords (+ Linear-embedded rot) → residual MLP → one latent. Fits `E[z \| goal]`. |
| `mlp` | `--use-z` | same, additionally conditioned on the latent of the current observation |
| `shortcut` | – | shortcut model (Frans et al. 2024) over latents: AdaLN-Zero DiT-style MLP, flow matching + self-consistency with an **EMA-target** bootstrap (the paper's recipe; live-weight targets measured unstable). Samples in 1, 2, 4, … steps — at 1 step one forward pass. |
| `shortcut` | `--use-z` | same, `z_now` joins the conditioning vector |

Positions go through the same log-spaced sinusoidal bands as the PointViT
encoder's `_SinusoidalPosEmbed`, after the encoder's own affine
`(p − norm_center)/norm_scale`. `z_now` conditioning means "the objects at this
pose, the rest of the scene continuing from now"; at eval it is recomputed at
every replan. Training pairs for it span offsets ±5…±25 env steps — the gaps
the replanning loop actually faces.

**Augmentation** (on by default, `--pos-jitter 0.005`): 5 mm Gaussian jitter on
the position channels. Positions are near-unique per frame and the Fourier
bands resolve ~1 cm, so without it a head uses the position as a lookup key and
memorizes single frames (train loss collapses while held-out R² drops).

## Pipeline

`precompute.py` runs the frozen encoder over a point-cloud dataset once and caches
`z` + the sensor-frame goal pose per frame; training then holds everything in
GPU memory and takes minutes. Splits are at **episode** granularity and every
episode named by a `starts_*.json` window file is excluded from training, so
closed-loop comparisons run on trajectories the goal model never saw.

`eval_3dtarget.py` (repo root) is `eval_lidar.py` with exactly one change:
`z_goal` is predicted from the dataset goal frame's privileged pose columns
instead of encoding the goal cloud. Same env, sensor, CEM, horizon, start
windows. `target_goal.goal_mode=true` encodes the recorded goal cloud instead —
the parity self-check that the rewritten cost path reproduces the baseline.

## Running it

All commands run from the repository root, inside the pixi environment
(`pixi run python ...`, or plain `python` in an activated `pixi shell`).
Paths follow `paths.py`: the datasets live under `$PLWM_DATA_ROOT`, the training
runs under `$PLWM_LOGS_ROOT`, and this package's caches and heads under
`$PLWM_T2L_ROOT` (default `experiment_logs/target2latent`); the sweep configs
read the same variable. `<env>` is one of `cube | pusht | tworoom | reacher`,
`<model>` one of `point_lewm | point_delta_jepa` (Point-LeWM / Point-Delta-JEPA).

```bash
export PLWM_T2L_ROOT=${PLWM_T2L_ROOT:-experiment_logs/target2latent}

# 0. self-check: the frame transforms against the stored clouds (all envs)
python -m target2latent.envs                 # or: python -m target2latent.envs cube

# 1. start lists. The 40 pinned window files ship in eval_starts/
#    (starts_<env>_seed<k>.json, seeds 1 25 35 36 40 42 65 80 83 99); they fix
#    the eval windows for paired comparisons AND name the episodes held out from
#    head training. Regenerate only if you must (existing files are kept):
python scripts/make_eval_starts.py --env cube --seeds 1 25 35 36 40 42 65 80 83 99 \
    --out-dir eval_starts

# 2. latent cache (once per env x encoder checkpoint, stride 5; resumable).
#    The released checkpoints live under checkpoints/<env>/<model>/
#    (pixi run download-checkpoints cube) and are addressed with --cache-dir . ;
#    a training run of your own with --run <folder> --policy <config>/weights_final.pt.
#    Use one --out per checkpoint: a cache resumes from existing shards without
#    checking which checkpoint wrote them.
python -m target2latent.precompute --env cube \
    --cache-dir . --policy cube/point-delta-jepa/weights.pt \
    --out $PLWM_T2L_ROOT/cache/cube_point_delta_jepa_s5.npz

# 3. the four heads (minutes each). Defaults are the paper's protocol:
#    60 epochs, AdamW lr 3e-4 (one-cycle), weight decay 1e-4, batch 4096,
#    --pos-jitter 0.005 (5 mm); --eval-dir defaults to eval_starts/.
CACHE=$PLWM_T2L_ROOT/cache/cube_point_delta_jepa_s5.npz
M=$PLWM_T2L_ROOT/models/cube_point_delta_jepa
python -m target2latent.train --env cube --cache $CACHE --head mlp              --out ${M}_mlp
python -m target2latent.train --env cube --cache $CACHE --head mlp      --use-z --out ${M}_mlp_z
python -m target2latent.train --env cube --cache $CACHE --head shortcut         --out ${M}_shortcut
python -m target2latent.train --env cube --cache $CACHE --head shortcut --use-z --out ${M}_shortcut_z
#    The heads of the paper are released next to their encoder,
#    checkpoints/<env>/<model>/target2latent/{mlp,mlp_z,shortcut,shortcut_z}/model.pt,
#    so steps 2 and 3 are only needed for a checkpoint of your own.

# 4. closed-loop: all four heads + the goal-cloud parity arm, paired on the
#    pinned windows, ten seeds x 50 episodes (the sweep addresses the released heads)
pixi run bash scripts/run_sweep.sh config/eval_sweep/3dtarget_cube_point_delta_jepa.conf
python scripts/summarize_evals.py eval_results/3dtarget_cube_point_delta_jepa

#    one head, one seed by hand (the sweep runs exactly this per entry):
python eval_3dtarget.py --config-name=3dtarget_cube \
    policy=cube/point-delta-jepa/weights.pt \
    target_goal.model_dir=checkpoints/cube/point-delta-jepa/target2latent/mlp \
    seed=42 eval.starts_file=eval_starts/starts_cube_seed42.json \
    output.dir=eval_results/3dtarget_cube_point_delta_jepa/t2l_mlp/seed42
#    your own encoder and heads: run=point_delta_jepa_cube policy=point_delta_jepa_cube/weights_final.pt
#                                target_goal.model_dir=${M}_mlp
#    parity self-check (recorded goal cloud, should match eval_lidar.py):
python eval_3dtarget.py --config-name=3dtarget_cube \
    policy=cube/point-delta-jepa/weights.pt \
    target_goal.goal_mode=true \
    seed=42 eval.starts_file=eval_starts/starts_cube_seed42.json \
    output.dir=eval_results/3dtarget_cube_point_delta_jepa/goalcloud/seed42
```

The naming convention ties the stages together: caches are
`cache/<env>_<model>_s5.npz`, heads `models/<env>_<model>_<head>` with `<head>`
in `mlp | mlp_z | shortcut | shortcut_z`. The sweep configs
`config/eval_sweep/3dtarget_<env>_<model>.conf` (eval config
`config/eval/3dtarget_<env>.yaml`) address the released heads,
`checkpoints/<env>/<model>/target2latent/<head>/`; point their `M=` line at your
own `models/` folder to evaluate heads you trained (the `${M}/<head>` entries then
need the `${M}_<head>` naming above, or name your folders the released way).
Keep the sweep's `SEEDS` a subset of the seeds that have a starts file, and
generate any new starts file BEFORE training the heads that will be evaluated
on it.

## Caveats

* The closed-loop metric is stochastic run-to-run (the encoder's ball query
  resamples every forward); single-seed differences mean nothing. Always
  compare paired on fixed `eval.starts_file` windows, pooled over seeds.
* Encoder checkpoints are pinned per cache (`_meta_run` / `_meta_policy` in the
  npz); train and eval must use the same one.
* The cube eval's raycast backend is `warp` (it reproduces the stored clouds
  bit-exactly), which needs `warp-lang` in the environment; pusht, tworoom and reacher use
  the `mujoco` backend (the pusht/tworoom mirror scenes are rebuilt per episode).
