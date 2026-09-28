<div align="center">
<h1>Does Latent Planning Survive Point Clouds?<br>Action-Conditioned JEPA World Models for Geometric Observations and Goals</h1>

[**Fabio F. Oberweger**](https://scholar.google.com/citations?user=njm6I3wAAAAJ)<sup>&ast;&dagger;</sup>,
[**Michael Schwingshackl**](https://scholar.google.com/citations?user=fsvMYQYAAAAJ)<sup>&ast;</sup> &
[**Markus Murschitz**](https://scholar.google.com/citations?user=S8yQbTQAAAAJ)

AIT Austrian Institute of Technology<br>
Center for Vision, Automation & Control

&ast;joint first authors &emsp; &dagger;correspondence

<a href="https://arxiv.org/abs/2608.29434"><img src="https://img.shields.io/badge/arXiv-2608.29434-red" alt="Paper"></a>
<a href="https://fafraob.github.io/point-lewm/"><img src="https://img.shields.io/badge/Project_Page-point--lewm-green" alt="Project Page"></a>
<a href="https://huggingface.co/collections/fafraob/point-lewm-point-delta-jepa-and-more-6aba3f4ea8842e5c1f72d6e5"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets_%26_Checkpoints-blue" alt="Hugging Face"></a>
</div>

World models that plan from **point clouds** instead of pixels. Point-LeWM and Point-Delta-JEPA
put a point-cloud encoder in front of the LeWorldModel and Delta-JEPA objectives, plan with CEM in
latent space, and reach goals given either a goal scan or just a **3-D target pose**. This
repository holds the code, the re-sensed point-cloud benchmarks, and every trained checkpoint of
the paper.

We also release two image baselines that had no public checkpoints on these benchmarks: our
**re-implementation of Delta-JEPA for images** and **DINO-WM trained with the
[official code](https://github.com/gaoyuezhou/dino_wm)**, both for all four environments. If you
use them, please cite the original papers together with this work (see [Citation](#citation)).

The code builds on [LeWorldModel](https://github.com/lucas-maes/le-wm) and
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (Maes et al.), which we
extend with a simulated range sensor, point-cloud encoders and the target-to-latent module, on the
[DINO-WM](https://github.com/gaoyuezhou/dino_wm) code (Zhou et al.) and on the
[Utonia](https://github.com/Pointcept/Utonia) backbone (Pointcept); see
[Acknowledgements](#acknowledgements).

## Released assets

| Environment | Point-cloud dataset | Checkpoints (all models) |
|---|---|---|
| Two-Room | [`fafraob/point-cloud-tworoom`](https://huggingface.co/datasets/fafraob/point-cloud-tworoom) (0.8 GB) | [`fafraob/point-cloud-tworoom`](https://huggingface.co/fafraob/point-cloud-tworoom) |
| Reacher | [`fafraob/point-cloud-reacher`](https://huggingface.co/datasets/fafraob/point-cloud-reacher) (7.1 GB) | [`fafraob/point-cloud-reacher`](https://huggingface.co/fafraob/point-cloud-reacher) |
| Push-T | [`fafraob/point-cloud-pusht`](https://huggingface.co/datasets/fafraob/point-cloud-pusht) (4.6 GB) | [`fafraob/point-cloud-pusht`](https://huggingface.co/fafraob/point-cloud-pusht) |
| OGB-Cube | [`fafraob/point-cloud-ogb-cube`](https://huggingface.co/datasets/fafraob/point-cloud-ogb-cube) (43 GB) | [`fafraob/point-cloud-ogb-cube`](https://huggingface.co/fafraob/point-cloud-ogb-cube) |

Each checkpoint repository holds one folder per model:

| Folder | Model | Observation |
|---|---|---|
| `point-lewm/` | Point-LeWM | point cloud |
| `point-delta-jepa/` | Point-Delta-JEPA | point cloud |
| `point-lewm/target2latent/`, `point-delta-jepa/target2latent/` | target-to-latent goal heads (`mlp`, `mlp_z`, `shortcut`, `shortcut_z`) | 3-D target pose |
| `image-delta-jepa/` | Delta-JEPA, our image re-implementation | RGB |
| `dino-wm/` | DINO-WM, trained with the official code | RGB |
| `utonia-wm/` | Utonia-WM, a JEPA predictor on frozen [Utonia](https://huggingface.co/Pointcept/Utonia) features | point cloud |

The image LeWM checkpoints of Maes et al. are evaluated unchanged from
[their release](https://huggingface.co/collections/quentinll/lewm), and Vox-WM (a parameter-free
voxel-statistics encoder) is trained from its config; neither is re-hosted here.

## Installation

Linux x86-64, an NVIDIA GPU with a CUDA 12 driver, EGL for headless MuJoCo, and
[pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | bash`). Evaluation and the
goal heads run on a 24 GB consumer GPU; the training configs assume A100s.

```bash
git clone https://github.com/fafraob/point-lewm && cd point-lewm
pixi install --locked        # the pinned environment (Python 3.10, PyTorch 2.7, CUDA 12.6)
pixi run build-kernels       # once: compiles torch-cluster and torch-scatter against that PyTorch (~20 min)
pixi run test                # optional
```

`TORCH_CUDA_ARCH_LIST="8.9"` (RTX 4090) or `"8.0"` (A100) before `build-kernels` compiles for one
GPU only. Every command below runs through `pixi run ...` from the repository root.

## Evaluate the released checkpoints

Everything below is shown for OGB-Cube. The other environments work identically: replace `cube`
by `pusht`, `reacher` or `tworoom` in the download commands, in the config name (`lidar_<env>`,
`3dtarget_<env>`, `image_<env>`) and in the checkpoint paths.

### 1. Download the checkpoints and the dataset

```bash
pixi run download-checkpoints cube      # all released models of OGB-Cube, 0.9 GB -> checkpoints/cube/
pixi run download-data cube             # the point-cloud dataset, 43 GB -> data/cube.lance
```

`PLWM_DATA_ROOT=/big/disk` before `download-data` puts the datasets elsewhere; the configs read the
same variable. `pixi run download-checkpoints` with no argument fetches all four environments
(3.7 GB).

### 2. One evaluation

```bash
pixi run python eval_lidar.py --config-name=lidar_cube seed=42
```

This plans 50 episodes with the released Point-Delta-JEPA world model: closed-loop CEM planning in
latent space from live scans of the simulated range sensor, toward the scan of a goal state 25
steps ahead in a recorded expert trajectory, with a budget of 50 environment steps. The seed
selects which 50 start-goal windows of the dataset are drawn. The script prints the success rate,
appends it with the resolved config to `eval_results/lidar_cube_results.txt`, and writes one video
per episode (agent | dataset | goal | scan panels) next to it. Useful overrides:

```bash
    ++output.video=false        # no videos
    eval.num_eval=10            # a quick look with fewer episodes
    output.dir=eval_results/my_run
```

Planning is stochastic (the encoder's ball query resamples points at every forward pass), so a
repeated seed differs by a few points. The paper reports mean and standard deviation over ten
seeds, see [step 5](#5-the-paper-protocol).

### 3. Every released model on the same windows

The checkpoint is selected with `policy=<env>/<model>/weights.pt`, relative to `checkpoints/`:

```bash
pixi run python eval_lidar.py --config-name=lidar_cube seed=42 policy=cube/point-lewm/weights.pt        # Point-LeWM
pixi run python eval_lidar.py --config-name=lidar_cube seed=42 policy=cube/point-delta-jepa/weights.pt  # Point-Delta-JEPA (the default)
pixi run python eval_lidar.py --config-name=utoniawm_cube seed=42                                       # Utonia-WM
pixi run python eval.py       --config-name=image_cube seed=42                                          # Delta-JEPA (image)
pixi run python eval.py       --config-name=image_cube seed=42 cache_dir=checkpoints/cube/dino-wm policy=model_latest.pth   # DINO-WM
pixi run python eval.py       --config-name=image_cube seed=42 policy=image_lewm_cube                   # LeWM (image)
pixi run python eval_lidar.py --config-name=lidar_cube seed=42 policy=random                            # random baseline
```

The image models plan on the same dataset (its `pixels` column is the original image data), so all
rows see the identical windows. Two of them need one more download: Utonia-WM the frozen Utonia
backbone (`pixi run download-checkpoints --utonia`, CC-BY-NC-4.0), and the image LeWM baseline the
checkpoints of Maes et al. (`pixi run download-checkpoints --image-lewm`). DINO-WM fetches DINOv2
through `torch.hub` on first use.

### 4. Plan toward a 3-D target instead of a goal scan

```bash
pixi run python eval_3dtarget.py --config-name=3dtarget_cube seed=42 eval.starts_file=eval_starts/starts_cube_seed42.json
```

Same planner, same windows, but the goal latent is predicted from the cube's target position by
a released target-to-latent head (default: the `mlp_z` head of Point-Delta-JEPA). The start list is
pinned because these episodes were held out when the heads were trained; `eval_starts/` has one
per seed and environment. Variants:

```bash
    target_goal.model_dir=checkpoints/cube/point-delta-jepa/target2latent/shortcut_z   # another head: mlp, mlp_z, shortcut, shortcut_z
    policy=cube/point-lewm/weights.pt target_goal.model_dir=checkpoints/cube/point-lewm/target2latent/mlp_z   # the Point-LeWM heads
    target_goal.goal_mode=true      # parity check: encode the recorded goal scan instead (reproduces eval_lidar.py)
```

Results go to `eval_results/3dtarget_cube_results.txt` and `summary.json`. The goal parameterization
of each environment (cube position; T pose and ball position; agent position; wrist and fingertip)
is documented in [`target2latent/README.md`](target2latent/README.md).

### 5. The paper protocol

Ten seeds (1, 25, 35, 36, 40, 42, 65, 80, 83, 99), 50 episodes each, paired across models by the
seed. The sweep files in `config/eval_sweep/` run exactly that, one evaluation per GPU:

```bash
pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_cube.conf     # --dry-run, --seeds "1 42", --gpus "0,1"
python scripts/summarize_evals.py eval_results/lidar_cube                # mean +- sd over seeds
```

[`docs/reproduction.md`](docs/reproduction.md) lists the sweep behind every table and figure.

### 6. A model you trained yourself

`run=<run folder> policy=<config name>/weights_final.pt` selects a training run under
`experiment_logs/` instead of the released checkpoints, for every evaluator above:

```bash
pixi run python eval_lidar.py --config-name=lidar_cube seed=42 run=point_lewm_cube policy=point_lewm_cube/weights_final.pt
```

### No dataset yet? Smoke test on the shipped sample

`data_sample/` holds three start-goal windows of the OGB-Cube dataset (10 MB) and the full dataset's
action statistics. With the checkpoints of step 1 downloaded:

```bash
SAMPLE="eval.dataset_name=$PWD/data_sample/cube_sample.lance eval.num_eval=3 ++dataset.stats_file=$PWD/data_sample/action_stats.json"
pixi run python eval_lidar.py    --config-name=lidar_cube    seed=42 $SAMPLE eval.starts_file=eval_starts/starts_cube_sample.json
pixi run python eval_3dtarget.py --config-name=3dtarget_cube seed=42 $SAMPLE eval.starts_file=eval_starts/starts_cube_sample.json
pixi run python eval.py          --config-name=image_cube    seed=42 $SAMPLE +eval.starts_file=eval_starts/starts_cube_sample.json
```

## Checkpoint and config names

| Paper | Code name | Training config | Evaluation config | Released checkpoint |
|---|---|---|---|---|
| Point-LeWM | `point_lewm` | `config/train/point_lewm_<env>.yaml` | `config/eval/lidar_<env>.yaml` | `<env>/point-lewm/weights.pt` |
| Point-Delta-JEPA | `point_delta_jepa` | `config/train/point_delta_jepa_<env>.yaml` | `config/eval/lidar_<env>.yaml` | `<env>/point-delta-jepa/weights.pt` |
| Target-to-latent | `target2latent` | `python -m target2latent.train` | `config/eval/3dtarget_<env>.yaml` | `<env>/<model>/target2latent/<head>/model.pt` |
| Utonia-WM | `utoniawm` | `config/train/utoniawm_<env>.yaml` | `config/eval/utoniawm_<env>.yaml` | `<env>/utonia-wm/weights.pt` (+ `--utonia` backbone) |
| Vox-WM | `voxstats` | `config/train/voxstats_<env>.yaml` | `config/eval/voxstats_<env>.yaml` | train it |
| Delta-JEPA (image) | `image_delta_jepa` | `config/train/image_delta_jepa_<env>.yaml` | `config/eval/image_<env>.yaml` | `<env>/image-delta-jepa/weights.pt` |
| DINO-WM (image) | `dinowm` | `dinowm_swm/conf/train_swm.yaml` | `config/eval/image_<env>.yaml` | `<env>/dino-wm/` |
| LeWM (image) | `image_lewm` | released by Maes et al. | `config/eval/image_<env>.yaml` | `--image-lewm` of the download script |

`<env>` is one of `tworoom`, `reacher`, `pusht`, `cube`. `load_pretrained` resolves
`<cache_dir>/checkpoints/<policy>`; the eval configs default `cache_dir` to the repository root,
and `run=` takes precedence over it.

## Training your own models

One config per model and environment; a run lands in `experiment_logs/<run_name>/` and exports its
weights after every epoch. For Two-Room:

```bash
pixi run download-data tworoom
pixi run train --config-name=point_lewm_tworoom      run_name=point_lewm_tworoom        # Point-LeWM
pixi run train --config-name=point_delta_jepa_tworoom run_name=point_delta_jepa_tworoom   # Point-Delta-JEPA
pixi run python eval_lidar.py --config-name=lidar_tworoom run=point_delta_jepa_tworoom policy=point_delta_jepa_tworoom/weights_final.pt
```

[`docs/training.md`](docs/training.md) covers all seven models (including the Utonia feature cache,
the image baselines and the goal heads), the protocol of the reported runs, multi-GPU settings, and
how to add an environment.

## Baselines we provide checkpoints for

**Delta-JEPA on images** ([paper](https://arxiv.org/abs/2606.31232)) had no released weights on
these benchmarks, so we re-implemented the objective (the LeWM image stack with SIGReg replaced by
the Latent Difference Action Decoder) and trained it on the four image datasets with the paper's
own protocol: `config/train/image_delta_jepa_<env>.yaml`, `config/train/model/image_delta_jepa.yaml`.

**DINO-WM** ([paper](https://arxiv.org/abs/2411.04983), [code](https://github.com/gaoyuezhou/dino_wm))
is trained with the unmodified official code, vendored under `third_party/dino_wm/`, through the
adapter in `dinowm_swm/` (frozen DINOv2 ViT-S/14, no proprioception, no decoder; every deviation
from the original recipe is listed in `dinowm_swm/conf/train_swm.yaml`). The checkpoints load with
the official code as well as with `eval.py` here.

**Utonia-WM** puts a JEPA predictor on the frozen [Utonia](https://github.com/Pointcept/Utonia)
point-cloud backbone. Only our trained predictor is released; the backbone is downloaded from
`Pointcept/Utonia` (`pixi run download-checkpoints --utonia`) and is licensed CC-BY-NC-4.0.

**Image LeWM** is evaluated from the checkpoints of
[Maes et al.](https://github.com/lucas-maes/le-wm) on the identical start-goal windows
(`pixi run download-checkpoints --image-lewm`).

If you use the Delta-JEPA image checkpoints or the DINO-WM checkpoints, please cite the respective
original paper together with ours, which provides the trained weights.

## Reproducing the paper

[`docs/reproduction.md`](docs/reproduction.md) maps every table and figure to its sweep config and
script: the evaluation protocol and seeds, planning toward 3-D targets, the sensor-degradation
ablations, goals the expert never commanded, latent probing, attention maps and the statistical
tests. [`docs/data.md`](docs/data.md) documents the datasets, their format, and the sensing pipeline
that produced them from the LeWM image datasets. `results/` holds the per-seed numbers behind the
main table.

## Repository layout

```
train.py, jepa.py, module.py, utils.py   world-model training (Hydra + Lightning)
eval_lidar.py                            closed-loop planning with a point-cloud world model
eval_3dtarget.py                         the same, goal latent predicted from a 3-D target
eval.py                                  closed-loop planning with an image world model
precompute_utonia.py                     frozen Utonia token cache (Utonia-WM)
probe_latents.py                         linear and MLP probes of the latents
paths.py                                 data / run / result locations (environment variables)
pc_encoders/                             PointViT, Utonia and Vox-WM encoders
target2latent/                           target-to-latent module (heads, data, training)
dinowm_swm/                              adapter running the official DINO-WM code on these datasets
utonia/, third_party/dino_wm/            vendored Utonia and DINO-WM
stable-worldmodel/                       vendored stable-worldmodel fork: environments, range sensor,
                                         dataset conversion, CEM planner
config/train, config/eval, config/eval_sweep   one config per model x environment; the paper's sweeps
eval_starts/                             fixed start-goal lists (10 seeds x 4 environments)
data_sample/                             three OGB-Cube windows + action statistics (quick start)
checkpoints/, data/                      downloads (git-ignored)
scripts/, tests/, docs/, results/
```

Data, runs and results default to `data/`, `experiment_logs/` and `eval_results/` under the
repository and move with `PLWM_DATA_ROOT`, `PLWM_LOGS_ROOT` and `PLWM_RESULTS_ROOT` (`paths.py`).

## Acknowledgements

This repository is a fork of [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al.,
[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)): `jepa.py`, `module.py`, `train.py`,
`eval.py` and `utils.py` descend from it, and the image LeWM baseline is their released model.
The environments, datasets, CEM planner and evaluation loop come from
[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (vendored under
`stable-worldmodel/` with our range sensor and dataset conversion on top). The DINO-WM baseline
runs the unmodified [dino_wm](https://github.com/gaoyuezhou/dino_wm) code (Zhou et al.,
[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)), vendored under `third_party/dino_wm/`. The
image Delta-JEPA baseline re-implements [Delta-JEPA](https://arxiv.org/abs/2606.31232). Utonia-WM
uses the frozen [Utonia](https://github.com/Pointcept/Utonia) backbone (Pointcept), vendored
under `utonia/`. Thank you to all of them for open-sourcing their work.

## License

The code is MIT (`LICENSE`); the released datasets and checkpoints are CC-BY-4.0. The repository
builds on LeWorldModel, stable-worldmodel, Utonia and DINO-WM, whose licenses and our modifications
are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{oberweger2026doeslatentplanningsurvive,
  title         = {Does Latent Planning Survive Point Clouds? Action-Conditioned JEPA World Models for Geometric Observations and Goals},
  author        = {Fabio F. Oberweger and Michael Schwingshackl and Markus Murschitz},
  year          = {2026},
  eprint        = {2608.29434},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.29434},
}
```

When using the baseline checkpoints, please also cite the original works:
[LeWorldModel](https://arxiv.org/abs/2603.19312) (Maes et al.),
[Delta-JEPA](https://arxiv.org/abs/2606.31232), [DINO-WM](https://arxiv.org/abs/2411.04983)
(Zhou et al.) and [Utonia](https://github.com/Pointcept/Utonia) (Pointcept).
