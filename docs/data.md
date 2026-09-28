# Data

The four benchmarks are the image datasets released with
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Two-Room, Reacher, Push-T, OGB-Cube),
re-sensed with a simulated range sensor. The conversion replays every recorded state in the
simulator, without stepping the physics, and stores the scan next to every original column.
Episodes, actions and goals are therefore identical to the image datasets, and the image baselines
are evaluated on the same datasets (their `pixels` column is the original image data).

## Download the point-cloud datasets

```bash
pixi run download-data                 # all four (56 GB of archives, ~870 GB extracted)
pixi run download-data tworoom cube    # a subset
```

The datasets land in `$PLWM_DATA_ROOT` (default `data/` under the repository), one
[Lance](https://lancedb.github.io/lance/) dataset per environment, and the archives are deleted after
extraction (`--keep-archives` keeps them). Equivalent by hand:

```bash
hf download fafraob/point-cloud-tworoom two_room.lance.tar.zst --repo-type dataset --local-dir $PLWM_DATA_ROOT
tar --zstd -xf $PLWM_DATA_ROOT/two_room.lance.tar.zst -C $PLWM_DATA_ROOT
```

| | Two-Room | Reacher | Push-T | OGB-Cube |
|---|---|---|---|---|
| Hugging Face dataset | `fafraob/point-cloud-tworoom` | `fafraob/point-cloud-reacher` | `fafraob/point-cloud-pusht` | `fafraob/point-cloud-ogb-cube` |
| Archive | `two_room.lance.tar.zst`, 0.8 GB | `reacher.lance.tar.zst`, 7.1 GB | `pusht.lance.tar.zst`, 4.6 GB | `cube.lance.tar.zst`, 43 GB |
| Dataset | `two_room.lance` | `reacher.lance` | `pusht.lance` | `cube.lance` |
| Sensor pose | derived oblique | derived oblique | derived oblique | scene camera |
| Episodes | 10,000 | 10,000 | 18,685 | 10,000 |
| Frames | 920,809 | 2,010,000 | 2,336,736 | 2,010,000 |
| Episode length | 31-101 | 201 | 49-246 | 201 |

Every frame stores a fixed `(10000, 3)` cloud in the sensor frame (x forward, y left, z up), with
missed rays set to `-1`, next to the original columns (`pixels`, `action`, the simulator state and
the privileged object poses). `stable-worldmodel/lidar_integration.md` documents the sensor
interface the world-model code consumes. `data_sample/cube_sample.lance` is a three-window excerpt
of the OGB-Cube dataset with bit-identical rows (episode and step indices renumbered), and
`data_sample/action_stats.json` the full dataset's action statistics, which a subset must plan with
(`++dataset.stats_file=`).

## Where data and results go

All locations are relative to the repository root by default and move with environment variables
(`paths.py`); the Hydra configs read the same variables:

| Variable | Default | Holds |
|---|---|---|
| `PLWM_DATA_ROOT` | `data/` | the point-cloud datasets (`.lance`) and, for the image baselines, the source `.h5` files |
| `PLWM_LOGS_ROOT` | `experiment_logs/` | training runs, one folder per run |
| `PLWM_RESULTS_ROOT` | `eval_results/` | evaluation sweeps |
| `PLWM_T2L_ROOT` | `experiment_logs/target2latent/` | target-to-latent latent caches and heads you train |
| `PLWM_PROBE_ROOT` | `experiment_logs/probing/` | probing caches and results |

Storage is substantial: 870 GB for the four datasets, 260 GB for the extracted image datasets (only
needed to train the image baselines or to re-run the conversion), and 100-240 GB for each Utonia-WM
token cache.

## The image datasets (image baselines, re-sensing)

The image Delta-JEPA baseline and DINO-WM train on the original image datasets (`.h5`), not on the
point-cloud datasets. Download them from the LeWM collection <https://huggingface.co/collections/quentinll/lewm>:

| Environment | Dataset repository | Archive | Extracted file |
|---|---|---|---|
| Two-Room | `quentinll/lewm-tworooms` | `tworoom.tar.zst` (3.4 GB) | `tworoom.h5` (13 GB) |
| Push-T | `quentinll/lewm-pusht` | `pusht_expert_train.h5.zst` (13 GB) | `pusht_expert_train.h5` (46 GB) |
| Reacher | `quentinll/lewm-reacher` | `reacher.tar.zst` (24 GB) | `reacher.h5` (99 GB) |
| OGB-Cube | `quentinll/lewm-cube` | `cube_single_expert.tar.zst` (46 GB) | `cube_single_expert.h5` (102 GB) |

```bash
export PLWM_DATA_ROOT=$PWD/data          # or any large disk
for repo in lewm-tworooms lewm-pusht lewm-reacher lewm-cube; do
  pixi run hf download "quentinll/$repo" --repo-type dataset --local-dir "$PLWM_DATA_ROOT"
done
for archive in tworoom.tar.zst reacher.tar.zst cube_single_expert.tar.zst; do
  pixi run tar --zstd -xvf "$PLWM_DATA_ROOT/$archive" -C "$PLWM_DATA_ROOT"
done
pixi run zstd -d "$PLWM_DATA_ROOT/pusht_expert_train.h5.zst"
```

## Re-sensing the image datasets yourself

The released datasets were produced by `stable-worldmodel/scripts/data/lidar_from_h5.py` from the
`.h5` files above; the four Hydra configs under `stable-worldmodel/scripts/data/config/` hold the
exact sensor settings of the paper (a 100 x 100 ray grid, the OGB-Cube scene camera, and a derived
oblique 45/45 viewpoint for the other three environments) and reproduce the datasets bit for bit.

```bash
REPO=$PWD                  # the repository root
cd "$(mktemp -d)"          # the script writes its Hydra log into the working directory
CONVERT="pixi run --manifest-path $REPO/pixi.toml python $REPO/stable-worldmodel/scripts/data/lidar_from_h5.py"
$CONVERT --config-name=lidar_from_h5_tworoom input=$PLWM_DATA_ROOT/tworoom.h5            output=$PLWM_DATA_ROOT/two_room.lance
$CONVERT --config-name=lidar_from_h5_pusht   input=$PLWM_DATA_ROOT/pusht_expert_train.h5 output=$PLWM_DATA_ROOT/pusht.lance
$CONVERT --config-name=lidar_from_h5_reacher input=$PLWM_DATA_ROOT/reacher.h5            output=$PLWM_DATA_ROOT/reacher.lance
$CONVERT --config-name=lidar_from_h5         input=$PLWM_DATA_ROOT/cube_single_expert.h5 output=$PLWM_DATA_ROOT/cube.lance
```

The OGB-Cube conversion casts rays on the GPU (`warp` backend) and takes about three hours. The
other three use the MuJoCo CPU raycaster on a mirror scene of the 2D environment. A run can be
resumed (the default `write_mode=append` skips finished episodes) and sharded over processes with
`ep_start=` / `ep_end=`.
