# Data

The four benchmarks are the image datasets released with
[LeWorldModel](https://github.com/lucas-maes/le-wm) (Two-Room, Reacher, Push-T, OGB-Cube),
re-sensed with a simulated range sensor. The conversion replays every recorded state in the
simulator, without stepping the physics, and stores the scan next to every original column.
Episodes, actions and goals are therefore identical to the image datasets. The image baselines
train on and are evaluated from the original image files (`.h5`,
[below](#the-image-datasets-image-baselines-re-sensing)), which hold the same episodes in the same
order, so every model is scored on the identical start-goal windows.

Since version 2 of the datasets (October 2026) the `pixels` column of the Two-Room, Reacher and
Push-T tables holds the sensor's own view: a 224 x 224 render from the range sensor's camera pose,
so image and scan share pose and field of view. The OGB-Cube sensor rides the scene camera, so its
`pixels` column has been that view from the start. See [Versions](#versions).

## Download the point-cloud datasets

```bash
pixi run download-data                 # all four (54 GB of archives, ~1.2 TB extracted)
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
| Archive | `two_room.lance.tar.zst`, 0.6 GB | `reacher.lance.tar.zst`, 7.7 GB | `pusht.lance.tar.zst`, 2.6 GB | `cube.lance.tar.zst`, 43 GB |
| Dataset | `two_room.lance` | `reacher.lance` | `pusht.lance` | `cube.lance` |
| Sensor pose | derived oblique | derived oblique | derived oblique | scene camera |
| `pixels` | sensor view (v2) | sensor view (v2) | sensor view (v2) | scene camera |
| Episodes | 10,000 | 10,000 | 18,685 | 10,000 |
| Frames | 920,809 | 2,010,000 | 2,336,736 | 2,010,000 |
| Episode length | 31-101 | 201 | 49-246 | 201 |

Every frame stores a fixed `(10000, 3)` cloud in the sensor frame (x forward, y left, z up), with
missed rays set to `-1`, next to the `pixels` column described above and the original columns
(`action`, the simulator state and the privileged object poses). `stable-worldmodel/lidar_integration.md` documents the sensor
interface the world-model code consumes. `data_sample/cube_sample.lance` is a three-window excerpt
of the OGB-Cube dataset with bit-identical rows (episode and step indices renumbered), and
`data_sample/action_stats.json` the full dataset's action statistics, which a subset must plan with
(`++dataset.stats_file=`).

## Versions

The Hub repositories keep both versions: `main` is version 2, and `--revision v1` fetches the
release the paper's point-cloud checkpoints were trained on, e.g.
`hf download fafraob/point-cloud-pusht pusht.lance.tar.zst --repo-type dataset --revision v1 --local-dir $PLWM_DATA_ROOT`.

| | v1 (September 2026) | v2 (October 2026) |
|---|---|---|
| Two-Room, Reacher `pixels` | original scene camera | sensor view |
| Two-Room, Reacher `lidar` | unchanged | unchanged |
| Push-T `pixels` | original scene camera | sensor view |
| Push-T `lidar` | T block displaced wherever it is rotated | recast with the corrected replay |
| OGB-Cube | unchanged | unchanged |

The v1 Push-T conversion restored the T block's position before its angle; pymunk resolves a
position against the body's centre of gravity at its previous rotation, so the T sat in the wrong
place in the scans wherever it is rotated. `state`, `pixels` and the actions were not affected, and
neither were the live environments, which do not replay. `stable_worldmodel/envs/replay.py` now
writes the angle first, and every pass over a table restores a frame through it. The released
point-cloud checkpoints and the numbers in the paper and in `results/` belong to v1.

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

Storage is substantial: 1.2 TB for the four datasets, 260 GB for the extracted image datasets (only
needed to train the image baselines or to re-run the conversion), and 100-240 GB for each Utonia-WM
token cache.

## The image datasets (image baselines, re-sensing)

The image baselines (image LeWM, image Delta-JEPA, DINO-WM) train on and are evaluated from the
original image datasets (`.h5`), not on the point-cloud tables: `config/train/data/image_<env>.yaml`
and `config/eval/image_<env>.yaml` read them (OGB-Cube keeps evaluating on `cube.lance`, whose
`pixels` are the original images). Download them from the LeWM collection
<https://huggingface.co/collections/quentinll/lewm>:

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

The scans were produced by `stable-worldmodel/scripts/data/lidar_from_h5.py` from the `.h5` files
above; the four Hydra configs under `stable-worldmodel/scripts/data/config/` hold the exact sensor
settings of the paper (a 100 x 100 ray grid, the OGB-Cube scene camera, and a derived oblique 45/45
viewpoint for the other three environments) and reproduce the scans of the current tables exactly,
with the original images in `pixels`.

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

The sensor-view images of version 2 are a second pass over a finished table,
`stable-worldmodel/scripts/data/scanview_from_lance.py`: it restores every recorded state in the
scene the scan was cast in, renders it from the sensor's camera and replaces the `pixels` column in
place (`--lidar` recasts the scans as well, which is how the Push-T table was repaired). Every step
is a new table version, the renders are produced fragment by fragment so several workers can share
a table, and one commit assembles the result:

```bash
S=$REPO/stable-worldmodel/scripts/data/scanview_from_lance.py
C=$REPO/stable-worldmodel/scripts/data/config/lidar_from_h5_tworoom.yaml
T=$PLWM_DATA_ROOT/two_room.lance
pixi run python $S plan     --table $T
pixi run python $S shard    --table $T --config $C --stage-dir stage/ --shard 0 --num-shards 1   # one call per shard
pixi run python $S commit   --table $T --stage-dir stage/
pixi run python $S finalize --table $T
pixi run python $S verify   --table $T --config $C --episodes 0 1 2
```
