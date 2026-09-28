# Reproducing the paper

Table and figure numbers refer to [arXiv:2608.29434](https://arxiv.org/abs/2608.29434). Every
command runs from the repository root inside the pixi environment, with the released checkpoints
downloaded (`pixi run download-checkpoints`) and the datasets in place (`pixi run download-data`,
[`docs/data.md`](data.md)).

| Paper | Where |
|---|---|
| Sec. 3.1, App. C: re-sensing the benchmarks | `stable-worldmodel/scripts/data/lidar_from_h5.py`, [`docs/data.md`](data.md) |
| Sec. 3.2, App. A: Point-LeWM and Point-Delta-JEPA | `jepa.py`, `module.py`, `pc_encoders/pointvit_encoder.py`, `config/train/{point_lewm,point_delta_jepa}_<env>.yaml` |
| Sec. 3.2, App. A: Utonia-WM and Vox-WM | `pc_encoders/utonia_encoder.py`, `pc_encoders/voxstats_encoder.py`, `utonia/`, `precompute_utonia.py`, `config/train/{utoniawm,voxstats}_<env>.yaml` |
| Sec. 3.2.2, App. A-B: target-to-latent module | `target2latent/` ([README](../target2latent/README.md)), `eval_3dtarget.py` |
| Table 2: image Delta-JEPA baseline | `config/train/image_delta_jepa_<env>.yaml`, `config/train/model/image_delta_jepa.yaml` |
| Table 2: planning success | `eval_lidar.py`, `eval.py`, `config/eval_sweep/{lidar,utoniawm,voxstats,image,dinowm}_<env>.conf`, `image_<env>_delta_jepa.conf` |
| Table 3: planning toward 3-D targets | `config/eval_sweep/3dtarget_<env>_<model>.conf` |
| Tables 4, 9, 10: sensor degradation | `config/eval_sweep/ablation_<env>.conf`, `ablation_t2l_<env>.conf`, `scripts/gen_ablation_confs.py`, `scripts/ablation_table.py` |
| Figure 5, Table 11, App. D.2.1: goals the expert never commanded | `config/eval/3dtarget_cube_grid.yaml`, `config/eval_sweep/3dtarget_cube_grid.conf`, `scripts/cube_grid_goals.py`, `cube_goal_scans.py`, `cube_grid_stats.py`, `plot_cube_goals_pair.py` |
| Tables 6-8: latent probing | `probe_latents.py`, `results/probing/` |
| Table 12: statistical tests | `scripts/paired_stats.py`, `results/table2_per_seed.csv` |
| Figure 3: scenes and their scans | `scripts/scene_pairs.py` |
| Figure 6: encoder attention | `scripts/attention_all.sh`, `scripts/viz_attention.py`, `scripts/figure_attention.py` |

## Evaluation protocol

Every evaluation is closed-loop receding-horizon CEM planning (300 samples, 30 iterations, planning
horizon 5 steps of frameskip 5) toward a goal taken 25 steps ahead in a recorded expert trajectory,
with a budget of 50 environment steps and 50 episodes per seed. Point-cloud models observe live
scans cast by the same sensor that generated the training data; the image models plan on the same
datasets, whose `pixels` column is the original image data, so every model sees the identical
start-goal windows.

One evaluation of one seed:

```bash
pixi run python eval_lidar.py --config-name=lidar_pusht seed=42                                   # Point-Delta-JEPA (config default)
pixi run python eval_lidar.py --config-name=lidar_pusht seed=42 policy=pusht/point-lewm/weights.pt # Point-LeWM
pixi run python eval_lidar.py --config-name=utoniawm_pusht seed=42                                # Utonia-WM (needs --utonia backbone)
pixi run python eval.py --config-name=image_pusht seed=42                                         # image Delta-JEPA
pixi run python eval.py --config-name=image_pusht seed=42 policy=image_lewm_pusht                 # image LeWM (--image-lewm download)
pixi run python eval.py --config-name=image_pusht seed=42 cache_dir=checkpoints/pusht/dino-wm policy=model_latest.pth
pixi run python eval_lidar.py --config-name=lidar_pusht seed=42 policy=random                     # random baseline
```

It prints the success rate, appends it to `eval_results/<config>_results.txt` and writes one video
per episode next to it (`++output.video=false` turns them off). Planning is stochastic, because the
encoder's ball query resamples points at every forward pass, so a single seed can differ from the
logged value by a few points. The reported quantity is the mean over ten seeds.

**Sweeps.** The paper reports mean and standard deviation over ten evaluation seeds (1, 25, 35,
36, 40, 42, 65, 80, 83, 99). The seed selects which 50 start-goal windows are drawn, identically for
every model, so comparisons are paired by seed. Each table is produced by the sweep configs in
`config/eval_sweep/`, which list the checkpoints (one `label | overrides` line each), the seeds and
the evaluation config:

```bash
pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf --dry-run   # show the plan
pixi run bash scripts/run_sweep.sh config/eval_sweep/lidar_tworoom.conf             # one evaluation per GPU
python scripts/summarize_evals.py eval_results/lidar_tworoom                        # mean +- sd over seeds
```

| Table 2 row | Sweep configs | Checkpoint |
|---|---|---|
| Random | `image_<env>.conf` (label `random`) | - |
| LeWM (image) | `image_<env>.conf` (label `image_lewm`) | Maes et al., `--image-lewm` |
| Point-LeWM, Point-Delta-JEPA | `lidar_<env>.conf` | released |
| Delta-JEPA (image) | `image_<env>_delta_jepa.conf` | released |
| DINO-WM | `dinowm_<env>.conf` | released |
| Utonia-WM | `utoniawm_<env>.conf` | released (+ Utonia backbone) |
| Vox-WM | `voxstats_<env>.conf` | train `voxstats_<env>` first ([`docs/training.md`](training.md)) |

`scripts/run_sweep.sh` runs the evaluations in parallel, one per visible GPU (`--gpus "0,1"` to
choose them, `--seeds "1 42"` for a subset), and writes
`eval_results/<sweep>/<label>/seed<k>/{results.txt,eval.log,exit_code}`. Utonia-WM rolls out a grid
of about 256 tokens per frame and is roughly 50 times slower to evaluate than the end-to-end models.

**Fixed start-goal lists.** `eval_starts/starts_<env>_seed<k>.json` pins the 50 windows of every
seed. They are exactly what the seed draws, made explicit so that the target-to-latent heads can
exclude these episodes from training. `scripts/make_eval_starts.py` regenerates them.

## Planning toward 3-D targets (Table 3)

`eval_3dtarget.py` replaces the goal scan by a goal latent predicted from the 3-D target by one of
the four released heads under `checkpoints/<env>/<model>/target2latent/`; the `goalcloud` row of
each sweep encodes the recorded goal scan through the same code path as a parity check.

```bash
pixi run bash scripts/run_sweep.sh config/eval_sweep/3dtarget_pusht_point_delta_jepa.conf   # Point-Delta-JEPA, Push-T
pixi run bash scripts/run_sweep.sh config/eval_sweep/3dtarget_pusht_point_lewm.conf        # Point-LeWM, Push-T
python scripts/summarize_evals.py eval_results/3dtarget_pusht_point_delta_jepa
```

Heads of your own ([`docs/training.md`](training.md)) are evaluated by pointing the sweep's `M=`
line, or `target_goal.model_dir=`, at their folder.

## Sensor degradation (Tables 4, 9, 10)

Range noise of 0.25 % and 0.5 % of the scan extent and dropout of 25 % and 50 % of the returns,
applied at evaluation time only. The levels are defined once in `scripts/gen_ablation_confs.py`,
which writes the eight sweeps:

```bash
python scripts/gen_ablation_confs.py
for env in tworoom reacher pusht cube; do
  pixi run bash scripts/run_sweep.sh config/eval_sweep/ablation_$env.conf       # Table 9, goal cloud
  pixi run bash scripts/run_sweep.sh config/eval_sweep/ablation_t2l_$env.conf   # Tables 4 and 10, 3-D target
done
python scripts/ablation_table.py           # writes tables/table9_degradation.tex and table10_t2l_degradation.tex
```

As in the paper, every entry is relative to the model's clean success rate (100 = no degradation):
the script divides each mean and standard deviation by the clean mean, taken from the `*_clean` row
of `ablation_<env>.conf` or, if that row has not been run, from `lidar_<env>.conf`. Table 4 is the
Two-Room and OGB-Cube excerpt of Table 10. `scripts/perturb_preview.py` and
`scripts/perturb_inspect.py` show what each level does to a scan.

## Goals the expert never commanded (Figure 5, Table 11, App. D.2.1)

Every recorded OGB-Cube goal is a placement on the table (z = 0.02 m), so a cube held above it is a
goal the expert never once aimed at while staying inside the arm's workspace. 201 commanded
positions on five height layers, each approached from five designed grasped starts, are planned for
twice with the same frozen world model and planner: once with the goal latent predicted from the
commanded position, once with the goal cloud *rebuilt* (the environment set to the commanded pose
and re-scanned, the strongest goal observation obtainable in simulation). Only the goal latent
differs between the two arms.

```bash
pixi run python scripts/cube_grid_goals.py       # the commanded goals, five grasped starts
pixi run python scripts/cube_goal_scans.py       # the rebuilt goal clouds, after its fidelity check
pixi run python -m pytest tests/test_cube_grid.py
pixi run python scripts/gen_cube_grid_conf.py    # 2 arms x 5 starts x 5 layers = 50 items
pixi run bash scripts/run_sweep.sh config/eval_sweep/3dtarget_cube_grid.conf
pixi run python scripts/cube_grid_stats.py       # Table 11
pixi run python scripts/plot_cube_goals_pair.py  # Figure 5
```

The artefacts land under `$PLWM_T2L_ROOT/grid/` (default `experiment_logs/target2latent/grid/`)
and the sweep uses the released OGB-Cube Point-Delta-JEPA checkpoint and its `mlp_z` head.
`cube_goal_scans.py` refuses to write unless the rebuilt clouds sit inside the spread of recorded
observations of the same pose: a rebuild that is an outlier would hand the baseline a scene the
encoder never saw, and it would then lose for reasons of rendering rather than of goal
representation.

Two properties of the measurement are worth knowing before reading the numbers.
`eval_3dtarget.py` suppresses environment termination inside the grid path and reports each
episode's *closest approach* instead: the environment stops the moment the cube is within its 4 cm
threshold, so a final distance would measure the threshold rather than the method's precision. And
the sensor's field of view does not reach the far corners at height, so for those goals no goal
observation exists at all, rebuilt or otherwise; `cube_goal_scans.py` labels them and
`cube_grid_stats.py` reports them as their own stratum rather than averaging them in. Replication
here is over the five starts, not over evaluation seeds: nothing is retrained, so a second seed
would only resample planner and encoder noise, while a second start re-asks the question from
another corner of the workspace.

## Latent probing (Tables 6-8)

Linear (ridge) and MLP probes regress privileged simulator state from the 192-dimensional latent
on episode-disjoint splits, by default from the released Point-LeWM and Point-Delta-JEPA
checkpoints:

```bash
for env in pusht reacher cube; do pixi run python probe_latents.py --env $env; done
pixi run python probe_latents.py --report
```

`results/probing/` holds the paper's numbers (`probe_report.txt` and one JSON file per
environment). To probe a run of your own, pass
`--logs-root experiment_logs --models "Point-LeWM=point_lewm_pusht:point_lewm_pusht/weights_final.pt"`.

## Encoder attention (Figure 6) and scenes (Figure 3)

`scripts/attention_all.sh` computes the CLS-token attention rollout of every released model and
environment (`scripts/viz_attention.py`) and renders it; `scripts/figure_attention.py` composes
the figure. `pixi run python scripts/scene_pairs.py` renders the scenes and their scans.

## Statistical tests (Table 12)

`results/table2_per_seed.csv` holds the per-seed success rates behind Table 2 (10 seeds x 50
episodes per cell; the image Delta-JEPA row is reproduced by its sweep). The paired t-tests, the
TOST equivalence tests at a margin of 10 points and the objective-family gaps follow from

```bash
pixi run python scripts/paired_stats.py results/table2_per_seed.csv
```
