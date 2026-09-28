"""Pin closed-loop eval windows: write ``starts_<env>_seed<k>.json`` files.

The release ships the lists every reported number was evaluated on, in
``<repo>/eval_starts/starts_<env>_seed<k>.json`` for the four environments
(cube, tworoom, pusht, reacher) and the ten seeds 1 25 35 36 40 42 65 80 83 99;
the sweep configs under ``config/eval_sweep/`` point ``eval.starts_file`` at
them. This script is how those files were made and how to add seeds.

A starts file is JSON ``{"episodes": [...], "start_steps": [...]}`` consumed in
two places:

* ``eval.starts_file`` in eval_lidar.py / eval_3dtarget.py -- fixes the eval
  windows, the precondition for paired checkpoint comparisons (the metric's
  run-to-run noise from encoder ball-query resampling is larger than most
  effects);
* ``target2latent.data.eval_episode_set`` -- every episode named by any
  ``starts_*.json`` in the goal-model trainer's ``--eval-dir`` (point it at
  ``eval_starts/``) is EXCLUDED from training, so the closed-loop eval runs on
  trajectories the goal model never saw. Generate the files BEFORE training a
  goal model.

Windows are drawn by ``eval_lidar.sample_eval_starts``, i.e. exactly what a
bare ``seed=<k>`` eval would sample from the same dataset -- the file just
makes the draw explicit, portable, and known to training. The episodes it
returns are POSITIONAL indices into ``dataset.lengths``; the trainer's
exclusion matches them against the ``episode_idx`` column, which is the same
numbering (verified on the cube table: episode_idx is 0..N-1 in storage
order). One file per seed::

    pixi run python scripts/make_eval_starts.py --env cube --seeds 42 1 25

Existing files are kept (delete to regenerate): a starts file that silently
changes after a goal model was trained would break the exclusion guarantee.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo root on the path: this script lives in scripts/ but imports eval_lidar
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, help="target2latent.envs.SPECS entry")
    p.add_argument("--dataset", default=None, help="override the spec's lance path")
    p.add_argument("--out-dir", default=None,
                   help="default <repo>/eval_starts, where the shipped lists live")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--max-start-step", type=int, default=None)
    p.add_argument("--min-goal-displacement", type=float, default=None,
                   help="cube only; null = the historical 'easy' metric")
    args = p.parse_args()

    from target2latent.envs import SPECS

    spec = SPECS[args.env]
    dataset_path = args.dataset or spec.dataset
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "eval_starts"
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [(s, out_dir / f"starts_{args.env}_seed{s}.json") for s in args.seeds]
    for seed, out in [t for t in todo if t[1].exists()]:
        print(f"[starts] {out} exists, keeping it (delete to regenerate)")
    todo = [t for t in todo if not t[1].exists()]
    if not todo:
        return

    import stable_worldmodel as swm
    from eval_lidar import sample_eval_starts

    dataset = swm.data.load_dataset(dataset_path)
    for seed, out in todo:
        eps, steps = sample_eval_starts(
            dataset, args.num_eval, args.goal_offset, seed,
            max_start_step=args.max_start_step,
            min_goal_displacement=args.min_goal_displacement,
        )
        payload = {
            "episodes": np.asarray(eps).astype(int).tolist(),
            "start_steps": np.asarray(steps).astype(int).tolist(),
            "meta": {
                "env": args.env, "dataset": str(dataset_path), "seed": seed,
                "num_eval": args.num_eval, "goal_offset": args.goal_offset,
                "max_start_step": args.max_start_step,
                "min_goal_displacement": args.min_goal_displacement,
            },
        }
        with open(out, "w") as f:
            json.dump(payload, f)
        print(f"[starts] wrote {out}  ({len(payload['episodes'])} windows)")


if __name__ == "__main__":
    main()
