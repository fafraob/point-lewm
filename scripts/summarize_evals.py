#!/usr/bin/env python3
"""Aggregate an evaluation sweep into one success-rate table.

A sweep (``scripts/run_sweep.sh`` over a ``config/eval_sweep/*.conf``) runs every
checkpoint at several seeds, each seed writing its own ``results.txt``. Each of those holds a
single 50-episode success rate, which on its own says very little: the seed
selects WHICH 50 (episode, start_step) windows are drawn from the dataset, so
two seeds of the same checkpoint routinely differ by several points. This
script reads all of them and reports mean +- standard deviation across seeds,
which is the number a comparison between arms should be based on.

    python3 scripts/summarize_evals.py eval_results/lidar_cube
    python3 scripts/summarize_evals.py eval_results/lidar_cube --csv out.csv

Standard library only, so it runs with any Python.
"""

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path


# eval_lidar.py appends "metrics: {...}" per run (single-quoted dict repr;
# numpy scalars may print as np.float64(46.0) depending on the numpy version,
# hence the optional wrapper); eval_3dtarget.py appends a JSON summary
# (double-quoted) -- the regex accepts both.
SUCCESS_RE = re.compile(r"['\"]success_rate['\"]:\s*(?:np\.float\d+\()?([0-9.eE+-]+)")
PLANNER_RE = re.compile(r"^planner:\s*(\S+)", re.M)
TIME_RE = re.compile(r"^evaluation_time:\s*([0-9.]+)", re.M)
NUM_EVAL_RE = re.compile(r"^\s*num_eval:\s*(\d+)", re.M)
SEED_DIR_RE = re.compile(r"^seed(-?\d+)$")


def parse_results_file(path):
    """Return the LAST result block in a results.txt, or None if unfinished.

    eval_lidar.py opens the file in append mode, so a re-run of the same
    (checkpoint, seed) adds a second block. The last one is the current answer;
    earlier blocks are kept in the file as history but not aggregated.
    """
    text = path.read_text(errors="replace")
    blocks = text.split("==== CONFIG ====")
    for block in reversed(blocks):
        m = SUCCESS_RE.search(block)
        if m:
            planner = PLANNER_RE.search(block)
            secs = TIME_RE.search(block)
            n = NUM_EVAL_RE.search(block)
            return {
                "success_rate": float(m.group(1)),
                "planner": planner.group(1) if planner else "?",
                "seconds": float(secs.group(1)) if secs else float("nan"),
                "num_eval": int(n.group(1)) if n else 0,
                "reruns": len(blocks) - 2,  # blocks[0] is the pre-first-run text
            }
    return None


def collect(sweep_dir):
    """Walk <sweep>/<label>/seed<k>/results.txt into {label: {seed: parsed}}."""
    runs = {}
    for results in sorted(sweep_dir.glob("*/seed*/results.txt")):
        seed_dir, label_dir = results.parent, results.parent.parent
        m = SEED_DIR_RE.match(seed_dir.name)
        if not m:
            continue
        parsed = parse_results_file(results)
        if parsed is None:
            print(f"  ! no metrics in {results} (task still running, or it crashed)", file=sys.stderr)
            continue
        runs.setdefault(label_dir.name, {})[int(m.group(1))] = parsed
    return runs


def expected_from_conf(sweep_dir):
    """(labels, seeds) the frozen sweep.conf asked for, so gaps are visible.

    A missing (label, seed) means that array task never wrote a result --
    almost always a task that hit the walltime or died. Reporting the mean of
    whatever happened to finish, silently, would be the wrong default.
    """
    conf = sweep_dir / "sweep.conf"
    if not conf.is_file():
        return None, None
    text = conf.read_text(errors="replace")
    seeds = None
    m = re.search(r"^SEEDS=\(([^)]*)\)", text, re.M)
    if m:
        seeds = [int(s) for s in m.group(1).split()]
    labels = None
    m = re.search(r"^EVALS=\((.*?)^\)", text, re.M | re.S)
    if m:
        labels = [
            line.strip().strip('"').split("|")[0].strip()
            for line in m.group(1).splitlines()
            if "|" in line and not line.strip().startswith("#")
        ]
    return labels, seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path, help="eval_results/<SWEEP_NAME>")
    ap.add_argument("--csv", type=Path, help="also write the per-seed rows to this CSV")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        sys.exit(f"no such sweep directory: {args.sweep_dir}")

    runs = collect(args.sweep_dir)
    if not runs:
        sys.exit(f"no results.txt with metrics under {args.sweep_dir}")

    exp_labels, exp_seeds = expected_from_conf(args.sweep_dir)

    print(f"\nsweep: {args.sweep_dir}")
    print("success rate (%), mean +- sd over seeds\n")
    print(f"{'checkpoint':<22} {'planner':<9} {'n':>2} {'mean':>7} {'sd':>6} "
          f"{'min':>6} {'max':>6}  per-seed")
    print("-" * 96)

    rows = []
    order = exp_labels if exp_labels else sorted(runs)
    for label in order:
        by_seed = runs.get(label)
        if not by_seed:
            print(f"{label:<22} {'--':<9} {'0':>2}   (no results)")
            continue
        seeds = sorted(by_seed)
        vals = [by_seed[s]["success_rate"] for s in seeds]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        planner = by_seed[seeds[0]]["planner"]
        per_seed = " ".join(f"{s}:{by_seed[s]['success_rate']:.0f}" for s in seeds)
        print(f"{label:<22} {planner:<9} {len(vals):>2} {statistics.fmean(vals):>7.1f} "
              f"{sd:>6.1f} {min(vals):>6.1f} {max(vals):>6.1f}  {per_seed}")
        for s in seeds:
            r = by_seed[s]
            rows.append({
                "label": label, "seed": s, "planner": r["planner"],
                "success_rate": r["success_rate"], "num_eval": r["num_eval"],
                "seconds": r["seconds"], "reruns": r["reruns"],
            })
        if exp_seeds:
            gaps = [s for s in exp_seeds if s not in by_seed]
            if gaps:
                print(f"{'':<22} MISSING seeds {gaps} -- those array tasks produced no result")
        if any(r["reruns"] for r in by_seed.values()):
            print(f"{'':<22} note: some seeds have appended re-runs; the last block is used")

    total_eps = sum(r["num_eval"] for r in rows)
    total_h = sum(r["seconds"] for r in rows if r["seconds"] == r["seconds"]) / 3600
    print("-" * 96)
    print(f"{len(rows)} evaluations, {total_eps} episodes, {total_h:.1f} GPU-hours\n")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
