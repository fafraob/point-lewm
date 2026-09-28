#!/usr/bin/env python3
"""Table 11: the commanded-goal grid, paired, stratified, over five starts.

From the repository root, after the sweep
``config/eval_sweep/3dtarget_cube_grid.conf`` has run::

    pixi run python scripts/cube_grid_stats.py
    pixi run python scripts/cube_grid_stats.py --results <sweep dir> --latex

WHAT IS PAIRED WITH WHAT. Every (goal, start) is run by both arms, so the unit
of comparison is one commanded goal reached from one start and the statistic is
the WITHIN-PAIR difference. That is the whole reason the grid files share one
goal set: an unpaired comparison of medians would be swamped by how hard the
individual goals are, which varies far more than the arms do.

THE STRATA. Two, decided before the runs and both carried by the goal clouds
themselves (``cube_returns`` in cube_goal_scans.npz):

  visible   the rebuilt goal cloud shows the commanded cube. The goal-cloud
            baseline is at full strength here -- an exact observation of the
            commanded state, which retrieval could never supply.
  blind     the sensor's field of view does not reach the commanded pose, so the
            rebuilt cloud shows the scene WITHOUT the cube. No goal observation
            of these poses exists at all; a typed goal names them regardless.

Reported per height layer and per stratum: each arm's median closest approach,
the paired median difference with a bootstrap CI, Wilcoxon's signed-rank p, and
each arm's success rate at the environment's own 4 cm threshold. Per-start
medians are printed too, so a start effect is visible rather than averaged away.

Distances are metres on disk and centimetres everywhere a human reads them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import RESULTS_ROOT, T2L_ROOT  # noqa: E402

TO_CM = 100.0
SUCCESS_CM = 4.0
ARMS = ("t2l_mlp_z", "goalcloud")
LABEL = {"t2l_mlp_z": "target-to-latent", "goalcloud": "rebuilt goal cloud",
         "t2l_mlp": "target-to-latent (no z)", "retrieval": "retrieval"}


def load_results(results, arms, starts):
    """``{(arm, layer, start): {goal_key: case}}`` plus the layer heights."""
    out, heights = {}, {}
    for d in sorted(Path(results).glob("*/seed*/grid_results.json")):
        label = d.parent.parent.name
        arm, _, tail = label.rpartition("_z")
        if arm not in arms:
            continue
        z_str, _, s_str = tail.partition("_s")
        if not s_str.isdigit():
            continue
        start = int(s_str)
        if start not in starts:
            continue
        r = json.loads(d.read_text())
        layer = int(r["block"])
        heights[layer] = float(r.get("height_m", int(z_str) / 100))
        out[(arm, layer, start)] = {tuple(np.round(c["goal"], 4)): c for c in r["cases"]}
    return out, heights


def paired(res, layer, starts, arms):
    """Per-goal paired closest approach, in cm, pooled over starts."""
    a, b, keys = [], [], []
    for s in starts:
        ha, hb = res.get((arms[0], layer, s)), res.get((arms[1], layer, s))
        if not ha or not hb:
            continue
        for k in sorted(set(ha) & set(hb)):
            a.append(ha[k]["closest_distance"] * TO_CM)
            b.append(hb[k]["closest_distance"] * TO_CM)
            keys.append((s, k))
    return np.asarray(a), np.asarray(b), keys


def boot_median_diff(d, n=10000, seed=0):
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.median(d[rng.integers(0, len(d), size=(n, len(d)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon(d):
    try:
        from scipy.stats import wilcoxon as w

        if len(d) < 6 or np.allclose(d, 0):
            return float("nan")
        return float(w(d).pvalue)
    except Exception:
        return float("nan")


def blind_goals(scans_file):
    """The commanded goals whose rebuilt cloud shows no cube at all."""
    if not Path(scans_file).is_file():
        return set()
    with np.load(scans_file) as z:
        g, seen = np.asarray(z["goals"]), np.asarray(z["cube_returns"])
    return {tuple(np.round(x, 4)) for x in g[seen == 0]}


def row(name, a, b, arms, n_note=""):
    d = a - b
    lo, hi = boot_median_diff(d)
    p = wilcoxon(d)
    star = "" if not np.isfinite(p) else (" *" if p < 0.05 else "")
    ci = f"[{lo:+.2f}, {hi:+.2f}]"
    print(f"  {name:<22s}{len(a):>5d}{np.median(a):>10.2f}{np.median(b):>10.2f}"
          f"{np.median(d):>+9.2f}  {ci:^16s}"
          f"{(a < SUCCESS_CM).mean() * 100:>6.0f}%{(b < SUCCESS_CM).mean() * 100:>6.0f}%"
          f"{'' if not np.isfinite(p) else f'{p:>9.3f}'}{star}{n_note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="default: $PLWM_RESULTS_ROOT/3dtarget_cube_grid")
    ap.add_argument("--scans", type=Path, default=None,
                    help="default: $PLWM_T2L_ROOT/grid/cube_goal_scans.npz; only used to "
                         "label the goals the sensor cannot see")
    ap.add_argument("--arms", nargs=2, default=list(ARMS), metavar=("A", "B"))
    ap.add_argument("--starts", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--latex", action="store_true", help="also emit the paper table")
    args = ap.parse_args()
    results = args.results or (Path(RESULTS_ROOT) / "3dtarget_cube_grid")
    scans = args.scans or (Path(T2L_ROOT) / "grid/cube_goal_scans.npz")

    res, heights = load_results(results, args.arms, set(args.starts))
    if not res:
        raise SystemExit(f"no grid_results.json for arms {args.arms} under {results}")
    have = sorted({s for (_, _, s) in res})
    layers = sorted(heights)
    print(f"{results}\n{len(res)} runs: arms {args.arms}, starts {have}, "
          f"layers {[f'{heights[l] * 100:.0f}cm' for l in layers]}")

    blind = blind_goals(scans)
    print(f"{len(blind)} commanded goals lie outside the sensor's view "
          f"(no goal observation of them exists)\n")

    print(f"  {'':<22s}{'n':>5s}{args.arms[0][:9]:>10s}{args.arms[1][:9]:>10s}"
          f"{'paired':>9s}  {'95% CI':^16s}{'ok':>7s}{'ok':>7s}{'p':>9s}")
    print(f"  {'':<22s}{'':>5s}{'median cm':>10s}{'median cm':>10s}{'diff':>9s}"
          f"  {'':^16s}{'A':>7s}{'B':>7s}")
    print("  " + "-" * 96)

    for layer in layers:
        a, b, keys = paired(res, layer, have, args.arms)
        if not len(a):
            continue
        vis = np.array([k[1] not in blind for k in keys])
        name = f"z = {heights[layer] * 100:.0f} cm"
        row(name + ("  (in-distribution)" if abs(heights[layer] - 0.02) < 1e-6 else ""),
            a, b, args.arms)
        if (~vis).any() and vis.any():
            row("    visible goals", a[vis], b[vis], args.arms)
            row("    outside sensor view", a[~vis], b[~vis], args.arms)

    print("  " + "-" * 96)
    elevated = [l for l in layers if heights[l] > 0.02]
    pooled = [paired(res, l, have, args.arms) for l in elevated]
    pooled = [p for p in pooled if len(p[0])]
    if pooled:
        a = np.concatenate([p[0] for p in pooled])
        b = np.concatenate([p[1] for p in pooled])
        keys = [k for p in pooled for k in p[2]]
        vis = np.array([k[1] not in blind for k in keys])
        row("elevated layers, all", a, b, args.arms)
        if (~vis).any():
            row("    visible goals", a[vis], b[vis], args.arms)
            row("    outside sensor view", a[~vis], b[~vis], args.arms)

    # per start: is the ranking a property of the method or of where we started?
    print(f"\n  per start (elevated layers, median cm):")
    for s in have:
        aa, bb = [], []
        for l in elevated:
            x, y, _ = paired(res, l, [s], args.arms)
            aa.append(x)
            bb.append(y)
        aa, bb = np.concatenate(aa), np.concatenate(bb)
        if len(aa):
            print(f"    start {s}: {args.arms[0]} {np.median(aa):5.2f}   "
                  f"{args.arms[1]} {np.median(bb):5.2f}   "
                  f"paired diff {np.median(aa - bb):+5.2f}   (n={len(aa)})")

    if args.latex:
        print("\n% ---- generated by scripts/cube_grid_stats.py ----")
        print(r"\begin{tabular}{lrrrr}")
        print(r"\toprule")
        print(r"height & $n$ & " + LABEL.get(args.arms[0], args.arms[0]) + " & "
              + LABEL.get(args.arms[1], args.arms[1]) + r" & paired diff \\")
        print(r"\midrule")
        for layer in layers:
            a, b, _ = paired(res, layer, have, args.arms)
            if not len(a):
                continue
            lo, hi = boot_median_diff(a - b)
            print(f"{heights[layer] * 100:.0f}\\,cm & {len(a)} & {np.median(a):.2f} & "
                  f"{np.median(b):.2f} & ${np.median(a - b):+.2f}$ "
                  f"$[{lo:+.2f}, {hi:+.2f}]$ \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
