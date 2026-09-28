#!/usr/bin/env python3
"""Paired statistical tests behind the paper's Appendix E (Table 12).

All tests operate on the ten per-seed success rates of two methods evaluated on
the SAME seeds (the seed fixes the 50 start-goal windows), so they are paired by
seed. Reproduces: paired difference of means with its 95% CI, the two-sided
paired t-test, and the TOST equivalence test at a +-margin (default 10 points).

    python scripts/paired_stats.py results/table2_per_seed.csv
    python scripts/paired_stats.py results/table2_per_seed.csv --margin 5
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict

from scipy import stats  # scipy ships with the pinned environment


def load(path):
    rows = defaultdict(dict)  # (table, env, method) -> {seed: rate}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["table"], r["environment"], r["method"])][int(r["seed"])] = float(r["success_rate"])
    return rows


def paired(a, b):
    """a, b: dicts seed -> rate. Returns stats of (a - b) over the shared seeds."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    t, p = stats.ttest_rel([a[s] for s in seeds], [b[s] for s in seeds])
    return dict(n=n, mean=mean, ci=ci, t=float(t), p=float(p), se=se)


def tost(a, b, margin):
    """Two one-sided tests: H0 |mean diff| >= margin. Returns the TOST p-value."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    mean = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n)
    t_low = (mean + margin) / se   # H0: diff <= -margin, reject for large t
    t_high = (mean - margin) / se  # H0: diff >= +margin, reject for small t
    p_low = 1 - stats.t.cdf(t_low, n - 1)
    p_high = stats.t.cdf(t_high, n - 1)
    return max(p_low, p_high)


def fmt_p(p):
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--table", default="Table 2")
    ap.add_argument("--margin", type=float, default=10.0, help="TOST equivalence margin in points")
    args = ap.parse_args()
    rows = load(args.csv)
    envs = ["Two-Room", "Reacher", "Push-T", "OGB-Cube"]

    print(f"Modality comparison, Point-LeWM - LeWM (paired by seed), TOST margin +-{args.margin:g}")
    print(f"{'env':10s} {'delta':>7s} {'95% CI':>18s} {'paired t p':>11s} {'TOST p':>9s}")
    for env in envs:
        a = rows.get((args.table, env, "Point-LeWM")); b = rows.get((args.table, env, "LeWM"))
        if not a or not b:
            print(f"{env:10s} (missing)"); continue
        r = paired(a, b); pt = tost(a, b, args.margin)
        print(f"{env:10s} {r['mean']:+7.1f} [{r['ci'][0]:+5.1f}, {r['ci'][1]:+5.1f}]  {fmt_p(r['p']):>10s} {fmt_p(pt):>9s}   (n={r['n']})")

    print()
    print("Objective-family gaps, Point-Delta-JEPA - Point-LeWM (paired by seed)")
    for env in envs:
        a = rows.get((args.table, env, "Point-Delta-JEPA")); b = rows.get((args.table, env, "Point-LeWM"))
        if not a or not b:
            print(f"{env:10s} (missing)"); continue
        r = paired(a, b)
        print(f"{env:10s} {r['mean']:+7.1f} [{r['ci'][0]:+5.1f}, {r['ci'][1]:+5.1f}]  p = {r['p']:.2e}   (n={r['n']})")


if __name__ == "__main__":
    main()
