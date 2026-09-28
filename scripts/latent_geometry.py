#!/usr/bin/env python3
"""Latent-geometry figures comparing world-model arms on one environment.

    python3 scripts/latent_geometry.py \\
        --dump LeWM=tmp/latents/point_lewm_reacher.npz \\
        --dump Delta-JEPA=tmp/latents/point_delta_jepa_reacher.npz \\
        --out figures/reacher_latent

Reads the ``.npz`` files written by ``scripts/dump_latents.py`` -- no GPU, no
model, no dataset -- and writes three figures plus ``metrics.json`` and
``summary.md``.

WHAT EACH PANEL CLAIMS, AND WHAT WOULD FALSIFY IT

**A -- spectrum.** Normalized eigenvalues of the latent covariance, log axis,
one curve per arm, with effective rank beside them. This is the collapse
diagnostic the SIGReg claim rests on: a latent that has quietly folded onto a
few directions shows a spectrum falling off a cliff, and its effective rank
drops far below the embedding width. Two scale-free summaries are reported
because they disagree in informative ways -- the participation ratio
``(Σλ)²/Σλ²`` (dominated by the top eigenvalues) and RankMe,
``exp(H(σ/Σσ))`` on the singular values (Garrido et al. 2023, sensitive to the
tail).

**B -- linear probe + PCA scatter.** Ridge regression from the frozen latent to
privileged state, fit and scored on DISJOINT EPISODES, and the top-2 PCs
colored by the same quantities. Frame-level splits are the classic way to get a
meaningless R² here: adjacent frames are near-duplicates, so a frame-split
probe scores high on a latent that has memorized nothing but frame identity.
Two controls pin the scale: ``qvel`` is not recoverable from a single frame
(a velocity is not in one cloud), and every factor is re-fit against shuffled
targets. A probe that scores well on either control is measuring leakage, not
structure, and the whole panel is void.

**D -- rollout drift.** Predicted latents ``horizon`` predictor applications
deep, against the true latents of the same frames, in units of that arm's own
latent variance -- so a collapsed latent cannot win by having a small error.
The no-change baseline (predict the last observed latent) is the bar every arm
must clear; the encoder's own sampling noise is the floor it cannot go below.
The Mahalanobis panel asks a different question: not "is the prediction right"
but "is it even a latent the encoder could have produced". A rollout that
leaves the encoder's distribution is being scored by CEM in a region the cost
was never calibrated on, which is a mechanism for planning failure rather than
a correlate of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------- styling
# Validated categorical pair (light surface, all-pairs: CVD dE 24.7, normal 33.6).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"
# Sequential ramp: one hue, monotone light -> dark (never a rainbow).
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fa", "#7fb0e4", "#2a78d6", "#123f77"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "medium",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def style(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    return ax


# ------------------------------------------------------------------- data load


class Dump:
    """One arm's dump, plus the episode-level split every metric is read on."""

    def __init__(self, name, path, fit_frac=0.7):
        self.name = name
        self.path = Path(path)
        f = np.load(self.path, allow_pickle=False)
        self.meta = json.loads(str(f["meta"]))
        self.z = f["z"]
        self.ep = f["ep"]
        self.step = f["step"]
        self.ep_ptr = f["ep_ptr"]
        self.roll_pred = f["roll_pred"]
        self.roll_true = f["roll_true"]
        self.roll_last = f["roll_last"]
        self.roll_ep = f["roll_ep"]
        self.roll_t0 = f["roll_t0"]
        self.state = {k[len("state_"):]: f[k] for k in f.files if k.startswith("state_")}
        self.noise_mse = float(f["noise_mse_per_dim"]) if "noise_mse_per_dim" in f.files else None

        # Episode-level split. Sorting first makes it deterministic given the
        # dump, and identical across arms whose dumps used the same --seed.
        eps = np.unique(self.ep)
        n_fit = int(round(fit_frac * len(eps)))
        self.fit_eps, self.test_eps = eps[:n_fit], eps[n_fit:]
        self.fit = np.isin(self.ep, self.fit_eps)
        self.test = ~self.fit
        self.roll_test = np.isin(self.roll_ep, self.test_eps)

    @property
    def dim(self):
        return self.z.shape[1]

    def episode_slice(self, ep_id):
        """(lo, hi) into ``z`` for one episode's frames, in time order."""
        starts = self.ep[self.ep_ptr[:-1]]
        j = int(np.flatnonzero(starts == ep_id)[0])
        return int(self.ep_ptr[j]), int(self.ep_ptr[j + 1])


def factors(dump):
    """Probe targets: privileged state, plus the two controls.

    ``qpos`` enters as (cos, sin) per joint -- reacher's first joint is an
    unbounded hinge, so regressing the raw angle would score a wrap-around as a
    2π error and understate every arm equally but arbitrarily.
    """
    s = dump.state
    out = {}
    if "finger_pos" in s:
        out["fingertip xy"] = s["finger_pos"]
    if "target_pos" in s:
        out["target xy"] = s["target_pos"]
    if "finger_pos" in s and "target_pos" in s:
        out["reach error"] = s["finger_pos"] - s["target_pos"]
    if "qpos" in s:
        out["joint angles (cos,sin)"] = np.concatenate([np.cos(s["qpos"]), np.sin(s["qpos"])], axis=1)
    if "qvel" in s:
        out["joint velocity (control)"] = s["qvel"]
    return out


# ----------------------------------------------------------------- panel A: PCA


def spectrum(dump):
    """Normalized covariance eigenspectrum and two scale-free rank summaries."""
    z = dump.z[dump.fit]
    zc = z - z.mean(axis=0, keepdims=True)
    # Singular values of the centered matrix; lambda = s^2/(n-1).
    s = np.linalg.svd(zc, compute_uv=False)
    lam = s**2 / max(len(zc) - 1, 1)
    p_lam = lam / lam.sum()
    p_s = s / s.sum()
    return {
        "eigenvalues_normalized": p_lam,
        "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
        "rankme": float(np.exp(-(p_s * np.log(p_s + 1e-12)).sum())),
        "dims_for_95pct": int(np.searchsorted(np.cumsum(p_lam), 0.95) + 1),
        "dim": int(dump.dim),
        "mean_variance": float(lam.mean()),
    }


def pca_basis(dump, k=2):
    z = dump.z[dump.fit]
    mu = z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z - mu, full_matrices=False)
    var = np.var((z - mu) @ vt.T, axis=0)
    total = np.var(z - mu, axis=0).sum()
    return mu, vt[:k], var[:k] / total


# --------------------------------------------------------------- panel B: probe


def _ridge(X, Y, lam):
    """Closed-form ridge on centered/scaled X; returns weights and intercept."""
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    W = np.linalg.solve(A, X.T @ Y)
    return W


def _r2(Y, P):
    """Uniform mean of per-dimension R^2, referenced to the TEST-set mean."""
    ss_res = ((Y - P) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - ss_res / np.maximum(ss_tot, 1e-12)))


def fit_probe(dump, Y, lams=np.logspace(-2, 5, 15)):
    """Fit the ridge probe on the fit episodes; return a decoder for any latent.

    X is standardized with fit-set statistics so a single lambda grid means the
    same thing for every arm (the arms' latents differ in scale by construction
    -- one is regularized toward N(0, I), the other is not). Lambda is chosen on
    an inner EPISODE split that never touches the test episodes.
    """
    Xf, Yf = dump.z[dump.fit], Y[dump.fit]
    mu, sd = Xf.mean(axis=0, keepdims=True), Xf.std(axis=0, keepdims=True) + 1e-8
    Xs = (Xf - mu) / sd
    ymu = Yf.mean(axis=0, keepdims=True)

    inner_eps = dump.fit_eps[: int(0.8 * len(dump.fit_eps))]
    inner = np.isin(dump.ep[dump.fit], inner_eps)
    best, best_r2 = lams[0], -np.inf
    for lam in lams:
        W = _ridge(Xs[inner], Yf[inner] - ymu, lam)
        r2 = _r2(Yf[~inner], Xs[~inner] @ W + ymu)
        if r2 > best_r2:
            best, best_r2 = lam, r2

    W = _ridge(Xs, Yf - ymu, best)

    def decode(Z):
        return ((Z - mu) / sd) @ W + ymu

    return decode, float(best), (mu, sd, ymu, Xs, Yf)


def probe(dump, Y, rng):
    """Held-out-episode R^2 of the linear probe, plus its shuffled-target control."""
    decode, lam, (mu, sd, ymu, Xs, Yf) = fit_probe(dump, Y)
    Yt = Y[dump.test]
    r2 = _r2(Yt, decode(dump.z[dump.test]))

    # Control: the same fit against targets shuffled across frames. Anything
    # this scores is what the probe's capacity buys on its own.
    perm = rng.permutation(len(Yf))
    Ws = _ridge(Xs, Yf[perm] - ymu, lam)
    r2_shuf = _r2(Yt, ((dump.z[dump.test] - mu) / sd) @ Ws + ymu)
    return {"r2": r2, "r2_shuffled": r2_shuf, "lambda": lam, "n_test": int(len(Yt))}


def rollout_physics(dump, Y):
    """Decode the ROLLOUT's latents with the probe fitted on true latents.

    Latent MSE is in arbitrary units; this reads the same rollout in the units
    of the thing being predicted. The ceiling is the probe applied to the TRUE
    latents of the same frames -- a rollout cannot be decoded better than the
    representation it is imitating, so the gap between the two curves is what
    the predictor lost, and the ceiling's distance from 1.0 is what the encoder
    never had.
    """
    decode, _, _ = fit_probe(dump, Y)
    m = np.flatnonzero(dump.roll_test)
    if not len(m):
        return None
    K = dump.roll_pred.shape[1]

    idx = np.empty((len(m), K), dtype=np.int64)
    for a, i in enumerate(m):
        lo, _ = dump.episode_slice(int(dump.roll_ep[i]))
        idx[a] = lo + int(dump.roll_t0[i]) + 1 + np.arange(K)

    Ytrue = Y[idx]
    Yhat = decode(dump.roll_pred[m].reshape(-1, dump.dim)).reshape(len(m), K, -1)
    Yceil = decode(dump.z[idx.reshape(-1)]).reshape(len(m), K, -1)
    return {
        "r2": np.array([_r2(Ytrue[:, k], Yhat[:, k]) for k in range(K)]),
        "r2_ceiling": np.array([_r2(Ytrue[:, k], Yceil[:, k]) for k in range(K)]),
        "rmse": np.array([float(np.sqrt(((Ytrue[:, k] - Yhat[:, k]) ** 2).mean())) for k in range(K)]),
        "rmse_ceiling": np.array([float(np.sqrt(((Ytrue[:, k] - Yceil[:, k]) ** 2).mean())) for k in range(K)]),
    }


# ------------------------------------------------------------- panel D: rollout


def rollout_metrics(dump, shrink=1e-3):
    """Per-horizon prediction error and distribution distance, on test episodes."""
    m = dump.roll_test
    pred, true, last = dump.roll_pred[m], dump.roll_true[m], dump.roll_last[m]
    if not len(pred):
        return None
    K, D = pred.shape[1], pred.shape[2]

    # Scale-free unit: this arm's own mean per-dimension latent variance, so an
    # error of 1.0 is "no better than predicting the dataset mean".
    zf = dump.z[dump.fit]
    mu = zf.mean(axis=0, keepdims=True)
    var = float(np.var(zf, axis=0).mean())

    mse = ((pred - true) ** 2).mean(axis=(0, 2)) / var
    mse_nochange = ((last[:, None, :] - true) ** 2).mean(axis=(0, 2)) / var
    mse_mean = ((mu[None] - true) ** 2).mean(axis=(0, 2)) / var

    # Mahalanobis, on the fit-episode latent distribution. Shrinkage is not
    # cosmetic: a partially collapsed latent has near-null directions, and
    # without it the distance is dominated by their inverses.
    cov = np.cov(zf - mu, rowvar=False)
    cov = cov + shrink * np.trace(cov) / D * np.eye(D)
    inv = np.linalg.inv(cov)

    def maha(x):
        # d / sqrt(D): under the fitted Gaussian a typical sample sits at ~1.0
        # regardless of the embedding width, so the arms are on one axis.
        d = x.reshape(-1, D) - mu
        q = np.einsum("ij,jk,ik->i", d, inv, d)
        return np.sqrt(np.maximum(q, 0.0) / D).reshape(x.shape[:-1])

    return {
        "horizon": np.arange(1, K + 1),
        "mse": mse,
        "mse_nochange": mse_nochange,
        "mse_mean": mse_mean,
        "noise_floor": (dump.noise_mse / var) if dump.noise_mse is not None else None,
        "maha_pred": maha(pred).mean(axis=0),
        "maha_true": maha(true).mean(axis=0),
        "n_rollouts": int(len(pred)),
        "latent_var": var,
    }


# ----------------------------------------------------------------------- figures


def fig_spectrum(dumps, spec, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    for i, d in enumerate(dumps):
        s = spec[d.name]
        p = s["eigenvalues_normalized"]
        x = np.arange(1, len(p) + 1)
        axes[0].plot(x, p, color=SERIES[i], label=d.name, zorder=3)
        axes[1].plot(x, np.cumsum(p), color=SERIES[i], label=d.name, zorder=3)
        axes[1].annotate(
            d.name, (x[-1], np.cumsum(p)[-1]), xytext=(-2, -8 - 10 * i),
            textcoords="offset points", ha="right", color=SERIES[i], fontsize=7.5,
        )

    style(axes[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("share of latent variance")
    axes[0].set_title("Eigenspectrum", loc="left", color=INK)
    axes[0].legend(loc="upper right")

    style(axes[1])
    axes[1].axhline(0.95, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=2)
    axes[1].annotate("95%", (1, 0.95), xytext=(2, 3), textcoords="offset points",
                     color=MUTED, fontsize=7)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("principal component")
    axes[1].set_ylabel("cumulative variance")
    axes[1].set_title("Cumulative", loc="left", color=INK)

    # The ranks belong beside the curves they summarize, not in a caption.
    for i, d in enumerate(dumps):
        s = spec[d.name]
        axes[0].annotate(
            f"{d.name}   PR {s['participation_ratio']:.1f} · RankMe {s['rankme']:.1f} · "
            f"{s['dims_for_95pct']}/{s['dim']} dims for 95%",
            (0.02, 0.14 - 0.08 * i), xycoords="axes fraction",
            fontsize=6.8, color=SERIES[i],
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5),
        )
    _save(fig, out)


def fig_pca(dumps, color_specs, out):
    """PCA scatter grid: rows = arms, columns = the quantity used for color."""
    n_r, n_c = len(dumps), len(color_specs)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.35 * n_c, 2.35 * n_r),
                             constrained_layout=True, squeeze=False)
    for r, d in enumerate(dumps):
        mu, basis, frac = pca_basis(d)
        P = (d.z[d.test] - mu) @ basis.T
        for c, (label, values, cmap, unit) in enumerate(color_specs):
            ax = axes[r][c]
            v = values(d)[d.test]
            sc = ax.scatter(P[:, 0], P[:, 1], c=v, cmap=cmap, s=3.5, linewidths=0,
                            alpha=0.75, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(GRID)
            if r == 0:
                ax.set_title(label, loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(d.name, color=INK, fontsize=8.5)
            ax.annotate(f"PC1 {frac[0]:.0%} · PC2 {frac[1]:.0%}", (0.03, 0.03),
                        xycoords="axes fraction", fontsize=6.5, color=INK_2,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
            if r == n_r - 1:
                cb = fig.colorbar(sc, ax=axes[:, c].tolist(), location="bottom",
                                  fraction=0.06, pad=0.02, aspect=28)
                cb.set_label(unit, fontsize=7, color=INK_2)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=6.5, length=2)
    _save(fig, out)


def fig_probe(dumps, results, out):
    names = list(next(iter(results.values())).keys())
    n = len(names)
    fig, ax = plt.subplots(figsize=(6.2, 0.26 * n * len(dumps) + 1.3), constrained_layout=True)
    h = 0.34
    for i, d in enumerate(dumps):
        y = np.arange(n) + (i - (len(dumps) - 1) / 2) * (h + 0.04)
        vals = [results[d.name][k]["r2"] for k in names]
        ax.barh(y, np.clip(vals, -0.25, 1.0), height=h, color=SERIES[i], label=d.name, zorder=3)
        for yy, v in zip(y, vals):
            ax.annotate(f"{v:.2f}", (max(v, 0), yy), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK_2)
        # The shuffled fit's R^2 is unbounded below and its magnitude carries no
        # information -- only "is it at zero" does, so it is clamped into view.
        ctrl = np.clip([results[d.name][k]["r2_shuffled"] for k in names], -0.2, 1.0)
        ax.scatter(ctrl, y, s=14, facecolor=SURFACE, edgecolor=MUTED, linewidths=1,
                   zorder=4, label="shuffled-target control" if i == 0 else None)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xlabel("held-out-episode $R^2$ of a linear probe from the frozen latent\n(bars clipped at $-0.25$; the label gives the true value)")
    ax.set_xlim(-0.25, 1.0)
    style(ax, grid_axis="x")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    _save(fig, out)


def fig_rollout(dumps, roll, phys, phys_name, out):
    from matplotlib.lines import Line2D

    fig, axgrid = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes = axgrid.ravel()

    ax = style(axes[0])
    lo = np.inf
    for i, d in enumerate(dumps):
        r = roll[d.name]
        ax.plot(r["horizon"], r["mse"], color=SERIES[i], marker="o", ms=3.5, zorder=4)
        ax.annotate(d.name, (r["horizon"][-1], r["mse"][-1]), xytext=(-3, 7),
                    textcoords="offset points", ha="right", color=SERIES[i], fontsize=7.5)
        ax.plot(r["horizon"], r["mse_nochange"], color=SERIES[i], lw=1.2,
                ls=(0, (4, 2)), alpha=0.75, zorder=3)
        lo = min(lo, r["mse"].min(), r["noise_floor"] or np.inf)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=2)
    ax.annotate("predict the mean", (1, 1.0), xytext=(2, -9), textcoords="offset points",
                fontsize=6.5, color=MUTED)
    floors = [roll[d.name]["noise_floor"] for d in dumps if roll[d.name]["noise_floor"]]
    if floors:
        ax.axhspan(lo / 3, max(floors), color=GRID, zorder=1)
        ax.annotate("encoder sampling noise", (1, max(floors)), xytext=(2, 2),
                    textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(lo / 3, 2.5)
    ax.set_xlabel("rollout steps (× frameskip env steps)")
    ax.set_ylabel("latent MSE / latent variance")
    ax.set_title("Prediction error", loc="left", color=INK)
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=1.8, label="rollout"),
            Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 2)), label="no-change baseline"),
        ],
        loc="lower right",
    )

    ax = style(axes[1])
    for i, d in enumerate(dumps):
        r = roll[d.name]
        ax.plot(r["horizon"], r["maha_pred"], color=SERIES[i], marker="o", ms=3.5, zorder=4)
        ax.plot(r["horizon"], r["maha_true"], color=SERIES[i], lw=1.2, ls=(0, (4, 2)),
                alpha=0.75, zorder=3)
        ax.annotate(d.name, (r["horizon"][-1], r["maha_pred"][-1]), xytext=(-3, 7),
                    textcoords="offset points", ha="right", color=SERIES[i], fontsize=7.5)
    ax.set_xlabel("rollout steps")
    ax.set_ylabel(r"Mahalanobis $d/\sqrt{D}$")
    # Anchored at 0 and past 1.0 on purpose: autoscaling zooms into the third
    # decimal and turns "no drift at all" into a dramatic-looking wiggle.
    top = max(1.15, 1.1 * max(np.max(roll[d.name]["maha_pred"]) for d in dumps))
    ax.set_ylim(0, top)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=2)
    ax.annotate("a typical encoded frame", (1, 1.0), xytext=(2, 3),
                textcoords="offset points", fontsize=6.5, color=MUTED)
    ax.set_title("Drift off the encoder's distribution", loc="left", color=INK)
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=1.8, label="predicted latents"),
            Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 2)), label="true latents"),
        ],
        loc="lower left",
    )

    ax = style(axes[2])
    for i, d in enumerate(dumps):
        if phys.get(d.name) is None:
            continue
        pk = phys[d.name]
        # RMSE in the factor's own units, not R^2: an R^2 axis spanning
        # 0.9986-0.9998 magnifies a difference of a couple of millimetres into
        # the whole panel. The R^2 values are in summary.md.
        ax.plot(roll[d.name]["horizon"], pk["rmse"], color=SERIES[i], marker="o", ms=3.5, zorder=4)
        ax.plot(roll[d.name]["horizon"], pk["rmse_ceiling"], color=SERIES[i], lw=1.2,
                ls=(0, (4, 2)), alpha=0.75, zorder=3)
        ax.annotate(d.name, (roll[d.name]["horizon"][-1], pk["rmse"][-1]), xytext=(-3, 7),
                    textcoords="offset points", ha="right", color=SERIES[i], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xlabel("rollout steps")
    ax.set_ylabel(f"RMSE of decoded {phys_name}\n(dataset units)")
    ax.set_title(f"Rollout read back as {phys_name}", loc="left", color=INK)
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=1.8, label="decoded from the rollout"),
            Line2D([], [], color=MUTED, lw=1.2, ls=(0, (4, 2)), label="ceiling: decoded from true latents"),
        ],
        loc="lower right",
    )

    ax = axes[3]
    d = dumps[0]
    mu, basis, frac = pca_basis(d)
    m = np.flatnonzero(d.roll_test)
    picks = m[np.linspace(0, len(m) - 1, min(3, len(m))).round().astype(int)]
    for j, idx in enumerate(picks):
        ep_id, t0 = int(d.roll_ep[idx]), int(d.roll_t0[idx])
        lo, hi = d.episode_slice(ep_id)
        traj = (d.z[lo:hi] - mu) @ basis.T
        ax.plot(traj[:, 0], traj[:, 1], color=GRID, lw=1.2, zorder=2)
        k = len(d.roll_true[idx])
        true = traj[t0 : t0 + k + 1]
        pred = np.concatenate([traj[t0 : t0 + 1], (d.roll_pred[idx] - mu) @ basis.T])
        ax.plot(true[:, 0], true[:, 1], color=SERIES[0], zorder=4)
        ax.plot(pred[:, 0], pred[:, 1], color=SERIES[1], ls=(0, (3, 2)), zorder=5)
        ax.scatter(true[:1, 0], true[:1, 1], s=20, color=INK, zorder=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GRID)
    ax.set_title(f"{d.name} in the PCA plane", loc="left", color=INK)
    ax.annotate("grey: whole episode\nblue: true future · orange: rollout",
                (0.03, 0.97), xycoords="axes fraction", va="top", fontsize=6.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    _save(fig, out)


def _save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out.with_suffix('.pdf')}  +  .png")


# --------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", action="append", required=True, metavar="NAME=PATH",
                   help="repeat once per arm; NAME labels it in the figures")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--fit-frac", type=float, default=0.7, help="episode fraction used to fit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dumps = []
    for spec in args.dump:
        # rpartition, not partition: a display name may itself contain "="
        # ("LeWM (h=3)"), while the path is the tail.
        name, _, path = spec.rpartition("=")
        assert path, f"--dump expects NAME=PATH, got {spec!r}"
        dumps.append(Dump(name, path, args.fit_frac))

    ref = dumps[0]
    for d in dumps[1:]:
        # The comparison is only clean if every arm was measured on the same
        # frames; the dumps carry the episode ids, so check rather than assume.
        assert np.array_equal(d.ep, ref.ep) and np.array_equal(d.step, ref.step), (
            f"{d.name} was dumped on different frames than {ref.name} -- "
            "re-run scripts/dump_latents.py with the same --seed/--episodes"
        )

    out = Path(args.out)
    print(f"[analysis] {len(dumps)} arms, {len(ref.z)} frames, "
          f"{len(ref.fit_eps)} fit / {len(ref.test_eps)} test episodes, D={ref.dim}")

    spec = {d.name: spectrum(d) for d in dumps}
    fig_spectrum(dumps, spec, out / "panelA_spectrum")

    results = {d.name: {k: probe(d, v, rng) for k, v in factors(d).items()} for d in dumps}
    fig_probe(dumps, results, out / "panelB_probe")

    color_specs = []
    if "finger_pos" in ref.state:
        color_specs.append(("colored by fingertip x", lambda d: d.state["finger_pos"][:, 0], SEQ, "fingertip x"))
        color_specs.append(("colored by fingertip y", lambda d: d.state["finger_pos"][:, 1], SEQ, "fingertip y"))
    if "qpos" in ref.state:
        # A hinge angle is circular: a cyclic map is the only honest ramp for it.
        color_specs.append(("colored by joint 0 angle",
                            lambda d: np.mod(d.state["qpos"][:, 0], 2 * np.pi), "twilight", "joint 0 angle (rad)"))
    if color_specs:
        fig_pca(dumps, color_specs, out / "panelB_pca")

    roll = {d.name: rollout_metrics(d) for d in dumps}
    # Read the rollouts back in physical units, using the best-probed factor --
    # a factor the encoder does not represent (reacher's target, R^2 ~ 0) would
    # only measure the probe's failure, not the predictor's.
    ref_probe = results[ref.name]
    phys_name = max(
        (k for k in ref_probe if "control" not in k),
        key=lambda k: ref_probe[k]["r2"],
    )
    phys = {d.name: rollout_physics(d, factors(d)[phys_name]) for d in dumps}
    if all(roll.values()):
        fig_rollout(dumps, roll, phys, phys_name, out / "panelD_rollout")

    # ------------------------------------------------------------------ numbers
    metrics = {
        "frames": int(len(ref.z)),
        "fit_episodes": int(len(ref.fit_eps)),
        "test_episodes": int(len(ref.test_eps)),
        "arms": {},
    }
    for d in dumps:
        s = dict(spec[d.name])
        s.pop("eigenvalues_normalized")
        r = roll[d.name]
        metrics["arms"][d.name] = {
            "meta": d.meta,
            "spectrum": s,
            "probe": results[d.name],
            "rollout": None if r is None else {
                "n_rollouts": r["n_rollouts"],
                "latent_var": r["latent_var"],
                "noise_floor": r["noise_floor"],
                "mse_by_horizon": r["mse"].tolist(),
                "mse_nochange_by_horizon": r["mse_nochange"].tolist(),
                "maha_pred_by_horizon": r["maha_pred"].tolist(),
                "maha_true_by_horizon": r["maha_true"].tolist(),
            },
            "decoded_physics": None if phys.get(d.name) is None else {
                "factor": phys_name,
                **{k: v.tolist() for k, v in phys[d.name].items()},
            },
        }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        "# Latent geometry",
        "",
        f"{len(ref.z)} frames from {len(ref.fit_eps) + len(ref.test_eps)} episodes "
        f"({len(ref.fit_eps)} fit / {len(ref.test_eps)} held out), D = {ref.dim}.",
        "",
        "| arm | checkpoint | rollout context | predictor.num_frames |",
        "|---|---|---|---|",
    ] + [
        f"| {d.name} | `{d.meta['run']}/{d.meta['policy']}` | {d.meta['history']} | "
        f"{d.meta['predictor_num_frames']} |"
        for d in dumps
    ] + [
        "",
        "## Effective rank (panel A)",
        "",
        "| arm | participation ratio | RankMe | dims for 95% var |",
        "|---|---|---|---|",
    ]
    for d in dumps:
        s = spec[d.name]
        lines.append(f"| {d.name} | {s['participation_ratio']:.1f} | {s['rankme']:.1f} | "
                     f"{s['dims_for_95pct']} / {s['dim']} |")

    # The scatter is illustrative, the probe is the claim -- say so wherever an
    # arm's top-2 plane holds too little variance to represent its latent.
    thin = [d.name for d in dumps if pca_basis(d)[2].sum() < 0.25]
    if thin:
        lines += ["", "> **Reading panelB_pca.png.** The top-2 plane carries under 25% of the "
                  f"latent variance for {', '.join(thin)}, so its scatter is a shadow of a "
                  "much higher-dimensional cloud and its apparent shape is largely a "
                  "projection artifact. The probe below uses all dimensions and is the "
                  "quantitative statement; the scatter only shows how the variance is "
                  "*organized*."]

    names = list(results[dumps[0].name].keys())
    lines += ["", "## Linear probe, held-out episodes (panel B)", "",
              "R² of a ridge probe from the frozen latent; the value in brackets is the "
              "same probe fit against shuffled targets.", "",
              "| factor | " + " | ".join(d.name for d in dumps) + " |",
              "|---" * (len(dumps) + 1) + "|"]
    for k in names:
        cells = [f"{results[d.name][k]['r2']:.3f} [{results[d.name][k]['r2_shuffled']:+.3f}]" for d in dumps]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    if all(roll.values()):
        lines += ["", "## Rollout (panel D)", "",
                  "Latent MSE in units of that arm's own latent variance; 1.0 = no better "
                  "than predicting the dataset mean.", "",
                  "| arm | 1 step | 5 steps | last step | no-change, last step | noise floor |",
                  "|---|---|---|---|---|---|"]
        for d in dumps:
            r = roll[d.name]
            k5 = min(4, len(r["mse"]) - 1)
            nf = f"{r['noise_floor']:.2e}" if r["noise_floor"] else "—"
            lines.append(
                f"| {d.name} | {r['mse'][0]:.3f} | {r['mse'][k5]:.3f} | {r['mse'][-1]:.3f} | "
                f"{r['mse_nochange'][-1]:.3f} | {nf} |"
            )
        if any(phys.values()):
            lines += ["", f"Rollout decoded as **{phys_name}** (R², held-out episodes; "
                      "the ceiling is the same probe on the true latents):", "",
                      "| arm | 1 step | last step | ceiling, last step |",
                      "|---|---|---|---|"]
            for d in dumps:
                pk = phys.get(d.name)
                if pk is None:
                    continue
                lines.append(f"| {d.name} | {pk['r2'][0]:.3f} | {pk['r2'][-1]:.3f} | "
                             f"{pk['r2_ceiling'][-1]:.3f} |")

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"[analysis] wrote {out / 'metrics.json'} and {out / 'summary.md'}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
