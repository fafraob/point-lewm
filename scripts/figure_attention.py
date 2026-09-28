#!/usr/bin/env python3
"""Compose the attention maps into one publication figure.

    pixi run python scripts/figure_attention.py
    pixi run python scripts/figure_attention.py --frames 2 --cmap magma_r

With the default `--layout wide`, rows are the two models and columns are
environments (with `--frames` samples each), as in Figure 6 of the paper, so a
reader compares Point-LeWM against Point-Delta-JEPA down a column and across
environments along a row. `--layout tall` transposes.

NORMALISATION IS PER PANEL, and the colourbar is labelled as such, because the
arms differ by two orders of magnitude in how peaked their attention is (the
OGB-Cube Point-LeWM checkpoint reaches max/mean 170x where Point-Delta-JEPA sits
near 7x). A shared absolute scale would render the flatter panels uniformly
blank and invite the reader to compare numbers that are not comparable;
per-panel scaling shows each map's own structure, and the concentration
statistics belong in the text or in `--stats`.

Points are drawn low-attention first so the high ones land on top -- otherwise a
dense floor paints over exactly the structure the figure is about -- and points
no token covers are dropped rather than coloured, since they carry no value.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_ply import read_ply, scene_frame  # noqa: E402  (same directory)


def find_files(root, env, arm, frames, suffix, subdir, ext):
    """Frames for one (env, arm), preferring the as-trained directory.

    `subdir` is searched first (the polyscope renders live in e.g. sensor/), then
    the directory itself. `*_centers.*` is excluded -- those are the token-centre
    views, a different picture from the cloud.
    """
    for name in (f"{env}_{arm}", f"{env}_{arm}_r074"):
        for place in ([root / name / subdir] if subdir else []) + [root / name]:
            found = sorted(f for f in place.glob(f"*_{suffix}.{ext}")
                           if not f.stem.endswith("_centers"))
            if found:
                return found[:frames], f"{place.relative_to(root)}"
    return [], None


def square_pad(image, fill=1.0):
    """Pad a crop to a square so every panel is the same shape and scale.

    Stretching to a square would distort the scene differently per panel;
    padding keeps the geometry honest and still gives a uniform grid.
    """
    h, w = image.shape[:2]
    side = max(h, w)
    shape = (side, side) + image.shape[2:]
    out = np.full(shape, fill, dtype=image.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top:top + h, left:left + w] = image
    return out


def crop_white(image, tolerance=0.995):
    """Trim the uniform border a renderer leaves around the cloud.

    Every panel is framed by its own content rather than by the renderer's
    canvas, so the grid reads evenly instead of showing wildly different amounts
    of empty page per environment.
    """
    grey = image[..., :3].mean(axis=2) if image.ndim == 3 else image
    mask = grey < tolerance
    if not mask.any():
        return image
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    pad = 1
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, image.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, image.shape[1])
    return image[r0:r1, c0:c1]


def project(xyz):
    """Orthographic view down the scene's own plane (PCA), so every env is flat-on."""
    center, _, axis_u, axis_v, _ = scene_frame(xyz)
    rel = xyz - center
    return rel @ axis_u, rel @ axis_v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("attention"))
    ap.add_argument("--envs", nargs="+", default=["tworoom", "reacher", "pusht", "cube"])
    ap.add_argument("--arms", nargs="+", default=["point_lewm", "point_delta_jepa"])
    ap.add_argument("--labels", nargs="+", default=["Point-LeWM", "Point-Delta-JEPA"],
                    help="display names for --arms")
    ap.add_argument("--frames", type=int, default=2, help="panels per arm per env")
    ap.add_argument("--suffix", default="rollout", help="rollout | occlusion | grad")
    ap.add_argument("--source", choices=("png", "ply"), default="png",
                    help="png: compose the renders as they were shot -- their viewpoint, shading "
                         "and occlusion are the point, and re-projecting throws that away. "
                         "ply: draw the points in matplotlib instead (vector output, top-down).")
    ap.add_argument("--subdir", default="sensor",
                    help="subdirectory of renders to prefer, e.g. sensor or log ('' for none)")
    ap.add_argument("--cmap", default="coolwarm",
                    help="sequential is the correct family for a positive quantity; "
                         "for --source png this only labels the colourbar, so it must MATCH the "
                         "ramp the renders were made with (coolwarm by default in render_ply.py)")
    ap.add_argument("--low", type=float, default=60.0, help="percentile mapped to the low colour")
    ap.add_argument("--high", type=float, default=99.5, help="percentile mapped to the high colour")
    ap.add_argument("--point-size", type=float, default=0.5)
    ap.add_argument("--stats", action="store_true", help="print max/mean in each panel corner")
    ap.add_argument("--group-gap", type=float, default=0.22,
                    help="gap between environments, as a fraction of a panel width. The gap "
                         "BETWEEN THE TWO FRAMES of one environment stays hairline, so the pair "
                         "reads as one unit rather than as four unrelated panels.")
    ap.add_argument("--max-px", type=int, default=320,
                    help="downsample each panel to this longest side before embedding. The "
                         "renders are 1000px squares and the printed panels are ~2 inches, so "
                         "the full resolution only inflates the PDF.")
    ap.add_argument("--width", type=float, default=7.0,
                    help="FINAL printed width in inches (7.0 = full width of a two-column page, "
                         "3.4 = one column). The figure is built at this size so the point sizes "
                         "below are the point sizes on paper -- drawing a 18-inch canvas and "
                         "letting LaTeX shrink it is what makes labels unreadable.")
    ap.add_argument("--font", type=float, default=8.0, help="base font size in points")
    ap.add_argument("--no-compress", action="store_true",
                    help="skip the ghostscript pass. By default the PDF is rewritten with its "
                         "images downsampled to 300 dpi of the PRINTED size -- the renders carry "
                         "far more pixels than a 2-inch panel can show, and flate cannot compress "
                         "a dot pattern well, so the raw file is many megabytes of invisible detail.")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="drop the colourbar entirely (per-panel normalisation means it carries "
                         "no absolute information anyway -- the caption can say low/high)")
    ap.add_argument("--layout", choices=("wide", "tall"), default="wide",
                    help="wide: environments across the columns, arms down the rows -- fits a "
                         "two-column paper without eating a page. tall: the transpose.")
    ap.add_argument("--out", type=Path, default=Path("figures/attention_grid"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.compression"] = 9   # flate level for embedded rasters
    # Embed text as TrueType, not matplotlib's default Type 3. Type 3 renders
    # fine but several venues (IEEE PDF eXpress among them) reject it outright.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    plt.rcParams.update({
        # Times, with metric-compatible fallbacks: the real font is often absent
        # on Linux, and Nimbus Roman / Liberation Serif have identical metrics,
        # so a figure built here still matches a Times-set paper.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",     # Times-like math to match
        "font.size": args.font,
        "axes.linewidth": 0.4,
        "savefig.bbox": "tight",
    })

    per_arm = args.frames
    if args.layout == "wide":
        # arms down the rows, (environment x frame) across: 2 x 8 for four envs
        # and two frames, which lies flat across a page instead of filling one.
        n_rows, n_cols = len(args.arms), len(args.envs) * per_arm
    else:
        n_rows, n_cols = len(args.envs), len(args.arms) * per_arm
    if args.layout == "wide":
        # Panel columns with a narrow SPACER column between environments: one
        # uniform wspace would either separate the two frames of an environment
        # as much as the environments themselves, or crowd everything equally.
        widths, spacer_at = [], set()
        for e in range(len(args.envs)):
            if e:
                spacer_at.add(len(widths))
                widths.append(args.group_gap)
            widths.extend([1.0] * per_arm)
        # Size the FIGURE so each axes box comes out square. Otherwise the boxes
        # are wider than tall, imshow centres a square image inside them, and the
        # leftover white reads as spacing that no wspace setting can remove.
        left, right = 0.055, 0.995
        top, bottom = 0.90, (0.17 if not args.no_colorbar else 0.04)
        fig_w = args.width
        panel_w = fig_w * (right - left) / sum(widths)
        fig_h = panel_w * n_rows / (top - bottom)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(n_rows, len(widths), width_ratios=widths,
                              wspace=0.02, hspace=0.02,
                              left=left, right=right, top=top, bottom=bottom)
        panel_cols = [i for i in range(len(widths)) if i not in spacer_at]
        axes = [[fig.add_subplot(gs[r, c]) for c in panel_cols] for r in range(n_rows)]
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.05 * n_rows),
                                 squeeze=False)

    pretty_env = {"cube": "Cube", "tworoom": "Two-room", "reacher": "Reacher", "pusht": "Push-T"}

    def cell(env_i, arm_i, frame_i):
        """Where a panel goes, for either orientation."""
        if args.layout == "wide":
            return axes[arm_i][env_i * per_arm + frame_i]
        return axes[env_i][arm_i * per_arm + frame_i]

    for e, env in enumerate(args.envs):
        for a, arm in enumerate(args.arms):
            panels, used = find_files(args.root, env, arm, args.frames, args.suffix,
                                      args.subdir, "png" if args.source == "png" else "ply")
            print(f"{env:8s} {arm:10s} <- {used or 'MISSING'} ({len(panels)} frames)")
            for f in range(args.frames):
                ax = cell(e, a, f)
                ax.set_axis_off()   # no ticks, no frame -- the cloud IS the panel
                if f >= len(panels):
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes, color="0.6")
                    continue

                if args.source == "png":
                    image = square_pad(crop_white(plt.imread(panels[f])))
                    if args.max_px and image.shape[0] > args.max_px:
                        step = int(np.ceil(image.shape[0] / args.max_px))
                        image = image[::step, ::step]
                    # RGB uint8: the alpha channel is a quarter of the bytes and
                    # carries nothing on an opaque render, and float RGBA would
                    # be embedded at 4 bytes per channel.
                    image = (np.clip(image[..., :3], 0, 1) * 255).astype(np.uint8)
                    ax.imshow(image, interpolation="antialiased", aspect="equal")
                    if args.stats:
                        ax.text(0.03, 0.05, panels[f].stem.split("_")[0],
                                transform=ax.transAxes, fontsize=6, color="0.35")
                    continue

                ax.set_aspect("equal")
                xyz, _, value = read_ply(panels[f])
                if value is None:
                    ax.text(0.5, 0.5, "no scalar", ha="center", va="center",
                            transform=ax.transAxes, color="0.6")
                    continue
                keep = value > 0
                xyz, value = xyz[keep], value[keep]
                u, v = project(xyz)
                order = np.argsort(value)
                lo, hi = np.percentile(value, [args.low, args.high])
                ax.scatter(u[order], v[order], c=value[order], s=args.point_size,
                           cmap=args.cmap, norm=colors.Normalize(lo, max(hi, lo + 1e-12)),
                           linewidths=0, rasterized=True)
                if args.stats:
                    ax.text(0.03, 0.03, f"{value.max() / value.mean():.0f}$\\times$",
                            transform=ax.transAxes, fontsize=6, color="0.35")

    if args.layout == "wide":
        pass                                   # the gridspec already set the margins
    else:
        fig.subplots_adjust(top=0.945, bottom=0.10, left=0.055, wspace=0.005, hspace=0.005)

    def span_label(first_ax, last_ax, text, where):
        """Label a group of panels, centred on the span it covers."""
        a, b = first_ax.get_position(), last_ax.get_position()
        if where == "top":
            fig.text((a.x0 + b.x1) / 2, 0.985, text, ha="center", va="top",
                     fontsize=args.font + 1)
        else:
            fig.text(0.008, (b.y0 + a.y1) / 2, text, ha="left", va="center",
                     fontsize=args.font + 1, rotation=90)

    for e, env in enumerate(args.envs):
        name = pretty_env.get(env, env)
        if args.layout == "wide":       # one heading over that env's frame columns
            span_label(cell(e, 0, 0), cell(e, 0, args.frames - 1), name, "top")
        else:
            span_label(cell(e, 0, 0), cell(e, len(args.arms) - 1, args.frames - 1), name, "left")
    for a, arm in enumerate(args.arms):
        label = args.labels[a] if a < len(args.labels) else arm
        if args.layout == "wide":
            span_label(cell(0, a, 0), cell(0, a, args.frames - 1), label, "left")
        else:
            span_label(axes[0][a * per_arm], axes[0][a * per_arm + per_arm - 1], label, "top")
    #if not args.no_colorbar:
        #bar = fig.add_axes([0.40, 0.075, 0.22, 0.028] if args.layout == "wide"
                          # else [0.36, 0.045, 0.30, 0.016])
        #fig.colorbar(cm.ScalarMappable(cmap=args.cmap), cax=bar, orientation="horizontal",
        #             ticks=[0, 1])
        #bar.set_xticklabels(["low", "high"])
        #bar.set_xlabel("encoder attention (per-panel normalised)", fontsize=8, labelpad=3)
        #bar.tick_params(labelsize=8, length=2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)

    pdf = Path(f"{args.out}.pdf")
    if not args.no_compress:
        from pdf_compress import compress_pdf

        compress_pdf(pdf)

    print(f"\nwrote {pdf} and {args.out}.png")


if __name__ == "__main__":
    main()
