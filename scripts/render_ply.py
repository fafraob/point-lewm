#!/usr/bin/env python3
"""Render point-cloud PLYs to images with polyscope, on a white background.

Made for the attention clouds from scripts/viz_attention.py, whose vertex
colours already encode the weights -- this only has to frame and shoot them:

    pixi run python scripts/render_ply.py attention/*.ply
    pixi run python scripts/render_ply.py attention/*.ply --views 3

CAMERA. The clouds are in the SENSOR frame: the sensor sits at the origin
looking down +x, so a camera placed AT the sensor pose sees every point
face-on -- no parallax, no occlusion, and a flat picture where a raised arm and
the floor behind it overlap into one blob. So the default drifts off that axis:
`--azimuth 25 --elevation 20` degrees away from the sensor's own bearing, far
enough to give the scene depth while keeping the view the sensor actually
covered (drift too far and you are looking at the cloud's empty back side, where
the sensor never saw anything). `--views N` renders N azimuths spread around
that default, which is the cheap way to not be fooled by one unlucky angle.

A file that stores a per-point `attention` scalar (what viz_attention.py
writes) is coloured here with --cmap; a file that already carries RGB is left
as it is. `--radius` sets point size relative to the
scene, `--size` the image resolution, `--backend openGL3_egl` forces headless
rendering when there is no display.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def read_ply(path):
    """xyz + rgb from an ASCII or binary-little-endian PLY (no dependency)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            sys.exit(f"{path}: not a PLY file")
        fmt, count, props = None, 0, []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                count = int(line.split()[2])
            elif line.startswith("element"):  # any later element: stop collecting props
                break
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
        names = [n for _, n in props]

        if fmt == "ascii":
            rows = np.array([f.readline().split() for _ in range(count)], dtype=np.float64)
        else:
            np_type = {"float": "f4", "float32": "f4", "double": "f8", "uchar": "u1",
                       "uint8": "u1", "int": "i4", "short": "i2", "ushort": "u2"}
            endian = "<" if "little" in fmt else ">"
            dtype = np.dtype([(n, endian + np_type[t]) for t, n in props])
            raw = np.frombuffer(f.read(dtype.itemsize * count), dtype=dtype, count=count)
            rows = np.stack([raw[n].astype(np.float64) for n in names], axis=1)

    xyz = rows[:, [names.index("x"), names.index("y"), names.index("z")]]
    rgb = None
    if {"red", "green", "blue"} <= set(names):
        rgb = rows[:, [names.index("red"), names.index("green"), names.index("blue")]] / 255.0
    scalar = None
    for candidate in ("attention", "quality", "intensity", "scalar", "value"):
        if candidate in names:
            scalar = rows[:, names.index(candidate)]
            break
    return xyz, rgb, scalar


def ramp(values, cmap_name, clip_percentile, log_scale=False, cmap_range=(0.0, 1.0)):
    """Scalar -> RGB, stretched so the interesting end is visible.

    Two choices matter on a white page. FIRST, direction: a ramp whose high end
    is yellow (viridis, plasma) hides exactly what you want to see -- pale
    yellow on white is nothing. The default here runs light grey -> deep red, so
    low attention recedes towards the background and high attention is the
    darkest, most saturated thing in the frame.

    SECOND, the stretch: attention is concentrated, so normalising by the max
    lets a handful of points own the top of the ramp and flattens everything
    else to the bottom. Clipping at a percentile (default 99) spends the ramp on
    the range the bulk of the points actually occupy.
    """
    v = np.asarray(values, dtype=np.float64)
    # Exactly-zero points are NOT low attention: FPS+radius grouping leaves part
    # of the cloud in no group at all, so those points were never scored. Fitting
    # the ramp over them drags the whole covered range into the middle of the
    # colormap -- which is why a nearly uniform field renders as a uniform wash.
    # Fit over the scored points and paint the unscored ones neutral grey.
    scored = v > 0
    if not scored.any():
        scored = np.ones_like(v, dtype=bool)
    values_scored = v[scored]
    if log_scale:
        # Attention spans two orders of magnitude on some checkpoints (max/mean
        # up to 170x on the cube arm). On a linear ramp that puts every ordinary
        # token at the very bottom and leaves a handful of red specks -- the map
        # says "one token matters" when what you wanted was the shape of the
        # rest. Log spreads the bulk back across the colours.
        floor = values_scored[values_scored > 0].min() if (values_scored > 0).any() else 1.0
        v = np.where(scored, np.log10(np.maximum(v, floor)), v)
        values_scored = v[scored]
    hi = np.percentile(values_scored, clip_percentile) if clip_percentile < 100 else values_scored.max()
    lo = values_scored.min()
    norm = np.clip((v - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(v)
    # Sequential colormaps start at white-ish, which on a white page erases the
    # bulk of a floor-dominated scan. --cmap-range samples a SLICE of the ramp
    # instead, so the low end can be a mid tone that still reads as an object.
    lo_frac, hi_frac = cmap_range
    norm = lo_frac + norm * (hi_frac - lo_frac)
    try:
        import matplotlib

        colors = matplotlib.colormaps[cmap_name](norm)[:, :3]
    except Exception:  # no matplotlib: light grey -> deep red
        colors = np.stack([0.85 - 0.25 * norm, 0.85 - 0.8 * norm, 0.85 - 0.8 * norm], axis=1)
    colors[~scored] = (0.82, 0.85, 0.88)
    return colors


def scene_frame(xyz):
    """Centre, radius and an orthonormal frame aligned to the cloud's own plane.

    These scans are dominated by a floor, so the useful reference is that plane,
    not the world axes: PCA's smallest-variance direction is its normal, the
    other two span it. Angles measured against this frame behave the same way on
    a two-room arena, a pusht table and a reacher workspace, none of which share
    an orientation in sensor coordinates.
    """
    center = xyz.mean(axis=0)
    centered = xyz - center
    _, _, vt = np.linalg.svd(centered[:: max(1, len(centered) // 20000)], full_matrices=False)
    normal = vt[2]
    if np.dot(normal, -center) < 0:  # look from the side the sensor was on
        normal = -normal
    return center, float(np.linalg.norm(centered, axis=1).max()), vt[0], vt[1], normal


def fit_distance(xyz, center, direction, fov_deg, aspect, margin):
    """Closest distance along `direction` at which every point is still in frame.

    Fitting to the bounding SPHERE (the obvious shortcut) parks the camera far
    too far away for these scans: a floor-dominated cloud seen at an angle
    projects to a thin ellipse, nothing like the sphere that encloses it, and
    the render ends up as a small lozenge in a sea of white. So project the
    points into the camera's own basis and solve for the distance where the
    widest one just fits: a point sitting `a` sideways and `c` towards the
    camera needs D >= c + |a| / tan(fov/2).
    """
    forward = direction / np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:          # looking straight down
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    rel = xyz - center
    a, b, c = rel @ right, rel @ up, rel @ forward
    fov_v = np.radians(fov_deg)
    fov_h = 2 * np.arctan(np.tan(fov_v / 2) * aspect)
    need = np.maximum(c + np.abs(a) / np.tan(fov_h / 2),
                      c + np.abs(b) / np.tan(fov_v / 2))
    return float(need.max()) * margin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="image directory (default: beside the PLY)")
    ap.add_argument("--size", default="1600x1200", help="WxH (default 1600x1200)")
    ap.add_argument("--azimuth", type=float, default=35.0, help="rotation within the scene plane (deg)")
    ap.add_argument("--elevation", type=float, default=55.0,
                help="degrees above the scene plane; 90 = straight down, "
                     "which is the most readable view of a 2D-derived arena")
    ap.add_argument("--distance", type=float, default=1.0,
                help="multiplier on the auto-fitted distance (1.0 fills the frame)")
    ap.add_argument("--views", type=int, default=1, help="render N azimuths around --azimuth")
    ap.add_argument("--radius", type=float, default=0.0035,
                    help="point radius; scene-relative by default (see --radius-mode)")
    ap.add_argument("--radius-mode", choices=("relative", "absolute"), default="relative",
                    help="relative: a fraction of THIS cloud's extent, so dots look the same size "
                         "in every image regardless of scene scale. absolute: world units (metres), "
                         "so a dot means the same physical size across scenes -- which is what you "
                         "want when comparing environments whose point spacing differs (reacher "
                         "scans ~0.04 m apart, pusht ~0.18 m), and which will look sparse on the "
                         "wide scenes and crowded on the tight ones.")
    ap.add_argument("--cmap", default="Reds",
                    help="matplotlib colormap for the scalar (default Reds: light -> deep red, "
                         "readable on white; try magma_r, inferno_r, cividis_r)")
    ap.add_argument("--cmap-range", type=float, nargs=2, default=(0.0, 1.0), metavar=("LO", "HI"),
                    help="use only this slice of the colormap (0-1). Sequential maps fade to "
                         "near-white at 0, which vanishes on a white page; --cmap-range 0.3 1 "
                         "keeps the low end a visible mid tone")
    ap.add_argument("--clip-percentile", type=float, default=99.0,
                    help="stretch the ramp to this percentile (100 = use the max)")
    ap.add_argument("--from-sensor", action="store_true",
                    help="start from the SENSOR's own viewpoint (the origin, since these clouds "
                         "are in sensor frame) and nudge off it by --azimuth/--elevation. Keeps "
                         "the framing the scan was taken with, while the small offset opens up "
                         "the occlusion shadows that are invisible dead-on.")
    ap.add_argument("--log", action="store_true",
                    help="log-scale the values before the ramp (for concentrated attention)")
    ap.add_argument("--hide-missing", action="store_true",
                    help="drop points no token covers instead of painting them grey -- with "
                         "coverage near 50%% the covered/uncovered checker otherwise dominates "
                         "the picture and reads as structure that is not attention")
    ap.add_argument("--camera-json", type=Path, default=None,
                    help="JSON with {\"eye\": [x,y,z], \"lookat\": [x,y,z]} in the cloud's own "
                         "coordinates. Used to render a cloud from exactly the camera another "
                         "tool rendered the scene from, so the two images can be paired.")
    ap.add_argument("--backend", default="auto",
                    choices=("auto", "openGL3_glfw", "openGL3_egl", "openGL_mock"))
    args = ap.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    if args.distance <= 0:
        # 0 puts the camera at the scene centre, which renders nothing useful.
        # It is a MULTIPLIER on the fitted distance: <1 pushes in, >1 pulls back.
        print(f"[render] --distance {args.distance} is not usable; using 1.0 "
              f"(it multiplies the auto-fitted distance: 0.8 = closer, 1.4 = further back)")
        args.distance = 1.0
    import polyscope as ps

    try:
        ps.init(args.backend)
    except Exception as exc:  # no display -> try headless EGL before giving up
        if args.backend != "auto":
            raise
        print(f"[render] {type(exc).__name__}: {exc}; retrying with openGL3_egl")
        ps.init("openGL3_egl")

    ps.set_window_size(width, height)
    ps.set_background_color((1.0, 1.0, 1.0))       # white, as asked
    ps.set_ground_plane_mode("none")               # no shadow plane under the cloud
    ps.set_SSAA_factor(3)                          # cheap antialiasing; points are 1px otherwise
    ps.set_up_dir("z_up")
    ps.set_transparency_mode("pretty")

    for path in args.ply:
        xyz, rgb, scalar = read_ply(path)
        if args.hide_missing and scalar is not None:
            keep = scalar > 0
            xyz, scalar = xyz[keep], scalar[keep]
            rgb = rgb[keep] if rgb is not None else None
        out_dir = args.out or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        cloud = ps.register_point_cloud(path.stem, xyz)
        cloud.set_radius(args.radius, relative=(args.radius_mode == "relative"))
        if scalar is not None:                     # value in the file, ramp chosen here
            cloud.add_color_quantity(
                "attention",
                ramp(scalar, args.cmap, args.clip_percentile, args.log, args.cmap_range),
                enabled=True)
        elif rgb is not None:                      # pre-coloured file: leave it alone
            cloud.add_color_quantity("color", rgb, enabled=True)

        center, radius, axis_u, axis_v, normal = scene_frame(xyz)

        if args.camera_json is not None:
            import json

            view = json.loads(args.camera_json.read_text())
            ps.look_at(view["eye"], view["lookat"])
            name = path.stem + ".png"
            ps.screenshot(str(out_dir / name), transparent_bg=False)
            print(f"  {out_dir / name}  ({len(xyz)} pts, camera from {args.camera_json.name})")
            ps.remove_point_cloud(path.stem)
            continue

        offsets = [0.0] if args.views == 1 else np.linspace(-60.0, 60.0, args.views)
        for i, delta in enumerate(offsets):
            az, el = np.radians(args.azimuth + delta), np.radians(args.elevation)
            if args.from_sensor:
                # The sensor sits at the origin. Swing ITS position about the
                # scene centre -- azimuth about the scene normal, elevation
                # about the horizontal -- so the view stays the one the scan was
                # taken from, only nudged. Dead-on, every point faces the camera
                # and the occlusion shadows read as nothing; a few degrees of
                # parallax is what makes them holes.
                offset = -center
                sideways = np.cross(normal, offset)
                if np.linalg.norm(sideways) < 1e-9:
                    sideways = axis_u
                sideways /= np.linalg.norm(sideways)
                rotated = np.cos(az) * offset + np.sin(az) * np.linalg.norm(offset) * sideways
                direction = rotated / np.linalg.norm(rotated) + np.tan(el) * normal
                direction /= np.linalg.norm(direction)
                # Keep the sensor's BEARING but not its distance: the scan camera
                # sits inside the scene, so standing where it stands crops most of
                # the cloud out of frame. Fit the distance as in the other mode.
                eye = center + direction * args.distance * fit_distance(
                    xyz, center, direction, 45.0, width / height, 1.06)
            else:
                direction = (np.cos(el) * (np.cos(az) * axis_u + np.sin(az) * axis_v)
                             + np.sin(el) * normal)
                direction /= np.linalg.norm(direction)
                distance = args.distance * fit_distance(
                    xyz, center, direction, 45.0, width / height, 1.06)
                eye = center + direction * distance
            ps.look_at(eye.tolist(), center.tolist())
            name = path.stem + (f"_view{i}" if args.views > 1 else "") + ".png"
            ps.screenshot(str(out_dir / name), transparent_bg=False)
            print(f"  {out_dir / name}  ({len(xyz)} pts, radius {radius:.2f}, "
                  f"az {args.azimuth + delta:.0f} deg, el {args.elevation:.0f} deg "
                  f"off the scene plane)")
        ps.remove_point_cloud(path.stem)

    print(f"\nrendered {len(args.ply)} cloud(s)")


if __name__ == "__main__":
    main()
