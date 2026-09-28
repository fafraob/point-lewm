#!/usr/bin/env python3
"""Matched pairs: the rendered scene and its LiDAR scan, per environment.

    pixi run python scripts/scene_pairs.py
    pixi run python scripts/scene_pairs.py --envs tworoom reacher --row 500000

WHY NOT THE STORED PIXELS. Every table carries a `pixels` column, but for three
of the four environments it is not the scan's scene: the mirrors store a
top-down 2D render while the LiDAR is scanned obliquely from the `iso` camera,
and reacher's pixels come from the source project's own renderer (which is why
its collection config sets check_render: false). Pairing those would show two
different scenes side by side.

So the render is made HERE, from the dataset row the scan came from:

  1. restore the row into the env  (the eval's own set_state call)
  2. put the row's cloud into WORLD coordinates -- exactly, see world_from_sensor
  3. render the scene with mujoco, and the cloud with polyscope, along ONE
     view direction, each framed on its own content

Step 3 is the compromise that makes the figure readable. Sharing the exact
camera (what an earlier revision did) sounds better and is worse: the sensor
sits inside the scene, so a camera at its pose crops the robot out of the
render, and the panels then disagree about what the environment even contains.
Sharing only the BEARING keeps the two panels in the same orientation -- the
wall runs the same way, the arm points the same way -- while each panel gets a
distance that fits what it has to show.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ helpers
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import DATA_ROOT  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen rendering, before mujoco loads

# AddRaycastLidarWrapper._CAM_FRAME_FIX. The sensor frame is x=forward, y=left,
# z=up; MuJoCo cameras look down -z with x=right, y=up, and the wrapper composes
# `rot = cam_xmat @ CAM_FRAME_FIX` to map one to the other. Copied rather than
# imported so this script keeps working against a checkout where the wrapper
# moved -- but it is a COPY OF A CONSTANT, not a guess: getting it wrong lands
# the cloud rotated half a turn about the vertical, which looks entirely
# plausible next to a render until you notice the wall is on the wrong side.
CAM_FRAME_FIX = np.array([[0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0]])


def write_ply(path, xyz, scalar):
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float attention\n")
        f.write("end_header\n")
        for point, value in zip(xyz, scalar):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {value:.6f}\n")


def _ensure_importable():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import stable_worldmodel  # noqa: F401
    except ModuleNotFoundError:
        # the stable-worldmodel checkout shipped in the repo root
        vendored = Path(__file__).resolve().parents[1] / "stable-worldmodel"
        if (vendored / "stable_worldmodel" / "__init__.py").is_file():
            sys.path.insert(0, str(vendored))


# env id | table | camera the sensor rides | how a row is restored | env kwargs
# `drift` swings the shared view direction off the sensor's own bearing
# (azimuth, elevation in degrees). `frame` and `pad` decide what has to fit in
# the panel, and they are per-env because the four scenes have wildly different
# ratios of interesting thing to empty floor: reacher's arm is 0.26 m across on
# a 2 m plane, two-room's arena IS the whole 4 m scan. Framing everything the
# same way gives either a cropped robot or a stamp-sized one.
#   frame "solid" -> the box around the geoms (the robot, the blocks, the walls)
#   frame "scene" -> that box unioned with the cloud's, for arenas the scan fills
#   pad           -> expand it, so the subject is not jammed against the edge
SCENES = {
    "tworoom": dict(
        env="swm/TwoRoomLidar-v1", table="two_room.lance", camera="iso",
        restore=[("_set_state", {"state": "pos_agent"})],
        kwargs=dict(goal_lidar_height=None, iso_azimuth=45.0, iso_elevation=45.0),
        # pad < 1 crops slightly INTO the arena corners. A square arena seen
        # corner-on projects to a diamond, so fitting its corners leaves four
        # triangles of dead white; letting them run off the edge fills the panel.
        drift=(12.0, 4.0), frame="scene", pad=0.63,
    ),
    "reacher": dict(
        env="swm/Reacher3D-v0", table="reacher.lance", camera="iso",
        restore=[("set_state", {"qpos": "qpos", "qvel": "qvel"})],
        kwargs=dict(iso_azimuth=45.0, iso_elevation=45.0),
        drift=(12.0, 4.0), frame="solid", pad=2.4,
        # This env inherits dm_control's dark-green checker floor, which makes
        # its panel the odd one out in a row of grey arenas. Retint it to the
        # two-room floor so the eye compares scenes, not palettes.
        floor_checker=((0.90, 0.90, 0.90), (0.80, 0.80, 0.82)),
    ),
    "pusht": dict(
        env="swm/PushTLidar-v1", table="pusht.lance", camera="iso",
        restore=[("_set_state", {"state": "state"})],
        kwargs=dict(goal_lidar_height=None, iso_azimuth=45.0, iso_elevation=45.0),
        # Only the T and the pusher are geoms here -- the arena is the ground
        # plane -- so the solid box is small and the padding does the framing.
        drift=(12.0, 4.0), frame="solid", pad=2.0,
    ),
    "cube": dict(
        env="swm/OGBCube-v0", table="cube.lance",
        camera="front_pixels",
        # qpos/qvel restore the blocks too: in OGBench's cube scene every block
        # is a free joint, so it is all in qpos.
        restore=[("set_state", {"qpos": "qpos", "qvel": "qvel"})],
        kwargs={}, drift=(18.0, 8.0), frame="solid", pad=0.92,
    ),

}


def read_row(table_path, row, columns):
    import lancedb

    p = Path(table_path).expanduser()
    ds = lancedb.connect(str(p.parent)).open_table(p.stem).to_lance()
    row = row % ds.count_rows()
    have = {f.name for f in ds.schema}
    columns = [c for c in columns if c in have]
    batch = ds.take([row], columns=columns)
    out = {}
    for name in columns:
        out[name] = np.asarray(batch.column(name).combine_chunks().flatten(), dtype=np.float64)
    return out, row


def mujoco_handles(env):
    """(model, data) for either flavour, after the scene exists."""
    base = env.unwrapped
    if getattr(base, "model", None) is not None:
        return base.model, base.data
    physics = getattr(base, "physics", None) or getattr(getattr(base, "env", None), "physics", None)
    if physics is not None:
        return physics.model._model, physics.data._data
    raise SystemExit(f"{type(base).__name__} exposes no MuJoCo model to render")


def world_from_sensor(cloud, cam_pos, cam_mat):
    """Sensor-frame hits -> world coordinates, by the wrapper's own transform."""
    rot = cam_mat @ CAM_FRAME_FIX
    return cloud @ rot.T + cam_pos


def retint_checker(model, mujoco, name, bright, dark):
    """Recolour a builtin checker texture in place.

    Setting `mat_rgba` -- the obvious move, and what an earlier revision did --
    changes nothing visible here: these floors are TEXTURED materials, and the
    texture's own texels are what the renderer paints. So rewrite the texels,
    keeping the checker pattern by mapping its two luminance levels onto the
    two replacement colours.
    """
    tex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    if tex < 0:
        return False
    channels = int(model.tex_nchannel[tex])
    start = int(model.tex_adr[tex])
    count = int(model.tex_width[tex]) * int(model.tex_height[tex]) * channels
    texels = model.tex_data[start:start + count].reshape(-1, channels)
    luminance = texels[:, :3].mean(axis=1)
    lighter = luminance > luminance.mean()
    texels[lighter, :3] = np.round(np.array(bright) * 255)
    texels[~lighter, :3] = np.round(np.array(dark) * 255)
    return True


def frame_box(model, data, cloud, mode, pad):
    """The box both panels are framed on: what the figure has to show.

    Framing on the cloud alone is what cropped the cube's robot out of its own
    panel -- the scan covers a 90-degree cone from a camera standing next to the
    arm, so its centroid sits well off the scene's. Framing on the cloud is also
    what shrank reacher's arm to a speck: the scan reaches 1.4 m across a
    workspace 0.26 m wide, so nearly the whole panel was empty floor.

    So take the box around the GEOMS, and union in the cloud only where the scan
    is the subject (`mode="scene"`, the two-room arena). `geom_rbound` is 0 for
    the infinite ground planes, which is exactly the filter wanted -- a plane
    has no extent to bound and would contribute nothing but noise.
    """
    lows, highs = [], []
    # geom_rbound is the bounding SPHERE, and for a 4 m wall 5 cm thick that is
    # a 2 m radius -- framing on it pushed the two-room camera back until the
    # arena was a stamp. geom_aabb is the tight box (centre + half-extents, in
    # the geom's own frame), so rotate its corners into the world instead. The
    # rbound test survives only as the "is this an infinite plane" filter.
    solid = np.flatnonzero(np.asarray(model.geom_rbound) > 0)
    if solid.size:
        aabb = np.asarray(model.geom_aabb).reshape(-1, 6)[solid]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        local = aabb[:, None, :3] + signs[None] * aabb[:, None, 3:]
        rot = np.asarray(data.geom_xmat)[solid].reshape(-1, 3, 3)
        corners = np.einsum("gij,gcj->gci", rot, local) + np.asarray(data.geom_xpos)[solid][:, None]
        corners = corners.reshape(-1, 3)
        lows.append(corners.min(axis=0))
        highs.append(corners.max(axis=0))
    if mode == "scene" or not lows:
        lows.append(cloud.min(axis=0))
        highs.append(cloud.max(axis=0))
    low, high = np.min(lows, axis=0), np.max(highs, axis=0)
    center = (low + high) / 2
    return center, center + (low - center) * pad, center + (high - center) * pad


def fit_distance(points, center, direction, fovy_deg, aspect, margin=1.05):
    """Nearest distance along `direction` at which every point is still in frame.

    Fitting the bounding SPHERE parks the camera much too far back for these
    scenes -- a wide flat arena seen obliquely projects to a thin ellipse,
    nothing like the sphere around it, and the panel becomes a lozenge in a sea
    of white. Project into the camera's own basis instead and solve for the
    distance where the widest point just fits: one sitting `a` sideways and `c`
    towards the camera needs D >= c + |a| / tan(fov/2).
    """
    forward = direction / np.linalg.norm(direction)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    rel = points - center
    a, b, c = rel @ right, rel @ up, rel @ forward
    fov_v = np.radians(fovy_deg)
    fov_h = 2 * np.arctan(np.tan(fov_v / 2) * aspect)
    need = np.maximum(c + np.abs(a) / np.tan(fov_h / 2),
                      c + np.abs(b) / np.tan(fov_v / 2))
    return float(need.max()) * margin


def box_corners(low, high):
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1]) for z in (low[2], high[2])])


def view_direction(cam_pos, center, azimuth_deg, elevation_deg):
    """Unit vector from the scene towards the eye: the sensor's bearing, drifted.

    Rendering from the sensor's exact bearing shows no occlusion at all -- the
    scan has one return per ray, so nothing can hide behind anything. A dozen
    degrees of parallax is what turns those shadows into holes. Both panels take
    THIS direction, which is what keeps them in the same orientation.
    """
    direction = cam_pos - center
    direction = direction / (np.linalg.norm(direction) or 1.0)
    up = np.array([0.0, 0.0, 1.0])
    sideways = np.cross(up, direction)
    norm = np.linalg.norm(sideways)
    sideways = sideways / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    az = np.radians(azimuth_deg)
    swung = np.cos(az) * direction + np.sin(az) * sideways
    swung = swung / np.linalg.norm(swung) + np.tan(np.radians(elevation_deg)) * up
    return swung / np.linalg.norm(swung)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", nargs="+", default=list(SCENES))
    ap.add_argument("--data", type=Path, default=Path(DATA_ROOT),
                    help="folder holding the .lance tables (default: paths.DATA_ROOT)")
    ap.add_argument("--row", type=int, default=None, help="dataset row (default: --seed picks one)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=900, help="render resolution (square)")
    ap.add_argument("--invalid-value", type=float, default=-1.0)
    ap.add_argument("--point-size", type=float, default=3.2,
                    help="point radius in per-mille of the cloud's own extent, so a dot looks the "
                         "same size in every panel however large the scene is")
    ap.add_argument("--cmap", default="Blues",
                    help="a quiet single-hue ramp: the floor comes out a soft blue-grey and "
                         "everything standing on it deepens to solid blue. A two-ended "
                         "cold-to-warm map separates the same things but shouts about it, "
                         "which is not what these panels are for")
    ap.add_argument("--cmap-range", type=float, nargs=2, default=(0.38, 0.95), metavar=("LO", "HI"),
                    help="slice of the colormap to use. Sequential maps start at near-white, "
                         "and polyscope's shading lightens them further, so a floor mapped to 0 "
                         "disappears into the page; starting partway up keeps it visible")
    ap.add_argument("--colour-by", choices=("height", "depth"), default="height",
                    help="height reads far better: walls, arms and blocks stand off the floor, "
                         "while depth varies smoothly across a floor-dominated scan and washes "
                         "the structure out")
    ap.add_argument("--drift", type=float, nargs=2, default=None,
                    metavar=("AZ", "EL"),
                    help="override the per-env swing off the sensor's bearing, in degrees")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="multiplier on the fitted distance for every panel (<1 = closer in)")
    ap.add_argument("--width", type=float, default=7.0, help="printed figure width in inches")
    ap.add_argument("--font", type=float, default=9.0)
    ap.add_argument("--out", type=Path, default=Path("figures/scene_pairs"))
    ap.add_argument("--no-compress", action="store_true",
                    help="skip the ghostscript pass. By default the PDF is rewritten with its "
                         "rasters JPEG-encoded at q=95: the cloud panels are dense dot patterns "
                         "that flate cannot compress at all, so the saved figure is several MB "
                         "of essentially incompressible noise")
    args = ap.parse_args()

    _ensure_importable()
    import gymnasium as gym
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.compression"] = 9
    # Embed text as TrueType, not matplotlib's default Type 3. Type 3 renders
    # fine but several venues (IEEE PDF eXpress among them) reject it outright.
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    import stable_worldmodel  # noqa: F401

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": args.font,
    })

    pretty = {"cube": "Cube", "tworoom": "Two-room", "reacher": "Reacher", "pusht": "Push-T"}
    panels = {}
    rng = np.random.default_rng(args.seed)
    work = args.out.parent / "_clouds"
    work.mkdir(parents=True, exist_ok=True)

    for name in args.envs:
        spec = SCENES[name]
        wanted = ["lidar"] + [col for _, args_map in spec["restore"] for col in args_map.values()]
        row = args.row if args.row is not None else int(rng.integers(0, 1_000_000))
        columns, row = read_row(args.data / spec["table"], row, wanted)
        if "lidar" not in columns:
            print(f"{name}: no lidar column, skipped")
            continue

        env = gym.make(spec["env"], **spec["kwargs"]).unwrapped
        env.reset(seed=0)
        for method, mapping in spec["restore"]:
            fn = getattr(env, method, None)
            if fn is None:
                print(f"{name}: env has no {method}(), state not restored")
                continue
            fn(**{arg: columns[col] for arg, col in mapping.items() if col in columns})

        model, data = mujoco_handles(env)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec["camera"])
        if cam_id < 0:
            print(f"{name}: no camera '{spec['camera']}', skipped")
            continue

        checker = spec.get("floor_checker")
        if checker is not None and not retint_checker(model, mujoco, "grid", *checker):
            print(f"{name}: no 'grid' texture to retint, floor left as it is")

        mujoco.mj_forward(model, data)
        cam_pos = np.array(data.cam_xpos[cam_id])
        cam_mat = np.array(data.cam_xmat[cam_id]).reshape(3, 3)

        cloud = columns["lidar"].reshape(-1, 3)
        cloud = cloud[~(cloud == args.invalid_value).all(axis=1)]
        world = world_from_sensor(cloud, cam_pos, cam_mat)

        center, low, high = frame_box(model, data, world,
                                     spec.get("frame", "solid"), spec.get("pad", 1.1))
        azimuth, elevation = args.drift if args.drift is not None else spec["drift"]
        direction = view_direction(cam_pos, center, azimuth, elevation)

        # ONE camera for both panels. The mujoco free camera and polyscope both
        # use a 45-degree vertical fov, so placing them at the same point on the
        # same bearing means the two images share a projection: the wall runs
        # the same way, the arm is the same size, and the only difference left
        # is the one the figure is about -- what a single-viewpoint scan misses.
        fovy = float(model.vis.global_.fovy)
        distance = args.zoom * fit_distance(box_corners(low, high), center, direction, fovy, 1.0)
        eye = center + direction * distance

        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.size)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = distance
        # MuJoCo's free camera angles describe the VIEW direction, not the eye:
        # measured on a fixed scene, azimuth 0 / elevation -40 puts the eye at
        # (-cos40, 0, +sin40) * distance. So both angles are negated relative to
        # the eye offset -- and getting azimuth wrong is invisible in a single
        # panel: it just renders the scene from the far side, which looks
        # perfectly reasonable until it is laid next to the matching cloud.
        camera.azimuth = float(np.degrees(np.arctan2(-direction[1], -direction[0])))
        camera.elevation = -float(np.degrees(np.arcsin(direction[2])))
        renderer = mujoco.Renderer(model, args.size, args.size)
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
        # Sky -> white, so the render sits on the same page colour as the cloud
        # beneath it. A segmentation pass is the exact way to find it: every
        # pixel a geom drew carries that geom's id, and only the background is
        # -1. (Colour-keying the black instead would eat the shadows.)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera)
        image[renderer.render()[:, :, 0] < 0] = 255
        renderer.disable_segmentation_rendering()
        renderer.close()
        env.close()

        # Cloud: same camera, coloured off the floor.
        if args.colour_by == "height":
            scalar = world[:, 2] - world[:, 2].min()
        else:
            scalar = np.linalg.norm(world - eye, axis=1)
            scalar = scalar.max() - scalar          # near = warm end of the ramp
        # render_ply paints non-positive values neutral grey (they mean "no
        # token scored this point" for the attention clouds it was written for),
        # so lift the floor off zero rather than have it come out grey.
        scalar = scalar + 1e-3
        write_ply(work / f"{name}.ply", world, scalar)
        (work / f"{name}.json").write_text(json.dumps({"eye": eye.tolist(),
                                                       "lookat": center.tolist()}))
        panels[name] = (image, work / f"{name}.png", row)
        print(f"{name:8s} row {row:8d} | bearing from '{spec['camera']}' drifted "
              f"{azimuth:+.0f}/{elevation:+.0f} deg | frame {np.round(high - low, 2)} m "
              f"| {len(cloud)} points")

    # polyscope in a separate process: it and MuJoCo both want a GL context, and
    # sharing one process between them is asking for a driver-level surprise.
    for name in list(panels):
        cmd = [sys.executable, str(Path(__file__).with_name("render_ply.py")),
               str(work / f"{name}.ply"), "--out", str(work), "--size", f"{args.size}x{args.size}",
               "--camera-json", str(work / f"{name}.json"), "--cmap", args.cmap,
               "--cmap-range", str(args.cmap_range[0]), str(args.cmap_range[1]),
               "--radius", str(args.point_size / 1000), "--clip-percentile", "99.5"]
        if subprocess.run(cmd, check=False).returncode != 0:
            print(f"{name}: cloud render failed")
            panels.pop(name)

    if not panels:
        sys.exit("nothing to draw")

    names = [n for n in args.envs if n in panels]
    left, right, top, bottom = 0.04, 0.997, 0.945, 0.005
    panel_w = args.width * (right - left) / len(names)
    fig = plt.figure(figsize=(args.width, panel_w * 2 / (top - bottom)))
    gs = fig.add_gridspec(2, len(names), wspace=0.015, hspace=0.015,
                          left=left, right=right, top=top, bottom=bottom)

    for c, name in enumerate(names):
        image, cloud_png, row = panels[name]
        for r in range(2):
            ax = fig.add_subplot(gs[r, c])
            ax.set_axis_off()
            panel = image if r == 0 else plt.imread(cloud_png)
            ax.imshow(panel, interpolation="antialiased", aspect="equal")
            if c == 0:
                ax.text(-0.03, 0.5, ["Rendering", "LiDAR"][r], transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=args.font)
        box = gs[0, c].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, 0.99, pretty.get(name, name),
                 ha="center", va="top", fontsize=args.font + 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)

    if not args.no_compress:
        from pdf_compress import compress_pdf

        compress_pdf(Path(f"{args.out}.pdf"))

    print(f"\nwrote {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
