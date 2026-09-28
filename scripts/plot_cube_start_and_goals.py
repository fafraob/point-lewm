#!/usr/bin/env python3
"""Rendering helpers for the commanded-goal grid figure, and a single-arm view.

``scripts/plot_cube_goals_pair.py`` imports this module for Figure 5 of the
paper; run directly it draws one arm on its own::

    pixi run python scripts/plot_cube_start_and_goals.py --arm t2l_mlp_z

The picture: a MuJoCo render of the actual start state -- the arm holding the
cube, restored from the dataset frame every commanded goal starts from
(set_state with that row's qpos/qvel), not an illustration -- with the commanded
goals drawn in 3-D on their height layers, coloured by how close the cube got
(closest approach over the budget, centimetres). z = 2 cm is the expert's own
placement height; every layer above it is a goal regime the training data never
contains.

Needs EGL for headless rendering; MUJOCO_GL is set below.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for paths.py
from paths import RESULTS_ROOT, T2L_ROOT, dataset_path  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

INK, MUTED = "#1a1a19", "#6b6a66"
#: green -> red, ordered by lightness as well as hue so it still reads when
#: printed grey or by a red-green colourblind reader
ERROR_RAMP = ["#1a9850", "#66bd63", "#a6d96a", "#fee08b", "#fdae61", "#f46d43", "#d73027"]


def styled_render(base, size, markers, style="grey", **kw):
    """Render the scene so the measurements do not read as scene objects.

    ``solid``  markers as lit spheres (they look like props -- the problem)
    ``ghost``  translucent markers, no shadows: clearly overlaid, not placed
    ``grey``   two passes, with and without markers; the scene is desaturated
               and only the marker pixels keep their colour, the classic
               "annotation over a dimmed scene" look
    ``pins``   small markers on thin stems down to the table, like measurement
               pins planted in the workspace
    """
    if style == "solid":
        return _overview_render(base, size, markers=markers, **kw)
    if style == "ghost":
        ghost = [(p, list(c[:3]) + [0.80], r) for p, c, r in markers]
        return _overview_render(base, size, markers=ghost, shadows=False, **kw)
    if style == "pins":
        small = [(p, c, r * 0.72) for p, c, r in markers]
        stems = [((np.array([p[0], p[1], 0.021]), np.asarray(p)), list(c[:3]) + [0.5], 0.0016)
                 for p, c, _ in markers]
        return _overview_render(base, size, markers=small, stems=stems, shadows=False, **kw)
    if style not in ("grey", "paper"):
        raise SystemExit(f"unknown style {style!r}")
    # two passes: the difference IS the marker mask, so the scene can be
    # restyled freely without touching a single data pixel
    plain, depth = _overview_render(base, size, markers=(), shadows=False, crop=False,
                                    want_depth=True, **kw)
    with_m = _overview_render(base, size, markers=markers, shadows=False, crop=False, **kw)
    mask = np.abs(with_m.astype(int) - plain.astype(int)).max(axis=2) > 8
    lum = plain.astype(float) @ np.array([0.299, 0.587, 0.114])
    if style == "grey":
        scene = np.clip(np.stack([lum] * 3, axis=2) * 0.75 + 46, 0, 255)
    else:
        # paper: grey ghost of the scene, faded to white with distance. The dark
        # region above the table is not empty background -- it is the floor
        # plane running to the horizon -- so it cannot be masked out by colour;
        # the depth buffer fades it instead, which also gives the figure an
        # unobtrusive white top edge.
        scene = np.stack([lum] * 3, axis=2)
        scene = 255 - (255 - scene) * 0.45          # lift the scene toward white
        near, far = np.percentile(depth, 2), np.percentile(depth, 75)
        fog = np.clip((depth - near) / max(far - near, 1e-6), 0, 1)[..., None] ** 1.3
        scene = scene * (1 - fog) + 255 * fog
    return crop_border(np.where(mask[..., None], with_m, scene.astype(np.uint8)))


def render_start(cfg, table, episode, step, size, markers=(), style="grey", **render_kw):
    """RGB render of the dataset frame the episodes start from."""
    import lance
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from stable_worldmodel.wrapper import make_lidar_pre_wrappers

    ds = lance.dataset(str(table))
    rows = ds.to_table(columns=["episode_idx", "step_idx", "qpos", "qvel"])
    ep = rows.column("episode_idx").to_numpy()
    st = rows.column("step_idx").to_numpy()
    i = int(np.flatnonzero((ep == episode) & (st == step))[0])
    qpos = np.asarray(rows.column("qpos").to_numpy(zero_copy_only=False)[i], dtype=float)
    qvel = np.asarray(rows.column("qvel").to_numpy(zero_copy_only=False)[i], dtype=float)

    sensor = OmegaConf.to_container(cfg.lidar.sensor, resolve=True)
    world = swm.World(env_name=cfg.world.env_name, num_envs=1, max_episode_steps=10,
                      image_shape=(size, size), env_type=cfg.world.env_type,
                      ob_type=cfg.world.ob_type, multiview=cfg.world.multiview,
                      width=size, height=size, visualize_info=False,
                      terminate_at_goal=False,
                      pre_wrappers=make_lidar_pre_wrappers(sensor))
    world.reset(seed=0)
    base = world.envs.envs[0].unwrapped
    base.set_state(qpos=qpos, qvel=qvel)
    img = styled_render(base, size, markers, style=style, **render_kw)
    print(f"[render] start frame ep {episode} step {step}, {len(markers)} goal markers "
          f"-> {img.shape}")
    return img


def crop_border(img, thresh=12):
    """Trim the empty border the disabled skybox leaves around the scene."""
    content = img.max(axis=2) > thresh
    rows, cols = np.where(content.any(1))[0], np.where(content.any(0))[0]
    if not len(rows) or not len(cols):
        return img
    pad = 8
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


#: name of the cube geom, recoloured so scene red does not compete with the
#: red end of the error scale
CUBE_GEOM = "object_0"
NEUTRAL_CUBE = (0.62, 0.64, 0.68, 1.0)


def _seg_mat(p0, p1):
    """Rotation whose z axis runs p0 -> p1 (mujoco capsules extend along z)."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.eye(3).reshape(-1), 0.0
    d = d / n
    a = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(a, d); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return np.stack([u, v, d], axis=1).reshape(-1), n / 2


def _flatten_lights(model):
    """Turn the scene's spotlights into directional light.

    The pusht/tworoom mirrors carry one spot at (0, 0, 3) with a 45 deg cutoff,
    which burns a bright cone into the middle of a top-down render and leaves
    the edges dark. Directional light has no position or falloff, so the whole
    board is lit evenly.
    """
    import mujoco

    for i in range(model.nlight):
        model.light_type[i] = int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        model.light_dir[i] = (-0.3, -0.3, -1.0)
        model.light_diffuse[i] = 0.45
        model.light_specular[i] = 0.05


def _overview_render(base, size, markers=(), stems=(), segments=(), boxes=(),
                     light=(0.55, 0.65),
                     flat_light=False,
                     lookat=(0.42, 0.0, 0.16), distance=1.12,
                     azimuth=138.0, elevation=-21.0, fovy=None, skybox=False, shadows=True,
                     neutral_cube=False, crop=True, want_depth=False):
    """Render the scene with the commanded goals added as real 3-D spheres.

    The goals are pushed into the mjvScene as geoms rather than drawn on top of
    the image afterwards, so they are lit, depth-sorted and occluded by the arm
    exactly like scene geometry -- one rendered picture, no overlay.

    The scene's own cameras ('front', 'front_pixels', 'side_pixels') are tight
    crops on the gripper and 'side' points at the sky, so the figure brings its
    own free camera. The skybox (a starfield) is off by default: it is noise in
    a paper figure.
    """
    import mujoco

    # ogbench envs expose _model/_data, the pusht/tworoom mirrors model/data
    model = getattr(base, "_model", None) or base.model
    data = getattr(base, "_data", None) or base.data
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    # the offscreen framebuffer is sized by the model's visual defaults (224 px
    # for the pusht/tworoom mirrors), which caps the Renderer
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), size)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), size)
    renderer = mujoco.Renderer(model, height=size, width=size,
                               max_geom=model.ngeom + len(markers) + len(boxes) + 1000)
    try:
        if fovy is not None:
            model.vis.global_.fovy = fovy
        model.vis.headlight.ambient[:] = light[0]
        model.vis.headlight.diffuse[:] = light[1]
        if flat_light:
            _flatten_lights(model)
        if neutral_cube:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, CUBE_GEOM)
            if gid >= 0:
                model.geom_rgba[gid] = NEUTRAL_CUBE
        renderer.update_scene(data, camera=cam)
        scene = renderer.scene
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = int(skybox)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
        eye = np.eye(3).reshape(-1)
        for p0, p1, rgba, radius in segments:  # thin line in any direction
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mat, half = _seg_mat(p0, p1)
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                size=np.array([radius, half, radius], dtype=np.float64),
                                pos=((np.asarray(p0, float) + p1) / 2).astype(np.float64),
                                mat=mat, rgba=np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1
        for (p0, p1), rgba, radius in stems:  # thin pin down to the table
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mid = (np.asarray(p0) + np.asarray(p1)) / 2
            half = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)) / 2)
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                size=np.array([radius, half, radius], dtype=np.float64),
                                pos=mid.astype(np.float64), mat=eye,
                                rgba=np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1
        for pos, rgba, half in boxes:  # axis-aligned boxes, e.g. start cube poses
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=np.array([half, half, half], dtype=np.float64),
                                pos=np.asarray(pos, dtype=np.float64), mat=eye,
                                rgba=np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1
        for pos, rgba, radius in markers:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                size=np.array([radius, radius, radius], dtype=np.float64),
                                pos=np.asarray(pos, dtype=np.float64), mat=eye,
                                rgba=np.asarray(rgba, dtype=np.float32))
            scene.ngeom += 1
        img = np.asarray(renderer.render())
        if not want_depth:
            return crop_border(img) if crop else img
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = np.asarray(renderer.render())
        renderer.disable_depth_rendering()
        return (crop_border(img) if crop else img), depth
    finally:
        renderer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=None,
                    help="default: $PLWM_RESULTS_ROOT/3dtarget_cube_grid")
    ap.add_argument("--table", type=Path, default=Path(dataset_path("cube")))
    ap.add_argument("--arm", default="t2l_mlp_z")
    ap.add_argument("--goals", type=Path, default=None,
                    help="default: $PLWM_T2L_ROOT/grid/cube_grid_goals_s0.json")
    ap.add_argument("--eval-config", default="3dtarget_cube_grid")
    ap.add_argument("--render-size", type=int, default=900)
    ap.add_argument("--style", default="solid",
                    choices=("solid", "ghost", "grey", "pins", "paper"))
    ap.add_argument("--neutral-cube", action="store_true",
                    help="grey out the cube so scene red does not clash with the scale")
    ap.add_argument("--vmax", type=float, default=4.0,
                    help="cm at the top of the colour ramp (default: the 4 cm threshold)")
    ap.add_argument("--azimuth", type=float, default=138.0)
    ap.add_argument("--elevation", type=float, default=-21.0)
    ap.add_argument("--distance", type=float, default=1.12)
    ap.add_argument("--out", type=Path, default=Path("figures/goal_coverage"))
    args = ap.parse_args()
    results = args.results or Path(RESULTS_ROOT) / "3dtarget_cube_grid"
    goals_file = args.goals or Path(T2L_ROOT) / "grid/cube_grid_goals_s0.json"

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path("config/eval") / f"{args.eval_config}.yaml")
    spec = json.loads(Path(goals_file).read_text())
    start = spec["meta"]["start"]

    cases = []
    for f in sorted(Path(results).glob(f"{args.arm}_z*/seed*/grid_results.json")):
        cases += json.loads(f.read_text())["cases"]
    if not cases:
        raise SystemExit(f"no {args.arm} results under {results}")
    goals = np.array([c["goal"] for c in cases]) * 100.0          # cm
    err = np.array([c["closest_distance"] for c in cases]) * 100.0  # cm
    ok = np.array([c["success"] for c in cases])
    print(f"[goals] {len(goals)} commanded goals, {(~ok).sum()} missed the 4 cm threshold, "
          f"median error {np.median(err):.2f} cm")

    markers = build_markers(goals / 100.0, err, ok, args.vmax)
    img = render_start(cfg, args.table,
                       start["episode"], start["step"], args.render_size, markers=markers,
                       style=args.style, neutral_cube=args.neutral_cube,
                       azimuth=args.azimuth, elevation=args.elevation, distance=args.distance)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.style == "solid" else f"_{args.style}"
    plot(args.out / f"cube_start_and_goals_{args.arm}{suffix}", img, err, ok, args.vmax, args.arm)
    print(f"wrote {args.out / f'cube_start_and_goals_{args.arm}{suffix}'}.{{png,pdf}}")


def ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("error_gr", ERROR_RAMP)


def build_markers(goals_m, err_cm, ok, vmax):
    """One sphere per commanded goal, coloured by the error reached."""
    import matplotlib.colors as mcolors

    cmap, norm = ramp(), mcolors.Normalize(0, vmax, clip=True)
    return [(p, list(cmap(norm(e))[:3]) + [1.0], 0.012) for p, e in zip(goals_m, err_cm)]


def plot(stem, img, err, ok, vmax, arm, extend=None):
    """The figure: the rendered scene and a colour scale. Nothing else.

    ``extend`` forces the colour bar's overflow arrow, so panels meant to sit
    side by side keep identical bars even when only one of them overflows.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.imshow(img)
    ax.set_axis_off()

    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=ramp()), ax=ax,
                      shrink=0.7, pad=0.02, aspect=26,
                      extend=extend or ("max" if err.max() > vmax else "neither"))
    cb.set_label("error (cm)", fontsize=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, labelsize=9)

    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
