"""FPS/grouping story renders from ONE fixed perspective (iterate on me).

Renders three transparent PNGs of the SAME leveled, ground-kept frame from the
same camera (the straight-on "facing the arm" perspective of the interactive
viz_radius.py view):

    fps_1_plain.png    the cloud, nothing else
    fps_2_centers.png  + all num_tokens FPS centers (red)
    fps_3_groups.png   + N example groups: each chosen center and the points
                       grouped with it (its sampled ball) in its own color

Same code path as viz_radius.py (leveled table, encoder's own FPS + ball
query), but headless & batch. Knobs live in the CONFIG block below on purpose
-- tweak, re-run, look, repeat. For a pixel-exact camera: navigate in
viz_radius.py, click "print view json", save it to a file and pass
--view-json-file (overrides the auto camera).

Usage (from the repo root; no display needed):
    pixi run python scripts/render_fps_views.py
    pixi run python scripts/render_fps_views.py --episode 3 --step 40
    pixi run python scripts/render_fps_views.py --view-json-file view.json
    # bigger balls than the config (the 5 cm cube radius is hard to see):
    pixi run python scripts/render_fps_views.py --group-radius 0.10 0.15
    pixi run python scripts/render_fps_views.py --group-radius 0.15 --all-members
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from paths import dataset_path  # noqa: E402
from viz_grid import ENVS, canon_transform  # noqa: E402
from viz_radius import build_encoder  # noqa: E402

# ---------------------------------------------------------------- CONFIG ---
# group highlight colors, in order (blue violet purple orange yellow ...);
# add more to highlight more groups
GROUP_COLORS = [
    (0.16, 0.42, 0.78),  # blue
    (0.54, 0.31, 0.89),  # violet
    (0.48, 0.10, 0.52),  # purple
    (0.95, 0.55, 0.13),  # orange
    (0.93, 0.83, 0.15),  # yellow
    (0.10, 0.55, 0.50),  # teal
    (0.45, 0.65, 0.25),  # green
    (0.85, 0.45, 0.60),  # pink
    (0.55, 0.37, 0.22),  # brown
    (0.35, 0.70, 0.85),  # cyan
]
R_CLOUD = 0.0018        # point radius: background cloud
R_CENTER = 0.0042       # all-FPS-centers dots (panel 2 + context in panel 3)
R_GROUP_PT = 0.0038     # a highlighted group's points
R_GROUP_CENTER = 0.0065  # a highlighted group's center
CLOUD_GREY = (0.72, 0.72, 0.72)
CENTER_RED = (0.85, 0.15, 0.15)
# auto camera: the sensor's own viewpoint (dense scan rows at the bottom of
# the image, arm at the top -- the perspective of the viz_radius.py view),
# pulled back and lifted so the whole table fits
CAM_BACK = 3.4          # eye = target + (sensor - target) * CAM_BACK
CAM_LIFT = 0.55         # extra eye height (m) to steepen the look-down angle
CAM_TARGET_Z = 0.05     # aim just above the tabletop
CAM_FOV = 26.0          # narrow fov + far eye = the flat "long lens" look
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--config-name", default="point_delta_jepa_cube",
                   help="train config (config/train/<name>.yaml) for the encoder block (ground-kept view)")
    p.add_argument("--dataset", default=dataset_path("cube"),
                   help="OGB-Cube LiDAR table (default: paths.dataset_path('cube'))")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--n-groups", type=int, default=len(GROUP_COLORS),
                   help="how many example groups to highlight (<= colors)")
    p.add_argument("--group-ids", type=int, nargs="*", default=None,
                   help="explicit FPS-center indices to highlight "
                   "(default: auto -- one on the arm, rest spread)")
    p.add_argument("--seed", type=int, default=0, help="group-subsample seed")
    p.add_argument("--group-radius", type=float, nargs="*", default=None,
                   help="ball-query radius/radii (m) to render instead of the "
                   "config's; one fps_3_groups_r<cm>cm.png per value (FPS "
                   "centers and the highlighted ids are radius-independent)")
    p.add_argument("--group-size", type=int, default=None,
                   help="points sampled per ball (default: config's group_size)")
    p.add_argument("--all-members", action="store_true",
                   help="colour EVERY point inside a ball, not just the "
                   "group_size sampled ones (shows the ball's true extent)")
    p.add_argument("--view-json-file", default=None,
                   help="polyscope view json (from viz_radius's 'print view "
                   "json' button); overrides the auto camera")
    p.add_argument("--out", default=str(REPO / "outputs" / "encoder_viz"))
    p.add_argument("--res", type=int, nargs=2, default=(2800, 2450),
                   metavar=("W", "H"),
                   help="render resolution BEFORE the transparent-margin crop "
                   "(the scene fills ~40%% of the frame, so the cropped "
                   "output is roughly res * 0.42)")
    p.add_argument("--invalid-value", type=float, default=-1.0)
    return p.parse_args()


def load_frame(dataset_path, episode, step, invalid_value):
    import lance

    ds = lance.dataset(dataset_path)
    tbl = ds.to_table(columns=["lidar"],
                      filter=f"episode_idx = {episode} AND step_idx = {step}")
    assert tbl.num_rows == 1, (episode, step, tbl.num_rows)
    pts = np.stack(tbl["lidar"].to_numpy(zero_copy_only=False))
    pts = pts.reshape(-1, 3).astype(np.float32)
    keep = ~np.all(pts == invalid_value, axis=1)
    print(f"[render] ep {episode} step {step}: {keep.sum()}/{len(pts)} valid points")
    return torch.from_numpy(pts[keep])


def pick_groups(centers, n, focus_xy, focus_r=0.55):
    """n spread-out center indices near the action: candidates within
    ``focus_r`` of the table center (corner picks make dull panels), start on
    the arm (highest z), then greedy farthest-point among the candidates."""
    c = centers.numpy()
    cand = np.flatnonzero(np.linalg.norm(c[:, :2] - focus_xy, axis=1) < focus_r)
    if len(cand) < n:
        cand = np.arange(len(c))
    cc = c[cand]
    picked = [int(cc[:, 2].argmax())]
    d = np.linalg.norm(cc - cc[picked[0]], axis=1)
    while len(picked) < n:
        nxt = int(d.argmax())
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(cc - cc[nxt], axis=1))
    return [int(cand[i]) for i in picked]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    enc = build_encoder(args.config_name)
    coord = load_frame(args.dataset, args.episode, args.step, args.invalid_value)
    batch = torch.zeros(len(coord), dtype=torch.long)

    # level exactly like viz_radius (table -> z=0), so any view json grabbed
    # there matches here
    plane = (enc.ground_plane.tolist()
             if getattr(enc, "ground_plane", None) is not None
             else ENVS["cube"]["plane"])
    R, t = canon_transform(plane, coord.mean(dim=0).numpy())
    coord = coord @ torch.from_numpy(R).float().T + torch.from_numpy(t).float()
    sensor = t  # leveled position of the raw-frame origin = the lidar sensor

    center_idx = enc._fps_centers(coord, batch, 1).reshape(-1)
    centers = coord[center_idx]
    if args.group_size is not None:
        enc.group_size = int(args.group_size)
        enc.group_max_neighbors = max(int(enc.group_max_neighbors), enc.group_size)
    radii = args.group_radius or [float(enc.group_radius)]

    def group_members(tok, radius):
        """Point indices drawn for center ``tok`` at ``radius``."""
        if args.all_members:
            return np.flatnonzero(
                ((coord - centers[tok]).norm(dim=1) <= radius).numpy())
        return torch.unique(nbr_idx[tok]).numpy()

    pts = coord.numpy()
    table_xy = np.median(pts[np.abs(pts[:, 2]) < 0.02, :2], axis=0)

    n = min(args.n_groups, len(GROUP_COLORS))
    chosen = (args.group_ids[:n] if args.group_ids is not None
              else pick_groups(centers, n, table_xy))
    print(f"[render] highlighted center ids: {chosen}")

    import polyscope as ps

    ps.set_allow_headless_backends(True)
    ps.init()
    ps.set_window_size(*args.res)
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    # alpha 0: a 3-tuple sets bg alpha to 1, which OVERRIDES transparent_bg
    ps.set_background_color((1.0, 1.0, 1.0, 0.0))
    try:
        ps.set_SSAA_factor(2)
    except AttributeError:
        pass

    if args.view_json_file:
        ps.set_view_from_json(Path(args.view_json_file).read_text())
        print(f"[render] camera from {args.view_json_file}")
    else:
        # sensor's-eye view: from the lidar origin, pulled back / lifted
        target = np.array([table_xy[0], table_xy[1], CAM_TARGET_Z])
        eye = target + (sensor - target) * CAM_BACK + np.array([0, 0, CAM_LIFT])
        ps.look_at(tuple(eye), tuple(target))
        try:  # narrow the fov (json round-trip; no direct python setter)
            import json
            view = json.loads(ps.get_view_as_json())
            view["fov"] = CAM_FOV
            ps.set_view_from_json(json.dumps(view))
        except Exception as e:  # noqa: BLE001 -- old polyscope: keep 45deg fov
            print(f"[render] fov override skipped: {e}")
        print(f"[render] auto camera: sensor {np.round(sensor, 2)}, "
              f"eye {np.round(eye, 2)} -> target {np.round(target, 2)}, "
              f"fov {CAM_FOV}")

    cloud_np, centers_np = coord.numpy(), centers.numpy()

    def snap(name):
        path = out / f"{name}.png"
        ps.screenshot(str(path), transparent_bg=True)
        try:  # trim the transparent margins (keeps a small border)
            from PIL import Image

            im = Image.open(path).convert("RGBA")
            # SSAA leaves faint nonzero alpha in the background -- threshold,
            # else the bbox is always the full frame
            bbox = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
            if bbox:
                m = 24
                im.crop((max(0, bbox[0] - m), max(0, bbox[1] - m),
                         min(im.width, bbox[2] + m),
                         min(im.height, bbox[3] + m))).save(path, optimize=True)
        except ImportError:
            pass
        print(f"[render] wrote {path}")

    # 1 -- plain
    ps.register_point_cloud("cloud", cloud_np, radius=R_CLOUD, color=CLOUD_GREY)
    snap("fps_1_plain")

    # 2 -- + FPS centers
    ps.register_point_cloud("fps centers", centers_np, radius=R_CENTER,
                            color=CENTER_RED)
    snap("fps_2_centers")

    # 3 -- example groups only (the red context centers would bury them);
    # one panel per requested radius, same centers, same camera
    ps.remove_point_cloud("fps centers")
    for radius in radii:
        enc.group_radius = float(radius)
        # the cap must not truncate a big ball in scan order
        enc.group_max_neighbors = max(int(enc.group_max_neighbors),
                                      enc.group_size, 4096)
        torch.manual_seed(args.seed)
        nbr_idx = enc._group(coord, batch, centers, 1)[0]  # (T, group_size)
        for i, (tok, col) in enumerate(zip(chosen, GROUP_COLORS)):
            members = group_members(tok, radius)
            ps.register_point_cloud(f"group {i} points", cloud_np[members],
                                    radius=R_GROUP_PT, color=col)
            ps.register_point_cloud(f"group {i} center", centers_np[tok : tok + 1],
                                    radius=R_GROUP_CENTER, color=col)
        tag = "" if args.group_radius is None else f"_r{radius * 100:g}cm"
        tag += "_all" if args.all_members else ""
        snap(f"fps_3_groups{tag}")
        for i in range(len(chosen)):
            ps.remove_point_cloud(f"group {i} points")
            ps.remove_point_cloud(f"group {i} center")

    print(f"[render] done -> {out}")


if __name__ == "__main__":
    main()
