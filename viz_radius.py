"""Interactive tuner for the PointViT ball-query grouping radius.

Loads a lidar frame from the dataset, runs the encoder's OWN code path
(deterministic FPS centers -> ``_group`` = radius query + random subsample, see
:class:`pc_encoders.pointvit_encoder.PointViTEncoder`) on the FULL cloud --
ground kept by default; ``--remove-ground`` applies the config's plane removal
first -- and shows in polyscope which points a given ``group_radius`` selects, with live
sliders for radius / group_size / group_max_neighbors and per-ball occupancy
stats. Left-click a red FPS center to isolate that center's ball (members +
sampled points only); uncheck "isolate one center" to show all groups again. Use it to pick the ``group_radius`` (and sanity-check
``group_max_neighbors``) for an environment, then write the values into the
``config/train/model/*.yaml`` encoder blocks.

What to look at while tuning:
  * the small object (cube) should be covered by several balls whose members
    stay ON the object -- a radius that swallows table + cube into one ball
    blurs the task signal, one that is too small leaves balls with only their
    center (degenerate all-zero-offset groups, shown as "starved");
  * "ball occupancy" tells whether ``group_max_neighbors`` is comfortably above
    the typical ball (torch_cluster truncates fuller balls in packed row order,
    i.e. scan order -- NOT randomly -- so a tight cap biases the candidates);
  * "cloud coverage" is the fraction of points inside at least one ball -- the
    geometry the ViT can actually see.

Usage (needs a display; run from the repo root):
    pixi run python viz_radius.py                       # point_lewm_cube config, frame 0
    pixi run python viz_radius.py --config-name point_lewm_pusht
    pixi run python viz_radius.py --episode 12 --step 80
    pixi run python viz_radius.py --dataset /path/to/other.lance
"""

import argparse

import numpy as np
import torch

from paths import dataset_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--config-name",
        default="point_lewm_cube",
        help="train config whose model.encoder block to instantiate "
        "(point_lewm_<env> | point_delta_jepa_<env>, e.g. point_lewm_cube)",
    )
    p.add_argument(
        "--dataset",
        default=dataset_path("cube"),
        help="lance dataset with the flat `lidar` column "
        "(default: the cube table under $PLWM_DATA_ROOT, see paths.py)",
    )
    p.add_argument("--episode", type=int, default=0, help="episode index")
    p.add_argument("--step", type=int, default=0, help="step within the episode")
    p.add_argument(
        "--invalid-value",
        type=float,
        default=-1.0,
        help="drop points whose coords all equal this (missing lidar returns)",
    )
    p.add_argument(
        "--remove-ground",
        action="store_true",
        help="apply the config's ground-plane removal before FPS/grouping "
        "(default: keep the ground, FPS and balls run on the full cloud)",
    )
    return p.parse_args()


def load_frame(dataset_path, episode, step, invalid_value):
    """One raw lidar cloud (N, 3) from the dataset, invalid returns dropped."""
    import stable_worldmodel as swm

    ds = swm.data.load_dataset(dataset_path)
    row = int(np.concatenate([[0], np.cumsum(ds.lengths)])[episode] + step)
    pts = np.asarray(ds[row]["lidar"], dtype=np.float32).reshape(-1, 3)
    keep = ~(pts == invalid_value).all(axis=1)
    print(f"[viz] episode {episode} step {step} (row {row}): "
          f"{keep.sum()}/{len(pts)} valid points")
    return torch.from_numpy(pts[keep])


def build_encoder(config_name):
    """Instantiate the config's PointViT encoder (CPU; grouping only)."""
    import hydra

    with hydra.initialize(config_path="config/train", version_base=None):
        cfg = hydra.compose(config_name=config_name)
    enc = hydra.utils.instantiate(cfg.model.encoder)
    enc.eval()
    return enc


def ball_stats(coord, centers, radius, cap=4096):
    """True per-ball occupancy (uncapped-ish: cap is only a safety bound)."""
    import torch_cluster

    batch = torch.zeros(coord.shape[0], dtype=torch.long)
    cbatch = torch.zeros(centers.shape[0], dtype=torch.long)
    edges = torch_cluster.radius(coord, centers, radius, batch, cbatch, max_num_neighbors=cap)
    counts = torch.bincount(edges[0], minlength=centers.shape[0])
    covered = torch.zeros(coord.shape[0], dtype=torch.bool)
    covered[edges[1]] = True
    return counts, covered


def main():
    args = parse_args()
    torch.manual_seed(0)

    enc = build_encoder(args.config_name)
    raw = load_frame(args.dataset, args.episode, args.step, args.invalid_value)

    # Ground is KEPT by default -- FPS and the ball query run on the full
    # cloud. --remove-ground applies the config's plane removal instead (the
    # encoder's own preprocessing when ground removal is enabled).
    packed = {"coord": raw, "batch": torch.zeros(raw.shape[0], dtype=torch.long), "feat": None}
    ground = None
    if args.remove_ground:
        data = enc.remove_ground(packed)
        coord, batch = data["coord"], data["batch"]
        if coord.shape[0] != raw.shape[0]:
            # rows the plane filter removed, kept around for faint context
            # rendering (same mask remove_ground computes)
            dist = (raw @ enc.ground_plane[:3] + enc.ground_plane[3]).abs()
            ground = raw[dist < enc.ground_thresh]
        print(f"[viz] after ground removal: {coord.shape[0]} points")
    else:
        coord, batch = packed["coord"], packed["batch"]
        print(f"[viz] ground kept: {coord.shape[0]} points "
              f"(pass --remove-ground for the ground-removed encoder path)")

    # deterministic FPS centers -- independent of the radius, computed once
    center_idx = enc._fps_centers(coord, batch, 1).reshape(-1)  # (num_tokens,)
    centers = coord[center_idx]
    T = centers.shape[0]

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")

    # A structure first drawn with the camera eye near the world origin renders
    # permanently blank in polyscope, so place the camera explicitly before the
    # first frame.
    c = coord.mean(dim=0).tolist()
    ps.look_at((c[0] + 1.2, c[1] + 1.2, c[2] + 0.8), c)

    cloud_np = coord.numpy()
    ps_cloud = ps.register_point_cloud("cloud", cloud_np, radius=0.0016, color=(0.62, 0.62, 0.62))
    if ground is not None and len(ground):
        g = ps.register_point_cloud("ground (removed)", ground.numpy(), radius=0.0010,
                                    color=(0.85, 0.85, 0.85))
        g.set_transparency(0.25)
    ps_centers = ps.register_point_cloud("fps centers", centers.numpy(), radius=0.004,
                                         color=(0.85, 0.15, 0.15))

    # mutable UI state, seeded from the composed config
    state = {
        "radius": float(enc.group_radius),
        "group_size": int(enc.group_size),
        "max_neighbors": int(enc.group_max_neighbors),
        "seed": 0,
        "isolate": False,
        "token": 0,
        "stats": "",
    }
    rng_colors = np.random.default_rng(0).random((T, 3)) * 0.75 + 0.25

    def recompute():
        enc.group_radius = max(1e-4, float(state["radius"]))
        enc.group_size = max(1, int(state["group_size"]))
        enc.group_max_neighbors = max(enc.group_size, int(state["max_neighbors"]))
        state["max_neighbors"] = enc.group_max_neighbors

        torch.manual_seed(state["seed"])  # reproducible draw until "resample"
        nbr_idx = enc._group(coord, batch, centers, 1)[0]  # (T, k)

        counts, covered = ball_stats(coord, centers, enc.group_radius)
        if state["isolate"]:
            t = int(state["token"]) % T
            members = torch.unique(nbr_idx[t])
            ps_sel = ps.register_point_cloud("sampled points", coord[members].numpy(),
                                             radius=0.0035)
            ps_sel.add_color_quantity("group", np.tile(rng_colors[t], (len(members), 1)),
                                      enabled=True)
            ball = (coord - centers[t]).norm(dim=1) <= enc.group_radius
            ps_ball = ps.register_point_cloud("ball members", coord[ball].numpy(),
                                              radius=0.0022, color=(0.95, 0.75, 0.2))
            ps_ball.set_transparency(0.6)
        else:
            flat = nbr_idx.reshape(-1)
            tok = torch.arange(T)[:, None].expand_as(nbr_idx).reshape(-1)
            ps_sel = ps.register_point_cloud("sampled points", coord[flat].numpy(),
                                             radius=0.0022)
            ps_sel.add_color_quantity("group", rng_colors[tok.numpy()], enabled=True)
            ps.remove_point_cloud("ball members", error_if_absent=False)

        starved = int((counts < enc.group_size).sum())
        dup = 1.0 - nbr_idx.reshape(-1).unique().numel() / nbr_idx.numel()
        capped = int((counts > enc.group_max_neighbors).sum())
        state["stats"] = (
            f"radius {enc.group_radius * 100:.1f} cm | "
            f"ball occupancy min/med/max: {int(counts.min())}/"
            f"{int(counts.median())}/{int(counts.max())}\n"
            f"starved balls (< group_size={enc.group_size}): {starved}/{T} "
            f"({starved / T:.0%}) | duplicated samples: {dup:.0%}\n"
            f"balls over max_neighbors cap ({enc.group_max_neighbors}, "
            f"scan-order-biased!): {capped}/{T} ({capped / T:.0%})\n"
            f"cloud coverage (in >=1 ball): {int(covered.sum())}/{coord.shape[0]} "
            f"({covered.float().mean():.0%})"
        )

    def callback():
        changed = False
        # Click-to-isolate: piggyback on polyscope's built-in left-click
        # selection. A click on a red FPS center isolates that center's ball;
        # the selection is consumed (reset) so unchecking the box sticks and
        # re-clicking the same center works again.
        if ps.have_selection():
            sel = ps.get_selection()
            if sel.is_hit and sel.structure_name == "fps centers":
                state["token"] = int(sel.local_index)
                state["isolate"] = True
                ps.reset_selection()
                changed = True
        ch, state["radius"] = psim.SliderFloat(
            "group_radius (m)", state["radius"], v_min=0.005, v_max=0.5,
            flags=psim.ImGuiSliderFlags_Logarithmic,
        )
        changed |= ch
        ch, state["group_size"] = psim.SliderInt("group_size", state["group_size"], 1, 128)
        changed |= ch
        ch, state["max_neighbors"] = psim.SliderInt(
            "group_max_neighbors", state["max_neighbors"], 8, 1024
        )
        changed |= ch
        if psim.Button("resample draw"):
            state["seed"] += 1
            changed = True
        psim.SameLine()
        ch, state["isolate"] = psim.Checkbox("isolate one center", state["isolate"])
        changed |= ch
        if state["isolate"]:
            psim.TextUnformatted(
                f"isolated center #{state['token']} -- click another red FPS "
                "center to switch, uncheck to show all"
            )
        else:
            psim.TextUnformatted("click a red FPS center to isolate its ball")
        if changed:
            recompute()
        psim.TextUnformatted(state["stats"])

    recompute()
    ps.set_user_callback(callback)
    print("[viz] settings live in the ImGui panel; copy the tuned values into "
          "the encoder block of config/train/model/*.yaml")
    ps.show()


if __name__ == "__main__":
    main()
