"""Interactive designer for the Utonia token grid (grid_dims / grid_bounds).

Loads a handful of lidar frames from an environment's dataset, maps them into
the CANONICAL frame exactly like :class:`pc_encoders.utonia_encoder
.UtoniaEncoder` (fitted table plane -> z = 0 with +z up, workspace center ->
xy origin; same Rodrigues rotation, same center-based orientation) and shows
in polyscope the clouds with a live-editable token grid on top: sliders for
``grid_x/y/z``, editable ``grid_bounds``, the occupied cells of the active
frame as translucent boxes colored by point count, and occupancy / cache-size
stats over all loaded frames. Use it to pick ``grid_dims`` + ``grid_bounds``
per environment, then write the values into the utoniawm data/model configs
("print YAML" dumps them to the terminal).

The per-env table planes were RANSAC-fitted on the released tables (a cube
refit matched the checked-in plane to 0.005 deg / 0.005 mm) and are stored
CUBE-STYLE: the raw normal
points DOWNWARD like the checked-in cube plane -- the encoder re-orients it
from canonical_center, and ground removal uses |dist|, so only consistency
matters. pusht / reacher / two_room all share an exact 45 deg camera tilt
(n = (1/sqrt2, 0, -1/sqrt2)); only d differs.

The suggested ``canonical_center`` (printed at startup, drawn as the origin
gizmo) is the mid-point of the loaded clouds' robust bbox, rounded to 2
decimals and then used EXACTLY like the encoder uses the config value -- so
what you see is what the config will produce.

What to look at while designing:
  * cell size vs object size: the moving object (T-block / fingertip / agent)
    should span >= a few cells along its travel axes, so its motion actually
    changes the token set -- one fat cell that always contains the object
    carries no dynamics signal;
  * z layering: ground is KEPT in this arm, so z_lo sits just below 0 (the
    plane inliers land in the bottom cell layer, cube uses -0.01) and z_hi
    just above the tallest geometry -- "clamped" points get smeared into edge
    cells, keep that fraction ~0;
  * empty cells are fine (EMPTY is part of the dynamics -- see
    config/train/model/utoniawm_cube.yaml)
    but mind the budget: cache size scales linearly with num_tokens, keep the
    product in the few-hundred range like DINO-WM's 256 patches;
  * walls (two_room): decide whether wall cells are worth their tokens --
    they never change, but they anchor the layout like DINO-WM's static
    background patches do.

Usage (needs a display; run from the repo root):
    pixi run python viz_grid.py --env pusht
    pixi run python viz_grid.py --env two_room --frames 12
    pixi run python viz_grid.py --env cube          # reference: shipped grid
    pixi run python viz_grid.py --dataset /path/t.lance --plane nx ny nz d
"""

import argparse

import numpy as np

from paths import dataset_path

GROUND_THRESH = 0.008  # coloring/stats only; the encoders' ground_thresh
EMBED_DIM = 192        # cache-size estimate: fp16 tokens at the default width

# Table planes (nx, ny, nz, d) in the RAW sensor frame, |n . xyz + d| = dist.
# Cube is the checked-in plane (config/train/model/utoniawm_cube.yaml); the
# rest were RANSAC-fitted the same way (see module docstring) and sign-matched
# to cube.
ENVS = {
    "cube": dict(
        dataset=dataset_path("cube"),
        plane=(0.628104, 0.0, -0.778129, -0.638995),
        center=(1.27, 0.0, 0.25),  # shipped canonical_center
    ),
    "pusht": dict(
        dataset=dataset_path("pusht"),
        plane=(0.707107, 0.0, -0.707107, -3.211194),
        center=None,  # suggested from data at startup
    ),
    "reacher": dict(
        dataset=dataset_path("reacher"),
        plane=(0.707106, 0.0, -0.707107, -0.567108),
        center=None,
    ),
    "two_room": dict(
        dataset=dataset_path("tworoom"),
        plane=(0.707105, 0.0, -0.707108, -2.821242),
        center=None,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--env", choices=sorted(ENVS), default=None,
                   help="environment preset (dataset path + fitted plane)")
    p.add_argument("--dataset", default=None,
                   help="lance dataset with the flat `lidar` column "
                   "(overrides the preset's path)")
    p.add_argument("--plane", type=float, nargs=4, default=None,
                   metavar=("NX", "NY", "NZ", "D"),
                   help="table plane in the raw frame (overrides the preset)")
    p.add_argument("--frames", type=int, default=8,
                   help="random frames to load (spread over the whole table)")
    p.add_argument("--seed", type=int, default=0, help="frame-sampling seed")
    p.add_argument("--invalid-value", type=float, default=-1.0,
                   help="drop points whose coords all equal this")
    return p.parse_args()


def load_frames(dataset_path, n, seed, invalid_value):
    """n random rows -> list of (Ni, 3) clouds + row labels + total row count."""
    import lance

    ds = lance.dataset(dataset_path)
    total = ds.count_rows()
    rows = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
    cols = ["lidar", "episode_idx", "step_idx"]
    tbl = ds.take(rows.tolist(), columns=cols)
    flat = np.stack(tbl["lidar"].to_numpy(zero_copy_only=False))
    pts = flat.reshape(n, -1, 3).astype(np.float32)
    clouds = [p[~np.all(p == invalid_value, axis=1)] for p in pts]
    labels = [f"ep {e} step {s}" for e, s in
              zip(tbl["episode_idx"].to_pylist(), tbl["step_idx"].to_pylist())]
    return clouds, labels, total


def canon_transform(plane, center):
    """(R, t) mapping raw -> canonical, EXACTLY like UtoniaEncoder.__init__."""
    plane = np.asarray(plane, dtype=np.float64)
    n, d = plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])
    center = np.asarray(center, dtype=np.float64)
    if n @ center + d < 0:  # orient: workspace center above table
        n, d = -n, -d
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), n @ z
    assert 1 + c > 1e-6, "plane normal anti-parallel to z; flip the plane"
    V = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + V + V @ V / (1 + c)  # Rodrigues
    t = np.concatenate([-(R @ center)[:2], [d]])
    return R, t


def suggest_center(plane, clouds):
    """Robust-bbox mid-point of the pooled clouds as a canonical_center.

    Returned in the RAW frame, rounded to 2 decimals (cube-config style), so
    the value can go into the YAML verbatim. Only its xy (through R) fixes the
    canonical origin; z just has to sit above the table for orientation.
    """
    plane = np.asarray(plane, dtype=np.float64)
    n, d = plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])
    pool = np.concatenate(clouds).astype(np.float64)
    # provisional center: the pooled centroid always sits above the plane
    # (points ON it average out, everything else is above), so it orients n
    prov = pool.mean(0)
    if prov @ n + d < 0:
        n, d = -n, -d
    R, _ = canon_transform(np.concatenate([n, [d]]), prov)
    canon = pool @ R.T
    canon[:, 2] += d
    lo, hi = np.percentile(canon, 1, axis=0), np.percentile(canon, 99, axis=0)
    mid = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, max(hi[2] / 2, 0.05)])
    raw = R.T @ (mid - np.array([0.0, 0.0, d]))
    return tuple(round(float(v), 2) + 0.0 for v in raw)  # +0.0 kills -0.0


def fit_bounds(pts, lo_pct=0.5, hi_pct=99.5, margin=0.02):
    """Percentile bbox of canonical points, z_lo pinned just below the plane."""
    lo = np.percentile(pts, lo_pct, axis=0) - margin
    hi = np.percentile(pts, hi_pct, axis=0) + margin
    lo[2] = -0.01  # ground kept: plane inliers land in the bottom cell layer
    return lo, hi


def cell_index(pts, lo, cell, dims):
    """Per-point cell index + clamped mask, matching the encoder's binning."""
    idx = np.floor((pts - lo) / cell).astype(np.int64)
    clamped = (idx < 0) | (idx >= dims)
    idx = np.clip(idx, 0, dims - 1)
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    return flat, clamped.any(axis=1)


def lattice(lo, hi, dims):
    """Grid-line nodes + edges for a curve network."""
    xs = [np.linspace(lo[a], hi[a], dims[a] + 1) for a in range(3)]
    nodes, edges, base = [], [], 0
    for a in range(3):  # full-extent lines along axis a at every (b, c) corner
        b, c = (a + 1) % 3, (a + 2) % 3
        B, C = np.meshgrid(xs[b], xs[c], indexing="ij")
        k = B.size
        start = np.zeros((k, 3)); end = np.zeros((k, 3))
        start[:, a], end[:, a] = lo[a], hi[a]
        start[:, b] = end[:, b] = B.ravel()
        start[:, c] = end[:, c] = C.ravel()
        nodes += [start, end]
        edges.append(np.stack([np.arange(k) + base, np.arange(k) + base + k], axis=1))
        base += 2 * k
    return np.concatenate(nodes), np.concatenate(edges)


def cell_boxes(flat_ids, counts, lo, cell, dims, shrink=0.94):
    """Quad-face cuboids for occupied cells (slightly shrunk), + per-face count."""
    ijk = np.stack(np.unravel_index(flat_ids, dims), axis=1).astype(np.float64)
    c0 = lo + (ijk + (1 - shrink) / 2) * cell
    c1 = lo + (ijk + (1 + shrink) / 2) * cell
    n = len(flat_ids)
    corners = np.zeros((n, 8, 3))
    for k in range(8):
        pick = np.array([(k >> 2) & 1, (k >> 1) & 1, k & 1], dtype=bool)
        corners[:, k] = np.where(pick, c1, c0)
    quads = np.array([[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                      [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]])
    verts = corners.reshape(-1, 3)
    faces = (quads[None] + 8 * np.arange(n)[:, None, None]).reshape(-1, 4)
    face_count = np.repeat(counts, 6)
    return verts, faces, face_count


def main():
    args = parse_args()
    preset = ENVS.get(args.env, {})
    dataset = args.dataset or preset.get("dataset")
    plane = tuple(args.plane) if args.plane else preset.get("plane")
    assert dataset and plane, "pass --env or both --dataset and --plane"
    env_name = args.env or "custom"

    clouds, labels, total_rows = load_frames(
        dataset, args.frames, args.seed, args.invalid_value)
    print(f"[viz] {env_name}: {len(clouds)} frames from {dataset} "
          f"({total_rows} rows total)")

    center = preset.get("center") or suggest_center(plane, clouds)
    R, t = canon_transform(plane, center)
    canon = [(c @ R.T + t).astype(np.float32) for c in clouds]
    print(f"[viz] canonical_plane:  {list(plane)}")
    print(f"[viz] canonical_center: {list(center)}"
          + ("  (suggested from data -- pin it in the config)"
             if preset.get("center") is None else "  (from config)"))

    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")  # the dataset's own plane sits at z=0

    # A structure first drawn with the camera eye near the world origin renders
    # permanently blank in polyscope, so place the camera explicitly before the
    # first frame (same workaround as in viz_radius.py).
    m = canon[0].mean(axis=0).tolist()
    ps.look_at((m[0] + 1.5, m[1] + 1.5, m[2] + 1.0), m)

    pool = np.concatenate(canon)
    lo0, hi0 = fit_bounds(pool)
    state = {
        "dims": [8, 8, 4],                       # cube default as a seed
        "lo": [float(v) for v in lo0],
        "hi": [float(v) for v in hi0],
        "frame": 0,
        "show_lattice": True,
        "show_cells": True,
        "show_others": True,
        "stats": "",
    }

    def rebuild():
        dims = np.maximum(1, np.asarray(state["dims"], dtype=np.int64))
        lo = np.asarray(state["lo"], dtype=np.float64)
        hi = np.asarray(state["hi"], dtype=np.float64)
        hi = np.maximum(hi, lo + 1e-3)
        state["hi"] = [float(v) for v in hi]
        cell = (hi - lo) / dims
        f = int(state["frame"]) % len(canon)
        active = canon[f]

        # active cloud: ground inliers faint, objects colored by height
        zdist = np.abs(active[:, 2])
        is_ground = zdist < GROUND_THRESH
        pc = ps.register_point_cloud("active frame", active, radius=0.0016)
        col = np.empty((len(active), 3))
        col[is_ground] = (0.72, 0.72, 0.72)
        h = active[:, 2]
        hn = np.clip((h - 0) / max(hi[2], 1e-6), 0, 1)
        col[~is_ground] = np.stack(
            [0.10 + 0.85 * hn, 0.45 + 0.3 * (1 - hn), 0.85 - 0.6 * hn],
            axis=1)[~is_ground]
        pc.add_color_quantity("ground faint / height", col, enabled=True)

        if state["show_others"] and len(canon) > 1:
            others = np.concatenate([c for i, c in enumerate(canon) if i != f])
            po = ps.register_point_cloud("other frames", others, radius=0.0008,
                                         color=(0.55, 0.55, 0.6))
            po.set_transparency(0.15)
        else:
            ps.remove_point_cloud("other frames", error_if_absent=False)

        if state["show_lattice"]:
            nodes, edges = lattice(lo, hi, dims)
            net = ps.register_curve_network("token grid", nodes, edges,
                                            radius=0.0005, color=(0.9, 0.35, 0.15))
            net.set_transparency(0.5)
        else:
            ps.remove_curve_network("token grid", error_if_absent=False)

        # occupancy: active frame boxes + stats over all loaded frames
        ids, clamped = cell_index(active, lo, cell, dims)
        counts = np.bincount(ids, minlength=int(dims.prod()))
        occ_ids = np.flatnonzero(counts)
        if state["show_cells"] and len(occ_ids):
            verts, faces, fc = cell_boxes(occ_ids, counts[occ_ids], lo, cell, dims)
            mesh = ps.register_surface_mesh("occupied cells", verts, faces,
                                            smooth_shade=False)
            mesh.add_scalar_quantity("log10 points", np.log10(fc),
                                     defined_on="faces", cmap="viridis",
                                     enabled=True)
            mesh.set_transparency(0.35)
        else:
            ps.remove_surface_mesh("occupied cells", error_if_absent=False)

        per_frame_occ, union = [], np.zeros(int(dims.prod()), dtype=bool)
        clamp_frac = []
        for c in canon:
            i, cl = cell_index(c, lo, cell, dims)
            occ = np.bincount(i, minlength=int(dims.prod())) > 0
            per_frame_occ.append(int(occ.sum()))
            union |= occ
            clamp_frac.append(float(cl.mean()))
        ntok = int(dims.prod())
        cache_gb = total_rows * ntok * EMBED_DIM * 2 / 1e9
        occ_arr = np.array(per_frame_occ)
        state["stats"] = (
            f"[{labels[f]}]  cell {cell[0]*100:.1f} x {cell[1]*100:.1f} x "
            f"{cell[2]*100:.1f} cm | num_tokens {ntok} "
            f"(cache ~{cache_gb:.0f} GB fp16 @ embed_dim {EMBED_DIM})\n"
            f"occupied cells/frame min/med/max: {occ_arr.min()}/"
            f"{int(np.median(occ_arr))}/{occ_arr.max()} "
            f"({occ_arr.mean() / ntok:.0%} of grid) | union over "
            f"{len(canon)} frames: {int(union.sum())} | never occupied: "
            f"{ntok - int(union.sum())}\n"
            f"points CLAMPED into edge cells (out of bounds): "
            f"{np.mean(clamp_frac):.1%} mean / {max(clamp_frac):.1%} max "
            f"| active-frame pts/occupied cell median: "
            f"{int(np.median(counts[occ_ids])) if len(occ_ids) else 0}"
        )

    def callback():
        changed = False
        ch, state["frame"] = psim.SliderInt("frame", state["frame"], 0,
                                            len(canon) - 1)
        changed |= ch
        for i, name in enumerate("grid_x grid_y grid_z".split()):
            ch, state["dims"][i] = psim.SliderInt(name, state["dims"][i], 1, 24)
            changed |= ch
        ch, state["lo"] = psim.InputFloat3("bounds lo", state["lo"])
        changed |= ch
        ch, state["hi"] = psim.InputFloat3("bounds hi", state["hi"])
        changed |= ch
        if psim.Button("fit bounds: all points"):
            lo, hi = fit_bounds(np.concatenate(canon))
            state["lo"], state["hi"] = [float(v) for v in lo], [float(v) for v in hi]
            changed = True
        psim.SameLine()
        if psim.Button("fit bounds: off-plane only"):
            pool = np.concatenate(canon)
            obj = pool[np.abs(pool[:, 2]) >= GROUND_THRESH]
            lo, hi = fit_bounds(obj)
            state["lo"], state["hi"] = [float(v) for v in lo], [float(v) for v in hi]
            changed = True
        ch, state["show_lattice"] = psim.Checkbox("lattice", state["show_lattice"])
        changed |= ch
        psim.SameLine()
        ch, state["show_cells"] = psim.Checkbox("occupied cells", state["show_cells"])
        changed |= ch
        psim.SameLine()
        ch, state["show_others"] = psim.Checkbox("other frames (faint)",
                                                 state["show_others"])
        changed |= ch
        if psim.Button("print YAML"):
            lo = [round(float(v), 2) for v in state["lo"]]
            hi = [round(float(v), 2) for v in state["hi"]]
            print(f"# {env_name} -- viz_grid.py verdict")
            print(f"  grid_dims: {list(state['dims'])}")
            print(f"  grid_bounds: [{lo}, {hi}]")
            print(f"  canonical_plane: {list(plane)}")
            print(f"  canonical_center: {list(center)}")
        if changed:
            rebuild()
        psim.TextUnformatted(state["stats"])

    rebuild()
    ps.set_user_callback(callback)
    print("[viz] tune grid_x/y/z + bounds in the panel; 'print YAML' dumps the "
          "values to paste into the utoniawm configs")
    ps.show()


if __name__ == "__main__":
    main()
