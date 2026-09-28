"""Linear + MLP probing of frozen world-model latents against 3-D ground truth.

For each environment the frozen checkpoints of the two point-cloud world models
(Point-LeWM and Point-Delta-JEPA) encode a strided sweep of the dataset --
``z = projector(encoder(cloud))``, exactly the representation the CEM cost
plans in (same composition as ``target2latent.precompute``) -- and two probes
per target regress the privileged state from ``z``:

* **linear**: ridge regression, closed form, lambda chosen on a validation
  split (log grid),
* **mlp**: 192 -> 512 -> 512 -> d, GELU, AdamW + cosine, early-stopped on the
  validation split.

Both report test-set MSE (native units) and pooled R^2. Splits are at EPISODE
granularity (72/8/20 train/val/test, seeded), so test frames come from
trajectories no probe ever saw.

Every probed quantity lives in 3-D and, where it is a pose, in the SENSOR
frame (the frame of the stored clouds; transforms + cached camera poses come
from ``target2latent.envs`` and were verified against the clouds there).
Angles enter as sensor-frame heading vectors -- the in-plane axis
``(cos a, sin a, 0)`` rotated into the sensor frame, the convention envs.py
already uses for pusht's T. Per env:

* pusht    agent location, block (T) location, block angle (heading vec)
* reacher  finger position, joint position (both FK 3-D elbow anchor and raw
           angles) and the link heading vectors
* cube     joint position (raw 6 angles AND FK 3-D anchors of the 5 moving
           arm joints), joint velocity (raw), end-effector position,
           end-effector yaw (heading vec), block position, block quaternion
           (raw as stored, world frame -- NOTE a cube is 24-fold symmetric,
           orientation is not identifiable from the cloud), block yaw
           (heading vec; same caveat mod 90 deg)

The reacher sensor pose is not in envs.py's SPECS; the constants below were
read off a live ``swm/Reacher3D-v0`` and verified against the
stored clouds: the transformed fingertip lands exactly its own 2.2 cm
half-extent from the nearest lidar return on every sampled frame, and
qpos-FK reproduces the table's ``finger_pos`` to float precision
(links 0.12 m + 0.12 m, arm plane z = 0.05).

Run (one env at a time; all stages cache + resume, so re-running is cheap)::

    python probe_latents.py --env pusht
    python probe_latents.py --env reacher
    python probe_latents.py --env cube
    python probe_latents.py --report          # combined table over all envs

One GPU is enough; the three envs plus the report take a few hours.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

from target2latent.envs import LOGS_ROOT, SPECS

# ---------------------------------------------------------------- constants

from paths import PROBE_ROOT  # noqa: E402  (<repo>/experiment_logs/probing or $PLWM_PROBE_ROOT)

#: display name -> (run folder, policy) per env; the checkpoints under probe,
#: resolved as <logs-root>/<run>/checkpoints/<policy>. The defaults are the
#: released checkpoints under checkpoints/<env>/<model>/ (run ".", --logs-root
#: defaults to the repository root). To probe a training run of your own:
#:   --logs-root experiment_logs --models "Point-LeWM=point_lewm_pusht:point_lewm_pusht/weights_final.pt"
MODELS = {
    "pusht": [
        ("Point-LeWM", ".", "pusht/point-lewm/weights.pt"),
        ("Point-Delta-JEPA", ".", "pusht/point-delta-jepa/weights.pt"),
    ],
    "reacher": [
        ("Point-LeWM", ".", "reacher/point-lewm/weights.pt"),
        ("Point-Delta-JEPA", ".", "reacher/point-delta-jepa/weights.pt"),
    ],
    "cube": [
        ("Point-LeWM", ".", "cube/point-lewm/weights.pt"),
        ("Point-Delta-JEPA", ".", "cube/point-delta-jepa/weights.pt"),
    ],
}

ENVS = ("pusht", "reacher", "cube")


def _ckpt_tag(run, policy):
    """Short name of a checkpoint for cache files: the run folder, or, for the
    released layout (run "."), the checkpoint folder, e.g. cube_point-lewm."""
    return run if run != "." else Path(policy).parent.as_posix().replace("/", "_")

#: reacher iso sensor pose (see module docstring). The rotation is the shared
#: 45/45 iso bearing -- numerically identical to SPECS['pusht'].sensor_rot.
_REACHER_ORIGIN = np.array([0.401, 0.401, 0.5671])
_REACHER_ROT = np.array(
    [
        [-0.5000000773622201, 0.7071067811865475, -0.4999999226377678],
        [-0.5000000773622201, -0.7071067811865475, -0.4999999226377679],
        [-0.7071066717798296, 0.0, 0.7071068905932484],
    ]
)
_REACHER_ARM_Z = 0.05  # arm plane height: joint anchors, links and finger
_REACHER_L1 = 0.12  # shoulder -> wrist
_REACHER_DATASET = str(Path(SPECS["pusht"].dataset).parent / "reacher.lance")

#: dataset columns to fetch per env (besides episode_idx/step_idx)
TARGET_COLUMNS = {
    "pusht": ["state"],
    "reacher": ["qpos", "finger_pos"],
    "cube": [
        "qpos",
        "proprio/joint_pos",
        "proprio/joint_vel",
        "proprio/effector_pos",
        "proprio/effector_yaw",
        "privileged/block_0_pos",
        "privileged/block_0_quat",
        "privileged/block_0_yaw",
    ],
}

#: probe target -> human label, in report order
TARGET_LABELS = {
    "pusht": {
        "agent_pos": "agent location (3D, sensor)",
        "block_pos": "block (T) location (3D, sensor)",
        "block_angle": "block angle (heading vec, sensor)",
    },
    "reacher": {
        "finger_pos": "finger position (3D, sensor)",
        "joint_pos_3d": "joint position (FK elbow anchor, 3D, sensor)",
        "joint_angles_raw": "joint position (raw angles, rad)",
        "joint_angle_headings": "joint angles (link heading vecs, sensor)",
    },
    "cube": {
        "joint_pos_raw": "joint position (raw 6 angles, rad)",
        "joint_pos_3d": "joint position (FK anchors j2-j6, 3D, sensor)",
        "joint_vel_raw": "joint velocity (raw, rad/s)",
        "effector_pos": "end-effector position (3D, sensor)",
        "effector_yaw": "end-effector yaw (heading vec, sensor)",
        "block_pos": "block position (3D, sensor)",
        "block_quat_raw": "block quaternion (raw wxyz, world)",
        "block_yaw": "block yaw (heading vec, sensor)",
    },
}


def _heading(angle, rot):
    """(N,) planar angle -> (N, 3) sensor-frame unit heading vector."""
    a = np.asarray(angle, dtype=np.float64).reshape(-1)
    d = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=1)
    return (d @ rot).astype(np.float32)


# ----------------------------------------------------------- target builders


def targets_pusht(raw):
    spec = SPECS["pusht"]
    st = np.asarray(raw["state"], dtype=np.float64).reshape(-1, 7)
    agent = spec.world_to_sensor(spec._px_to_world(st[:, 0:2]))
    block = spec.world_to_sensor(spec._px_to_world(st[:, 2:4]))
    # the mirror's y-flip reflects the frame -> world yaw = -angle (envs.py)
    return {
        "agent_pos": agent.astype(np.float32),
        "block_pos": block.astype(np.float32),
        "block_angle": _heading(-st[:, 4], spec.sensor_rot),
    }


def targets_reacher(raw):
    qp = np.asarray(raw["qpos"], dtype=np.float64).reshape(-1, 2)
    fp = np.asarray(raw["finger_pos"], dtype=np.float64).reshape(-1, 2)
    th0, th1 = qp[:, 0], qp[:, 0] + qp[:, 1]

    def to_sensor(xy):
        w = np.concatenate([xy, np.full((len(xy), 1), _REACHER_ARM_Z)], axis=1)
        return ((w - _REACHER_ORIGIN) @ _REACHER_ROT).astype(np.float32)

    elbow = _REACHER_L1 * np.stack([np.cos(th0), np.sin(th0)], axis=1)
    return {
        "finger_pos": to_sensor(fp),
        "joint_pos_3d": to_sensor(elbow),
        "joint_angles_raw": qp.astype(np.float32),
        "joint_angle_headings": np.concatenate(
            [_heading(th0, _REACHER_ROT), _heading(th1, _REACHER_ROT)], axis=1
        ),
    }


def _cube_fk_anchors(qpos):
    """(N, 21) qpos -> (N, 15) world anchors of arm joints 2..6.

    Joint 1 (shoulder_pan) anchors at the fixed base -- constant, so excluded.
    Verified against the table: the model's pinch site reproduces the stored
    ``proprio/effector_pos`` to < 1 mm on sampled rows.
    """
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401  (registers swm/OGBCube-v0)

    env = gym.make("swm/OGBCube-v0")
    env.reset(seed=0)
    m, d = env.unwrapped.model, env.unwrapped.data
    assert m.nq == qpos.shape[1], (m.nq, qpos.shape)
    out = np.empty((len(qpos), 15), dtype=np.float32)
    t0 = time.time()
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        out[i] = d.xanchor[1:6].reshape(-1)
        if (i + 1) % 50000 == 0:
            print(f"[targets] cube FK {i + 1}/{len(qpos)} "
                  f"({(i + 1) / (time.time() - t0):.0f} rows/s)", flush=True)
    env.close()
    return out


def targets_cube(raw):
    spec = SPECS["cube"]
    anchors_w = _cube_fk_anchors(np.asarray(raw["qpos"], dtype=np.float64))
    anchors_s = spec.world_to_sensor(anchors_w.reshape(-1, 3)).reshape(-1, 15)
    eff = np.asarray(raw["proprio/effector_pos"], dtype=np.float64).reshape(-1, 3)
    blk = np.asarray(raw["privileged/block_0_pos"], dtype=np.float64).reshape(-1, 3)
    return {
        "joint_pos_raw": np.asarray(raw["proprio/joint_pos"], np.float32),
        "joint_pos_3d": anchors_s.astype(np.float32),
        "joint_vel_raw": np.asarray(raw["proprio/joint_vel"], np.float32),
        "effector_pos": spec.world_to_sensor(eff).astype(np.float32),
        "effector_yaw": _heading(raw["proprio/effector_yaw"], spec.sensor_rot),
        "block_pos": spec.world_to_sensor(blk).astype(np.float32),
        "block_quat_raw": np.asarray(raw["privileged/block_0_quat"], np.float32),
        "block_yaw": _heading(raw["privileged/block_0_yaw"], spec.sensor_rot),
    }


TARGET_BUILDERS = {"pusht": targets_pusht, "reacher": targets_reacher, "cube": targets_cube}


def dataset_path(env, override=None):
    if override:
        return override
    return SPECS[env].dataset if env in SPECS else _REACHER_DATASET


# --------------------------------------------------------------- data stages


def _take_columns(ds, rows, columns, chunk=50000):
    """ds.take in chunks -> dict[column] = stacked np array (len(rows), ...)."""
    parts = {c: [] for c in columns}
    for i in range(0, len(rows), chunk):
        tbl = ds.take(rows[i : i + chunk].tolist(), columns=columns)
        for c in columns:
            col = tbl.column(c).to_numpy(zero_copy_only=False)
            parts[c].append(np.stack(col) if col.dtype == object else np.asarray(col))
    return {c: np.concatenate(v) for c, v in parts.items()}


def build_targets(env, ds, keep, out_path, dataset):
    """Cache {target: (N, d)} + ep/step for the strided rows; load if present."""
    meta = {"env": env, "dataset": str(dataset), "n": len(keep),
            "row0": int(keep[0]) if len(keep) else 0,
            "stride": int(keep[1] - keep[0]) if len(keep) > 1 else 1}
    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as f:
            old = json.loads(str(f["_meta"]))
            if old == meta:
                print(f"[targets] reusing {out_path}")
                return {k: f[k] for k in f.files if k != "_meta"}
        print(f"[targets] {out_path} is stale (meta mismatch), rebuilding")
    cols = ["episode_idx", "step_idx", *TARGET_COLUMNS[env]]
    raw = _take_columns(ds, keep, cols)
    out = TARGET_BUILDERS[env](raw)
    out["ep"] = np.asarray(raw["episode_idx"], dtype=np.int32)
    out["step"] = np.asarray(raw["step_idx"], dtype=np.int32)
    for k, v in out.items():
        assert len(v) == len(keep), (k, v.shape)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, _meta=json.dumps(meta), **out)
    print(f"[targets] wrote {out_path}: "
          + ", ".join(f"{k}{v.shape[1:]}" for k, v in out.items() if v.ndim > 1))
    return out


def build_latents(env, ds, keep, run, policy, logs_root, out_path, batch_size,
                  invalid_value=-1.0, rows_per_shard=20000):
    """Encode the strided rows with one frozen checkpoint; sharded + resumable."""
    meta = {"env": env, "run": run, "policy": policy, "n": len(keep)}
    if out_path.exists():
        with np.load(out_path) as f:
            if json.loads(str(f["_meta"])) == meta:
                print(f"[latents] reusing {out_path}")
                return f["z"]
        print(f"[latents] {out_path} is stale (meta mismatch), rebuilding")

    from target2latent.precompute import load_encoder, pack

    encode = load_encoder(run, policy, logs_root)
    shard_dir = Path(str(out_path) + ".shards")
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = [keep[i : i + rows_per_shard] for i in range(0, len(keep), rows_per_shard)]
    t0, done = time.time(), 0
    for si, rows in enumerate(shards):
        path = shard_dir / f"shard_{si:05d}.npy"
        if path.exists():
            done += len(rows)
            continue
        tbl = ds.take(rows.tolist(), columns=["lidar"])
        lidar = np.stack(tbl.column("lidar").to_numpy(zero_copy_only=False))
        lidar = lidar.reshape(len(lidar), -1, 3).astype(np.float32)
        zs = [
            encode(pack(lidar[i : i + batch_size], invalid_value, "cuda")).float().cpu()
            for i in range(0, len(lidar), batch_size)
        ]
        np.save(path, torch.cat(zs).numpy().astype(np.float32))
        done += len(rows)
        rate = done / max(time.time() - t0, 1e-9)
        print(f"[latents] {run}/{policy} shard {si + 1}/{len(shards)} "
              f"{done}/{len(keep)} ({rate:.0f} f/s, eta "
              f"{(len(keep) - done) / max(rate, 1e-9) / 60:.1f} min)", flush=True)
    z = np.concatenate([np.load(shard_dir / f"shard_{si:05d}.npy")
                        for si in range(len(shards))])
    assert len(z) == len(keep), (z.shape, len(keep))
    np.savez(out_path, _meta=json.dumps(meta), z=z)
    print(f"[latents] wrote {out_path} {z.shape}")
    return z


# -------------------------------------------------------------------- probes


def episode_split(ep, seed, frac_train=0.72, frac_val=0.08):
    """Episode-level train/val/test masks (test = 1 - train - val)."""
    eps = np.unique(ep)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_tr = int(round(frac_train * len(eps)))
    n_va = int(round(frac_val * len(eps)))
    tr, va, te = eps[:n_tr], eps[n_tr : n_tr + n_va], eps[n_tr + n_va :]
    return (np.isin(ep, tr), np.isin(ep, va), np.isin(ep, te))


def _metrics(pred, y):
    err = pred - y
    mse = float(np.mean(err**2))
    sst = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    return {"mse": mse, "r2": 1.0 - float(np.sum(err**2)) / max(sst, 1e-12)}


def ridge_probe(X, Y, masks):
    """Closed-form ridge on standardized X, lambda picked on val -> test metrics."""
    tr, va, te = masks
    mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    Xs = ((X - mu) / sd).astype(np.float64)
    ymu = Y[tr].mean(axis=0)
    Yc = (Y - ymu).astype(np.float64)
    G = Xs[tr].T @ Xs[tr]
    b = Xs[tr].T @ Yc[tr]
    eye = np.eye(X.shape[1])
    # standardized X -> diag(G) ~ n_train, so this absolute grid spans
    # "no regularization" to "heavily damped"
    best = None
    for lam in 10.0 ** np.arange(-2, 9):
        W = np.linalg.solve(G + lam * eye, b)
        val_mse = float(np.mean((Xs[va] @ W - Yc[va]) ** 2))
        if best is None or val_mse < best[0]:
            best = (val_mse, lam, W)
    _, lam, W = best
    out = _metrics(Xs[te] @ W + ymu, Y[te].astype(np.float64))
    out["lambda"] = lam
    return out


class _MLP(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def mlp_probe(X, Y, masks, seed=0, max_epochs=100, batch=8192, patience=15,
              device="cuda" if torch.cuda.is_available() else "cpu"):
    """Small MLP on standardized X/Y, early-stopped on val -> test metrics."""
    tr, va, te = masks
    torch.manual_seed(seed)
    xmu, xsd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
    ymu, ysd = Y[tr].mean(axis=0), Y[tr].std(axis=0)
    ysd[ysd < 1e-8] = 1.0  # constant target dims: predict via bias only
    Xt = torch.as_tensor((X - xmu) / xsd, dtype=torch.float32, device=device)
    Yt = torch.as_tensor((Y - ymu) / ysd, dtype=torch.float32, device=device)
    itr = torch.as_tensor(np.flatnonzero(tr), device=device)
    iva = torch.as_tensor(np.flatnonzero(va), device=device)

    model = _MLP(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = max(1, len(itr) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs * steps_per_epoch
    )
    best_val, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        perm = itr[torch.randperm(len(itr), device=device)]
        for i in range(0, steps_per_epoch * batch, batch):
            idx = perm[i : i + batch]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(Xt[iva]), Yt[iva]))
        if val < best_val - 1e-7:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ite = torch.as_tensor(np.flatnonzero(te), device=device)
        preds = torch.cat([model(Xt[ite[i : i + 65536]])
                           for i in range(0, len(ite), 65536)])
    pred = preds.cpu().numpy() * ysd + ymu
    out = _metrics(pred.astype(np.float64), Y[te].astype(np.float64))
    out["best_epoch"] = best_epoch
    return out


# ---------------------------------------------------------------------- main


def run_env(args):
    import lance

    env = args.env
    root = Path(args.probe_root)
    dataset = dataset_path(env, args.dataset)
    ds = lance.dataset(dataset)
    n_rows = ds.count_rows()
    keep = np.arange(0, n_rows, args.stride)
    print(f"[probe] {env}: {n_rows} rows, stride {args.stride} -> {len(keep)} frames")

    targets = build_targets(env, ds, keep, root / "cache" / f"{env}_targets_s{args.stride}.npz",
                            dataset)
    ep = targets["ep"]
    masks = episode_split(ep, args.seed)
    print(f"[probe] episode split (seed {args.seed}): "
          f"train {int(masks[0].sum())} / val {int(masks[1].sum())} / "
          f"test {int(masks[2].sum())} frames")

    models = MODELS[env] if not args.models else [
        (s.split("=")[0], *s.split("=")[1].split(":", 1)) for s in args.models
    ]
    results = {
        "env": env, "dataset": str(dataset), "stride": args.stride,
        "seed": args.seed, "n_frames": len(keep),
        "n_test": int(masks[2].sum()), "models": {},
    }
    for name, run, policy in models:
        z = build_latents(env, ds, keep, run, policy, args.logs_root,
                          root / "cache" / f"{env}_{_ckpt_tag(run, policy)}_s{args.stride}.npz",
                          args.batch_size)
        entry = {"run": run, "policy": policy, "latent_dim": int(z.shape[1]),
                 "targets": {}}
        for tname in TARGET_LABELS[env]:
            Y = targets[tname].astype(np.float32)
            t0 = time.time()
            lin = ridge_probe(z, Y, masks)
            mlp = mlp_probe(z, Y, masks, seed=args.seed)
            entry["targets"][tname] = {"dim": int(Y.shape[1]),
                                       "linear": lin, "mlp": mlp}
            print(f"[probe] {env} | {name:18s} | {tname:22s} "
                  f"lin R2 {lin['r2']:+.3f} mse {lin['mse']:.3e} | "
                  f"mlp R2 {mlp['r2']:+.3f} mse {mlp['mse']:.3e} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        results["models"][name] = entry

    out = root / "results" / f"probe_{env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"[probe] wrote {out}")
    print(render_env(results))


# -------------------------------------------------------------------- report


def render_env(res):
    env = res["env"]
    names = list(res["models"])
    head = f"\n=== {env}  (stride {res['stride']}, {res['n_test']} test frames, seed {res['seed']}) ==="
    lines = [head]
    for name in names:
        m = res["models"][name]
        lines.append(f"    {name}: {m['run']}/checkpoints/{m['policy']} "
                     f"(D={m['latent_dim']})")
    cols = "".join(f"{n[:24]:>50s}" for n in names)
    lines.append(f"{'target':44s}{'dim':>4s}" + cols)
    sub = "".join(f"{'lin MSE':>12s}{'lin R2':>9s}{'mlp MSE':>12s}{'mlp R2':>9s}"
                  + " " * 8 for _ in names)
    lines.append(" " * 48 + sub)
    for tname, label in TARGET_LABELS[env].items():
        row = f"{label:44s}{res['models'][names[0]]['targets'][tname]['dim']:>4d}"
        for n in names:
            t = res["models"][n]["targets"][tname]
            row += (f"{t['linear']['mse']:>12.3e}{t['linear']['r2']:>9.3f}"
                    f"{t['mlp']['mse']:>12.3e}{t['mlp']['r2']:>9.3f}" + " " * 8)
        lines.append(row)
    return "\n".join(lines)


def run_report(args):
    root = Path(args.probe_root)
    blocks, missing = [], []
    for env in ENVS:
        p = root / "results" / f"probe_{env}.json"
        if not p.exists():
            missing.append(env)
            continue
        blocks.append(render_env(json.loads(p.read_text())))
    report = "\n".join(blocks) + "\n"
    if missing:
        report += f"\n(missing envs: {', '.join(missing)})\n"
    out = root / "results" / "probe_report.txt"
    out.write_text(report)
    print(report)
    print(f"[report] wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--env", choices=ENVS)
    p.add_argument("--report", action="store_true",
                   help="render the combined table from existing per-env jsons")
    p.add_argument("--dataset", default=None, help="override the env's lance path")
    p.add_argument("--logs-root", default=".")
    p.add_argument("--probe-root", default=PROBE_ROOT)
    p.add_argument("--models", nargs="*", default=None, metavar="NAME=RUN:POLICY",
                   help="override the checkpoints under probe")
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.report:
        return run_report(args)
    if not args.env:
        p.error("--env or --report is required")
    torch.manual_seed(args.seed)
    run_env(args)


if __name__ == "__main__":
    main()
