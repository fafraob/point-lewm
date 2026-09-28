"""Train a 3-D-target -> latent goal model on a precomputed cache.

    python -m target2latent.train --env cube \\
        --cache $PLWM_T2L_ROOT/cache/cube_point_delta_jepa_s5.npz \\
        --head mlp --out $PLWM_T2L_ROOT/models/cube_point_delta_jepa_mlp

(from the repository root; ``--use-z`` for the z-conditioned variants,
``--head shortcut`` for the diffusion heads). Every episode named by a
``starts_*.json`` file in ``--eval-dir`` (default: the shipped ``eval_starts/``)
is excluded from training.

The cache lives on the GPU, so an epoch is a few hundred optimizer steps and a
run takes minutes -- the expensive part of this project is the closed-loop
evaluation, not this fit.

The four architectures are ``--head {mlp,shortcut}`` x ``[--use-z]``.

Offline metrics (held-out episodes)
-----------------------------------
``r2_point``      1 - MSE(point estimate) / Var(z): the share of latent variance
                  the goal spec explains.
``r2_best_of_K``  (shortcut) same with the best of K samples -- a diffusion head
                  is allowed to be bad at the point estimate, so validation
                  model selection uses this quantity for it.
``retrieval_cm``  predict a latent, find its nearest latent neighbour among
                  held-out frames, report how far *that frame's* objects are
                  from the queried pose. A latent can be far from the true one
                  in L2 and still name the right pose, or close and name the
                  wrong one -- this is the number that matters for goal
                  grounding. ``retrieval_cm_true`` is the same lookup with the
                  true latent (the encoder's own ceiling), ``retrieval_cm_floor``
                  the nearest pool frame by pose (pure pool density).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

import numpy as np
import torch

from target2latent import data as D
from target2latent.envs import EVAL_STARTS_DIR, SPECS
from target2latent.models import build_model


# ------------------------------------------------------------------ metrics


@torch.no_grad()
def offline_metrics(model, cond, z, coords, ks=(1,), pool=8000, queries=2000,
                    seed=0, chunk=512):
    """Latent-space + retrieval metrics on held-out frames (module docstring)."""
    dev = cond.device
    D_ = z.shape[1]
    var = z.var(dim=0, unbiased=False).mean().item()
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {"latent_var_per_dim": var, "n_val": int(len(z))}

    for k in ks:
        se_best, se_point, n = 0.0, 0.0, 0
        for i in range(0, len(cond), chunk):
            c, zz = cond[i : i + chunk], z[i : i + chunk]
            goals = model.goals(c, n=k, generator=g)  # (b, k, D)
            sq = ((goals - zz.unsqueeze(1)) ** 2).sum(-1)  # (b, k)
            se_best += sq.min(dim=1).values.sum().item()
            se_point += sq[:, 0].sum().item()
            n += len(c)
        out[f"mse_best_of_{k}"] = se_best / (n * D_)
        out[f"r2_best_of_{k}"] = 1.0 - out[f"mse_best_of_{k}"] / var
        if k == 1:
            out["mse_point"] = se_point / (n * D_)
            out["r2_point"] = 1.0 - out["mse_point"] / var

    # -- retrieval faithfulness: pool and queries are DISJOINT --------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(z))
    n_q = min(queries, len(z) // 2)
    n_pool = min(pool, len(z) - n_q)
    q_idx = torch.as_tensor(np.sort(perm[:n_q]), device=dev)
    pool_idx = torch.as_tensor(np.sort(perm[n_q : n_q + n_pool]), device=dev)
    z_pool, c_pool = z[pool_idx], coords[pool_idx]

    def retrieve(z_query, q_rows):
        errs = []
        for i in range(0, len(z_query), chunk):
            d = torch.cdist(z_query[i : i + chunk], z_pool)
            errs.append((c_pool[d.argmin(dim=1)] - coords[q_rows[i : i + chunk]]).norm(dim=1))
        return torch.cat(errs)

    pred = model.point_estimate(cond[q_idx])
    err = retrieve(pred, q_idx)
    out["retrieval_cm"] = float(err.mean() * 100)
    out["retrieval_cm_median"] = float(err.median() * 100)
    err_t = retrieve(z[q_idx], q_idx)
    out["retrieval_cm_true"] = float(err_t.mean() * 100)
    dp = []
    for i in range(0, len(q_idx), chunk):
        dp.append(torch.cdist(coords[q_idx[i : i + chunk]], c_pool).min(dim=1).values)
    out["retrieval_cm_floor"] = float(torch.cat(dp).mean() * 100)
    return out


@torch.no_grad()
def latency(model, cond, ns=(1,), reps=50):
    """Microseconds per goals() call for a batch of 50 (the eval's num_envs)."""
    c = cond[:50].contiguous()
    out = {}
    for n in ns:
        for _ in range(5):
            model.goals(c, n=n)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            model.goals(c, n=n)
        torch.cuda.synchronize()
        out[f"us_per_call_n{n}"] = (time.time() - t0) / reps * 1e6
    return out


# -------------------------------------------------------------------- train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=sorted(SPECS))
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--head", required=True, choices=["mlp", "shortcut"])
    p.add_argument("--use-z", action="store_true",
                   help="condition on the latent of the current observation")
    p.add_argument("--offsets", default="-25,-20,-15,-10,-5,5,10,15,20,25",
                   help="context->goal step offsets for --use-z pairs "
                        "(the eval's goal_offset_steps is 25)")
    p.add_argument("--max-pairs", type=int, default=4000000)
    p.add_argument("--eval-dir", default=EVAL_STARTS_DIR,
                   help="dir with starts_*.json; those episodes are excluded from training "
                        "(default: the shipped eval_starts/)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-bands", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--flow-steps", type=int, default=1,
                   help="shortcut: default sampling steps (power of two)")
    p.add_argument("--ema", type=float, default=0.999,
                   help="shortcut: EMA decay of the bootstrap-target network "
                        "(the paper's bootstrap_ema; 0 disables -- measured unstable)")
    p.add_argument("--pos-jitter", type=float, default=0.005,
                   help="std (m) of Gaussian jitter on the position channels; "
                        "regularizes against position-as-lookup-key memorization")
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = SPECS[args.env]

    cache, meta = D.load_cache(args.cache)
    assert meta.get("env", args.env) == args.env, \
        f"cache was precomputed for env {meta.get('env')!r}, not {args.env!r}"
    # conditioning-layout guard: a cache with a different rotation layout
    # (quaternion columns, or the wrong rot width) must be rebuilt, not
    # silently trained on
    assert "quat" not in cache, \
        "cache layout mismatch: delete it (+ its .shards/) and re-run precompute"
    cache_rot = cache["rot"].shape[1] if "rot" in cache else 0
    assert cache_rot == spec.rot_dim, \
        f"cache rot channels ({cache_rot}) != spec.rot_dim ({spec.rot_dim}): re-run precompute"
    eval_eps = D.eval_episode_set(args.eval_dir)
    train_m, val_m = D.make_splits(cache["ep"], eval_eps)

    if args.use_z:
        offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
        now, goal = D.build_pairs_multi(
            cache["ep"], cache["step"], offsets, max_pairs=args.max_pairs, seed=args.seed
        )
        # a pair belongs to a split iff BOTH ends do (no leakage across the split)
        tr = train_m[now] & train_m[goal]
        va = val_m[now] & val_m[goal]
        cond_tr = D.assemble(cache, True, rows=goal[tr], z_now_rows=now[tr])
        cond_va = D.assemble(cache, True, rows=goal[va], z_now_rows=now[va])
        z_tr, z_va = cache["z"][goal[tr]], cache["z"][goal[va]]
        coords_va = cache["coords"][goal[va]]
    else:
        cond_tr = D.assemble(cache, False, rows=train_m)
        cond_va = D.assemble(cache, False, rows=val_m)
        z_tr, z_va = cache["z"][train_m], cache["z"][val_m]
        coords_va = cache["coords"][val_m]
    print(f"[train] cond {cond_tr.shape} -> z {z_tr.shape}   val {cond_va.shape}")

    z_mean = z_tr.mean(axis=0)
    z_scale = float(np.sqrt(((z_tr - z_mean) ** 2).mean()))
    print(f"[train] z scalar scale {z_scale:.4f}")

    rot_dim = spec.rot_dim
    model = build_model(
        args.head, latent_dim=z_tr.shape[1], z_mean=z_mean, z_scale=z_scale,
        norm_center=spec.norm_center, norm_scale=spec.norm_scale,
        coord_dim=spec.coord_dim, rot_dim=rot_dim, use_z=args.use_z,
        width=args.width, hidden=args.hidden, depth=args.depth,
        num_bands=args.num_bands, dropout=args.dropout,
        flow_steps=args.flow_steps, pos_jitter=args.pos_jitter,
    ).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f"[train] {args.env}/{args.head}{'+z' if args.use_z else ''}: "
          f"{n_par / 1e6:.2f} M params")

    ema_model = None
    if args.head == "shortcut" and args.ema > 0:
        ema_model = copy.deepcopy(model).eval()
        ema_model.requires_grad_(False)
        # plain attribute, NOT a registered submodule: keeps state_dict,
        # parameters() and the optimizer untouched
        object.__setattr__(model, "_target_net", ema_model)

    bank = D.TensorBank(cond_tr, z_tr, device=dev)
    cond_va_t = D.to_gpu(cond_va, dev)
    z_va_t = D.to_gpu(z_va, dev)
    coords_va_t = D.to_gpu(coords_va, dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(bank.n // args.batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
        pct_start=0.05, div_factor=10.0, final_div_factor=100.0,
    )
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    # Model selection follows the quantity the planner uses: the point estimate
    # for the MLP, the best of 16 samples for the diffusion head (which is
    # allowed to be bad at the conditional mean).
    sel_k = 16 if args.head == "shortcut" else 1
    best = (-float("inf"), -1, None)
    history = []

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        agg, nb = {}, 0
        for c, zz in bank.batches(args.batch_size, generator=gen):
            loss, stats = model.loss(c, zz)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.lerp_(pm, 1.0 - args.ema)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            agg["loss"] = agg.get("loss", 0.0) + float(loss)
            nb += 1
        row = {"epoch": ep, "lr": sched.get_last_lr()[0],
               **{k: v / nb for k, v in agg.items()}}
        if ep % args.val_every == 0 or ep == args.epochs - 1:
            model.eval()
            m = offline_metrics(model, cond_va_t, z_va_t, coords_va_t,
                                ks=(1, sel_k) if sel_k > 1 else (1,), seed=args.seed)
            sel = m[f"r2_best_of_{sel_k}"]
            row.update({"val_r2_point": m["r2_point"], "val_r2_sel": sel,
                        "val_retr_cm": m["retrieval_cm"]})
            if sel > best[0]:
                best = (sel, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
            print(f"[train] ep {ep:4d} " + "  ".join(
                f"{k}={v:.4f}" for k, v in row.items() if k != "epoch"), flush=True)
        history.append(row)
    print(f"[train] done in {time.time() - t0:.0f}s")

    # Restore the best-validation weights: with the LR schedule and the
    # lookup-key overfitting the last epoch is routinely not the best one.
    if best[2] is not None and best[1] != args.epochs - 1:
        print(f"[train] restoring best epoch {best[1]} (val_r2={best[0]:.4f})")
        model.load_state_dict(best[2])

    cfg = {
        "env": args.env, "head": args.head, "use_z": bool(args.use_z),
        "latent_dim": int(z_tr.shape[1]),
        "z_mean": z_mean.tolist(), "z_scale": z_scale,
        "norm_center": list(spec.norm_center), "norm_scale": spec.norm_scale,
        "coord_dim": spec.coord_dim, "rot_dim": rot_dim,
        "width": args.width, "hidden": args.hidden, "depth": args.depth,
        "num_bands": args.num_bands, "dropout": args.dropout,
        "flow_steps": args.flow_steps, "pos_jitter": args.pos_jitter,
        "selected_epoch": int(best[1]), "cache": args.cache,
        "encoder_run": meta.get("run"), "encoder_policy": meta.get("policy"),
        # folder-checkpoint root (--cache-dir) + modality; defaults apply
        # when a cache's meta lacks these fields
        "encoder_cache_dir": meta.get("cache_dir", ""),
        "encoder_modality": meta.get("modality", "pointcloud"),
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out / "model.pt")

    model.eval()
    metrics = offline_metrics(model, cond_va_t, z_va_t, coords_va_t,
                              ks=(1, 16) if args.head == "shortcut" else (1,),
                              seed=args.seed)
    metrics.update(latency(model, cond_va_t))
    metrics.update({"env": args.env, "head": args.head, "use_z": bool(args.use_z),
                    "args": vars(args)})
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "history.json", "w") as f:
        json.dump(history, f)
    print("[train] metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:24s} {v:.5f}")
    print(f"[train] wrote {out}")


if __name__ == "__main__":
    main()
