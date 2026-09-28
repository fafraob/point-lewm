#!/usr/bin/env python
"""Package a dinowm_swm run for publication (Hugging Face model repo / tarball)
in the layout upstream github.com/gaoyuezhou/dino_wm loads:

    <out>/
      hydra.yaml                 resolved training config (upstream plan.py reads it)
      checkpoints/model_latest.pth   epoch + predictor + action_encoder + proprio_encoder
      swm_h5_norm_stats.json     action / proprio normalization used in training
      README.md                  model card with the loading snippet

    python scripts/export_dinowm_checkpoint.py experiment_logs/dinowm_pusht publish/dinowm_pusht
    python scripts/export_dinowm_checkpoint.py <run> <out> --ckpt model_latest.pth

``<run>`` is the dinowm_swm run folder (``run_dir`` of dinowm_swm.train, by
default ``$PLWM_LOGS_ROOT/dinowm_<env>``).

The optimizer states (2x the model size) are dropped; everything else is the
file upstream wrote, so `torch.load(..., weights_only=False)` on their side
returns the same pickled modules (their code must be importable as
`models.*`, exactly as for their own released checkpoints).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dinowm_swm import add_upstream_to_path  # noqa: E402

add_upstream_to_path()
import torch  # noqa: E402

KEEP = ("epoch", "predictor", "action_encoder", "proprio_encoder", "decoder", "encoder")

CARD = """---
license: mit
tags: [world-model, dino-wm, planning, stable-worldmodel]
---
# DINO-WM ({env}, no proprioception)

DINO-WM (Zhou et al., 2024, [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)) trained with the
unmodified official code (https://github.com/gaoyuezhou/dino_wm, commit {upstream}) on the
stable-worldmodel `{dataset}` dataset (LeWM benchmark). Frozen DINOv2 ViT-S/14 patch tokens,
ViT predictor (depth 6, 16 heads), 10-d action embedding concatenated onto every patch token,
history {num_hist}, frameskip {frameskip}, batch {batch}, {epochs} epochs, mixed precision {mp}.
No proprioception reaches the model: the proprio stream is a constant zero placeholder
(`proprio_key: null`), and the VQ-VAE decoder (visualisation only) was not trained.

## Files
* `checkpoints/model_latest.pth` -- dict with `epoch`, `predictor`, `action_encoder`, `proprio_encoder`
  (pickled `torch.nn.Module`s from `models/` of the dino_wm repo; the frozen encoder is rebuilt from
  `hydra.yaml`, exactly as `plan.py` does).
* `hydra.yaml` -- the resolved training config.
* `swm_h5_norm_stats.json` -- action mean/std (population std over all frames) applied before the
  action encoder. Actions fed to the model must be z-scored with these.

## Loading with the dino_wm code
```python
import torch, hydra
from omegaconf import OmegaConf
from models.visual_world_model import VWorldModel          # dino_wm repo on sys.path

cfg = OmegaConf.load("hydra.yaml")
ckpt = torch.load("checkpoints/model_latest.pth", map_location="cuda", weights_only=False)
encoder = hydra.utils.instantiate(cfg.encoder).cuda()     # torch.hub dinov2_vits14
wm = VWorldModel(image_size=cfg.img_size, num_hist=cfg.num_hist, num_pred=cfg.num_pred,
                 encoder=encoder, proprio_encoder=ckpt["proprio_encoder"],
                 action_encoder=ckpt["action_encoder"], decoder=None, predictor=ckpt["predictor"],
                 proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
                 concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat,
                 num_proprio_repeat=cfg.num_proprio_repeat).cuda()
wm.eval()   # VWorldModel.eval() returns None, do not chain it
# obs["visual"]: (B, T, 3, 224, 224) in [0, 1] -> Normalize(0.5, 0.5); obs["proprio"]: zeros (B, T, 1)
# act: (B, T, frameskip * action_dim) z-scored; z_obses, z = wm.rollout(obs_0, act)
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--ckpt", default="model_latest.pth")
    args = ap.parse_args()
    run, out = Path(args.run_dir), Path(args.out_dir)
    src = run / "checkpoints" / args.ckpt
    payload = torch.load(src, map_location="cpu", weights_only=False)
    slim = {k: payload[k] for k in KEEP if k in payload and payload[k] is not None}
    dropped = sorted(set(payload) - set(slim))
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(slim, out / "checkpoints" / "model_latest.pth")
    shutil.copy(run / "hydra.yaml", out / "hydra.yaml")
    stats = run / "swm_h5_norm_stats.json"
    if stats.exists():
        shutil.copy(stats, out / stats.name)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(run / "hydra.yaml")
    upstream = (REPO / "third_party" / "dino_wm" / "UPSTREAM_COMMIT.txt").read_text().split()[0][:12]
    (out / "README.md").write_text(CARD.format(
        env=cfg.env.name.replace("swm_", ""), upstream=upstream,
        dataset=os.path.basename(str(cfg.env.dataset.data_path)), num_hist=cfg.num_hist,
        frameskip=cfg.frameskip, batch=cfg.training.batch_size, epochs=payload.get("epoch"),
        mp=cfg.get("mixed_precision", "no")))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"exported epoch {payload.get('epoch')} to {out} ({size:.0f} MB); kept {sorted(slim)}, dropped {dropped}")


if __name__ == "__main__":
    main()
