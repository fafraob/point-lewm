#!/usr/bin/env python
"""Download the released checkpoints from the Hugging Face Hub into ``checkpoints/``.

    pixi run download-checkpoints                     # every environment
    pixi run download-checkpoints cube pusht          # a subset
    pixi run download-checkpoints cube --only point-lewm point-delta-jepa
    pixi run download-checkpoints --image-lewm        # also the image LeWM checkpoints (Maes et al.)
    pixi run download-checkpoints --utonia            # also the Utonia backbone (Utonia-WM)

One model repository per environment, ``fafraob/point-cloud-<env>`` (OGB-Cube:
``fafraob/point-cloud-ogb-cube``), lands in ``checkpoints/<env>/`` with the
repository's own layout::

    checkpoints/<env>/
        point-lewm/         weights.pt  config.json  target2latent/{mlp,mlp_z,shortcut,shortcut_z}/model.pt
        point-delta-jepa/   weights.pt  config.json  target2latent/{mlp,mlp_z,shortcut,shortcut_z}/model.pt
        image-delta-jepa/   weights.pt  config.json
        utonia-wm/          weights.pt  config.json
        dino-wm/            hydra.yaml  swm_h5_norm_stats.json  checkpoints/model_latest.pth

which is what the evaluation configs and sweeps address as
``cache_dir=. policy=<env>/<model>/weights.pt`` (``load_pretrained`` resolves
``<cache_dir>/checkpoints/<policy>``). The download is resumable and skips
files that are already complete.

``--image-lewm`` fetches the image LeWM checkpoints of Maes et al.
(``quentinll/lewm-<env>``) into ``checkpoints/image_lewm_<env>/`` and migrates
their parameter names to the pinned ``transformers`` version
(``scripts/convert_image_lewm_ckpts.py``). ``--utonia`` fetches the frozen
Utonia backbone (``Pointcept/Utonia``, CC-BY-NC-4.0) that Utonia-WM needs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO_ROOT / "checkpoints"
ENVS = ["tworoom", "reacher", "pusht", "cube"]
HF_REPO = {
    "tworoom": "fafraob/point-cloud-tworoom",
    "reacher": "fafraob/point-cloud-reacher",
    "pusht": "fafraob/point-cloud-pusht",
    "cube": "fafraob/point-cloud-ogb-cube",
}
MODELS = ["point-lewm", "point-delta-jepa", "image-delta-jepa", "utonia-wm", "dino-wm"]
LEWM_IMAGE = {"tworoom": "quentinll/lewm-tworooms", "reacher": "quentinll/lewm-reacher",
              "pusht": "quentinll/lewm-pusht", "cube": "quentinll/lewm-cube"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("envs", nargs="*", choices=ENVS + [[]], default=ENVS, help="default: all four")
    ap.add_argument("--only", nargs="+", choices=MODELS, default=None,
                    help="restrict to these model folders (default: the whole repository)")
    ap.add_argument("--image-lewm", action="store_true", help="also the image LeWM checkpoints")
    ap.add_argument("--utonia", action="store_true", help="also the Utonia backbone")
    args = ap.parse_args()
    envs = args.envs or ENVS

    for env in envs:
        patterns = ["README.md"] + [f"{m}/*" for m in args.only] if args.only else None
        path = snapshot_download(HF_REPO[env], local_dir=CHECKPOINTS / env, allow_patterns=patterns)
        print(f"[{env}] {HF_REPO[env]} -> {Path(path).relative_to(REPO_ROOT)}")

    if args.image_lewm:
        for env in envs:
            dest = CHECKPOINTS / f"image_lewm_{env}"
            snapshot_download(LEWM_IMAGE[env], local_dir=dest)
            print(f"[{env}] {LEWM_IMAGE[env]} -> {dest.relative_to(REPO_ROOT)}")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "convert_image_lewm_ckpts.py")]
                       + [str(CHECKPOINTS / f"image_lewm_{e}") for e in envs], check=True)

    if args.utonia:
        p = hf_hub_download("Pointcept/Utonia", "utonia.pth", local_dir=CHECKPOINTS / "utonia")
        print(f"[utonia] Pointcept/Utonia -> {Path(p).relative_to(REPO_ROOT)}  (CC-BY-NC-4.0)")


if __name__ == "__main__":
    main()
