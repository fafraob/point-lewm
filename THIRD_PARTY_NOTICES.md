# Third-party code and assets

This repository builds on and redistributes the following third-party code. Each component keeps
its original license; the license texts are in the locations given.

| Component | Where | License | Notes |
|---|---|---|---|
| LeWorldModel (LeWM) reference implementation | repository root (`jepa.py`, `module.py`, `train.py`, `eval.py`, `utils.py` descend from it) | MIT, `LICENSE` | The point-cloud encoders, the Delta-JEPA objective, the frozen-encoder baselines, the target-to-latent module and the analysis scripts are additions of this work. |
| stable-worldmodel 0.1.1 | `stable-worldmodel/` | MIT (see `stable-worldmodel/pyproject.toml`) | Vendored fork; the additions are listed in `stable-worldmodel/README.md`. |
| Utonia (Point Transformer V3 cross-domain encoder) | `utonia/` | Apache-2.0 (code), `utonia/LICENSE` | Vendored package with two local patches documented in `utonia/README.md`. The pretrained weights are downloaded separately from the Hugging Face Hub (`Pointcept/Utonia`) and are licensed CC-BY-NC-4.0 (see below). |
| DINO-WM official implementation | `third_party/dino_wm/` | MIT, `third_party/dino_wm/LICENSE` | Unmodified upstream code at the commit recorded in `third_party/dino_wm/UPSTREAM_COMMIT.txt`; the `env/` directory of the upstream repository (simulators the paper does not use) was removed. `dinowm_swm/` is our adapter. |

Assets released with this work (Hugging Face, CC-BY-4.0): the four re-sensed point-cloud
datasets (`fafraob/point-cloud-{tworoom,reacher,pusht,ogb-cube}`, derived from the LeWM image
datasets below) and the checkpoint repositories of the same names (Point-LeWM, Point-Delta-JEPA and
their target-to-latent heads, our image Delta-JEPA re-implementation, DINO-WM trained with the
official code, and the Utonia-WM predictor; the Utonia backbone itself is not redistributed).

External assets used but not redistributed:

| Asset | Source | License |
|---|---|---|
| Image datasets (OGB-Cube, Two-Room, Push-T, Reacher) and the LeWM image checkpoints | Hugging Face collection `quentinll/lewm` | MIT |
| OGBench 1.2.1 (cube environment) | `ogbench` package | MIT |
| Push-T environment | Chi et al., Diffusion Policy | MIT |
| MuJoCo 3.10 | `mujoco` package | Apache-2.0 |
| Utonia pretrained backbone | Hugging Face `Pointcept/Utonia` | CC-BY-NC-4.0 (non-commercial) |
| DINOv2 ViT-S/14 weights (DINO-WM baseline) | torch hub `facebookresearch/dinov2` | Apache-2.0 |
| PyTorch 2.7 | `torch` package | BSD-3-Clause |
