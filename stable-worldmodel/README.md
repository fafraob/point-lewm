# stable-worldmodel (vendored fork)

This directory is a fork of [`stable-worldmodel`](https://github.com/galilai-group/stable-worldmodel)
(Maes et al., 2026; MIT license; upstream version 0.1.1), the platform whose environments,
datasets, planners and evaluation loop the paper builds on. It is installed as a local package by the
root `pixi.toml`, so nothing has to be cloned or installed separately.

What the fork adds on top of upstream:

| Component | Files |
|---|---|
| Simulated range sensor (raycast LiDAR as a Gymnasium wrapper; `mujoco` and `warp` backends) | `stable_worldmodel/wrapper/lidar.py`, `stable_worldmodel/wrapper/_lidar_warp.py` |
| MuJoCo mirror scenes for the 2D environments, so the sensor can scan them (`swm/TwoRoomLidar-v1`, `swm/PushTLidar-v1`) | `stable_worldmodel/envs/two_room/lidar_env.py`, `stable_worldmodel/envs/pusht/lidar_env.py`, `stable_worldmodel/envs/camera_xml.py` |
| 3D Reacher (`swm/Reacher3D-v0`), a MuJoCo scene replaying the released reacher dataset | `stable_worldmodel/envs/reacher3d.py` |
| Dataset re-sensing: replay a released image dataset (`.h5`) state by state and store the scan next to every original column | `scripts/data/lidar_from_h5.py`, `scripts/data/config/lidar_from_h5*.yaml` |
| Point-cloud helpers and visualisation | `stable_worldmodel/data/utils.py`, `stable_worldmodel/plot/lidar_render.py`, `stable_worldmodel/viz/lidar.py` |
| Tests | `tests/test_lidar.py` |

`lidar_integration.md` documents the sensor interface (point-cloud layout, sentinel for misses,
frames, goal handling) that the world-model code in the parent directory consumes.

Upstream's documentation site, notebooks, benchmark scripts and the collection/training scripts that
the paper does not use were removed to keep the release small; the Python package itself is complete.
