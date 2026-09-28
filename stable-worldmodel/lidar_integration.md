# LiDAR Sensor Interface

This document describes the LiDAR sensor that this fork of **stable-worldmodel**
adds, the datasets it produces, and the contract that the point-cloud world
models in the parent repository (`train.py`, `eval_lidar.py`, `eval_3dtarget.py`
and their configs under `config/`) rely on to train, validate and evaluate on them.

`stable-worldmodel` is the **producer**: it casts the scans, writes them to disk,
and exposes them live at evaluation time. It does **not** contain the point-cloud
encoders or world models; those live in the parent repository.

---

## 1. What the interface is

A `gymnasium` wrapper (`AddRaycastLidarWrapper`) casts rays each `reset()`/`step()`
and writes an **unordered point cloud** to `info['lidar']`:

- Shape: **`(n_rays, 3)`** float32, xyz in **metres**.
- **Misses** (rays that hit nothing within `max_range`) are filled with a sentinel
  `miss_value` (default **`-1.0`**) in all 3 coordinates.
- Fixed size (`n_rays` constant) so it batches like an image and stores as a
  fixed-width column. **Miss filtering / subsampling is the encoder's job, not the
  data format's.**
- Frame: `'sensor'` (default; relative to the sensor pose) or `'world'` (absolute).
- Optional per-point colors (`add_rgb=True`): `info['lidar_rgb']`, `(n_rays, 3)`,
  sampled from the mount camera's image through a pinhole projection; misses get
  `rgb_fill_value` (default `-1.0`).
- Goal conditioning (task mode only, `add_goal_lidar=True`): `info['goal_lidar']`,
  same shape, is the cloud of the goal configuration (see §7).

Two raycast backends: `warp` (GPU) and `mujoco` (CPU, `mj_multiRay`). They agree
up to mesh tessellation: `mujoco` casts against analytic primitives while `warp`
casts against triangle meshes, so a few rays on curved silhouettes differ. Use the
backend that generated a dataset when evaluating on it (the configs do).

---

## 2. Dataset schema (Lance)

The LiDAR tables are produced by `scripts/data/lidar_from_h5.py`, which replays an
existing swm-format `.h5` image dataset state by state and casts the scan against
each restored state (§3). The output `.lance` table holds every source column plus
the computed LiDAR columns. The parent repository expects the tables under
`$PLWM_DATA_ROOT` (default `<repo>/data`) as `cube.lance`, `two_room.lance`,
`pusht.lance` and `reacher.lance`.

Per-step columns relevant to a world model:

| Column | On-disk type | Meaning |
|---|---|---|
| `lidar` | `fixed_size_list<float32>[n_rays*3]` | **flattened** cloud; `reshape(-1, 3)` → `(n_rays, 3)`, misses = `-1.0` |
| `lidar_rgb` | `fixed_size_list<float32>[n_rays*3]` | per-point colors, row-aligned with `lidar` (only when `add_rgb` is on) |
| `pixels` | `binary` (JPEG) → decodes to `(3, H, W) uint8` | the source RGB view, kept for interpretability |
| `proprio/*` | `fixed_size_list<float32>` | robot proprioception (joint pos/vel, gripper, effector) |
| `privileged/*` | `fixed_size_list<float32>` | block/target poses (for analysis only) |
| `qpos`, `qvel` | `fixed_size_list<float32>` | full MuJoCo state (MuJoCo-native sources) |
| `action` | `fixed_size_list<float32>` | expert action |
| `reward`, `terminated`, `truncated`, `success` | `[1]` | step flags |
| `episode_idx`, `step_idx` | `int32` | indices |

Mirror environments (Two-Room, Push-T) carry their 2D state columns
(`pos_agent`, `pos_target`, `block_pose`, `goal_pose`, ...) instead of
`qpos`/`qvel`.

Notes:
- `lidar` is stored **flat** (`n_rays*3`). E.g. 128×128 grid → `n_rays = 16384` →
  column width `49152`; `reshape(-1, 3)` → `(16384, 3)`. The tables of this
  repository use a 100×100 grid (`n_rays = 10000`, column width `30000`).
- **No `goal_lidar` in the training tables.** The scans are cast in
  `data_collection` mode where the env has no goal; the training goal comes from
  future frames (§7).
- `n_rays` is **constant within a dataset**. You cannot mix resolutions in one
  `.lance`. Different `h_res`/`v_res` or `az`/`el` ⇒ different datasets.

---

## 3. (Re)generating data

`scripts/data/lidar_from_h5.py` is a Hydra script. Each environment has its own
config under `scripts/data/config/`, whose `lidar:` block is exactly the sensor of
the corresponding table:

| Environment | Config | Env id | Sensor |
|---|---|---|---|
| OGB-Cube | `lidar_from_h5.yaml` (default) | `swm/OGBCube-v0` | camera-matched grid on `front_pixels`, `warp`, colors on |
| Two-Room | `lidar_from_h5_tworoom.yaml` | `swm/TwoRoomLidar-v1` | oblique 45/45 `iso` camera, `mujoco`, no colors |
| Push-T | `lidar_from_h5_pusht.yaml` | `swm/PushTLidar-v1` | oblique 45/45 `iso` camera, `mujoco`, no colors |
| Reacher | `lidar_from_h5_reacher.yaml` | `swm/Reacher3D-v0` | oblique 45/45 `iso` camera, `mujoco`, no colors |

Run from the repository root with EGL:

```bash
MUJOCO_GL=egl pixi run python stable-worldmodel/scripts/data/lidar_from_h5.py \
    input=data/<source>.h5 output=data/cube.lance

MUJOCO_GL=egl pixi run python stable-worldmodel/scripts/data/lidar_from_h5.py \
    --config-name=lidar_from_h5_tworoom \
    input=data/<source>.h5 output=data/two_room.lance
```

`write_mode=append` (the default) resumes an interrupted run, and
`ep_start`/`ep_end` shard the episode range across processes. Any `lidar.*` key can
be overridden on the command line, but the shipped values reproduce the tables the
world models were trained on. Full wrapper parameters are documented in
`stable_worldmodel/wrapper/lidar.py::AddRaycastLidarWrapper`. Key ones:

| Param | Default | Notes |
|---|---|---|
| `backend` | `mujoco` | `warp` = GPU, 5–25× faster at high res |
| `h_res`, `v_res` | 128, 32 | grid scan; `n_rays = h_res * v_res` |
| `h_fov`, `v_fov` | (−180,180), (−15,15) | degrees |
| `match_camera_fov` | False | build the ray grid from the mount camera's own fov (`h_fov`/`v_fov` are then ignored) |
| `az`, `el` | — | explicit per-ray arrays or file paths (degrees); override grid |
| `frame` | `sensor` | `sensor` (ego, translation-invariant) or `world` |
| `mount_camera` | `front_pixels` | sensor rides the camera pose; or `mount_body`/`mount_site`/explicit |
| `max_range` | 20.0 | metres; beyond ⇒ miss |
| `miss_value` | −1.0 | sentinel for no-hit rays |
| `min_alpha` | None (cube config sets 0.5) | skip translucent geoms (goal markers, arena walls) |
| `add_rgb`, `rgb_fill_value` | False, −1.0 | per-point colors from the mount camera |
| `ray_noise_std`, `range_noise_std` | 0.0 | sensor noise (radians / metres) |

---

## 4. Loading the data (train / val)

```python
import stable_worldmodel as swm

ds = swm.data.load_dataset(
    'data/cube.lance', num_steps=4, frameskip=5, keys_to_load=['lidar', 'action']
)
ep = ds.load_episode(0)          # dict of full-episode columns
lidar = ep['lidar']              # (T, n_rays*3) float32

sample = ds[0]
sample['lidar']                  # (num_steps, n_rays*3)  <-- single key, time-first
```

- Multi-step returns **one `lidar` key of shape `(num_steps, n_rays*3)`** (time
  first). Reshape to `(num_steps, n_rays, 3)` with `swm.data.reshape_lidar`.
- Val is a held-out split of the same dataset (e.g. `spt.data.random_split`), so it
  uses the exact same columns/loader: whatever works for train works for val.

**Helpers provided** (in `stable_worldmodel/data/utils.py`, exported as
`swm.data.*`):

```python
swm.data.reshape_lidar(flat, n_dims=3)      # (..., n_rays*3) -> (..., n_rays, 3)
swm.data.drop_lidar_misses(points, miss_value=-1.0)  # (n_rays,3) -> (n_valid,3)
```

> `drop_lidar_misses` yields a **variable-length** cloud, which suits visualization
> and single samples but **not** batched training (it will not collate). For
> training, keep fixed `n_rays` and either pass a **mask** (see §5) or pack the
> valid points of every cloud into one flat batch.

---

## 5. Encoder input contract

Feed a point encoder the **fixed `(n_rays, 3)` cloud plus a validity mask**:

```python
pts = swm.data.reshape_lidar(sample['lidar'])        # (num_steps, n_rays, 3)
mask = (pts != MISS_VALUE).all(dim=-1)               # (num_steps, n_rays) bool, True = real hit
# optional: zero the misses so they don't leak coordinates
pts = torch.where(mask.unsqueeze(-1), pts, torch.zeros_like(pts))
```

- A PointNet / Point-JEPA-style encoder should **mask or FPS-sample** the valid
  points. Do not rely on a fixed count of *valid* points, since it varies per
  frame; the fixed count is `n_rays` (hits + misses).
- If you prefer fixed-N valid points, do farthest-point-sampling over the masked
  set inside the dataloader or encoder.

The parent repository takes the packing route: its data configs set
`obs.invalid_value: -1.0`, and the training collate function drops every point
whose coordinates all equal it before packing the clouds into a batched
`{coord, batch, feat}` dict.

---

## 6. Normalization (a common source of silent errors)

A z-score column normalizer applied to `lidar` lets the `-1.0` miss sentinels
**poison the mean/std**.

Do one of:
- **Don't** z-score `lidar`, and instead scale xyz yourself, e.g. divide by
  `max_range` (bounded to ~[−1, 1]) **after masking**:
  ```python
  pts = pts / MAX_RANGE          # apply only where mask is True
  ```
- Or compute mask-aware stats (mean/std over valid points only).

Whatever you choose, **exclude misses from statistics**. The parent repository's
`train.py` z-scores every loaded column except the observation keys, so `lidar`
reaches the encoder raw, with misses removed by the collate function.

---

## 7. Goal conditioning

- **Train time (from the dataset):** pass `goal_keys={'lidar': 'goal_lidar'}` to
  `GoalDataset` (the goal-sampling wrapper; see
  `stable_worldmodel/data/dataset.py`). It copies a **future frame's `lidar`** into
  `sample['goal_lidar']`, exactly as it does for pixels. This works generically, so
  no stored `goal_lidar` is needed.
- **Eval time, goal from the env:** in **task mode** with `add_goal_lidar=True` the
  wrapper casts a second scan against the goal state and writes
  **`info['goal_lidar']`** (constant per episode).
- **Eval time, goal from the dataset:** the parent repository's evaluations replay
  a start state and take the goal cloud from a later frame of the same recorded
  episode, so they set `add_goal_lidar: false` and receive the goal as
  `goal_lidar` from the dataset instead of from the sensor.

---

## 8. Eval / planning integration

Evaluation builds a `World` with the LiDAR wrapper as a pre-wrapper, driven by the
sensor block of the eval config (in the parent repository, `lidar.sensor` in
`config/eval/lidar_<env>.yaml`):

```python
world = swm.World(**cfg.world, image_shape=(224, 224),
                  pre_wrappers=swm.wrapper.make_lidar_pre_wrappers(cfg.lidar.sensor))
```

During the rollout the policy receives the stacked info dict via
`policy.get_action(infos)` (`World._get_actions` in
`stable_worldmodel/world/world.py`). It contains:

- `infos['lidar']`: shape **`(num_envs, history, n_rays, 3)`** (leading env + time
  dims added by the pool/wrapper; reshape/mask as in §5).
- `infos['goal_lidar']`: the goal cloud, same trailing shape.

The world model and planning cost consume `lidar` (+ `goal_lidar`) instead of
`pixels`/`goal`; the parent repository's `LidarPlanAdapter` (`eval_lidar.py`) does
this conversion.

**Critical:** the eval sensor config (`n_rays`, `h_fov`/`v_fov` or
`match_camera_fov`, `mount_*`, `frame`, `max_range`, `miss_value`, `min_alpha`,
`backend`) **must match the one that generated the training table**, or the encoder
sees a distribution shift. Keep each eval config's sensor block identical to the
`lidar:` block of the matching `scripts/data/config/lidar_from_h5*.yaml`.

---

## 9. Storage budgeting

Misses are stored too (fixed width). Per-frame LiDAR bytes ≈ `n_rays * 3 * 4`:

| Grid | n_rays | bytes/frame | 200 steps × 10k eps |
|---|---|---|---|
| 64×64 | 4 096 | 49 KB | ~98 GB |
| 128×128 | 16 384 | 196 KB | ~393 GB |
| 256×256 | 65 536 | 786 KB | ~1.5 TB |
| 512×512 | 262 144 | 3.1 MB | ~6.3 TB |

Pick `n_rays` for your disk budget. This, not compute, is the practical ceiling
(Warp makes 512² cheap to *compute* at ~14 ms/cast, but ~6 TB to *store*).

---

## 10. API quick reference

| Symbol | Location |
|---|---|
| `AddRaycastLidarWrapper` | `stable_worldmodel/wrapper/lidar.py` |
| `make_lidar_pre_wrappers(cfg)` | `stable_worldmodel/wrapper/lidar.py` (→ `swm.wrapper.*`) |
| `reshape_lidar`, `drop_lidar_misses` | `stable_worldmodel/data/utils.py` (→ `swm.data.*`) |
| Warp backend internals | `stable_worldmodel/wrapper/_lidar_warp.py` |
| LiDAR mirror envs | `stable_worldmodel/envs/two_room/lidar_env.py`, `stable_worldmodel/envs/pusht/lidar_env.py`, `stable_worldmodel/envs/reacher3d.py` |
| Dataset generation script + configs | `scripts/data/lidar_from_h5.py`, `scripts/data/config/lidar_from_h5*.yaml` |
| Panel renderer for eval videos | `stable_worldmodel/plot/lidar_render.py` |
| Visualizer (polyscope) | `swm-lidar-viz` console command (`stable_worldmodel/viz/lidar.py`) |
| Tests | `tests/test_lidar.py` |

---

## 11. Checklist for a new point-cloud model

- [ ] Generate the table at the chosen `n_rays` (§3); confirm a `lidar` column with
      `swm inspect` / `python -m stable_worldmodel.cli inspect <path>`.
- [ ] **Load `lidar`** as the observation modality (`keys_to_load`).
- [ ] Reshape `(num_steps, n_rays*3)` → `(num_steps, n_rays, 3)` and build a
      **miss mask** or drop the misses while packing (§5).
- [ ] **Do not z-score `lidar`** with a column normalizer: mask misses, scale by
      `max_range` or use mask-aware stats (§6).
- [ ] Train goal conditioning: `goal_keys={'lidar': 'goal_lidar'}` (§7).
- [ ] Eval: make the model/cost read `infos['lidar']` and `infos['goal_lidar']`,
      and keep the eval sensor config identical to the training table's (§8).
- [ ] Sanity-check visually with the `swm-lidar-viz` command (installed by the
      `viz` extra): `swm-lidar-viz --dataset <path> --episode 0`.

If those hold, train, val and eval see the same representation: train and val
share the dataset loader, and eval gets the same scan live from the env.
