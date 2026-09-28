"""Raycast LiDAR wrapper for MuJoCo-backed environments.

A single wrapper, :class:`AddRaycastLidarWrapper`, writes an **unordered**
point cloud to ``info['lidar']`` as a fixed-size ``(n_rays, 3)`` float32 array
(one xyz per ray, misses filled with ``miss_value``). Fixed size keeps the
Lance column width constant and lets the cloud batch like a ``(H, W, 3)``
image; miss filtering / subsampling is left to the (downstream) point encoder.

Rays are cast with MuJoCo's ``mj_multiRay`` from a mounted sensor origin, so
360 spinning patterns, multiple vertical channels, and arbitrary
azimuth/elevation lists are all supported.

Backends (auto-detected at the first reset/step):

* **dm_control** — the unwrapped env (or its inner ``.env``) exposes a
  ``physics`` object.
* **raw MuJoCo** (e.g. OGBench) — the unwrapped env exposes ``model`` /
  ``_model`` (+ ``data`` / ``_data``).

Goal conditioning (Version A): when the wrapped env stashes a goal state on
``_cur_goal_qpos`` / ``_cur_goal_qvel`` (OGBench task mode), the wrapper casts a
second scan against that state and writes it to ``info['goal_lidar']``.

Colored points (``add_rgb=True``): the sensor rides the render camera, so each
ray maps to exactly one pixel of that camera's image — same origin, no
parallax. Every cast renders the mount camera, projects the hit rays through
the pinhole model (``cam_fovy``), and writes per-point colors to
``info['lidar_rgb']`` (and ``info['goal_lidar_rgb']``) as ``(n_rays, 3)``
float32 in ``[0, 1]``; misses and points outside the camera frustum are filled
with ``rgb_fill_value``. Costs one extra offscreen render per step.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np


def make_lidar_pre_wrappers(cfg: Any | None) -> list:
    """Build the ``pre_wrappers`` list for :class:`~stable_worldmodel.World`.

    ``cfg`` is a mapping (plain dict or OmegaConf node) of
    :class:`AddRaycastLidarWrapper` kwargs, or ``None`` to disable LiDAR. Pass
    the result straight to ``swm.World(..., pre_wrappers=...)``.
    """
    if cfg is None:
        return []
    try:  # unwrap OmegaConf nodes to plain python without a hard dependency.
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return [partial(AddRaycastLidarWrapper, **dict(cfg))]


def _load_angles(spec: Sequence[float] | str) -> np.ndarray:
    """Load an azimuth/elevation list from a sequence or a file path.

    Files may be ``.npy`` (numpy array) or plain text / ``.csv`` (one angle per
    line or comma-separated). Values are returned in degrees, unchanged.
    """
    if isinstance(spec, str):
        path = Path(spec)
        if path.suffix == '.npy':
            arr = np.load(path)
        else:
            arr = np.loadtxt(path, delimiter=',')
        return np.asarray(arr, dtype=np.float64).ravel()
    return np.asarray(spec, dtype=np.float64).ravel()


class AddRaycastLidarWrapper(gym.Wrapper):
    """Cast a fixed grid of rays and write the hit cloud to ``info['lidar']``.

    Scan pattern is either a **grid** (``h_res``/``v_res`` over ``h_fov``/
    ``v_fov``), an explicit **az/el list** (arrays or file paths, in degrees),
    which takes precedence when both ``az`` and ``el`` are given, or a
    **camera-matched grid** (``match_camera_fov=True``): rays through the
    pixel centers of an ``h_res`` x ``v_res`` image of the mount camera, so
    the scan covers exactly the camera frustum and (with ``add_rgb``) every
    hit ray gets a color.

    Args:
        env: Environment to wrap.
        h_res: Horizontal samples per ring (grid mode).
        v_res: Number of vertical beams / rings (grid mode).
        h_fov: ``(min, max)`` azimuth in degrees. The endpoint is dropped so a
            full 360 sweep does not duplicate the seam.
        v_fov: ``(min, max)`` elevation in degrees.
        az: Explicit per-ray azimuths (degrees) as a sequence or file path.
            Overrides grid mode when set together with ``el``.
        el: Explicit per-ray elevations (degrees); see ``az``.
        match_camera_fov: Replace the az/el scan with a pinhole grid through
            the pixel centers of an ``h_res`` x ``v_res`` image of the mount
            camera (``cam_fovy`` vertical, aspect from the grid shape —
            render the camera at the same aspect). ``h_fov``/``v_fov`` are
            ignored; requires ``mount_camera``.
        max_range: Rays beyond this distance (metres) report a miss.
        ray_noise_std: Std (radians) of per-ray direction jitter. 0 disables.
        range_noise_std: Std (metres) of additive range noise on hits. 0
            disables.
        miss_value: Value written to all 3 coords of a no-hit ray.
        frame: ``'sensor'`` (points relative to the sensor origin/orientation)
            or ``'world'`` (absolute world coordinates).
        backend: ``'mujoco'`` (``mj_multiRay`` on CPU) or ``'warp'`` (GPU
            triangle-soup raycast; needs ``warp-lang`` and a CUDA GPU).
            ``'warp'`` is far faster at high res. NOT bit-identical:
            ``mj_multiRay`` intersects analytic primitives while warp casts
            against their triangulated meshes, so rays grazing curved geoms
            differ (measured on the cube env: ~4% of rays >1mm, ~0.1% >10cm).
            Use the SAME backend for dataset generation and eval.
        mount_camera: Camera name the sensor rides (default ``'front_pixels'``,
            i.e. the LiDAR sits at the camera pose). Takes precedence over
            ``mount_site`` / ``mount_body``.
        mount_site: Site name to mount on (precedence over ``mount_body``).
        mount_body: Body name to mount on.
        mount_offset: ``(x, y, z)`` offset from the mount point, in the mount's
            local frame when ``use_mount_rotation`` else world frame. Without a
            mount, this is the absolute world origin of the sensor.
        use_mount_rotation: Orient the scan by the mount rotation when True;
            otherwise keep the scan world-axis-aligned.
        include_static: Whether rays may hit static (worldbody) geoms (floors).
        geomgroup: Which of MuJoCo's 6 geom groups rays may hit, as a tuple of
            indices. ``None`` (default) hits all geometry.
        min_alpha: If set, rays pass *through* geoms with rgba alpha below this
            (e.g. translucent goal markers) and continue to the solid surface
            behind them.
        add_goal_lidar: Also cast against the env's goal state (if exposed) and
            write ``info['goal_lidar']``. See module docstring.
        add_rgb: Sample a per-point color from the mount camera's rendered
            image and write ``info['lidar_rgb']`` / ``info['goal_lidar_rgb']``
            as ``(n_rays, 3)`` float32 in ``[0, 1]``. Requires ``mount_camera``
            (the shared origin is what makes the ray -> pixel lookup exact)
            and a renderable env. See module docstring.
        rgb_fill_value: Color written for misses and for hits outside the
            camera frustum (rays wider than the camera's fov have no pixel).
    """

    # Sensor frame is x=forward, y=left, z=up. MuJoCo cameras look down -z with
    # x=right, y=up, so remap to make a camera-mounted scan face the view dir.
    _CAM_FRAME_FIX = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    # Reserved geom group translucent geoms are moved into so rays skip them.
    _IGNORE_GROUP = 5

    def __init__(
        self,
        env: gym.Env,
        h_res: int = 128,
        v_res: int = 32,
        h_fov: tuple[float, float] = (-180.0, 180.0),
        v_fov: tuple[float, float] = (-15.0, 15.0),
        az: Sequence[float] | str | None = None,
        el: Sequence[float] | str | None = None,
        match_camera_fov: bool = False,
        max_range: float = 20.0,
        ray_noise_std: float = 0.0,
        range_noise_std: float = 0.0,
        miss_value: float = -1.0,
        frame: str = 'sensor',
        backend: str = 'mujoco',
        mount_camera: str | None = 'front_pixels',
        mount_site: str | None = None,
        mount_body: str | None = None,
        mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_mount_rotation: bool = True,
        include_static: bool = True,
        geomgroup: tuple[int, ...] | None = None,
        min_alpha: float | None = None,
        add_goal_lidar: bool = True,
        add_rgb: bool = False,
        rgb_fill_value: float = -1.0,
    ) -> None:
        super().__init__(env)
        if frame not in ('sensor', 'world'):
            raise ValueError(
                f"frame must be 'sensor' or 'world', got {frame!r}"
            )
        if backend not in ('mujoco', 'warp'):
            raise ValueError(
                f"backend must be 'mujoco' or 'warp', got {backend!r}"
            )
        if add_rgb and mount_camera is None:
            raise ValueError(
                'add_rgb=True requires mount_camera: colors are sampled by '
                'projecting rays into that camera image, which is only exact '
                'when the sensor shares the camera origin.'
            )
        if match_camera_fov:
            if mount_camera is None:
                raise ValueError(
                    'match_camera_fov=True requires mount_camera: the scan '
                    "pattern is built from that camera's fovy."
                )
            if az is not None or el is not None:
                raise ValueError(
                    'match_camera_fov=True conflicts with an explicit az/el '
                    'pattern; set one or the other.'
                )
        self.backend = backend
        self._warp = None  # lazily built WarpRaycaster (backend='warp')
        self.max_range = float(max_range)
        self.ray_noise_std = float(ray_noise_std)
        self.range_noise_std = float(range_noise_std)
        self.miss_value = float(miss_value)
        self.frame = frame
        self.mount_camera = mount_camera
        self.mount_site = mount_site
        self.mount_body = mount_body
        self.mount_offset = np.asarray(mount_offset, dtype=np.float64)
        self.use_mount_rotation = use_mount_rotation
        self.include_static = include_static
        self.min_alpha = min_alpha
        self.add_goal_lidar = add_goal_lidar
        self.add_rgb = add_rgb
        self.rgb_fill_value = float(rgb_fill_value)

        # mj_multiRay geom-group mask (None = all groups).
        if geomgroup is None:
            self._geomgroup = None
        else:
            mask = np.zeros(6, dtype=np.uint8)
            for g in geomgroup:
                mask[g] = 1
            self._geomgroup = mask

        # Static per-ray unit directions in the sensor frame, (n_rays, 3).
        self.match_camera_fov = match_camera_fov
        if match_camera_fov:
            # Needs the model's cam_fovy: built lazily at the first cast.
            self._dirs_sensor = None
            self._grid_hw = (int(v_res), int(h_res))
            self._nray = int(v_res) * int(h_res)
        else:
            self._dirs_sensor = self._build_ray_grid(
                h_res, v_res, h_fov, v_fov, az, el
            )
            self._nray = self._dirs_sensor.shape[0]

        self._backend: str | None = None
        self._physics = None
        self._mujoco_env = None
        self._rng = np.random.default_rng()
        # mj_multiRay output buffers, reused across casts.
        self._geomid = np.empty(self._nray, dtype=np.int32)
        self._dist = np.empty(self._nray, dtype=np.float64)
        # Goal cloud is fixed per episode; computed on reset, re-emitted /step.
        self._goal_cloud: np.ndarray | None = None
        self._goal_rgb: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Scan pattern
    # ------------------------------------------------------------------

    def _build_ray_grid(
        self, h_res, v_res, h_fov, v_fov, az, el
    ) -> np.ndarray:
        """Return ``(n_rays, 3)`` unit ray directions in the sensor frame."""
        if az is not None and el is not None:
            az_r = np.deg2rad(_load_angles(az))
            el_r = np.deg2rad(_load_angles(el))
            if az_r.shape != el_r.shape:
                raise ValueError(
                    'az and el must have the same length, got '
                    f'{az_r.shape} and {el_r.shape}'
                )
        else:
            el_lin = np.deg2rad(np.linspace(v_fov[0], v_fov[1], v_res))
            # Drop the endpoint so a full 360 sweep does not double the seam.
            az_lin = np.deg2rad(
                np.linspace(h_fov[0], h_fov[1], h_res, endpoint=False)
            )
            EL, AZ = np.meshgrid(el_lin, az_lin, indexing='ij')
            el_r, az_r = EL.ravel(), AZ.ravel()

        dirs = np.stack(
            [
                np.cos(el_r) * np.cos(az_r),
                np.cos(el_r) * np.sin(az_r),
                np.sin(el_r),
            ],
            axis=-1,
        )
        return np.ascontiguousarray(dirs, dtype=np.float64)

    def _camera_grid(self, model) -> np.ndarray:
        """Rays through the mount camera's pixel centers -> ``(n_rays, 3)``.

        Pinhole model: ``cam_fovy`` spans the grid vertically, the horizontal
        span follows from the grid aspect (matching the projection used by
        ``_sample_rgb``, which assumes square pixels). Every ray therefore
        lands strictly inside the camera image — with ``add_rgb`` no hit is
        ever left uncolored — unlike an az/el grid over the same angles,
        whose corners poke outside the frustum.
        """
        v_res, h_res = self._grid_hw
        cid = model.camera(self.mount_camera).id
        tan_y = np.tan(0.5 * np.deg2rad(float(model.cam_fovy[cid])))
        tan_x = tan_y * (h_res / v_res)
        # Pixel-center coordinates on the image plane at unit depth
        # (camera frame: x right, y up, looking down -z).
        xs = ((np.arange(h_res) + 0.5) / h_res * 2.0 - 1.0) * tan_x
        ys = (1.0 - (np.arange(v_res) + 0.5) / v_res * 2.0) * tan_y
        x, y = np.meshgrid(xs, ys, indexing='xy')
        d_cam = np.stack([x, y, -np.ones_like(x)], axis=-1).reshape(-1, 3)
        d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
        # Rows: sensor<-camera, the inverse of _CAM_FRAME_FIX.
        dirs = d_cam @ self._CAM_FRAME_FIX
        return np.ascontiguousarray(dirs, dtype=np.float64)

    # ------------------------------------------------------------------
    # Backend detection / raw handles
    # ------------------------------------------------------------------

    def _detect_backend(self) -> None:
        if self._backend is not None:
            return
        base = self.env.unwrapped

        physics = getattr(base, 'physics', None)
        if physics is None:
            inner = getattr(base, 'env', None)
            physics = getattr(inner, 'physics', None) if inner else None
        if physics is not None:
            self._backend = 'dmc'
            self._physics = physics
            return

        has_model = (
            getattr(base, 'model', None) is not None
            or getattr(base, '_model', None) is not None
        )
        if has_model:
            self._backend = 'mujoco'
            self._mujoco_env = base
            return

        raise NotImplementedError(
            'AddRaycastLidarWrapper requires a MuJoCo-backed environment with '
            "an accessible model/data. Could not find a dm_control 'physics' "
            f"object or a raw-MuJoCo 'model' on {type(base).__name__!r}."
        )

    def _get_model_data(self):
        """Return raw ``(MjModel, MjData)`` for either backend."""
        if self._backend == 'dmc':
            model = self._physics.model
            data = self._physics.data
            model = getattr(model, 'ptr', model)
            data = getattr(data, 'ptr', data)
            return model, data
        env = self._mujoco_env
        model = getattr(env, 'model', None)
        if model is None:
            model = getattr(env, '_model', None)
        data = getattr(env, 'data', None)
        if data is None:
            data = getattr(env, '_data', None)
        return model, data

    # ------------------------------------------------------------------
    # Sensor pose
    # ------------------------------------------------------------------

    def _sensor_pose(self, model, data):
        """Return ``(origin (3,), rot (3,3) world<-sensor, exclude_body_id)``."""
        rot = np.eye(3, dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64)
        exclude = -1

        if self.mount_camera is not None:
            cid = model.camera(self.mount_camera).id
            origin = np.array(data.cam_xpos[cid], dtype=np.float64)
            cam_mat = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(
                3, 3
            )
            rot = cam_mat @ self._CAM_FRAME_FIX
            exclude = int(model.cam_bodyid[cid])
        elif self.mount_site is not None:
            sid = model.site(self.mount_site).id
            origin = np.array(data.site_xpos[sid], dtype=np.float64)
            rot = np.array(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
            exclude = int(model.site_bodyid[sid])
        elif self.mount_body is not None:
            bid = model.body(self.mount_body).id
            origin = np.array(data.xpos[bid], dtype=np.float64)
            rot = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)
            exclude = int(bid)

        if self.use_mount_rotation:
            origin = origin + rot @ self.mount_offset
        else:
            rot = np.eye(3, dtype=np.float64)
            origin = origin + self.mount_offset

        # Body 0 is the worldbody; excluding it would drop all static returns.
        if exclude == 0:
            exclude = -1
        return origin, rot, exclude

    # ------------------------------------------------------------------
    # Casting
    # ------------------------------------------------------------------

    def _alpha_geomgroup(self, model):
        """Move translucent geoms to a reserved group; return the ray mask.

        Reassigning ``geom_group`` (visual-only metadata, does not affect
        physics) makes ``mj_multiRay`` skip the geom and continue to the solid
        surface behind it. Reapplied each call to survive model rebuilds.
        """
        if self.min_alpha is None:
            return self._geomgroup
        model.geom_group[model.geom_rgba[:, 3] < self.min_alpha] = (
            self._IGNORE_GROUP
        )
        mask = (
            self._geomgroup.copy()
            if self._geomgroup is not None
            else np.ones(6, dtype=np.uint8)
        )
        mask[self._IGNORE_GROUP] = 0
        return mask

    def _raycast_mujoco(self, model, data, origin, dirs_world, exclude):
        """CPU ``mj_multiRay`` cast -> ``(dist (n,), hit (n,) bool)``."""
        import mujoco

        geomgroup = self._alpha_geomgroup(model)
        vec = np.ascontiguousarray(dirs_world.reshape(-1), dtype=np.float64)
        self._geomid.fill(-1)
        self._dist.fill(-1.0)
        mujoco.mj_multiRay(
            model,
            data,
            np.ascontiguousarray(origin, dtype=np.float64),
            vec,
            geomgroup,
            self.include_static,
            exclude,
            self._geomid,
            self._dist,
            None,
            self._nray,
            self.max_range,
        )
        hit = (self._geomid >= 0) & (self._dist >= 0.0)
        dist = np.where(hit, self._dist, 0.0)
        return dist, hit

    def _raycast_warp(self, model, data, origin, dirs_world, exclude):
        """GPU triangle-soup cast -> ``(dist (n,), hit (n,) bool)``."""
        if self._warp is None:
            from ._lidar_warp import WarpRaycaster

            self._warp = WarpRaycaster(
                self.max_range,
                geomgroup=self._geomgroup,
                min_alpha=self.min_alpha,
                include_static=self.include_static,
            )
        dist, hit = self._warp.raycast(
            model, data, origin, dirs_world, exclude
        )
        return dist.astype(np.float64), hit

    def _render_rgb(self) -> np.ndarray:
        """Render the mount camera -> ``(H, W, 3)`` uint8."""
        base = self.env.unwrapped
        try:
            img = base.render(camera=self.mount_camera)
        except TypeError:  # render() without a camera kwarg
            img = base.render()
        if img is None:
            raise RuntimeError(
                'add_rgb=True needs an RGB frame but render() returned None; '
                "construct the env with render_mode='rgb_array'."
            )
        return np.asarray(img)

    def _sample_rgb(self, model, data, dirs_world, hit) -> np.ndarray:
        """Color each hit ray from the mount camera image -> ``(n_rays, 3)``.

        The sensor origin coincides with the camera, so a ray's color is just
        the pixel it passes through: pinhole projection of the ray direction
        expressed in the camera frame (camera looks down -z, x right, y up).
        """
        img = self._render_rgb()
        h, w = img.shape[:2]
        cid = model.camera(self.mount_camera).id
        cam_mat = np.array(data.cam_xmat[cid], dtype=np.float64).reshape(3, 3)

        d_cam = dirs_world @ cam_mat  # rows: camera<-world applied per ray
        depth = -d_cam[:, 2]
        in_front = depth > 1e-9
        safe = np.where(in_front, depth, 1.0)
        # MuJoCo's fovy is vertical and pixels are square: fx = fy.
        focal = 0.5 * h / np.tan(0.5 * np.deg2rad(model.cam_fovy[cid]))
        u = np.floor(0.5 * w + focal * d_cam[:, 0] / safe).astype(np.int64)
        v = np.floor(0.5 * h - focal * d_cam[:, 1] / safe).astype(np.int64)

        valid = hit & in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rgb = np.full(
            (dirs_world.shape[0], 3), self.rgb_fill_value, dtype=np.float32
        )
        rgb[valid] = img[v[valid], u[valid], :3].astype(np.float32) / 255.0
        return rgb

    def _cast(self, model, data) -> tuple[np.ndarray, np.ndarray | None]:
        """Cast the scan against the current sim state.

        Returns ``(points (n_rays, 3), rgb (n_rays, 3) | None)`` — colors are
        ``None`` unless ``add_rgb`` is set.
        """
        origin, rot, exclude = self._sensor_pose(model, data)

        if self._dirs_sensor is None:  # deferred camera-matched pattern
            self._dirs_sensor = self._camera_grid(model)
        dirs_sensor = self._dirs_sensor
        if self.ray_noise_std > 0.0:
            # Jitter directions, then renormalize (small-angle ~= radians std).
            noise = self._rng.normal(
                0.0, self.ray_noise_std, size=dirs_sensor.shape
            )
            dirs_sensor = dirs_sensor + noise
            dirs_sensor /= np.linalg.norm(dirs_sensor, axis=1, keepdims=True)

        dirs_world = dirs_sensor @ rot.T

        if self.backend == 'warp':
            dist, hit = self._raycast_warp(
                model, data, origin, dirs_world, exclude
            )
        else:
            dist, hit = self._raycast_mujoco(
                model, data, origin, dirs_world, exclude
            )

        if self.range_noise_std > 0.0:
            dist = dist + hit * self._rng.normal(
                0.0, self.range_noise_std, size=dist.shape
            )
            dist = np.clip(dist, 0.0, None)
        dist = dist[:, None]

        if self.frame == 'sensor':
            pts = dist * dirs_sensor
        else:  # 'world'
            pts = origin[None, :] + dist * dirs_world

        pts = np.where(hit[:, None], pts, self.miss_value)
        rgb = (
            self._sample_rgb(model, data, dirs_world, hit)
            if self.add_rgb
            else None
        )
        return pts.astype(np.float32), rgb

    def _get_point_cloud(self) -> tuple[np.ndarray, np.ndarray | None]:
        self._detect_backend()
        model, data = self._get_model_data()
        return self._cast(model, data)

    def _get_goal_cloud(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Cast against the env's goal state, if it exposes one.

        The env stashes ``_cur_goal_qpos`` / ``_cur_goal_qvel`` (OGBench task
        mode). We temporarily apply them, cast, then restore the live state.
        Returns ``None`` when no goal state is available (e.g. data-collection
        mode or 2D envs), in which case ``goal_lidar`` is omitted.
        """
        import mujoco

        self._detect_backend()
        base = self.env.unwrapped
        goal_qpos = getattr(base, '_cur_goal_qpos', None)
        goal_qvel = getattr(base, '_cur_goal_qvel', None)
        if goal_qpos is None:
            return None, None

        model, data = self._get_model_data()
        saved_qpos = data.qpos.copy()
        saved_qvel = data.qvel.copy()
        try:
            data.qpos[:] = goal_qpos
            if goal_qvel is not None:
                data.qvel[:] = goal_qvel
            mujoco.mj_forward(model, data)
            # _cast renders the goal state for colors while it is applied.
            cloud, rgb = self._cast(model, data)
        finally:
            data.qpos[:] = saved_qpos
            data.qvel[:] = saved_qvel
            mujoco.mj_forward(model, data)
        return cloud, rgb

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict]:
        seed = kwargs.get('seed')
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(*args, **kwargs)
        info['lidar'], rgb = self._get_point_cloud()
        if rgb is not None:
            info['lidar_rgb'] = rgb
        if self.add_goal_lidar:
            self._goal_cloud, self._goal_rgb = self._get_goal_cloud()
            if self._goal_cloud is not None:
                info['goal_lidar'] = self._goal_cloud
            if self._goal_rgb is not None:
                info['goal_lidar_rgb'] = self._goal_rgb
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['lidar'], rgb = self._get_point_cloud()
        if rgb is not None:
            info['lidar_rgb'] = rgb
        if self.add_goal_lidar and self._goal_cloud is not None:
            info['goal_lidar'] = self._goal_cloud
            if self._goal_rgb is not None:
                info['goal_lidar_rgb'] = self._goal_rgb
        return obs, reward, terminated, truncated, info
