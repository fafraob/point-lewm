"""Render a LiDAR point cloud to an RGB image for video panels.

``LidarPanelRenderer`` renders with **polyscope's headless (EGL) backend** —
true perspective projection and round shaded points, so the panels read far
better than the previous matplotlib 3D scatter. A ``(n_rays, 3)`` cloud becomes
a ``(size, size, 3)`` uint8 image, colored by distance from the sensor — or by
recorded per-point colors (the lidar wrapper's ``lidar_rgb``) when passed to
:meth:`~LidarPanelRenderer.render`. Used by ``World`` to add a LiDAR panel next
to the RGB rollout in eval videos, and by ``swm-lidar-viz`` for the panel video.

Polyscope holds global state and needs a working GL context (a display, or EGL
on headless machines — the same requirement ``MUJOCO_GL=egl`` env rendering
already has). :func:`ensure_polyscope_init` initializes it once per process and
is safe to call again (e.g. before the interactive viewer).
"""

from __future__ import annotations

import numpy as np

# Camera framing is locked on the first non-empty cloud and reused for every
# subsequent frame, so all panels of a video share viewpoint and scale.

# Polyscope's EGL handles, captured right after a headless init while its
# context is current. MuJoCo's offscreen renderer shares the thread and makes
# ITS OWN EGL context current on every env render; polyscope never re-binds,
# so GL calls would hit MuJoCo's context and die with "Invalid value". Re-bind
# these before touching polyscope (mirroring what MuJoCo does on its side).
_EGL_STATE: tuple | None = None


def _capture_egl_state() -> None:
    global _EGL_STATE
    try:
        from OpenGL import EGL

        dpy = EGL.eglGetCurrentDisplay()
        ctx = EGL.eglGetCurrentContext()
        if dpy and ctx:
            _EGL_STATE = (
                dpy,
                EGL.eglGetCurrentSurface(EGL.EGL_DRAW),
                EGL.eglGetCurrentSurface(EGL.EGL_READ),
                ctx,
            )
    except Exception:  # pragma: no cover - PyOpenGL absent/exotic platform
        _EGL_STATE = None


def make_polyscope_current() -> None:
    """Re-bind polyscope's (headless EGL) GL context on this thread.

    No-op unless a headless init captured the handles -- windowed backends
    manage their own current-context via GLFW.
    """
    if _EGL_STATE is not None:
        from OpenGL import EGL

        EGL.eglMakeCurrent(*_EGL_STATE)


def ensure_polyscope_init(prefer_headless: bool = True):
    """Initialize polyscope once per process and return the module.

    ``prefer_headless=True`` (the renderer's default) tries the EGL backend
    FIRST, even when ``DISPLAY`` is set: offscreen video rendering must not
    depend on a (possibly broken, e.g. forwarded-X) window server, and must
    not flash a window. Pass ``prefer_headless=False`` — before anything else
    initializes polyscope — when an interactive window is wanted (the
    ``swm-lidar-viz`` viewer). A second call is a no-op either way, so
    whichever intent comes first in the process wins.
    """
    import polyscope as ps

    if ps.is_initialized():
        return ps
    ps.set_allow_headless_backends(True)
    if prefer_headless:
        try:
            ps.init('openGL3_egl')
            _capture_egl_state()
            return ps
        except Exception:
            pass  # no EGL on this machine: fall back to the default chain
    ps.init()
    if ps.is_headless():  # auto-selected EGL (no display found)
        _capture_egl_state()
    return ps


def _turbo(v: np.ndarray) -> np.ndarray:
    """Map values in ``[0, 1]`` to ``(N, 3)`` RGB via Google's turbo colormap
    (5th-degree polynomial approximation), matching the old matplotlib cmap."""
    v = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
    p = np.stack([np.ones_like(v), v, v**2, v**3, v**4, v**5], axis=-1)
    coeff = np.array(
        [
            [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
            [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
            [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
        ],
        dtype=np.float32,
    )
    return np.clip(p @ coeff.T, 0.0, 1.0)


class LidarPanelRenderer:
    """Reusable polyscope headless renderer: ``(n_rays, 3)`` cloud -> RGB frame.

    Args:
        size: Output image side length in pixels (square).
        miss_value: Sentinel filled into no-hit rays; those points are dropped.
        max_view: Drop points farther than this (metres) before rendering.
            ``None`` = keep everything; framing ignores far outliers anyway
            via ``zoom_percentile``.
        elev, azim: Camera elevation / azimuth (degrees) relative to the
            scene's dominant plane (the table/floor, fit to the first cloud) —
            NOT the sensor axes. The defaults (40 / 35) orbit the scene center
            in a 3/4 view, anchored to the fitted plane: the sensor's own
            occlusion shadows open up as visible holes and height differences
            separate. A view from the sensor pose itself cannot show either —
            from there every shadow hides exactly behind the object casting it,
            so the cloud reads as a flat image. Pass ``elev=None`` with
            ``azim=0`` for that first-person view instead; it matches the
            mounted RGB camera image ray-for-ray, at the cost of depth.
        zoom_percentile: Scene half-extent is this percentile of the first
            cloud's distance-to-center — far outliers don't shrink the scene.
        distance: Camera distance in units of the scene half-extent. 2.0
            keeps the whole scan inside the panel for the sensor layouts here;
            smaller values crop the near edge.
        flip_y: Negate the sensor 'left' (+y) axis when drawing. Leave False:
            the camera is placed on the sensor's own bearing, so the render
            already matches the RGB camera image (verified against the v1
            dataset's paired pixels/lidar rows) — negating y left-right
            mirrors it. Only for clouds whose own y-sign is flipped.
        point_radius: Point radius relative to the scene extent. Large
            enough (0.016) that neighbouring returns merge into readable
            surfaces at panel resolution; the old 0.005 rendered as isolated
            specks that vanished in a video re-encode.
        ssaa: Polyscope supersampling factor (antialiasing).
    """

    _CLOUD_NAME = 'lidar_panel'

    def __init__(
        self,
        size: int = 512,
        miss_value: float = -1.0,
        max_view: float | None = None,
        elev: float | None = 40.0,
        azim: float = 35.0,
        zoom_percentile: float = 95.0,
        distance: float = 2.0,
        flip_y: bool = False,
        point_radius: float = 0.016,
        ssaa: int = 2,
    ) -> None:
        self._ps = ensure_polyscope_init()
        self.size = int(size)
        self.miss_value = miss_value
        self.max_view = max_view
        self.elev = None if elev is None else float(elev)
        self.azim = float(azim)
        self.zoom_percentile = float(zoom_percentile)
        self.distance = float(distance)
        self.flip_y = flip_y
        self.point_radius = float(point_radius)
        # Eye pull-back (units of the scene half-extent) for the first-person
        # sensor view; grown by the blank-frame retry in render().
        self._back_scale = 0.2

        ps = self._ps
        ps.set_ground_plane_mode('none')
        ps.set_background_color((1.0, 1.0, 1.0))
        ps.set_SSAA_factor(int(ssaa))
        # (target, half-extent) locked on the first non-empty cloud.
        self._frame: tuple[np.ndarray, float] | None = None

    def _clean(
        self, cloud, colors=None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        arr = np.asarray(cloud)
        if arr.ndim > 2:  # (history, n_rays, 3) -> last frame
            arr = arr[-1]
        pts = arr.reshape(-1, 3)
        rgb = None
        if colors is not None:
            rgb = np.asarray(colors)
            if rgb.ndim > 2:
                rgb = rgb[-1]
            rgb = rgb.reshape(-1, 3)
        keep = ~np.all(pts == self.miss_value, axis=-1)
        pts = pts[keep]
        rgb = rgb[keep] if rgb is not None else None
        if self.max_view is not None and len(pts):
            near = np.linalg.norm(pts, axis=1) <= self.max_view
            pts = pts[near]
            rgb = rgb[near] if rgb is not None else None
        return pts, rgb

    def _point_colors(self, dist: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Recorded RGB where sampled, depth colormap elsewhere.

        Rows of ``rgb`` outside ``[0, 1]`` are the wrapper's fill (rays with
        no camera pixel) and fall back to depth coloring.
        """
        span = float(dist.max() - dist.min())
        norm = (dist - dist.min()) / (span if span > 0 else 1.0)
        depth = _turbo(norm)
        sampled = np.all((rgb >= 0.0) & (rgb <= 1.0), axis=1)
        return np.where(sampled[:, None], rgb, depth).astype(np.float32)

    def _set_camera(self, pts: np.ndarray) -> None:
        """Lock scene extents + the camera on the first non-empty cloud.

        Default (``elev=None``, ``azim=0``): a **first-person sensor view**.
        swm's lidar wrapper emits sensor-frame clouds with camera-aligned
        axes — x=forward, y=left, z=up, sensor at the origin (see
        ``AddRaycastLidarWrapper._CAM_FRAME_FIX``) — so rendering from the
        origin along +x with +z up reproduces the mounted RGB camera's view
        exactly; the vertical fov is recovered from the cloud's own angular
        span. The eye is pulled back slightly along -x (polyscope blanks
        structures first drawn with the eye at the origin, see below) with
        the fov widened to compensate, so the framing still matches the
        camera image.

        With an ``elev``/``azim`` override — or a cloud that doesn't face +x —
        an orbit view anchored to the scene's dominant plane is used instead:
        the plane normal (smallest-covariance eigenvector) is the up-vector,
        and the eye orbits the scene center ``elev`` degrees above the plane /
        ``azim`` degrees off the sensor-to-scene axis.
        """
        ps = self._ps
        # Robust scene center/extent: percentile box, so stray far returns
        # neither shift the target nor inflate the framing.
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        target = (lo + hi) / 2
        lim = float(
            np.percentile(
                np.linalg.norm(pts - target, axis=1), self.zoom_percentile
            )
        )
        lim = lim if lim > 0 else 1.0
        # Freeze extents so the relative point radius and camera stay stable
        # across frames regardless of each cloud's own bounding box.
        ps.set_automatically_compute_scene_extents(False)
        ps.set_length_scale(lim)
        ps.set_bounding_box(target - lim, target + lim)

        # First-person sensor view (the default): only for forward-facing
        # clouds — a scan with returns behind the sensor has no camera to
        # mimic, so it falls through to the orbit view.
        fwd_facing = bool((pts[:, 0] > 0).all())
        if self.elev is None and self.azim == 0.0 and fwd_facing:
            depth = pts[:, 0]
            # Vertical half-angle of the scan: for the camera-matched grid
            # z/x == the image-plane row coordinate, so its extremes span the
            # camera's fovy (percentile against stray returns; symmetric about
            # the +x optical axis like the camera itself).
            tan_half = float(np.percentile(np.abs(pts[:, 2] / depth), 99.9))
            tan_half = tan_half if tan_half > 0 else 0.5
            # Pull the eye back along -x to clear polyscope's origin blank
            # zone (see the orbit branch below), and widen the fov so objects
            # at the median depth keep their apparent size.
            back = self._back_scale * lim
            med = float(np.median(depth))
            fov = 2 * np.degrees(np.arctan(tan_half * med / (med + back)))
            ps.set_vertical_fov_degrees(float(np.clip(fov, 10.0, 120.0)))
            eye = np.array([-back, 0.0, 0.0])
            ps.look_at_dir(eye, eye + np.array([1.0, 0.0, 0.0]), (0.0, 0.0, 1.0))
            self._frame = (target, lim)
            return

        # Plane normal, oriented toward the sensor (origin) side.
        if len(pts) >= 10:
            _, eigvec = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))
            up = eigvec[:, 0]
        else:  # degenerate cloud: fall back to the sensor z axis
            up = np.array([0.0, 0.0, 1.0])
        if np.dot(up, -target) < 0:
            up = -up
        # Direction from the scene center back to the sensor (origin) -- the
        # RGB camera's own bearing -- split into its in-plane component and
        # its natural elevation above the plane.
        to_sensor = -target
        fwd = to_sensor - np.dot(to_sensor, up) * up
        norm = np.linalg.norm(fwd)
        fwd = fwd / norm if norm > 1e-6 else np.array([-1.0, 0.0, 0.0])
        sensor_elev = np.arctan2(np.dot(to_sensor, up), norm)

        el = sensor_elev if self.elev is None else np.deg2rad(self.elev)
        az = np.deg2rad(self.azim)
        side = np.cross(up, fwd)
        direction = (
            np.cos(el) * (np.cos(az) * fwd + np.sin(az) * side)
            + np.sin(el) * up
        )
        t_eye = self.distance * lim
        # Polyscope degeneracy (observed on 2.3): a structure whose FIRST draw
        # happens with the camera eye within ~0.1 * length-scale of the WORLD
        # ORIGIN renders blank, and every subsequent draw of that structure
        # stays blank -- with the camera locked per video and the cloud
        # re-registered per frame, one unlucky first cloud whites out the whole
        # video. Sensor-frame clouds put the sensor AT the origin and the eye
        # on the sensor bearing, so the eye crosses the origin whenever
        # distance * lim ≈ |target|. Keep a 0.25 * lim clearance by sliding the
        # eye further out along the same bearing (origin ends up in front of
        # the camera, which renders fine); the view direction is unchanged.
        clearance = 0.25 * lim
        if np.linalg.norm(target + t_eye * direction) < clearance:
            b = float(np.dot(target, direction))
            disc = b * b - float(np.dot(target, target)) + clearance**2
            t_eye = -b + np.sqrt(max(disc, 0.0))
        eye = target + t_eye * direction
        ps.look_at_dir(eye, target, up)
        self._frame = (target, lim)

    def render(self, cloud, colors=None, _retry=0) -> np.ndarray:
        """Return a ``(size, size, 3)`` uint8 image of the cloud.

        Args:
            cloud: ``(n_rays, 3)`` points (misses included; they are dropped).
            colors: Optional ``(n_rays, 3)`` per-point RGB in ``[0, 1]``,
                row-aligned with ``cloud`` (the lidar wrapper's ``lidar_rgb``).
                Fill rows fall back to the depth colormap; ``None`` keeps
                depth coloring for every point.
        """
        ps = self._ps
        make_polyscope_current()  # MuJoCo may have re-bound the GL context
        ps.set_window_size(self.size, self.size)  # global; re-pin per render

        pts, rgb = self._clean(cloud, colors)
        if len(pts):
            dist = np.linalg.norm(pts, axis=1)
            if rgb is None:
                span = float(dist.max() - dist.min())
                c = _turbo((dist - dist.min()) / (span if span > 0 else 1.0))
            else:
                c = self._point_colors(dist, rgb)
            draw = pts * np.array([1.0, -1.0, 1.0]) if self.flip_y else pts
            draw = np.ascontiguousarray(draw, dtype=np.float32)
            if self._frame is None:
                self._set_camera(draw)
            pc = ps.register_point_cloud(
                self._CLOUD_NAME,
                draw,
                radius=self.point_radius,
                point_render_mode='sphere',
            )
            pc.add_color_quantity('color', c, enabled=True)
        elif ps.has_point_cloud(self._CLOUD_NAME):
            ps.remove_point_cloud(self._CLOUD_NAME)  # blank panel, not stale

        buf = np.asarray(ps.screenshot_to_buffer(transparent_bg=False))
        # With a WINDOWED (GLFW) backend -- claimed by swm-lidar-viz so its
        # interactive viewer can open later -- there is a real window we never
        # service, because panel frames are offscreen screenshots. X/GL flow
        # control then blocks the THIRD screenshot forever (reproduced on a
        # software-GL X display: frames 0-1 fine, frame 2 hangs). Pumping one
        # frame keeps that window's event loop alive; headless EGL has no
        # window and needs nothing.
        if not ps.is_headless():
            ps.frame_tick()
        img = buf[..., :3]
        if img.shape[:2] != (self.size, self.size):  # defensive: window mgr
            yi = np.linspace(0, img.shape[0] - 1, self.size).astype(int)
            xi = np.linspace(0, img.shape[1] - 1, self.size).astype(int)
            img = img[yi][:, xi]
        # Safety net for the origin degeneracy documented in _set_camera: a
        # non-empty cloud framed by _set_camera can never legitimately fill a
        # uniform frame, so a constant image means the locked camera is broken
        # (e.g. an eye inside polyscope's blank zone despite the clearance).
        # Re-lock from this cloud with the eye pushed further out and retry.
        if len(pts) and _retry < 3 and img.min() == img.max():
            self._frame = None
            self.distance *= 1.6
            self._back_scale *= 1.6
            return self.render(cloud, colors, _retry=_retry + 1)
        return img.copy()

    def close(self) -> None:
        """Remove this renderer's structure (polyscope itself stays alive —
        it is process-global and cannot be re-initialized)."""
        if self._ps.has_point_cloud(self._CLOUD_NAME):
            self._ps.remove_point_cloud(self._CLOUD_NAME)
