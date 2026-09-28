"""Camera snippets for the LiDAR-ready MuJoCo scenes.

The mirror envs (TwoRoom / PushT) and the native Reacher3D scene all need the
same two viewpoints, so the geometry lives here instead of being re-derived per
env:

* :func:`topdown_camera_xml` -- straight down, fovy solved from the scene's
  half-extent so the view covers EXACTLY the arena. For the mirror envs that
  arena is the 2D renderer's image, which is what makes a ``match_camera_fov``
  scan line up with the stored pixels ray-for-ray (MuJoCo's default 45 deg fovy
  does not, and silently crops the outer rim).
* :func:`iso_camera_xml` -- an oblique view from an azimuth/elevation bearing,
  distance and fovy solved from the scene's bounding sphere so the whole scene
  stays framed at any angle. A LiDAR mounted here is much harder to learn from
  than a top-down one: geometry occludes, depth varies across the scan, and the
  ground plane is foreshortened.

Angles are degrees; lengths are metres in the scene's own world frame. Both
helpers assume the scene is centred on the world origin (true for all three
LiDAR scenes) and +z is up.
"""

from __future__ import annotations

import numpy as np


def look_at_origin_xyaxes(eye: np.ndarray) -> np.ndarray:
    """MuJoCo ``xyaxes`` (camera x=right, y=up) aiming ``eye`` at the origin.

    A MuJoCo camera looks down its own -z, so the pair (right, up) fixes the
    orientation: right is horizontal (perpendicular to world up), up completes
    the right-handed frame. ``eye`` must not be exactly on the world z axis --
    use :func:`topdown_camera_xml` for that degenerate case.
    """
    eye = np.asarray(eye, dtype=np.float64)
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        raise ValueError(
            f'eye {eye.tolist()} is on the world z axis; a look-at-origin '
            'camera there has no defined roll (use a top-down camera)'
        )
    right /= norm
    return np.concatenate([right, np.cross(right, fwd)])


def orbit_eye(azimuth: float, elevation: float, distance: float) -> np.ndarray:
    """Eye position at ``azimuth``/``elevation`` (deg) and ``distance`` (m)."""
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    return distance * np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
    )


def topdown_camera_xml(
    name: str = 'topdown', *, height: float, half_extent: float
) -> str:
    """Camera straight above the origin, framed exactly on ``half_extent``.

    Args:
        height: camera altitude (m).
        half_extent: half-width of the region to cover at ground level -- for a
            mirror env, the world size of the 2D render's half-image.
    """
    fovy = 2 * np.degrees(np.arctan(half_extent / height))
    return (
        f'<camera name="{name}" pos="0 0 {height:g}"'
        f' xyaxes="1 0 0 0 1 0" fovy="{fovy:.4f}"/>'
    )


def iso_camera_xml(
    name: str = 'iso',
    *,
    radius: float,
    azimuth: float = 45.0,
    elevation: float = 45.0,
    distance: float | None = None,
    fill: float = 2.5,
) -> str:
    """Oblique camera on the ``azimuth``/``elevation`` bearing, framing a sphere.

    Args:
        radius: bounding-sphere radius of the scene (m), origin-centred.
        azimuth: bearing CCW from +x. 45 with ``elevation`` 45 is the "45/45"
            view; true isometric foreshortening would be elevation 35.264.
        elevation: degrees above the ground plane.
        distance: eye distance (m); defaults to ``fill * radius``.
        fill: distance in units of ``radius`` when ``distance`` is None. Larger
            = further away and a narrower fov (less perspective distortion).
    """
    dist = float(distance) if distance is not None else fill * radius
    eye = orbit_eye(azimuth, elevation, dist)
    fovy = 2 * np.degrees(np.arcsin(min(radius / dist, 1.0)))
    axes = ' '.join(f'{v:.6f}' for v in look_at_origin_xyaxes(eye))
    return (
        f'<camera name="{name}" pos="{eye[0]:.4f} {eye[1]:.4f} {eye[2]:.4f}"'
        f' xyaxes="{axes}" fovy="{fovy:.4f}"/>'
    )
