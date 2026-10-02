"""Look and offscreen rendering of the LiDAR-ready MuJoCo scenes.

The mirror scenes (TwoRoom / PushT) exist for the raycast sensor, so how they
*look* never mattered: a checker ground, MuJoCo's default background, whatever
rgba the geoms were given. Rendering RGB from the sensor's own camera (the
"scanview" image data, which puts an image model at exactly the sensor's
viewpoint and field of view) makes it matter. Two helpers keep that concern out
of the env classes:

* :class:`SceneStyle` -- the few knobs that decide the look: ground (checker or
  one flat colour), background (MuJoCo's default or one flat colour), shadows,
  and per-slot geom colours. The defaults reproduce the scenes exactly as they
  were, so a scan cast against a styled scene is the scan it always was: style
  changes colours and textures only, never geometry.
* :class:`CameraRenderer` -- owns the ``mujoco.Renderer`` needed to draw a
  named camera, rebuilding it when the model object or the calling thread
  changes (a GL context belongs to the thread that made it current), plus a
  depth pass. Depth is what ties a render to a scan: rays of a
  ``match_camera_fov`` sensor pass through the pixel centres of an
  ``h_res x v_res`` image from the same camera, so a depth image of that size
  must agree with the scan's forward range at every pixel.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import mujoco
import numpy as np


def _rgb_str(value) -> str:
    """Normalise ``'r g b'`` / ``(r, g, b)`` to the ``'r g b'`` MuJoCo spells."""
    if isinstance(value, str):
        parts = value.split()
    else:
        parts = [f'{float(v):g}' for v in value]
    if len(parts) != 3:
        raise ValueError(f'expected 3 colour components, got {value!r}')
    return ' '.join(parts)


@dataclass(frozen=True)
class SceneStyle:
    """How a LiDAR-ready scene is drawn. Defaults = the scene as it always was.

    Args:
        ground: ``'checker'`` keeps the checker grid the scenes were built
            with; anything else is one flat ``'r g b'`` colour for the ground
            plane.
        background: ``None`` leaves MuJoCo's default background; otherwise
            an ``'r g b'`` colour drawn as a flat skybox.
        shadows: whether the scene light casts shadows (MuJoCo's default is
            yes). A render flag, not scene geometry.
        light: ``'spot'`` keeps the scene's own point light (a spotlight
            above the arena: bright centre, darker rim); ``'uniform'`` replaces
            it with a directional light straight down, so illumination does
            not fall off across the arena while faces still shade by their
            orientation. Intensities are balanced so an upward face renders
            at its nominal colour.
        checker: the two checker colours when ``ground == 'checker'``; ``None``
            keeps the scene's own.
        floor_matte: strip specular and reflectance from the ground material
            (no highlight, no mirror image of the objects). A flat-colour
            ground is always matte.
        rgba: per-slot colour overrides, ``{'agent': '0.9 0.2 0.2 1', ...}``.
            Each env documents its slots; unknown slots are ignored so one
            style can serve several scenes.
    """

    ground: str = 'checker'
    background: str | None = None
    shadows: bool = True
    light: str = 'spot'
    checker: tuple[str, str] | None = None
    floor_matte: bool = False
    rgba: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def coerce(cls, obj, default: SceneStyle | None = None) -> SceneStyle:
        """Accept a :class:`SceneStyle`, a mapping (Hydra config) or ``None``.

        ``None`` gives ``default`` (an env's own look) or the plain defaults.
        """
        if obj is None:
            return default if default is not None else cls()
        if isinstance(obj, cls):
            return obj
        kwargs = dict(obj)
        if 'rgba' in kwargs and kwargs['rgba'] is not None:
            kwargs['rgba'] = dict(kwargs['rgba'])
        if 'checker' in kwargs and kwargs['checker'] is not None:
            kwargs['checker'] = tuple(kwargs['checker'])
        return cls(**kwargs)

    def color(self, slot: str, default: str) -> str:
        return self.rgba.get(slot, default)

    def asset_xml(
        self,
        *,
        checker_rgb1: str,
        checker_rgb2: str,
        texrepeat: str,
        material_attrs: str = '',
    ) -> str:
        """Body of the scene's ``<asset>`` block (textures + materials).

        ``material_attrs`` are the scene's own extra attributes on its checker
        material (e.g. ``reflectance="0.1"``); ``floor_matte`` overrides them.
        """
        parts = []
        if self.ground == 'checker':
            c1, c2 = self.checker or (checker_rgb1, checker_rgb2)
            attrs = (
                ' specular="0" reflectance="0" shininess="0"'
                if self.floor_matte
                else (' ' + material_attrs if material_attrs else '')
            )
            parts.append(
                f'<texture name="grid" type="2d" builtin="checker" '
                f'rgb1="{_rgb_str(c1)}" rgb2="{_rgb_str(c2)}" '
                f'width="256" height="256"/>\n    '
                f'<material name="grid" texture="grid" '
                f'texrepeat="{texrepeat}"{attrs}/>'
            )
        else:
            parts.append(
                f'<material name="ground" rgba="{_rgb_str(self.ground)} 1" '
                f'specular="0" reflectance="0" shininess="0"/>'
            )
        if self.background is not None:
            bg = _rgb_str(self.background)
            parts.append(
                f'<texture name="sky" type="skybox" builtin="flat" '
                f'rgb1="{bg}" rgb2="{bg}" width="32" height="32"/>'
            )
        return '\n    '.join(parts)

    def ground_attr(self) -> str:
        """Attribute that colours the ground ``<geom>`` (see :meth:`asset_xml`)."""
        return 'material="grid"' if self.ground == 'checker' else 'material="ground"'

    def light_xml(self, *, pos: str, dir: str, attrs: str = '') -> str:
        """The scene's ``<light>``: its own spotlight, or a uniform directional one.

        ``attrs`` are the scene's own extra light attributes, kept verbatim in
        ``'spot'`` mode so an unstyled scene is exactly what it was.
        """
        if self.light == 'spot':
            extra = ' ' + attrs if attrs else ''
            return f'<light pos="{pos}" dir="{dir}"{extra}/>'
        if self.light == 'uniform':
            # With the headlight below, an upward face receives
            # 0.25 + 0.55 + 0.30*cos(45 deg) ~= 1.0 of its colour from a 45 deg
            # camera, so a ground set to a grey renders as that grey.
            return (
                f'<light directional="true" pos="{pos}" dir="{dir}" '
                f'diffuse="0.55 0.55 0.55" specular="0.3 0.3 0.3"/>'
            )
        raise ValueError(
            f"light must be 'spot' or 'uniform', got {self.light!r}"
        )

    def visual_xml(self) -> str:
        """Extra ``<visual>`` children: the headlight that pairs with 'uniform'."""
        if self.light == 'uniform':
            return (
                '<headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" '
                'specular="0.3 0.3 0.3"/>'
            )
        return ''


class CameraRenderer:
    """Offscreen RGB / depth renders of a named camera.

    One instance per scene; it keeps a ``mujoco.Renderer`` per pass (RGB,
    depth) and rebuilds them when the model object changes (the mirror envs
    rebuild their model every episode), when the output size changes, or when
    another thread calls (``World.collect`` runs envs on a background thread).
    """

    def __init__(self) -> None:
        self._renderers: dict[bool, mujoco.Renderer] = {}
        self._keys: dict[bool, tuple] = {}

    def _get(self, model, height: int, width: int, depth: bool):
        key = (id(model), int(height), int(width), threading.get_ident())
        if self._keys.get(depth) != key:
            old = self._renderers.pop(depth, None)
            if old is not None:
                old.close()
            r = mujoco.Renderer(model, int(height), int(width))
            if depth:
                r.enable_depth_rendering()
            self._renderers[depth] = r
            self._keys[depth] = key
        return self._renderers[depth]

    def render(
        self,
        model,
        data,
        camera: str | int,
        *,
        height: int,
        width: int,
        shadows: bool = True,
    ) -> np.ndarray:
        """RGB ``(height, width, 3)`` uint8 from ``camera`` (name or id)."""
        r = self._get(model, height, width, depth=False)
        r.update_scene(data, camera=camera)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(bool(shadows))
        return np.array(r.render(), dtype=np.uint8)

    def depth(
        self, model, data, camera: str | int, *, height: int, width: int
    ) -> np.ndarray:
        """Depth ``(height, width)`` float32, metres along the camera axis.

        This is z-depth (distance to the image plane), which for a ray through
        a pixel centre equals the sensor-frame ``x`` of the hit a
        ``match_camera_fov`` scan stores for that ray.
        """
        r = self._get(model, height, width, depth=True)
        r.update_scene(data, camera=camera)
        return np.array(r.render(), dtype=np.float32)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        self._keys.clear()
