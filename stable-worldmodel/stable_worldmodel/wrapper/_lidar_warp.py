"""GPU raycasting backend for the LiDAR wrapper, via NVIDIA Warp.

``mj_multiRay`` is CPU and scales ~linearly with rays x scene meshes, which
gets slow at high resolution (the UR5e arm alone is ~185k triangles). This
backend turns the scene into a single triangle soup, builds a ``wp.Mesh`` BVH
once, refits vertex positions each step from ``geom_xpos``/``geom_xmat``, and
casts all rays in parallel on the GPU with ``wp.mesh_query_ray``.

Geometry is tessellated to triangles: meshes use MuJoCo's own triangles;
box / plane / cylinder / capsule geoms are tessellated analytically. Geoms are
filtered at soup-build time by ``geomgroup`` / ``min_alpha`` / a mounted body
to exclude, so those knobs behave the same as the MuJoCo backend.

Only ``WarpRaycaster.raycast`` is used by the wrapper; it returns per-ray
``(distance, hit)`` and the wrapper builds the point cloud (frame, misses,
noise) exactly as for the MuJoCo backend.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Primitive tessellation (local geom frame, MuJoCo conventions)
# --------------------------------------------------------------------------


def _box_tris(size: np.ndarray) -> np.ndarray:
    """(3,) half-extents -> (12, 3, 3) triangle vertices in the box frame."""
    hx, hy, hz = size
    c = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    quads = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    tris = []
    for a, b, cc, d in quads:
        tris.append(c[[a, b, cc]])
        tris.append(c[[a, cc, d]])
    return np.asarray(tris, dtype=np.float32)


def _plane_tris(size: np.ndarray, bound: float) -> np.ndarray:
    """MuJoCo plane (normal +z) -> (2, 3, 3). Zero half-extent = infinite."""
    ex = size[0] if size[0] > 0 else bound
    ey = size[1] if size[1] > 0 else bound
    c = np.array(
        [[-ex, -ey, 0.0], [ex, -ey, 0.0], [ex, ey, 0.0], [-ex, ey, 0.0]],
        dtype=np.float32,
    )
    return np.asarray([c[[0, 1, 2]], c[[0, 2, 3]]], dtype=np.float32)


def _ring(radius: float, z: float, nseg: int) -> np.ndarray:
    a = np.linspace(0.0, 2.0 * np.pi, nseg, endpoint=False)
    return np.stack(
        [radius * np.cos(a), radius * np.sin(a), np.full(nseg, z)], axis=1
    ).astype(np.float32)


def _cylinder_tris(size: np.ndarray, nseg: int = 16) -> np.ndarray:
    """MuJoCo cylinder (radius size[0], half-length size[1], along z)."""
    r, h = float(size[0]), float(size[1])
    bot, top = _ring(r, -h, nseg), _ring(r, h, nseg)
    tris = []
    for i in range(nseg):
        j = (i + 1) % nseg
        tris += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]]]
        tris.append([[0, 0, -h], bot[j], bot[i]])  # bottom cap
        tris.append([[0, 0, h], top[i], top[j]])  # top cap
    return np.asarray(tris, dtype=np.float32)


def _capsule_tris(
    size: np.ndarray, nseg: int = 16, nring: int = 4
) -> np.ndarray:
    """MuJoCo capsule: cylinder body + two hemispherical caps (along z)."""
    r, h = float(size[0]), float(size[1])
    bot, top = _ring(r, -h, nseg), _ring(r, h, nseg)
    tris = []
    for i in range(nseg):
        j = (i + 1) % nseg
        tris += [[bot[i], bot[j], top[j]], [bot[i], top[j], top[i]]]
    # hemispheres via latitude rings
    for sign, cap_z in ((1.0, h), (-1.0, -h)):
        prev = top if sign > 0 else bot
        for k in range(1, nring + 1):
            phi = 0.5 * np.pi * k / nring
            rr, zz = r * np.cos(phi), sign * r * np.sin(phi)
            cur = _ring(rr, cap_z + zz, nseg)
            for i in range(nseg):
                j = (i + 1) % nseg
                if k < nring:
                    tris += [
                        [prev[i], prev[j], cur[j]],
                        [prev[i], cur[j], cur[i]],
                    ]
                else:  # close to the pole
                    pole = [0.0, 0.0, cap_z + sign * r]
                    tris.append([prev[i], prev[j], pole])
            prev = cur
    return np.asarray(tris, dtype=np.float32)


# --------------------------------------------------------------------------
# Raycaster
# --------------------------------------------------------------------------


class WarpRaycaster:
    """Build a scene triangle soup once, refit + cast on the GPU per step."""

    def __init__(
        self,
        max_range: float,
        device: str | None = None,
        geomgroup: np.ndarray | None = None,
        min_alpha: float | None = None,
        include_static: bool = True,
        plane_bound: float = 10.0,
    ) -> None:
        import warp as wp

        wp.init()
        _ensure_kernels()
        self.wp = wp
        self.max_range = float(max_range)
        if device is None:
            device = 'cuda:0' if wp.get_cuda_device_count() > 0 else 'cpu'
        self.device = device
        self.geomgroup = geomgroup
        self.min_alpha = min_alpha
        self.include_static = include_static
        self.plane_bound = plane_bound

        self._model_key = None  # (id(model), ngeom) — rebuild soup on change
        self._mesh = None
        self._local_v = None  # wp.array(vec3) static local verts
        self._vert_geom = None  # wp.array(int32) geom id per vertex
        self._world_v = None  # wp.array(vec3) refit target (= mesh.points)
        self._n_geom = 0
        self._dist = None
        self._hit = None
        self._n_rays = 0

    # -- geometry --------------------------------------------------------

    def _geom_local_tris(self, model, g: int):
        """Return (T, 3, 3) local triangle verts for geom ``g`` or None."""
        import mujoco

        gt = int(model.geom_type[g])
        size = np.asarray(model.geom_size[g], dtype=np.float32)
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            return _box_tris(size)
        if gt == mujoco.mjtGeom.mjGEOM_PLANE:
            return _plane_tris(size, self.plane_bound)
        if gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return _cylinder_tris(size)
        if gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return _capsule_tris(size)
        if gt == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            va = int(model.mesh_vertadr[mid])
            vn = int(model.mesh_vertnum[mid])
            fa = int(model.mesh_faceadr[mid])
            fn = int(model.mesh_facenum[mid])
            verts = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32)
            faces = np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int64)
            return verts[faces]  # (fn, 3, 3)
        return None  # SPHERE/ELLIPSOID/HFIELD/SDF: unsupported, skipped

    def _included(self, model, g: int, exclude_body: int) -> bool:
        body = int(model.geom_bodyid[g])
        if exclude_body >= 0 and body == exclude_body:
            return False
        if not self.include_static and body == 0:
            return False
        if self.geomgroup is not None:
            grp = int(model.geom_group[g])
            if grp < 0 or grp >= 6 or self.geomgroup[grp] == 0:
                return False
        if self.min_alpha is not None:
            if float(model.geom_rgba[g, 3]) < self.min_alpha:
                return False
        return True

    def _build(self, model, exclude_body: int) -> None:
        wp = self.wp
        local, vert_geom, faces = [], [], []
        voff = 0
        for g in range(model.ngeom):
            if not self._included(model, g, exclude_body):
                continue
            tris = self._geom_local_tris(model, g)
            if tris is None or len(tris) == 0:
                continue
            v = tris.reshape(-1, 3)  # (T*3, 3)
            n = v.shape[0]
            idx = np.arange(voff, voff + n, dtype=np.int32).reshape(-1, 3)
            local.append(v)
            vert_geom.append(np.full(n, g, dtype=np.int32))
            faces.append(idx)
            voff += n

        local = np.concatenate(local, axis=0)
        vert_geom = np.concatenate(vert_geom, axis=0)
        faces = np.concatenate(faces, axis=0).reshape(-1)

        self._n_geom = int(model.ngeom)
        self._local_v = wp.array(local, dtype=wp.vec3, device=self.device)
        self._vert_geom = wp.array(
            vert_geom, dtype=wp.int32, device=self.device
        )
        self._world_v = wp.zeros(
            local.shape[0], dtype=wp.vec3, device=self.device
        )
        indices = wp.array(faces, dtype=wp.int32, device=self.device)
        # 'sah' gives a high-quality BVH (fast ray queries) whose refit only
        # updates node bounds (~0.1 ms); warp's default 'lbvh' re-sorts on
        # every refit (~240 ms for this scene), which dominated per-step cost.
        self._mesh = wp.Mesh(
            points=self._world_v, indices=indices, bvh_constructor='sah'
        )

    # -- per-step update + cast -----------------------------------------

    def _ensure(self, model, exclude_body: int) -> None:
        key = (id(model), int(model.ngeom))
        if key != self._model_key:
            self._build(model, exclude_body)
            self._model_key = key

    def _update_poses(self, model, data) -> None:
        wp = self.wp
        xpos = np.asarray(data.geom_xpos, dtype=np.float32)
        xmat = np.asarray(data.geom_xmat, dtype=np.float32).reshape(-1, 3, 3)
        xpos_w = wp.array(xpos, dtype=wp.vec3, device=self.device)
        xmat_w = wp.array(xmat, dtype=wp.mat33, device=self.device)
        wp.launch(
            _kernel_xform,
            dim=self._local_v.shape[0],
            inputs=[self._local_v, self._vert_geom, xpos_w, xmat_w],
            outputs=[self._world_v],
            device=self.device,
        )
        self._mesh.refit()

    def raycast(self, model, data, origin, dirs_world, exclude_body=-1):
        """Cast ``dirs_world`` (N,3) from ``origin`` -> (dist (N,), hit (N,))."""
        wp = self.wp
        self._ensure(model, exclude_body)
        self._update_poses(model, data)

        n = dirs_world.shape[0]
        if self._n_rays != n:
            self._dist = wp.zeros(n, dtype=wp.float32, device=self.device)
            self._hit = wp.zeros(n, dtype=wp.int32, device=self.device)
            self._n_rays = n
        dirs_w = wp.array(
            np.ascontiguousarray(dirs_world, dtype=np.float32),
            dtype=wp.vec3,
            device=self.device,
        )
        wp.launch(
            _kernel_cast,
            dim=n,
            inputs=[
                self._mesh.id,
                wp.vec3(*[float(x) for x in origin]),
                dirs_w,
                self.max_range,
            ],
            outputs=[self._dist, self._hit],
            device=self.device,
        )
        return self._dist.numpy(), self._hit.numpy().astype(bool)


# --------------------------------------------------------------------------
# Warp kernels (compiled once, lazily)
# --------------------------------------------------------------------------


def _define_kernels():
    import warp as wp

    @wp.kernel
    def xform(
        local_v: wp.array(dtype=wp.vec3),
        vert_geom: wp.array(dtype=wp.int32),
        xpos: wp.array(dtype=wp.vec3),
        xmat: wp.array(dtype=wp.mat33),
        world_v: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        g = vert_geom[i]
        world_v[i] = xpos[g] + xmat[g] * local_v[i]

    @wp.kernel
    def cast(
        mesh: wp.uint64,
        origin: wp.vec3,
        dirs: wp.array(dtype=wp.vec3),
        max_range: wp.float32,
        dist: wp.array(dtype=wp.float32),
        hit: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        query = wp.mesh_query_ray(mesh, origin, dirs[i], max_range)
        if query.result:
            dist[i] = query.t
            hit[i] = 1
        else:
            dist[i] = 0.0
            hit[i] = 0

    return xform, cast


_kernel_xform = None
_kernel_cast = None


def _ensure_kernels():
    global _kernel_xform, _kernel_cast
    if _kernel_xform is None:
        _kernel_xform, _kernel_cast = _define_kernels()
