"""Deformable body builder for the 3-DOF AVBD `Solver`.

Builds a tet-mesh + tet-edge graph + surface-triangle list for a soft body,
then binds it to the existing 3-DOF `Solver` as one particle per vertex with
one `DISTANCE` constraint per unique tet edge.

This is a v1 mass-spring stand-in for proper per-element FEM. The AVBD paper
(SIGGRAPH 2025) and the VBD predecessor (Chen et al. 2024) treat deformables
with per-element strain energy (Neo-Hookean / co-rotated StVK), assembling
a per-vertex 3×3 Hessian block from contributing tets. Edge-spring is what
PBD/XPBD demos use — cheap, well-defined, but no volume preservation and no
shear/bending stiffness. It's enough to:

  - demonstrate the unified-solver claim (same `Solver`, same kernels,
    just one new constraint subgraph the solver does not distinguish);
  - exercise the constraint pool at ~hundreds of particles and thousands
    of constraints;
  - validate the rendering path (deformed surface + vertex point cloud).

Upgrade path to paper-faithful: add a TET_FEM constraint type whose row
references all 4 tet vertices and whose primal/dual kernels accumulate
∂W/∂x_i and diag(∂²W/∂x_i∂x_iᵀ) for vertex i ∈ {0..3} of the tet, using
co-rotated linear FEM with `wp.svd3` for the polar decomposition.

Mesh source: Stanford bunny OBJ from alecjacobson/common-3d-test-models
(cached under ~/.cache/avbd3d/). A procedural-sphere fallback runs if the
network is unreachable.
"""

from __future__ import annotations

import hashlib
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

from .scene import Body
from .solver import Solver

BUNNY_URL = (
    "https://raw.githubusercontent.com/alecjacobson/"
    "common-3d-test-models/master/data/stanford-bunny.obj"
)


# -----------------------------------------------------------------------------
# Mesh loading + caching
# -----------------------------------------------------------------------------

def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    out = base / "avbd3d"
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_bunny(url: str = BUNNY_URL, force: bool = False) -> Path:
    """Cache the Stanford bunny OBJ locally. Returns the cached path.

    First call hits the network (~2.4 MB); subsequent calls are free.
    """
    path = _cache_dir() / "stanford-bunny.obj"
    if path.exists() and not force and path.stat().st_size > 1024:
        return path
    print(f"[avbd3d.deformable] downloading {url} → {path}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    path.write_bytes(data)
    return path


def _close_boundary_loops(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Seal every open boundary loop with a centroid-fan patch.

    A boundary edge is one used by exactly one face. We walk each connected
    loop of boundary edges (following the directed half-edge opposite the
    existing face so the fan inherits outward winding), then add a single
    centroid vertex per loop and triangulate it as a fan. The patch may be
    non-planar — that's fine: it's a triangulation, not a flat cap.

    Used at load time on the Stanford bunny, whose original scan ships with
    a large opening on the underbelly plus a few small head/ear gaps.
    Without this the underside renders as a see-through hole.
    """
    from collections import defaultdict
    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    # Count each undirected edge and remember one directed instance.
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    edge_dir: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            edge_count[key] += 1
            edge_dir.setdefault(key, (u, v))
    # Boundary half-edges: traverse opposite to the existing face's edge
    # direction so the new fan triangles' winding matches the surface.
    nxt: dict[int, int] = {}
    for key, count in edge_count.items():
        if count != 1:
            continue
        u, v = edge_dir[key]
        nxt[v] = u
    if not nxt:
        return mesh
    visited: set[int] = set()
    new_faces: list[list[int]] = list(faces.tolist())
    new_verts: list[list[float]] = list(verts.tolist())
    for start in list(nxt.keys()):
        if start in visited:
            continue
        loop: list[int] = []
        cur = start
        guard = 0
        while cur not in visited and guard < len(nxt) + 4:
            visited.add(cur)
            loop.append(cur)
            cur = nxt.get(cur, -1)
            if cur < 0 or cur == start:
                break
            guard += 1
        if len(loop) < 3:
            continue
        centroid = np.mean([verts[i] for i in loop], axis=0)
        c_idx = len(new_verts)
        new_verts.append(centroid.tolist())
        for i in range(len(loop)):
            new_faces.append([c_idx, loop[i], loop[(i + 1) % len(loop)]])
    return trimesh.Trimesh(
        vertices=np.asarray(new_verts, dtype=np.float64),
        faces=np.asarray(new_faces, dtype=np.int64),
        process=False,
    )


def load_bunny_surface(scale: float = 1.0,
                       center: tuple[float, float, float] = (0.0, 0.6, 0.0),
                       upright: bool = True) -> trimesh.Trimesh:
    """Load + normalize the Stanford bunny.

    Normalization: centre at origin, longest bbox extent → unit length, then
    scale and translate. `upright=True` flips the bunny so its head points
    +Y (the OBJ ships head-up but ear-direction varies between repos; we
    pick a known frame).
    """
    try:
        path = download_bunny()
        # process=True merges duplicate verts and fixes inconsistent face
        # winding from the OBJ. Without it the Stanford bunny ships with
        # cracks along the scan-stitch seams and a few isolated flipped
        # triangles, both of which render as black/transparent tears with
        # any solid-surface shader.
        mesh = trimesh.load(path, force="mesh", process=True)
    except Exception as e:
        # Fallback: a coarse procedural sphere — keeps the demo runnable
        # offline. Same downstream API, just less recognizable.
        print(f"[avbd3d.deformable] bunny download failed ({e!r}); "
              f"falling back to procedural icosphere.")
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    if not isinstance(mesh, trimesh.Trimesh):
        # OBJ with material groups loads as a Scene
        mesh = mesh.dump(concatenate=True)
    # Patch the remaining holes left by the original scan. trimesh's
    # built-in fill_holes() only patches tiny (3–4 edge) loops and won't
    # touch the bunny's big open underbelly. _close_boundary_loops below
    # walks every open boundary, fan-triangulates from its centroid, and
    # is enough to make the rendered mesh fully opaque under side="front".
    mesh = _close_boundary_loops(mesh)
    # Normalize to unit bbox at origin
    centroid = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    mesh.apply_translation(-centroid)
    extent = float((mesh.bounds[1] - mesh.bounds[0]).max())
    if extent > 1e-9:
        mesh.apply_scale(1.0 / extent)
    if upright:
        # The common-3d-test-models bunny ships z-up; rotate it y-up so the
        # rest of avbd3d (gravity along -y) makes sense.
        R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, (1, 0, 0))
        mesh.apply_transform(R)
    mesh.apply_scale(scale)
    mesh.apply_translation(np.asarray(center, dtype=float))
    return mesh


# -----------------------------------------------------------------------------
# BCC voxel → tet lattice
# -----------------------------------------------------------------------------

@dataclass
class TetMesh:
    """Output of `build_tet_lattice`.

    - `vertices`     : (N, 3) float — world-space initial positions.
    - `tets`         : (M, 4) int — vertex indices into `vertices`.
    - `edges`        : (E, 2) int — unique tet edges (i < j), used for
      `DISTANCE` constraints.
    - `surface_tris` : (K, 3) int — triangles on the tet-mesh boundary
      (each face shared by exactly one tet). For rendering only.
    - `surface_verts`: (S,) int — indices into `vertices` that lie on the
      boundary. Used to apply per-vertex floor contact / collision only on
      the skin.
    - `volumes`      : (M,) float — signed rest volume of each tet (used for
      mass distribution).
    """
    vertices: np.ndarray
    tets: np.ndarray
    edges: np.ndarray
    surface_tris: np.ndarray
    surface_verts: np.ndarray
    volumes: np.ndarray


@dataclass
class RenderSkin:
    """High-res surface mesh skinned to a coarse tet mesh.

    For each render vertex `r_v`, we store the containing tet `tet_id` and
    the barycentric weights `(b0, b1, b2, b3)` such that

        r_v = b0·p0 + b1·p1 + b2·p2 + b3·p3

    where `p0..p3` are the tet's four vertex positions. Each frame, we
    re-evaluate this with the deformed tet positions to get the deformed
    high-res surface — that's the visual mesh the user actually sees.

    - `render_verts`  : (R, 3) — original (rest-pose) render vertex positions
    - `render_tris`   : (T, 3) — render triangle indices
    - `tet_ids`       : (R,)   — index into `TetMesh.tets` for each render vert
    - `weights`       : (R, 4) — barycentric weights summing to 1 per row
    """
    render_verts: np.ndarray
    render_tris: np.ndarray
    tet_ids: np.ndarray
    weights: np.ndarray

    def apply(self, tet_positions: np.ndarray, tets: np.ndarray) -> np.ndarray:
        """Skin: produce (R, 3) deformed render verts from (N, 3) deformed
        tet verts. Tets indexed by `tets` (M, 4). Pure linear blend — fast
        enough to run every frame on hundreds of thousands of render verts.
        """
        tet_verts = tet_positions[tets[self.tet_ids]]  # (R, 4, 3)
        return (self.weights[:, :, None] * tet_verts).sum(axis=1)


def _tet_barycentric_batch(q: np.ndarray,
                           p0: np.ndarray, p1: np.ndarray,
                           p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Vectorised barycentric coordinates of point `q` (3,) against an
    array of tets (T,3 each). Returns (T, 4) array; b_i = V_i / V_total
    where V_i is the signed volume of the tet with vertex i replaced by q.
    All four are positive ⇔ q lies inside that tet (with our orientation
    convention). NaN-safe via a tiny epsilon to avoid div-by-zero on
    degenerate tets.
    """
    # Volumes via scalar triple products. The shared-divisor form (× 6
    # cancels out of the b_i ratio) is numerically fine.
    def stp(a, b, c):  # (a × b) · c
        cross = np.cross(a, b)
        return np.einsum("ij,ij->i", cross, c)
    V = stp(p1 - p0, p2 - p0, p3 - p0)
    V0 = stp(p1 - q, p2 - q, p3 - q)
    V1 = stp(q - p0, p2 - p0, p3 - p0)
    V2 = stp(p1 - p0, q - p0, p3 - p0)
    V3 = stp(p1 - p0, p2 - p0, q - p0)
    eps = np.where(np.abs(V) < 1.0e-20, 1.0e-20, V)
    return np.stack([V0 / eps, V1 / eps, V2 / eps, V3 / eps], axis=-1)


def build_render_skin(render_verts: np.ndarray,
                      render_tris: np.ndarray,
                      tet: TetMesh) -> RenderSkin:
    """For each render vertex, find the tet that contains it (or, if it
    sits just outside the tet lattice, the tet whose barycentric extrapolation
    is closest to interior) and store barycentric weights.

    Speed: an rtree spatial index of tet AABBs gives O(log T) per-vertex
    AABB query, and each vertex usually has ≤ 10 candidate tets to
    barycentric-test. For 35k bunny verts × 1000 tets this completes in
    well under a second.
    """
    import rtree.index
    p0 = tet.vertices[tet.tets[:, 0]]
    p1 = tet.vertices[tet.tets[:, 1]]
    p2 = tet.vertices[tet.tets[:, 2]]
    p3 = tet.vertices[tet.tets[:, 3]]
    tet_min = np.minimum.reduce([p0, p1, p2, p3]).astype(float)
    tet_max = np.maximum.reduce([p0, p1, p2, p3]).astype(float)
    # Pad AABBs by 1% of the smallest cell so verts on the boundary still
    # land in some box's query result.
    pad = 0.005 * float((tet_max - tet_min).max(initial=1.0))
    properties = rtree.index.Property()
    properties.dimension = 3
    idx = rtree.index.Index(properties=properties)
    for tid in range(len(tet.tets)):
        idx.insert(tid, (
            float(tet_min[tid, 0] - pad), float(tet_min[tid, 1] - pad),
            float(tet_min[tid, 2] - pad),
            float(tet_max[tid, 0] + pad), float(tet_max[tid, 1] + pad),
            float(tet_max[tid, 2] + pad),
        ))
    tet_centroids = 0.25 * (p0 + p1 + p2 + p3)

    n_render = len(render_verts)
    tet_ids = np.full(n_render, -1, dtype=np.int32)
    weights = np.zeros((n_render, 4), dtype=np.float32)
    n_outside = 0
    for vi, q in enumerate(render_verts):
        qf = (float(q[0]), float(q[1]), float(q[2]))
        cands = list(idx.intersection((*qf, *qf)))
        if not cands:
            # Vert is outside every AABB — happens for surface verts that
            # poke beyond the voxel padding. Fall back to nearest tet
            # centroid.
            d2 = np.sum((tet_centroids - q) ** 2, axis=1)
            cands = [int(np.argmin(d2))]
        cands_arr = np.asarray(cands, dtype=np.int32)
        b = _tet_barycentric_batch(
            q.astype(np.float64),
            p0[cands_arr], p1[cands_arr], p2[cands_arr], p3[cands_arr],
        )
        # Pick the tet whose minimum bary coord is largest (closest to
        # interior). If any tet has all b_i ≥ 0 we're inside — done.
        min_b = b.min(axis=1)
        best = int(np.argmax(min_b))
        if min_b[best] < -1.0e-3:
            n_outside += 1
        tet_ids[vi] = cands_arr[best]
        weights[vi] = b[best].astype(np.float32)
    if n_outside:
        print(f"[avbd3d.deformable] {n_outside}/{n_render} render verts fall "
              f"outside the tet lattice and use barycentric extrapolation "
              f"(this is fine — they still deform smoothly).")
    return RenderSkin(
        render_verts=render_verts.astype(np.float32),
        render_tris=render_tris.astype(np.int32),
        tet_ids=tet_ids,
        weights=weights,
    )


# The 6 tetrahedra of the Freudenthal/Kuhn unit-cube subdivision. Every tet
# touches the (0,0,0)–(1,1,1) main diagonal; faces across the cube boundary
# are face-conforming for the standard split. Indices encode corner bitmask
# (x | y<<1 | z<<2) so corner_index((i,j,k)) lines up with voxel deduplication.
_KUHN_TETS = np.array([
    [0, 1, 3, 7],  # 000 100 110 111
    [0, 1, 5, 7],  # 000 100 101 111
    [0, 2, 3, 7],  # 000 010 110 111
    [0, 2, 6, 7],  # 000 010 011 111
    [0, 4, 5, 7],  # 000 001 101 111
    [0, 4, 6, 7],  # 000 001 011 111
], dtype=np.int32)


def _corner_offsets() -> np.ndarray:
    """Return the 8 cube-corner offsets ordered by bitmask (x | y<<1 | z<<2)."""
    return np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)
                     if True], dtype=np.int32)


# The bitmask ordering above is what _KUHN_TETS indexes into. Make it explicit:
_CORNER_OFFSETS = np.array([
    (0, 0, 0),  # 0
    (1, 0, 0),  # 1
    (0, 1, 0),  # 2
    (1, 1, 0),  # 3
    (0, 0, 1),  # 4
    (1, 0, 1),  # 5
    (0, 1, 1),  # 6
    (1, 1, 1),  # 7
], dtype=np.int32)


def build_tet_lattice(surface: trimesh.Trimesh,
                      resolution: int = 12,
                      padding_cells: float = 0.5) -> TetMesh:
    """Voxelize `surface` at `resolution` cells along the longest bbox axis,
    keep voxels whose center is inside the surface, then split each kept
    voxel into 6 tets via the Kuhn diagonal subdivision.

    Resolution 12 on the Stanford bunny → ~200-400 tet vertices,
    ~1000-2000 tets, ~3000-6000 unique edges (= DISTANCE constraints).
    """
    if not surface.is_watertight:
        # mesh.contains needs an "inside" definition — use the ray-test
        # fallback that trimesh ships (a winding-number-ish algorithm
        # that handles small holes).
        pass

    bounds = surface.bounds.copy()  # (2, 3)
    cell = float((bounds[1] - bounds[0]).max()) / float(resolution)
    if cell <= 0.0:
        raise ValueError("degenerate bounds")
    lo = bounds[0] - padding_cells * cell
    hi = bounds[1] + padding_cells * cell
    n = np.ceil((hi - lo) / cell).astype(int) + 1  # vertex grid dims (Nx+1)

    # Generate voxel centers
    ix, iy, iz = np.meshgrid(np.arange(n[0] - 1), np.arange(n[1] - 1),
                             np.arange(n[2] - 1), indexing="ij")
    centers = lo + (np.stack([ix, iy, iz], axis=-1).astype(float) + 0.5) * cell
    centers_flat = centers.reshape(-1, 3)
    inside_flat = surface.contains(centers_flat)
    inside = inside_flat.reshape(n[0] - 1, n[1] - 1, n[2] - 1)

    # Dilate the kept set by one cell so the tet mesh fully covers the
    # surface shell: any voxel adjacent (6-conn) to an inside voxel is also
    # kept, even if its own center is outside. Without this, ~60% of the
    # high-res render verts fall just outside the lattice (the surface is
    # closer to the kept-voxel face than the voxel center is) and the skin
    # is forced to extrapolate everywhere. With dilation, only verts that
    # poke into thin extremities (ear tips, leg corners) need any
    # extrapolation at all.
    dilated = inside.copy()
    dilated[1:, :, :] |= inside[:-1, :, :]
    dilated[:-1, :, :] |= inside[1:, :, :]
    dilated[:, 1:, :] |= inside[:, :-1, :]
    dilated[:, :-1, :] |= inside[:, 1:, :]
    dilated[:, :, 1:] |= inside[:, :, :-1]
    dilated[:, :, :-1] |= inside[:, :, 1:]
    inside = dilated
    kept_ijk = np.argwhere(inside)  # (M, 3)

    if len(kept_ijk) == 0:
        raise RuntimeError(
            "no voxels landed inside the surface — bump --bunny-resolution "
            "or check the mesh."
        )

    # Build the vertex pool. A vertex exists at (i, j, k) iff some kept voxel
    # references it as a corner. We assign each used (i, j, k) a sequential
    # vertex index via a dict.
    idx_of: dict[tuple[int, int, int], int] = {}
    verts_xyz: list[np.ndarray] = []

    def vid(i: int, j: int, k: int) -> int:
        key = (int(i), int(j), int(k))
        h = idx_of.get(key)
        if h is None:
            h = len(verts_xyz)
            idx_of[key] = h
            xyz = lo + np.array([i, j, k], dtype=float) * cell
            verts_xyz.append(xyz)
        return h

    tets_list: list[tuple[int, int, int, int]] = []
    for vi, vj, vk in kept_ijk:
        # Each cube corner c (bitmask 0..7) maps to global (i,j,k).
        corner_ids = np.empty(8, dtype=np.int32)
        for c in range(8):
            dx, dy, dz = _CORNER_OFFSETS[c]
            corner_ids[c] = vid(vi + dx, vj + dy, vk + dz)
        for tet_local in _KUHN_TETS:
            a, b, c, d = corner_ids[tet_local]
            tets_list.append((int(a), int(b), int(c), int(d)))

    vertices = np.stack(verts_xyz, axis=0).astype(np.float32)
    tets = np.array(tets_list, dtype=np.int32)

    # Rest volumes (signed; absolute used for mass)
    a = vertices[tets[:, 0]]
    b = vertices[tets[:, 1]]
    c = vertices[tets[:, 2]]
    d = vertices[tets[:, 3]]
    volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), (d - a)) / 6.0

    # Unique tet edges (4-choose-2 = 6 per tet)
    edge_pairs = np.concatenate([
        tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
        tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
    ], axis=0)
    edge_pairs = np.sort(edge_pairs, axis=1)
    edges = np.unique(edge_pairs, axis=0).astype(np.int32)

    # Surface = faces that appear exactly once across the tet pool.
    # 4 faces per tet; each face is the sorted 3-tuple of vertex ids.
    face_a = tets[:, [1, 2, 3]]
    face_b = tets[:, [0, 2, 3]]
    face_c = tets[:, [0, 1, 3]]
    face_d = tets[:, [0, 1, 2]]
    all_faces = np.concatenate([face_a, face_b, face_c, face_d], axis=0)
    all_faces_sorted = np.sort(all_faces, axis=1)
    # Find unique faces and how many times each appears.
    uniq_sorted, inv, counts = np.unique(
        all_faces_sorted, axis=0, return_inverse=True, return_counts=True,
    )
    surface_mask = counts == 1
    # Recover one ORIGINAL (unsorted) face per surface group so winding is
    # preserved for rendering. For each surface group, pick the first
    # original face that landed there.
    surface_sorted_ids = np.where(surface_mask)[0]
    # Map sorted-id → first original face index
    first_original = np.full(uniq_sorted.shape[0], -1, dtype=np.int64)
    for orig_idx, group in enumerate(inv):
        if first_original[group] == -1:
            first_original[group] = orig_idx
    pick = first_original[surface_sorted_ids]
    surface_tris = all_faces[pick].astype(np.int32)

    surface_verts = np.unique(surface_tris.reshape(-1)).astype(np.int32)

    return TetMesh(
        vertices=vertices,
        tets=tets,
        edges=edges,
        surface_tris=surface_tris,
        surface_verts=surface_verts,
        volumes=volumes.astype(np.float32),
    )


# -----------------------------------------------------------------------------
# Scene binding
# -----------------------------------------------------------------------------

@dataclass
class DeformableBody:
    """Handle returned by `bind_tet_mesh_to_solver`.

    `bodies` is in 1-to-1 correspondence with `tet.vertices`. `tet` is the
    immutable rest configuration; query `solver.positions()[indices]` each
    frame to get the deformed state.
    """
    tet: TetMesh
    bodies: list[Body]
    indices: np.ndarray  # (N,) solver body indices for tet.vertices
    edge_constraints: list[object]   # ConstraintHandle per edge
    floor_constraints: list[object]  # ConstraintHandle per surface vertex
    volume_constraints: list[object] = field(default_factory=list)  # one per tet (TET_VOLUME)
    skin: RenderSkin | None = None  # high-res render mesh bound to the tet pool

    def skin_positions(self, current_tet_positions: np.ndarray) -> np.ndarray:
        """Compute the deformed high-res render mesh positions from the
        current tet vertex positions. `current_tet_positions` is (N, 3) in
        SOLVER body-index order — pass `solver.positions()[deform.indices]`.
        Returns (R, 3) deformed render-vertex positions.
        """
        if self.skin is None:
            raise RuntimeError("This DeformableBody has no skin attached. "
                               "Build with `make_bunny(...)` (which attaches "
                               "the Stanford bunny surface), or call "
                               "`attach_render_skin` manually.")
        return self.skin.apply(current_tet_positions, self.tet.tets)


def bind_tet_mesh_to_solver(solver: Solver,
                            tet: TetMesh,
                            density: float = 1000.0,
                            edge_stiffness: float = 5.0e4,
                            edge_fracture: float = math.inf,
                            volume_stiffness: float = 1.0e4,
                            floor_y: float | None = 0.0,
                            friction: float = 0.5,
                            surface_collide: bool = False,
                            surface_radius_scale: float = 0.35) -> DeformableBody:
    """Add one particle per tet vertex + one DISTANCE constraint per tet edge.

    `edge_stiffness` is the per-edge spring `k` (matches the AVBD penalty
    clamp ceiling). For mass-spring tet networks `k ≈ E · (mean_edge_length)`
    gives roughly Young-modulus-equivalent stiffness; the default tuned to a
    rubbery soft body (E ≈ 1 MPa-ish at the demo's geometric scale).

    Mass per vertex = density × (¼ × sum of |volume| of incident tets).

    `surface_collide=True` enables sphere-sphere self-collision among surface
    vertices, with each surface vertex given a tiny collision radius equal to
    `surface_radius_scale × mean_edge_length`. Off by default because the
    O(N²) CPU broadphase is the bottleneck on Apple Silicon at this scale.

    `volume_stiffness > 0` emits one `TET_VOLUME` constraint per tet (soft,
    AVBD penalty clamped to this ceiling). Without volume preservation the
    tet network can pancake under floor contact since edge-only mass-spring
    has zero shear / inversion resistance. Set to 0 to disable (edge-only).
    """
    # Per-vertex mass = density × ¼ Σ incident tet |volume|.
    masses = np.zeros(len(tet.vertices), dtype=np.float32)
    abs_vol = np.abs(tet.volumes)
    for vid in range(4):
        np.add.at(masses, tet.tets[:, vid], abs_vol)
    masses *= 0.25 * density
    # Floor of 1e-6 in case a vertex has no incident tets (shouldn't happen
    # but defensive — a zero mass would make Solver treat the vertex as
    # kinematic-static, which is NOT what we want).
    masses = np.maximum(masses, 1e-6)

    # Mean edge length for the optional surface-collide radius.
    if len(tet.edges):
        edge_lens = np.linalg.norm(
            tet.vertices[tet.edges[:, 0]] - tet.vertices[tet.edges[:, 1]],
            axis=1,
        )
        mean_edge = float(edge_lens.mean())
    else:
        mean_edge = 0.0

    bodies: list[Body] = []
    indices = np.empty(len(tet.vertices), dtype=np.int32)
    surface_set = set(int(i) for i in tet.surface_verts)
    surface_radius = surface_radius_scale * mean_edge if mean_edge > 0 else 0.0
    for i, p in enumerate(tet.vertices):
        on_surface = i in surface_set
        b = solver.add_particle(
            position=tuple(float(v) for v in p),
            mass=float(masses[i]),
            collide=bool(surface_collide and on_surface and surface_radius > 0),
            radius=(surface_radius if (surface_collide and on_surface) else None),
            friction=friction,
        )
        bodies.append(b)
        indices[i] = b.index

    edge_constraints: list[object] = []
    for a, b in tet.edges:
        body_a = bodies[int(a)]
        body_b = bodies[int(b)]
        rest = float(np.linalg.norm(tet.vertices[int(a)] - tet.vertices[int(b)]))
        h = solver.add_distance(
            body_a, body_b, rest=rest,
            stiffness=edge_stiffness, fracture=edge_fracture,
        )
        edge_constraints.append(h)

    floor_constraints: list[object] = []
    if floor_y is not None:
        for sv in tet.surface_verts:
            h = solver.add_floor_contact(
                bodies[int(sv)], floor_y=float(floor_y), friction=friction,
            )
            floor_constraints.append(h)

    volume_constraints: list[object] = []
    if volume_stiffness > 0.0:
        for v0, v1, v2, v3 in tet.tets:
            volume_constraints.append(
                solver.add_tet_volume(
                    bodies[int(v0)], bodies[int(v1)],
                    bodies[int(v2)], bodies[int(v3)],
                    stiffness=float(volume_stiffness),
                )
            )

    return DeformableBody(
        tet=tet,
        bodies=bodies,
        indices=indices,
        edge_constraints=edge_constraints,
        floor_constraints=floor_constraints,
        volume_constraints=volume_constraints,
    )


# -----------------------------------------------------------------------------
# One-shot convenience
# -----------------------------------------------------------------------------

def make_bunny(solver: Solver,
               resolution: int = 12,
               scale: float = 1.2,
               center: tuple[float, float, float] = (0.0, 0.8, 0.0),
               density: float = 1000.0,
               edge_stiffness: float = 5.0e4,
               volume_stiffness: float = 1.0e4,
               friction: float = 0.5,
               floor_y: float | None = 0.0,
               surface_collide: bool = False,
               attach_skin: bool = True) -> DeformableBody:
    """Load + tetrahedralize the Stanford bunny and bind it to `solver`.

    Returns a `DeformableBody` whose `tet.surface_tris` you can hand to a
    viewer for the coarse voxel boundary, whose `indices` you can use to
    query deformed positions each frame, and (when `attach_skin=True`)
    whose `skin` carries the high-res Stanford bunny surface bound to the
    tet pool via barycentric coordinates — that's what you want to render
    if you want it to look like a bunny rather than a pile of voxels.
    """
    surf = load_bunny_surface(scale=scale, center=center)
    tet = build_tet_lattice(surf, resolution=resolution)
    deform = bind_tet_mesh_to_solver(
        solver, tet,
        density=density,
        edge_stiffness=edge_stiffness,
        volume_stiffness=volume_stiffness,
        friction=friction,
        floor_y=floor_y,
        surface_collide=surface_collide,
    )
    if attach_skin:
        deform.skin = build_render_skin(
            render_verts=np.asarray(surf.vertices, dtype=np.float32),
            render_tris=np.asarray(surf.faces, dtype=np.int32),
            tet=tet,
        )
    return deform
