"""AVBD 3D rigid-body solver (6-DOF).

Sibling of the particle Solver in `solver.py`. Each body now has full SE(3)
state: position (vec3) + orientation (quat) + linear velocity (vec3) +
angular velocity (vec3) + mass + body-local inverse inertia tensor.

Constraint pool follows the same per-row pattern as the 3-DOF solver but
with per-side **body-local** anchor offsets so the contact / pin point
rotates with the body each iteration. Per-body primal solve is a 6×6
SPD system (kernels solve via Schur-complement of 3×3 blocks).

MVP scope:
  - Bodies: rigid box (with rotation).
  - Constraints: FLOOR_CONTACT_6DOF per body corner + CONTACT_TANGENT_6DOF
    friction rows + PIN_6DOF (3 axis rows per pin).
  - No body-body contact yet (next iteration; needs OBB-OBB SAT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from . import kernels_6dof as K
from .coloring import build_body_edges, color_summary, greedy_color, spatial_8color

# Re-export the constraint type codes for callers / tests.
FLOOR_CONTACT_6DOF = 0
CONTACT_TANGENT_6DOF = 1
PIN_6DOF = 2
BOX_BOX_CONTACT_6DOF = 3


def _obb_sat(c_A: np.ndarray, R_A: np.ndarray, e_A: np.ndarray,
             c_B: np.ndarray, R_B: np.ndarray, e_B: np.ndarray,
             margin: float = 0.005):
    """Full 15-axis SAT for two OBBs (Ericson, Real-Time Collision Detection
    §4.4). Returns (axis_idx, n_hat, overlap) where n_hat points from B to A
    (i.e., the push direction for A), or None if separated by more than
    `margin` along any axis.

    The `margin` (default 5 mm) is the 2D demo's `COLLISION_MARGIN` idea —
    bodies separated by a hair still emit a contact so the warm-start λ
    persists across the gap. Without it, freshly-stacked boxes bounce as
    the contact blinks off mid-settle and λ resets from PENALTY_MIN.

    axis_idx ∈ [0,3) : A's face axis k → reference body = A.
    axis_idx ∈ [3,6) : B's face axis k → reference body = B.
    axis_idx ∈ [6,15): edge×edge cross product → edge-edge contact.
    """
    t = c_B - c_A
    eps = 1e-6
    best_idx, best_overlap, best_axis = -1, np.inf, None
    # 3 face axes from A
    for k in range(3):
        L = R_A[:, k]
        rA = e_A[k]  # only one term survives — L is A's own axis
        rB = sum(e_B[m] * abs(np.dot(L, R_B[:, m])) for m in range(3))
        sep = abs(np.dot(t, L))
        overlap = rA + rB - sep
        if overlap < -margin:
            return None
        if overlap < best_overlap:
            best_overlap = overlap
            best_idx = k
            # n_hat from B to A: opposite of L's sign relative to t.
            best_axis = -L if np.dot(t, L) > 0 else L
    # 3 face axes from B
    for k in range(3):
        L = R_B[:, k]
        rA = sum(e_A[m] * abs(np.dot(L, R_A[:, m])) for m in range(3))
        rB = e_B[k]
        sep = abs(np.dot(t, L))
        overlap = rA + rB - sep
        if overlap < -margin:
            return None
        if overlap < best_overlap:
            best_overlap = overlap
            best_idx = 3 + k
            best_axis = -L if np.dot(t, L) > 0 else L
    # 9 edge × edge cross products
    for i in range(3):
        for j in range(3):
            L = np.cross(R_A[:, i], R_B[:, j])
            n = np.linalg.norm(L)
            if n < eps:
                continue  # parallel edges — degenerate axis, skip
            L = L / n
            rA = sum(e_A[m] * abs(np.dot(L, R_A[:, m])) for m in range(3))
            rB = sum(e_B[m] * abs(np.dot(L, R_B[:, m])) for m in range(3))
            sep = abs(np.dot(t, L))
            overlap = rA + rB - sep
            if overlap < 0:
                return None
            if overlap < best_overlap:
                best_overlap = overlap
                best_idx = 6 + 3 * i + j
                best_axis = -L if np.dot(t, L) > 0 else L
    return (best_idx, best_axis.astype(np.float32), float(best_overlap))


@dataclass
class RigidBody:
    """Handle returned by add_box. Holds the body index + half-extents so
    the caller can find corners (for add_floor_contact_box) or render."""
    index: int
    half_extents: tuple[float, float, float]


@dataclass
class _Row:
    type: int
    body_a: int
    body_b: int = -1  # for PIN_6DOF: re-purposed to hold the axis (0/1/2)
    world_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    off_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    off_b: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest: float = 0.0
    stiffness: float = math.inf
    fracture: float = math.inf
    fmin: float = -math.inf
    fmax: float = math.inf
    sibling: int = -1
    friction: float = 0.0          # μ_d (dynamic / kinetic)
    friction_static: float = 0.0   # μ_s ≥ μ_d (AVBD Sec 3.3)
    partner: int = -1              # other tangent row of the (t,b) pair


def box_inertia_local(mass: float, hx: float, hy: float, hz: float) -> np.ndarray:
    """Uniform-density solid-box body-local inertia tensor (diagonal).
    Standard result: I_xx = m/3 (h_y² + h_z²) for half-extents (hx, hy, hz)
    (equivalent to m/12 · (full_y² + full_z²)). Mass m = 0 ⇒ zero matrix
    (treated as static by the kernels)."""
    if mass <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)
    Ixx = mass / 3.0 * (hy * hy + hz * hz)
    Iyy = mass / 3.0 * (hx * hx + hz * hz)
    Izz = mass / 3.0 * (hx * hx + hy * hy)
    return np.diag([Ixx, Iyy, Izz]).astype(np.float32)


def box_inv_inertia_local(mass: float, hx: float, hy: float, hz: float) -> np.ndarray:
    """Inverse of `box_inertia_local`. Zero matrix for static bodies — the
    primal kernel early-outs on m≤0 so this is never actually inverted, but
    we ship zeros to keep numpy/Warp arrays uniform."""
    if mass <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)
    I = box_inertia_local(mass, hx, hy, hz)
    return np.linalg.inv(I).astype(np.float32)


def box_inertia_local_or_zero(mass: float, hx: float, hy: float, hz: float) -> np.ndarray:
    """Body-local inertia or zero for static bodies — mirror of
    box_inv_inertia_local but for the un-inverted tensor. Used by the
    optimized primal_update_6dof so the per-iter wp.inverse() can be
    eliminated."""
    if mass <= 0.0:
        return np.zeros((3, 3), dtype=np.float32)
    return box_inertia_local(mass, hx, hy, hz)


class Solver6DOF:
    """6-DOF AVBD rigid body solver. Sibling of `Solver` for full SE(3) bodies."""

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        iterations: int = 10,
        gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
        alpha: float = 0.99,
        beta: float = 1.0e5,
        gamma: float = 0.99,
        post_stabilize: bool = True,
        device: str = "cpu",
        max_linear_speed: float = 30.0,
        max_angular_speed: float = 50.0,
        substeps: int = 1,
        friction_static_mult: float = 1.5,
    ):
        wp.init()
        self.device = device
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.gravity = tuple(float(g) for g in gravity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.post_stabilize = bool(post_stabilize)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        # Sub-stepping (AVBD paper Fig. 6 uses 5 substeps × 5 iters for the
        # card-tower demo). Splits step() into N inner solves with dt = dt/N;
        # each substep does its own warm-start + iter loop + post-stab. The
        # smaller per-substep correction cuts the post-stab-snap impulse that
        # otherwise makes stacked boxes bounce.
        self.substeps = max(1, int(substeps))
        # Static-vs-dynamic friction (AVBD Sec 3.3). Each tangent row carries
        # both μ_d (kinetic) and μ_s ≥ μ_d (stiction); the update_static_
        # friction_6dof kernel picks whichever is appropriate at substep
        # start based on the previous frame's ||λ_tb||. 1.5× is a standard
        # textbook ratio (e.g., dry steel-on-steel μ_s/μ_d ≈ 1.4–1.6); call
        # set_friction_static_mult to override per scene.
        self.friction_static_mult = max(1.0, float(friction_static_mult))

        # Per-body state (numpy buffers; flushed to Warp lazily).
        self._x: list[tuple[float, float, float]] = []
        self._q: list[tuple[float, float, float, float]] = []  # xyzw
        self._v: list[tuple[float, float, float]] = []
        self._omega: list[tuple[float, float, float]] = []
        self._mass: list[float] = []
        self._inv_I_local: list[np.ndarray] = []  # 3×3 each
        self._I_local: list[np.ndarray] = []      # 3×3 each (forward, for primal opt)
        self._half_extents: list[tuple[float, float, float]] = []
        self._friction: list[float] = []

        self._rows: list[_Row] = []
        self._dirty = True
        # Body-body contact (OBB-OBB) — optional, enabled via enable_self_collision.
        # Rebuilt every step: strip BOX_BOX_CONTACT_6DOF rows + their tangents,
        # rerun SAT + face clip on each colliding pair, append new rows.
        self._self_collide: bool = False
        self._self_friction: float = 0.0

        # Warp arrays — built in _flush.
        self.x = self.q = self.v = self.omega = None
        self.prev_v = self.prev_omega = None
        self.mass = self.inv_inertia_local = self.inv_inertia_world = None
        self.inertia_local = self.inertia_world = None
        self.x_initial = self.q_initial = None
        self.x_inertial = self.q_inertial = None
        self.body_color = None
        self.c_type = self.c_body_a = self.c_body_b = None
        self.c_world_anchor = self.c_off_a = self.c_off_b = None
        self.c_rest = self.c_stiffness = None
        self.c_lambda = self.c_penalty = self.c_fmin = self.c_fmax = None
        self.c_alpha_C0 = self.c_active = self.c_fracture = None
        self.c_sibling = self.c_friction = None
        self.c_friction_static = self.c_partner = self.c_was_static = None
        self.body_con_starts = self.body_con_indices = None
        self.num_colors = 0
        # Broadphase scratch (AVBD Alg 1 line 1 — LBVH on device-side AABBs).
        # See _gpu_emit_dynamic_contacts for the per-substep pipeline.
        self._bp_half_extents = None    # wp.array(vec3) — body half-extents
        self._bp_aabb_lo = None         # wp.array(vec3)
        self._bp_aabb_hi = None
        self._bp_pair_count = None      # wp.array(int, shape=1) atomic counter
        self._bp_pair_a = None
        self._bp_pair_b = None
        self._bp_pair_overlap = None
        self._bp_pair_sat_idx = None
        self._bp_pair_n_hat = None
        self._bp_pair_depth = None
        self._bp_max_pairs = 0
        self._bp_bvh = None             # wp.Bvh — rebuilt each call
        # Tracks broadphase wall time (seconds) of the most recent rebuild
        # so the viewer can surface it without polling internals.
        self.broadphase_ms = 0.0
        # Manifold-kernel output buffers — lazy-allocated alongside the
        # broadphase pair buffers when the kernel path is active.
        self._mf_poly_scratch = None
        self._mf_contact_count = None
        self._mf_ref_is_a = None
        self._mf_n_hat = None
        self._mf_t_hat = None
        self._mf_b_hat = None
        self._mf_off_ref = None
        self._mf_off_inc = None
        # Viewer staging buffers — built lazily on first read_batched() call,
        # shared across frames, reuploaded only when body/row counts change.
        self._viewer_pack_bodies = None
        self._viewer_pack_rows = None
        self._viewer_pack_bodies_n = 0
        self._viewer_pack_rows_n = 0

        # ---- GPU-resident dynamic constraint pool (AVBD_PERFORMANCE_GAP §1) ---
        # After _init_gpu_pool runs (lazily on first step), all c_* arrays are
        # allocated at N_static + N_dyn_capacity and the hot path no longer
        # rebuilds them. Static rows occupy [0, n_static); dynamic BOX_BOX +
        # tangent rows are atomically appended into the dynamic region by the
        # gpu_pool_emit_rows kernel each substep. The pair hash table seeds
        # warm-start λ/k from the previous substep without any CPU dict.
        self._gpu_pool_ready = False
        self._gpu_pool_n_static = 0
        self._gpu_pool_n_capacity = 0
        self._gpu_pool_n_dyn_capacity = 0
        self._gpu_pool_max_pairs = 0
        self._gpu_pool_hash_cap = 0
        self._gpu_pool_pool_max = 0
        # GPU buffers — allocated once in _init_gpu_pool.
        self.n_active_rows = None    # wp.array(int, 1) atomic row count
        self.body_friction = None    # wp.array(float, n_b)
        self.body_con_counts = None  # wp.array(int, n_b) — CSR atomic histogram
        self.body_con_cursor = None  # wp.array(int, n_b) — CSR scatter cursor
        self.pool_count = None       # wp.array(int, 1) atomic pool size
        self.pool_idx_n = None       # wp.array(int, pool_max)
        self.pool_idx_t = None
        self.pool_idx_b = None
        self.pool_idx_c = None       # contact ordinal within pair [0, 4)
        self.hash_keys = None        # wp.array(int, hash_cap) — -1 = empty
        self.hash_state = None       # wp.array(float, hash_cap*8)
        # Spatial coloring cell size, set during _init_gpu_pool.
        self._spatial_cell_size = 0.0
        # One-shot warning gate for row-pool overflow on dense clusters.
        self._row_overflow_warned = False
        # Live-contact-graph coloring fingerprint — set by _maybe_recolor.
        # `None` means "first substep, run the recolor unconditionally".
        self._color_topology_sig: int | None = None
        # CUDA-graph cache for the per-substep kernel sequence (Phase B).
        # `None` means "recapture on next step" (also the CPU fallback).
        self._graph = None
        # Signature of the Python-side scalars baked into _graph at
        # capture time. Compared against the live signature on each step;
        # mismatch → invalidate. Self-protects the solver against viewer
        # GUI mutations (iterations / dt / gravity sliders) that would
        # otherwise let the old graph replay with the new config.
        self._graph_signature: tuple | None = None
        # Cached probe: does this Warp build expose graph capture?
        self._graph_supported: bool | None = None

    # ---- Scene building -----------------------------------------------------

    def add_box(
        self,
        position: tuple[float, float, float],
        half_extents: tuple[float, float, float],
        mass: float,
        orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        friction: float = 0.0,
    ) -> RigidBody:
        """Add a rigid box. `mass ≤ 0` marks a static body (kernels skip it).
        `orientation` is a quaternion in XYZW order (Warp's wp.quat convention);
        identity = (0, 0, 0, 1)."""
        idx = len(self._x)
        self._x.append(tuple(float(v) for v in position))
        self._q.append(tuple(float(v) for v in orientation))
        self._v.append(tuple(float(v) for v in velocity))
        self._omega.append(tuple(float(v) for v in angular_velocity))
        self._mass.append(float(mass))
        hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
        self._half_extents.append((hx, hy, hz))
        self._inv_I_local.append(box_inv_inertia_local(mass, hx, hy, hz))
        self._I_local.append(box_inertia_local_or_zero(mass, hx, hy, hz))
        self._friction.append(max(0.0, float(friction)))
        self._dirty = True
        return RigidBody(index=idx, half_extents=(hx, hy, hz))

    def add_floor_contact_box(
        self,
        body: RigidBody,
        floor_y: float = 0.0,
        friction: float | None = None,
        stiffness: float = math.inf,
    ) -> list[int]:
        """Emit one FLOOR_CONTACT_6DOF row per corner of the box (8 rows) plus
        tangent friction rows along x̂ and ẑ for each corner if μ > 0.
        Returns the list of normal-row indices for the 8 corners (callers can
        track these for fracture/disable purposes)."""
        mu = float(friction) if friction is not None else self._friction[body.index]
        hx, hy, hz = body.half_extents
        normal_indices: list[int] = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    off = (sx * hx, sy * hy, sz * hz)
                    n_idx = len(self._rows)
                    self._rows.append(
                        _Row(
                            type=FLOOR_CONTACT_6DOF,
                            body_a=body.index,
                            world_anchor=(0.0, float(floor_y), 0.0),
                            off_a=off,
                            stiffness=stiffness,
                            fmin=-math.inf,
                            fmax=0.0,
                        )
                    )
                    normal_indices.append(n_idx)
                    if mu > 0.0:
                        # Two tangent rows per corner: along world x̂ and ẑ.
                        # Record them as a partner pair so update_static_
                        # friction_6dof can compute the joint ||λ_tb|| from
                        # (λ_t, λ_b) per AVBD Sec 3.3 instead of each row's
                        # |λ| in isolation.
                        mu_s = mu * self.friction_static_mult
                        t_x_idx = len(self._rows)
                        self._rows.append(
                            _Row(
                                type=CONTACT_TANGENT_6DOF,
                                body_a=body.index,
                                world_anchor=(1.0, 0.0, 0.0),
                                off_a=off,
                                stiffness=stiffness,
                                sibling=n_idx,
                                friction=mu,
                                friction_static=mu_s,
                            )
                        )
                        t_z_idx = len(self._rows)
                        self._rows.append(
                            _Row(
                                type=CONTACT_TANGENT_6DOF,
                                body_a=body.index,
                                world_anchor=(0.0, 0.0, 1.0),
                                off_a=off,
                                stiffness=stiffness,
                                sibling=n_idx,
                                friction=mu,
                                friction_static=mu_s,
                            )
                        )
                        self._rows[t_x_idx].partner = t_z_idx
                        self._rows[t_z_idx].partner = t_x_idx
        self._dirty = True
        return normal_indices

    def add_pin_corner(
        self,
        body: RigidBody,
        body_local: tuple[float, float, float],
        world_point: tuple[float, float, float],
        stiffness: float = math.inf,
        fracture: float = math.inf,
    ) -> list[int]:
        """Pin a body-local point to a fixed world point along all three axes."""
        start = len(self._rows)
        out = []
        for axis in (0, 1, 2):
            out.append(len(self._rows))
            self._rows.append(
                _Row(
                    type=PIN_6DOF,
                    body_a=body.index,
                    body_b=axis,  # re-purposed: axis index 0/1/2
                    world_anchor=tuple(float(p) for p in world_point),
                    off_a=tuple(float(v) for v in body_local),
                    stiffness=stiffness,
                    fracture=fracture,
                )
            )
        self._dirty = True
        return out

    def enable_self_collision(self, enabled: bool = True,
                              default_friction: float = 0.0) -> None:
        """Turn on per-substep OBB-OBB contact generation. The Warp BVH
        broadphase + 15-axis SAT + face-clip manifold runs entirely on the
        GPU; the gpu_pool_emit_rows kernel writes BOX_BOX_CONTACT_6DOF +
        CONTACT_TANGENT_6DOF rows directly into the dynamic region of the
        c_* arrays. See AVBD_PERFORMANCE_GAP §1/§2."""
        self._self_collide = bool(enabled)
        self._self_friction = max(0.0, float(default_friction))

    def _ensure_manifold_buffers(self, cap: int) -> None:
        """Allocate or grow the manifold-kernel output buffers to match the
        broadphase pair-buffer capacity (cap = self._bp_max_pairs). Each
        pair owns 16 vec3 polygon-scratch slots + 4 contact slots."""
        if (self._mf_poly_scratch is not None
                and self._mf_poly_scratch.shape[0] >= cap * 16):
            return
        dev = self.device
        self._mf_poly_scratch = wp.zeros(cap * 16, dtype=wp.vec3, device=dev)
        self._mf_contact_count = wp.zeros(cap, dtype=int, device=dev)
        self._mf_ref_is_a = wp.zeros(cap, dtype=int, device=dev)
        self._mf_n_hat = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self._mf_t_hat = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self._mf_b_hat = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self._mf_off_ref = wp.zeros(cap * 4, dtype=wp.vec3, device=dev)
        self._mf_off_inc = wp.zeros(cap * 4, dtype=wp.vec3, device=dev)

    def _maybe_recolor(self, n_active: int) -> None:
        """Rebuild body coloring from the substep's live contact graph.

        Coloring correctness requires the adjacency to match the constraint
        set the primal sweep will see. Any static approximation (static
        rows, inflated initial AABBs, parity hash) is structurally unsound
        for a simulation where bodies move and new contacts form: two
        bodies whose initial AABBs do not overlap can still collide later
        and race in the per-color update.

        Topology-hash short-circuit: when the (a, b) adjacency is unchanged
        from the previous substep (the common case for settled stacks) we
        skip the recolor — it stays a free no-op."""
        if n_active <= 0:
            return
        n_b = len(self._x)
        if n_b < 2:
            return
        # Readback shares the bus with the n_active_rows readback we
        # already do for overflow detection.
        ba_full = self.c_body_a.numpy()
        bb_full = self.c_body_b.numpy()
        ct_full = self.c_type.numpy()
        ba = ba_full[:n_active]
        bb = bb_full[:n_active]
        ct = ct_full[:n_active]
        # PIN_6DOF re-uses c_body_b as an axis id (not a body), and
        # negative indices encode "world / no second body".
        mask = ((ct != PIN_6DOF) & (ba >= 0) & (bb >= 0) & (ba != bb))
        a_edges = ba[mask].astype(np.int64, copy=False)
        b_edges = bb[mask].astype(np.int64, copy=False)
        if a_edges.size == 0:
            # No body-body edges this substep — every body is its own
            # color class. Leave coloring alone; the static seed from
            # _flush is already a valid partition.
            sig = 0
        else:
            # Canonical (min, max) packed into one int64 → topology key.
            lo_e = np.minimum(a_edges, b_edges)
            hi_e = np.maximum(a_edges, b_edges)
            canon = np.sort((lo_e << 32) | hi_e)
            sig = hash(canon.tobytes())
        if sig == self._color_topology_sig:
            return
        self._color_topology_sig = sig

        adj = build_body_edges(n_b, a_edges.tolist(), b_edges.tolist())
        color_np = greedy_color(adj)
        new_num_colors = int(color_np.max() + 1) if n_b > 0 else 0
        color_bodies_np = np.argsort(color_np, kind="stable").astype(np.int32)
        counts = np.bincount(
            color_np, minlength=max(new_num_colors, 1)).astype(np.int32)
        offsets = np.zeros(new_num_colors + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(counts[:new_num_colors])

        dev = self.device
        # Reuse the existing body_color Warp array (size n_b is fixed
        # across substeps) but rebuild color_bodies on size change.
        self.body_color.assign(color_np)
        if (self.color_bodies is None
                or self.color_bodies.shape[0] != color_bodies_np.shape[0]):
            self.color_bodies = wp.array(color_bodies_np, dtype=int, device=dev)
        else:
            self.color_bodies.assign(color_bodies_np)
        self._color_offsets_np = offsets
        self.num_colors = new_num_colors
        self.color_counts = color_summary(color_np)
        # Per-color launch shape changed → invalidate any cached CUDA graph.
        self._graph = None

    # ---- Runtime perturbations ---------------------------------------------

    def set_position(self, body: RigidBody, p: tuple[float, float, float]) -> None:
        self._flush()
        xs = self.x.numpy().copy()
        xs[body.index] = np.array(p, dtype=np.float32)
        self.x = wp.array(xs, dtype=wp.vec3, device=self.device)

    def set_orientation(self, body: RigidBody, q_xyzw: tuple[float, float, float, float]) -> None:
        self._flush()
        qs = self.q.numpy().copy()
        qs[body.index] = np.array(q_xyzw, dtype=np.float32)
        self.q = wp.array(qs, dtype=wp.quat, device=self.device)

    def set_velocity(self, body: RigidBody, v: tuple[float, float, float]) -> None:
        self._flush()
        vs = self.v.numpy().copy()
        vs[body.index] = np.array(v, dtype=np.float32)
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    def set_angular_velocity(self, body: RigidBody, w: tuple[float, float, float]) -> None:
        self._flush()
        ws = self.omega.numpy().copy()
        ws[body.index] = np.array(w, dtype=np.float32)
        self.omega = wp.array(ws, dtype=wp.vec3, device=self.device)

    # ---- Warp upload --------------------------------------------------------

    def _flush(self) -> None:
        """Allocate GPU-resident pool. Called once when self._dirty is True.

        After this returns, all c_* arrays live in GPU memory at fixed
        capacity = N_static + N_dyn_capacity. The hot path in _step_one
        never reallocates them and never rebuilds rows from Python.

        See AVBD_PERFORMANCE_GAP §1 and the GPU-resident pool plan at
        ~/.claude/plans/here-is-major-gaps-jaunty-flute.md."""
        if not self._dirty:
            return
        n_b = len(self._x)
        n_static = len(self._rows)
        dev = self.device

        # Snapshot existing GPU state so we preserve already-simulated bodies
        # when add_box is called mid-simulation. Static rows (pins, floor
        # contacts) cannot be appended mid-sim in the GPU-resident path
        # because the layout assumes [0, n_static) is fixed; rebuild from
        # the Python list each time _flush runs is fine for that case (rare).
        cur_x = self.x.numpy() if self.x is not None else None
        cur_q = self.q.numpy() if self.q is not None else None
        cur_v = self.v.numpy() if self.v is not None else None
        cur_w = self.omega.numpy() if self.omega is not None else None
        cur_prev_v = self.prev_v.numpy() if self.prev_v is not None else None
        cur_prev_w = self.prev_omega.numpy() if self.prev_omega is not None else None
        n_b_prev = 0 if cur_x is None else int(cur_x.shape[0])

        # Capacity for the dynamic constraint region. Each broadphase pair
        # can yield up to GPU_POOL_C_PER_PAIR contact points and each
        # contact owns 3 rows (normal + 2 tangents). The pair budget
        # scales with n_bodies — see _bp_max_pairs heuristic in the
        # broadphase emitter (16·n_b).
        if self._self_collide:
            max_pairs = max(64, 16 * max(1, n_b))
            n_dyn_capacity = max_pairs * 4 * 3   # 4 contacts × 3 rows
        else:
            max_pairs = 0
            n_dyn_capacity = 0
        n_cap = n_static + n_dyn_capacity

        # Body arrays — sized to n_b. Bodies are static after init in the
        # default path; if add_box was called mid-sim, copy old state into
        # the prefix.
        x_np = np.array(self._x, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        q_np = np.array(self._q, dtype=np.float32).reshape(-1, 4) if n_b else np.zeros((0, 4), np.float32)
        v_np = np.array(self._v, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        w_np = np.array(self._omega, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        m_np = np.array(self._mass, dtype=np.float32) if n_b else np.zeros(0, np.float32)
        prev_v_np = np.zeros((n_b, 3), dtype=np.float32)
        prev_w_np = np.zeros((n_b, 3), dtype=np.float32)
        inv_I_np = (np.stack(self._inv_I_local).astype(np.float32)
                    if n_b else np.zeros((0, 3, 3), dtype=np.float32))
        I_np = (np.stack(self._I_local).astype(np.float32)
                if n_b else np.zeros((0, 3, 3), dtype=np.float32))
        fric_np = np.asarray(self._friction, dtype=np.float32) if n_b else np.zeros(0, np.float32)

        if n_b_prev > 0 and n_b_prev <= n_b:
            x_np[:n_b_prev] = cur_x[:n_b_prev]
            q_np[:n_b_prev] = cur_q[:n_b_prev]
            v_np[:n_b_prev] = cur_v[:n_b_prev]
            w_np[:n_b_prev] = cur_w[:n_b_prev]
            if cur_prev_v is not None:
                prev_v_np[:n_b_prev] = cur_prev_v[:n_b_prev]
            if cur_prev_w is not None:
                prev_w_np[:n_b_prev] = cur_prev_w[:n_b_prev]

        self.x = wp.array(x_np, dtype=wp.vec3, device=dev)
        self.q = wp.array(q_np, dtype=wp.quat, device=dev)
        self.v = wp.array(v_np, dtype=wp.vec3, device=dev)
        self.omega = wp.array(w_np, dtype=wp.vec3, device=dev)
        self.prev_v = wp.array(prev_v_np, dtype=wp.vec3, device=dev)
        self.prev_omega = wp.array(prev_w_np, dtype=wp.vec3, device=dev)
        self.mass = wp.array(m_np, dtype=float, device=dev)
        self.inv_inertia_local = wp.array(inv_I_np, dtype=wp.mat33, device=dev)
        self.inertia_local = wp.array(I_np, dtype=wp.mat33, device=dev)
        self.body_friction = wp.array(fric_np, dtype=float, device=dev)
        self.x_initial = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.q_initial = wp.zeros(n_b, dtype=wp.quat, device=dev)
        self.x_inertial = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.q_inertial = wp.zeros(n_b, dtype=wp.quat, device=dev)
        self.inv_inertia_world = wp.zeros(n_b, dtype=wp.mat33, device=dev)
        self.inertia_world = wp.zeros(n_b, dtype=wp.mat33, device=dev)

        # Body coloring — initial seed only. Edges from static body-body
        # rows (e.g. floor/PIN don't qualify); the *runtime* coloring is
        # rebuilt each substep by `_maybe_recolor` from the live
        # `c_body_a` / `c_body_b` set after broadphase fills the dynamic
        # region. Static seeding here gives the first substep a sane
        # `num_colors` before the runtime recolor takes over.
        # spatial_8color() is intentionally NOT used — it can assign the
        # same color to two bodies that share a grid cell, violating the
        # Gauss-Seidel independence invariant.
        if n_b > 0:
            body_body_a = [r.body_a for r in self._rows
                           if r.type == BOX_BOX_CONTACT_6DOF]
            body_body_b = [r.body_b for r in self._rows
                           if r.type == BOX_BOX_CONTACT_6DOF]
            adj = build_body_edges(n_b, body_body_a, body_body_b)
            color_np = greedy_color(adj)
        else:
            color_np = np.zeros(0, dtype=np.int32)
        # Reset the topology fingerprint so the first substep recolor
        # always runs against the live contact graph.
        self._color_topology_sig = None
        self.num_colors = int(color_np.max() + 1) if n_b > 0 else 0
        self.color_counts = color_summary(color_np)
        self.body_color = wp.array(color_np, dtype=int, device=dev)

        # Build the per-color body bucket arrays the primal kernel indexes
        # into. `color_bodies` is body ids sorted by color (stable);
        # `color_offsets[c..c+1]` is the slice for color c. Each per-color
        # primal launch uses dim = bucket size and color_offsets[c] as the
        # base offset, so the kernel does no per-thread color filtering and
        # launches no wasted threads.
        if n_b > 0:
            color_bodies_np = np.argsort(color_np, kind="stable").astype(np.int32)
            color_counts_np = np.bincount(
                color_np, minlength=max(self.num_colors, 1)
            ).astype(np.int32)
            color_offsets_np = np.zeros(self.num_colors + 1, dtype=np.int32)
            color_offsets_np[1:] = np.cumsum(color_counts_np[: self.num_colors])
        else:
            color_bodies_np = np.zeros(0, dtype=np.int32)
            color_offsets_np = np.zeros(1, dtype=np.int32)
        self.color_bodies = wp.array(color_bodies_np, dtype=int, device=dev)
        self._color_offsets_np = color_offsets_np

        # ---- c_* arrays at fixed capacity (no per-substep realloc) ---------
        # Static prefix [0, n_static) seeded from self._rows; dynamic
        # region [n_static, n_cap) zeroed (kernel will overwrite).
        def _f32_static(attr_fn) -> np.ndarray:
            arr = np.zeros(n_cap, dtype=np.float32)
            for i, r in enumerate(self._rows):
                arr[i] = float(attr_fn(r))
            return arr

        def _i32_static(attr_fn) -> np.ndarray:
            arr = np.zeros(n_cap, dtype=np.int32)
            for i, r in enumerate(self._rows):
                arr[i] = int(attr_fn(r))
            return arr

        anchor_np = np.zeros((n_cap, 3), dtype=np.float32)
        off_a_np = np.zeros((n_cap, 3), dtype=np.float32)
        off_b_np = np.zeros((n_cap, 3), dtype=np.float32)
        for i, r in enumerate(self._rows):
            anchor_np[i] = r.world_anchor
            off_a_np[i] = r.off_a
            off_b_np[i] = r.off_b

        self.c_type = wp.array(_i32_static(lambda r: r.type), dtype=int, device=dev)
        self.c_body_a = wp.array(_i32_static(lambda r: r.body_a), dtype=int, device=dev)
        self.c_body_b = wp.array(_i32_static(lambda r: r.body_b), dtype=int, device=dev)
        self.c_world_anchor = wp.array(anchor_np, dtype=wp.vec3, device=dev)
        self.c_off_a = wp.array(off_a_np, dtype=wp.vec3, device=dev)
        self.c_off_b = wp.array(off_b_np, dtype=wp.vec3, device=dev)
        self.c_rest = wp.array(_f32_static(lambda r: r.rest), dtype=float, device=dev)
        self.c_stiffness = wp.array(
            _f32_static(lambda r: r.stiffness), dtype=float, device=dev)
        self.c_fmin = wp.array(_f32_static(lambda r: r.fmin), dtype=float, device=dev)
        self.c_fmax = wp.array(_f32_static(lambda r: r.fmax), dtype=float, device=dev)
        self.c_fracture = wp.array(
            _f32_static(lambda r: r.fracture), dtype=float, device=dev)
        self.c_friction = wp.array(
            _f32_static(lambda r: r.friction), dtype=float, device=dev)
        self.c_friction_static = wp.array(
            _f32_static(lambda r: r.friction_static), dtype=float, device=dev)

        # Sibling/partner default to -1 in dynamic region (no sibling) so the
        # kernels' sib >= 0 checks early-out for unused slots.
        sib_np = np.full(n_cap, -1, dtype=np.int32)
        partner_np = np.full(n_cap, -1, dtype=np.int32)
        for i, r in enumerate(self._rows):
            sib_np[i] = int(r.sibling)
            partner_np[i] = int(r.partner)
        self.c_sibling = wp.array(sib_np, dtype=int, device=dev)
        self.c_partner = wp.array(partner_np, dtype=int, device=dev)

        # λ / penalty seeded for STATIC rows only; dynamic rows are seeded
        # by the emit kernel from the pair hash each substep.
        lam_np = np.zeros(n_cap, dtype=np.float32)
        pen_np = np.full(n_cap, 1.0, dtype=np.float32)
        act_np = np.zeros(n_cap, dtype=np.int32)
        for i in range(n_static):
            act_np[i] = 1
        self.c_lambda = wp.array(lam_np, dtype=float, device=dev)
        self.c_penalty = wp.array(pen_np, dtype=float, device=dev)
        self.c_alpha_C0 = wp.zeros(n_cap, dtype=float, device=dev)
        self.c_active = wp.array(act_np, dtype=int, device=dev)
        self.c_was_static = wp.zeros(n_cap, dtype=int, device=dev)

        # ---- GPU-resident pool scratch -------------------------------------
        # CSR adjacency: rebuilt by the gpu_csr_* kernels each substep so it
        # tracks dynamic contacts; capacity covers worst-case (each row
        # touches up to 2 bodies, for n_cap rows total).
        self.body_con_counts = wp.zeros(max(n_b, 1), dtype=int, device=dev)
        self.body_con_cursor = wp.zeros(max(n_b, 1), dtype=int, device=dev)
        self.body_con_starts = wp.zeros(max(n_b + 1, 2), dtype=int, device=dev)
        self.body_con_indices = wp.zeros(max(2 * n_cap, 1), dtype=int, device=dev)
        # Single-element atomic counter for the dynamic-row tail.
        n_active_np = np.array([n_static], dtype=np.int32)
        self.n_active_rows = wp.array(n_active_np, dtype=int, device=dev)
        # Allocate the broadphase pair counter and the pool entry counter
        # unconditionally. Both are 1-element ints — cost is negligible — and
        # having them always live lets the fused reset+csr-zero kernel run on
        # both self-collide and non-self-collide paths without branching.
        if self._bp_pair_count is None:
            self._bp_pair_count = wp.zeros(1, dtype=int, device=dev)
        self.pool_count = wp.zeros(1, dtype=int, device=dev)

        if self._self_collide:
            # Pair hash for warm-start: keyed on (a, b, contact_ordinal).
            # Capacity should be ≥ 2 × expected entries to keep load ≤ 0.5.
            hash_cap = max(1024, max_pairs * 8)
            self._gpu_pool_hash_cap = hash_cap
            self.hash_keys = wp.full(hash_cap, -1, dtype=int, device=dev)
            self.hash_state = wp.zeros(hash_cap * 8, dtype=float, device=dev)

            pool_max = max(64, max_pairs * 4)  # up to 4 contacts per pair
            self._gpu_pool_pool_max = pool_max
            self.pool_idx_n = wp.zeros(pool_max, dtype=int, device=dev)
            self.pool_idx_t = wp.zeros(pool_max, dtype=int, device=dev)
            self.pool_idx_b = wp.zeros(pool_max, dtype=int, device=dev)
            self.pool_idx_c = wp.zeros(pool_max, dtype=int, device=dev)

        self._gpu_pool_n_static = n_static
        self._gpu_pool_n_capacity = n_cap
        self._gpu_pool_n_dyn_capacity = n_dyn_capacity
        self._gpu_pool_max_pairs = max_pairs
        self._gpu_pool_ready = True
        self._dirty = False
        # Layout changed → previous capture (if any) is invalid.
        self._graph = None

    # ---- The step -----------------------------------------------------------

    def step(self) -> None:
        if self.substeps <= 1:
            self._step_one()
            return
        full_dt = self.dt
        sub_dt = full_dt / self.substeps
        self.dt = sub_dt
        try:
            for _ in range(self.substeps):
                self._step_one()
        finally:
            self.dt = full_dt

    def _step_one(self) -> None:
        """One AVBD substep. After the initial _flush() upload from the CPU
        scene description, the entire hot path runs on the GPU:

            1. gpu_pool_reset_and_csr_zero — reset substep atomic counters
               and zero body_con_counts in one launch
            2. broadphase + SAT     — existing kernels (compute_body_aabb,
                                      bvh_broadphase_pairs, obb_sat_pairs)
            3. obb_contact_manifold — face-clip / edge-edge → per-pair geom
            4. gpu_pool_emit_rows   — atomically appends BOX_BOX + tangent
                                      rows into the dynamic region of c_*,
                                      looks up pair hash for warm-start
            5. gpu_csr_* (3 passes) — rebuild body→constraint CSR adjacency
            6. predict + prelude + iter loop (existing kernels)
            7. gpu_pool_hash_collect — refresh hash from current state

        Two cheap readbacks remain per substep: _bp_pair_count (sizes SAT /
        manifold / emit launches) and n_active_rows (sizes row-side launches).
        Each is a single int — no per-substep Python loop and no array
        reallocation. See AVBD_PERFORMANCE_GAP.md and the GPU-resident pool
        plan at ~/.claude/plans/here-is-major-gaps-jaunty-flute.md.
        """
        if self._dirty:
            self._flush()
        n_b = len(self._x)
        if n_b == 0:
            return
        dev = self.device
        n_static = self._gpu_pool_n_static

        # ---- 1. Reset substep counters + zero CSR counts (fused) ----
        # One launch at dim=n_b: thread 0 resets the atomic counters
        # (n_active_rows, pool_count, _bp_pair_count); every thread zeros
        # its body_con_counts entry. Replaces the prior dim=1 gpu_pool_reset
        # + dim=n_b gpu_csr_zero_counts pair.
        wp.launch(
            K.gpu_pool_reset_and_csr_zero, dim=n_b,
            inputs=[self.n_active_rows, self.pool_count,
                    self._bp_pair_count, self.body_con_counts, n_static],
            device=dev,
        )

        # ---- 2-4. Broadphase + SAT + manifold + kernel row emission ----
        n_active = n_static
        if self._self_collide and n_b >= 2:
            self._gpu_emit_dynamic_contacts(n_b)
            # Single int readback — sizes the row-kernel launches below.
            # n_active_rows is the *reservation* counter; the kernel-side
            # guard rejects contacts that would write past _gpu_pool_n_capacity
            # but still bumps the counter. Clamp before using as a launch dim,
            # and grow the c_* arrays next substep on overflow so the rejected
            # contacts get a slot.
            n_active_raw = int(self.n_active_rows.numpy()[0])
            cap = self._gpu_pool_n_capacity
            if n_active_raw > cap:
                if not self._row_overflow_warned:
                    import warnings
                    warnings.warn(
                        f"avbd3d row pool overflow: reserved {n_active_raw} "
                        f"rows > capacity {cap}; growing for next substep. "
                        "Tune solver._gpu_pool_n_dyn_capacity if frequent.",
                        RuntimeWarning, stacklevel=2)
                    self._row_overflow_warned = True
                self._grow_row_pool(max(2 * self._gpu_pool_n_dyn_capacity,
                                          (n_active_raw - n_static) * 2))
            n_active = min(n_active_raw, cap)
            # Live-contact-graph recolor (round 2 §1). Must run before the
            # CSR build below uses self.color_bodies / _color_offsets_np.
            self._maybe_recolor(n_active)

        # Post-Phase-A: row-side launches all use fixed dim=row_dim with
        # device-side bounds via `self.n_active_rows[0]`. Lets the inner
        # solve loop be CUDA-graph-captured (see Phase B below).
        row_dim = self._gpu_pool_n_capacity

        # ---- 5. Rebuild body→constraint CSR adjacency on GPU ----
        # 3 passes: atomic histogram → serial scan → atomic scatter.
        # (Step 1 above already zeroed body_con_counts.)
        if n_active > 0:
            wp.launch(
                K.gpu_csr_count, dim=row_dim,
                inputs=[self.n_active_rows, self.c_type,
                        self.c_body_a, self.c_body_b, self.body_con_counts],
                device=dev,
            )
            wp.launch(
                K.gpu_csr_starts_from_counts, dim=1,
                inputs=[self.body_con_counts, self.body_con_starts, n_b],
                device=dev,
            )
            wp.copy(self.body_con_cursor, self.body_con_starts, count=n_b)
            wp.launch(
                K.gpu_csr_scatter, dim=row_dim,
                inputs=[self.n_active_rows, self.c_type,
                        self.c_body_a, self.c_body_b,
                        self.body_con_cursor, self.body_con_indices],
                device=dev,
            )

        # ---- 6. Inertial target + warm-started x⁰, q⁰ ----
        wp.launch(
            K.predict_inertial_6dof, dim=n_b,
            inputs=[self.x, self.q, self.v, self.omega, self.prev_v,
                    self.mass, self.inv_inertia_local, self.inertia_local,
                    self.dt, wp.vec3(*self.gravity)],
            outputs=[self.x_initial, self.q_initial,
                     self.x_inertial, self.q_inertial,
                     self.inv_inertia_world, self.inertia_world,
                     self.x, self.q],
            device=dev,
        )

        if n_active > 0:
            # Fused warmstart_duals + update_static_friction + cache_alpha_C0.
            main_alpha = 1.0 if self.post_stabilize else self.alpha
            wp.launch(
                K.substep_prelude_6dof, dim=row_dim,
                inputs=[self.n_active_rows,
                        self.x_initial, self.q_initial,
                        self.c_type, self.c_body_a, self.c_body_b,
                        self.c_world_anchor, self.c_off_a, self.c_off_b,
                        self.c_rest, self.c_stiffness,
                        self.c_sibling, self.c_partner, self.c_friction_static,
                        self.c_lambda, self.c_penalty, self.c_active,
                        self.c_was_static, self.c_alpha_C0,
                        self.alpha, self.gamma,
                        1 if self.post_stabilize else 0,
                        main_alpha],
                device=dev,
            )

        total_iters = self.iterations + (1 if self.post_stabilize else 0)
        # Phase B — CUDA-graph capture for the inner solve loop. On CUDA
        # the iter loop is ~234 individual launches (26 iters × 9 colors
        # + duals); capturing once and replaying with `wp.capture_launch`
        # collapses that into a single host dispatch. On CPU Warp graph
        # capture is unsupported (per Warp 1.13 docs) — fall back to the
        # uncaptured launch sequence, which is what was running before.
        # The capture region launches at fixed dims that all bound against
        # device-side counts (Phase A), so the same graph replays correctly
        # across substeps with varying `n_active_rows[0]`.
        # Invalidate cached graph whenever any captured-in scalar changes.
        # Computed even on the CPU fallback path so the signature stays
        # up-to-date across device switches and so tests can assert on it.
        sig = self._current_graph_signature()
        if sig != self._graph_signature:
            self._graph = None
            self._graph_signature = sig

        use_graph = (n_active > 0
                     and str(dev).startswith("cuda")
                     and self._graph_cuda_supported())
        if use_graph:
            if self._graph is None:
                with wp.ScopedCapture(device=dev) as cap:
                    self._run_iter_loop(total_iters, row_dim, n_b, dev)
                self._graph = cap.graph
            wp.capture_launch(self._graph)
        else:
            self._run_iter_loop(total_iters, row_dim, n_b, dev)

        # ---- 7. Refresh pair hash for next-substep warm-start ----
        if self._self_collide and self._gpu_pool_hash_cap > 0:
            wp.launch(K.gpu_pool_hash_clear, dim=self._gpu_pool_hash_cap,
                      inputs=[self.hash_keys], device=dev)
            wp.launch(
                K.gpu_pool_hash_collect, dim=self._gpu_pool_pool_max,
                inputs=[self.pool_count,
                        self.pool_idx_n, self.pool_idx_t,
                        self.pool_idx_b, self.pool_idx_c,
                        self.c_body_a, self.c_body_b,
                        self.c_lambda, self.c_penalty,
                        self.c_active, self.c_was_static,
                        self._gpu_pool_hash_cap,
                        self.hash_keys, self.hash_state],
                device=dev,
            )

    def _current_graph_signature(self) -> tuple:
        """Snapshot of every Python-side scalar the capture region bakes
        into the recorded launches. Any change here means the cached
        graph is stale and must be recaptured."""
        return (
            int(self.iterations),
            int(self.post_stabilize),
            int(self.num_colors),
            float(self.dt),
            float(self.alpha),
            float(self.beta),
            float(self.gamma),
            float(self.max_linear_speed),
            float(self.max_angular_speed),
            tuple(self._color_offsets_np.tolist()),
        )

    def _graph_cuda_supported(self) -> bool:
        """Whether the current Warp build can capture a CUDA graph. Caches
        the answer so we don't probe `wp.ScopedCapture` on every step."""
        if self._graph_supported is not None:
            return self._graph_supported
        supported = False
        try:
            supported = hasattr(wp, "ScopedCapture") and hasattr(
                wp, "capture_launch")
        except Exception:
            supported = False
        self._graph_supported = supported
        return supported

    def _run_iter_loop(self, total_iters: int, row_dim: int,
                        n_b: int, dev) -> None:
        """The capture-eligible inner solve loop: cache_alpha_C0 (post-stab
        only), per-color primal_update, dual_update, and the final
        finalize_and_cap. All launches use fixed dims (Phase A); device-side
        counts come from `self.n_active_rows[0]` / `c_active[j]`. No Python
        readbacks inside — safe to enclose in `wp.ScopedCapture`."""
        if (self.max_linear_speed > 0.0
                and math.isfinite(self.max_linear_speed)
                and self.max_angular_speed > 0.0
                and math.isfinite(self.max_angular_speed)):
            max_lin = float(self.max_linear_speed)
            max_ang = float(self.max_angular_speed)
        else:
            max_lin = math.inf
            max_ang = math.inf

        for it in range(total_iters):
            if self.post_stabilize and it == self.iterations:
                wp.launch(
                    K.cache_alpha_C0_6dof, dim=row_dim,
                    inputs=[self.n_active_rows,
                            self.x, self.q, self.x_initial, self.q_initial,
                            self.c_type, self.c_body_a, self.c_body_b,
                            self.c_world_anchor, self.c_off_a, self.c_off_b,
                            self.c_rest, self.c_active, 0.0],
                    outputs=[self.c_alpha_C0],
                    device=dev,
                )

            for color_id in range(self.num_colors):
                base = int(self._color_offsets_np[color_id])
                count = int(self._color_offsets_np[color_id + 1]) - base
                if count <= 0:
                    continue
                wp.launch(
                    K.primal_update_6dof, dim=count,
                    inputs=[self.x, self.q, self.mass,
                            self.inv_inertia_world, self.inertia_world,
                            self.x_inertial, self.q_inertial,
                            self.c_type, self.c_body_a, self.c_body_b,
                            self.c_world_anchor, self.c_off_a, self.c_off_b,
                            self.c_rest, self.c_stiffness,
                            self.c_lambda, self.c_penalty,
                            self.c_fmin, self.c_fmax,
                            self.c_alpha_C0, self.c_active,
                            self.c_sibling, self.c_friction,
                            self.c_friction_static, self.c_was_static,
                            self.body_con_starts, self.body_con_indices,
                            self.color_bodies, base, self.dt],
                    device=dev,
                )

            if it < self.iterations:
                wp.launch(
                    K.dual_update_6dof, dim=row_dim,
                    inputs=[self.n_active_rows,
                            self.x, self.q,
                            self.c_type, self.c_body_a, self.c_body_b,
                            self.c_world_anchor, self.c_off_a, self.c_off_b,
                            self.c_rest, self.c_stiffness,
                            self.c_lambda, self.c_penalty,
                            self.c_fmin, self.c_fmax,
                            self.c_alpha_C0, self.c_active, self.c_fracture,
                            self.c_sibling, self.c_friction,
                            self.c_friction_static, self.c_was_static,
                            self.beta],
                    device=dev,
                )

            if it == self.iterations - 1:
                wp.launch(
                    K.finalize_and_cap_6dof, dim=n_b,
                    inputs=[self.x, self.q, self.x_initial, self.q_initial,
                            self.mass, self.dt, max_lin, max_ang],
                    outputs=[self.v, self.omega, self.prev_v, self.prev_omega],
                    device=dev,
                )

    def _ensure_pair_buffers(self, cap: int) -> None:
        """Allocate or grow the broadphase pair-buffer set to at least `cap`.
        Sets `self._bp_max_pairs` to the new capacity. Buffers are zeroed
        on grow — counts are reset at the top of each step via the fused
        reset kernel, so callers don't need to re-zero."""
        if (self._bp_pair_a is not None
                and self._bp_pair_a.shape[0] >= cap):
            return
        dev = self.device
        self._bp_pair_a = wp.zeros(cap, dtype=int, device=dev)
        self._bp_pair_b = wp.zeros(cap, dtype=int, device=dev)
        self._bp_pair_overlap = wp.zeros(cap, dtype=int, device=dev)
        self._bp_pair_sat_idx = wp.zeros(cap, dtype=int, device=dev)
        self._bp_pair_n_hat = wp.zeros(cap, dtype=wp.vec3, device=dev)
        self._bp_pair_depth = wp.zeros(cap, dtype=float, device=dev)
        self._bp_max_pairs = cap

    def _grow_row_pool(self, new_dyn_capacity: int) -> None:
        """Grow the dynamic tail of every `c_*` array to fit at least
        `new_dyn_capacity` rows past `n_static`. Preserves the static
        prefix [0, n_static) verbatim. Called after a row-pool overflow
        so the *next* substep has the headroom that this one lacked.

        Also grows pool_idx_* to match (pool_max scales with max_pairs)."""
        n_static = self._gpu_pool_n_static
        new_cap = n_static + new_dyn_capacity
        if new_cap <= self._gpu_pool_n_capacity:
            return
        dev = self.device

        def _grow_int(arr):
            old = arr.numpy()
            buf = np.zeros(new_cap, dtype=np.int32)
            buf[: old.shape[0]] = old
            return wp.array(buf, dtype=int, device=dev)

        def _grow_f32(arr):
            old = arr.numpy()
            buf = np.zeros(new_cap, dtype=np.float32)
            buf[: old.shape[0]] = old
            return wp.array(buf, dtype=float, device=dev)

        def _grow_vec3(arr):
            old = arr.numpy()
            buf = np.zeros((new_cap, 3), dtype=np.float32)
            buf[: old.shape[0]] = old
            return wp.array(buf, dtype=wp.vec3, device=dev)

        # Scalar int / float / vec3 row arrays.
        self.c_type = _grow_int(self.c_type)
        self.c_body_a = _grow_int(self.c_body_a)
        self.c_body_b = _grow_int(self.c_body_b)
        self.c_world_anchor = _grow_vec3(self.c_world_anchor)
        self.c_off_a = _grow_vec3(self.c_off_a)
        self.c_off_b = _grow_vec3(self.c_off_b)
        self.c_rest = _grow_f32(self.c_rest)
        self.c_stiffness = _grow_f32(self.c_stiffness)
        self.c_fmin = _grow_f32(self.c_fmin)
        self.c_fmax = _grow_f32(self.c_fmax)
        self.c_fracture = _grow_f32(self.c_fracture)
        self.c_friction = _grow_f32(self.c_friction)
        self.c_friction_static = _grow_f32(self.c_friction_static)
        self.c_lambda = _grow_f32(self.c_lambda)
        self.c_penalty = _grow_f32(self.c_penalty)
        self.c_alpha_C0 = _grow_f32(self.c_alpha_C0)
        self.c_active = _grow_int(self.c_active)
        self.c_was_static = _grow_int(self.c_was_static)
        # sibling / partner default to -1 in the grown tail.
        sib_np = np.full(new_cap, -1, dtype=np.int32)
        old_sib = self.c_sibling.numpy()
        sib_np[: old_sib.shape[0]] = old_sib
        self.c_sibling = wp.array(sib_np, dtype=int, device=dev)
        par_np = np.full(new_cap, -1, dtype=np.int32)
        old_par = self.c_partner.numpy()
        par_np[: old_par.shape[0]] = old_par
        self.c_partner = wp.array(par_np, dtype=int, device=dev)
        # body→constraint CSR scratch grows in step.
        self.body_con_indices = wp.zeros(max(2 * new_cap, 1), dtype=int,
                                         device=dev)

        self._gpu_pool_n_dyn_capacity = new_dyn_capacity
        self._gpu_pool_n_capacity = new_cap
        # Pool_idx_* sizing: 1 pool entry per emitted contact (≤ 4 per pair).
        new_pool_max = max(self._gpu_pool_pool_max, self._bp_max_pairs * 4)
        if new_pool_max > self._gpu_pool_pool_max:
            self.pool_idx_n = wp.zeros(new_pool_max, dtype=int, device=dev)
            self.pool_idx_t = wp.zeros(new_pool_max, dtype=int, device=dev)
            self.pool_idx_b = wp.zeros(new_pool_max, dtype=int, device=dev)
            self.pool_idx_c = wp.zeros(new_pool_max, dtype=int, device=dev)
            self._gpu_pool_pool_max = new_pool_max
        # Force re-capture next step if graph caching is wired up.
        self._graph = None

    def _gpu_emit_dynamic_contacts(self, n_b: int) -> None:
        """One-substep GPU pipeline: broadphase → SAT → manifold → emit rows.
        Atomically appends BOX_BOX + tangent rows into the dynamic region of
        c_* and records pool entries for the post-solve hash collect.

        Robustness: broadphase can overshoot `_bp_max_pairs` when a body
        cluster forms (or after a sudden coincident drop). On overflow we
        grow `_bp_*` and re-run broadphase up to `MAX_PAIR_RETRIES` times
        before clamping. The kernel-side guard in `gpu_pool_emit_rows`
        keeps the c_* writes in-bounds even if the host bookkeeping is
        ever a step behind."""
        import time as _t
        t_bp0 = _t.perf_counter()
        dev = self.device
        margin = 0.005
        MAX_PAIR_RETRIES = 2

        # AABB inputs are body state; they don't depend on the pair cap.
        he_np = np.asarray(self._half_extents, dtype=np.float32).reshape(-1, 3)
        if (self._bp_half_extents is None
                or self._bp_half_extents.shape[0] != n_b):
            self._bp_half_extents = wp.array(he_np, dtype=wp.vec3, device=dev)
            self._bp_aabb_lo = wp.zeros(n_b, dtype=wp.vec3, device=dev)
            self._bp_aabb_hi = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        else:
            self._bp_half_extents.assign(he_np)

        # Initial pair-buffer sizing matches the _flush heuristic.
        self._ensure_pair_buffers(max(256, 16 * n_b))

        wp.launch(
            K.compute_body_aabb_6dof, dim=n_b,
            inputs=[self.x, self.q, self._bp_half_extents, margin],
            outputs=[self._bp_aabb_lo, self._bp_aabb_hi],
            device=dev,
        )
        constructor = "lbvh" if str(dev).startswith("cuda") else "sah"
        self._bp_bvh = wp.Bvh(self._bp_aabb_lo, self._bp_aabb_hi,
                              constructor=constructor)

        n_pairs = 0
        for attempt in range(MAX_PAIR_RETRIES + 1):
            self._bp_pair_count.zero_()
            wp.launch(
                K.bvh_broadphase_pairs, dim=n_b,
                inputs=[self._bp_bvh.id, self._bp_aabb_lo, self._bp_aabb_hi,
                        self.mass, self._bp_pair_count,
                        self._bp_pair_a, self._bp_pair_b, self._bp_max_pairs],
                device=dev,
            )
            n_pairs = int(self._bp_pair_count.numpy()[0])
            if n_pairs <= self._bp_max_pairs or attempt == MAX_PAIR_RETRIES:
                break
            # Grow and retry — the BVH itself is unaffected by the cap.
            self._ensure_pair_buffers(max(2 * self._bp_max_pairs,
                                           n_pairs * 2))
            # Pair cap changed → emit-side row pool may need a re-capture.
            self._graph = None

        if n_pairs == 0:
            self.broadphase_ms = (_t.perf_counter() - t_bp0) * 1000.0
            return
        # Post-Phase-A: bounds for SAT / manifold / emit come from the
        # device-side `_bp_pair_count` array, not a Python scalar. We launch
        # at the *upper bound* `_bp_max_pairs` so the launch dim is fixed at
        # graph-capture time; threads past `pair_count[0]` early-return.
        # Trailing-tail wasted threads on CPU are negligible.
        launch_dim = self._bp_max_pairs

        wp.launch(
            K.obb_sat_pairs, dim=launch_dim,
            inputs=[self.x, self.q, self._bp_half_extents,
                    self._bp_pair_a, self._bp_pair_b,
                    self._bp_pair_count, margin],
            outputs=[self._bp_pair_overlap, self._bp_pair_sat_idx,
                     self._bp_pair_n_hat, self._bp_pair_depth],
            device=dev,
        )
        self._ensure_manifold_buffers(self._bp_max_pairs)
        wp.launch(
            K.obb_contact_manifold_6dof, dim=launch_dim,
            inputs=[self.x, self.q, self._bp_half_extents,
                    self._bp_pair_a, self._bp_pair_b,
                    self._bp_pair_overlap, self._bp_pair_sat_idx,
                    self._bp_pair_n_hat,
                    self._bp_pair_count, margin, self._mf_poly_scratch],
            outputs=[self._mf_contact_count, self._mf_ref_is_a,
                     self._mf_n_hat, self._mf_t_hat, self._mf_b_hat,
                     self._mf_off_ref, self._mf_off_inc],
            device=dev,
        )
        wp.launch(
            K.gpu_pool_emit_rows, dim=launch_dim,
            inputs=[self._bp_pair_count,
                    self._gpu_pool_n_capacity,
                    self._gpu_pool_pool_max,
                    self._bp_pair_a, self._bp_pair_b,
                    self._mf_contact_count, self._mf_ref_is_a,
                    self._mf_n_hat, self._mf_t_hat, self._mf_b_hat,
                    self._mf_off_ref, self._mf_off_inc,
                    self.body_friction, self._self_friction,
                    self.friction_static_mult,
                    self.n_active_rows, self.pool_count,
                    self.pool_idx_n, self.pool_idx_t,
                    self.pool_idx_b, self.pool_idx_c,
                    self._gpu_pool_hash_cap,
                    self.hash_keys, self.hash_state,
                    self.c_type, self.c_body_a, self.c_body_b,
                    self.c_world_anchor, self.c_off_a, self.c_off_b,
                    self.c_rest, self.c_stiffness,
                    self.c_fmin, self.c_fmax,
                    self.c_alpha_C0, self.c_active, self.c_fracture,
                    self.c_sibling, self.c_partner,
                    self.c_friction, self.c_friction_static,
                    self.c_lambda, self.c_penalty, self.c_was_static],
            device=dev,
        )
        self.broadphase_ms = (_t.perf_counter() - t_bp0) * 1000.0

    # ---- Read-back ----------------------------------------------------------

    def positions(self) -> np.ndarray:
        if self.x is None:
            return np.array(self._x, dtype=np.float32).reshape(-1, 3)
        return self.x.numpy().reshape(-1, 3)

    def orientations(self) -> np.ndarray:
        if self.q is None:
            return np.array(self._q, dtype=np.float32).reshape(-1, 4)
        return self.q.numpy().reshape(-1, 4)

    def velocities(self) -> np.ndarray:
        if self.v is None:
            return np.array(self._v, dtype=np.float32).reshape(-1, 3)
        return self.v.numpy().reshape(-1, 3)

    def angular_velocities(self) -> np.ndarray:
        if self.omega is None:
            return np.array(self._omega, dtype=np.float32).reshape(-1, 3)
        return self.omega.numpy().reshape(-1, 3)

    def lambdas(self) -> np.ndarray:
        if self.c_lambda is None:
            return np.zeros(len(self._rows), dtype=np.float32)
        return self.c_lambda.numpy()

    def active(self) -> np.ndarray:
        if self.c_active is None:
            return np.ones(len(self._rows), dtype=np.int32)
        return self.c_active.numpy()

    # ---- Batched readback (AVBD_PERFORMANCE_GAP §6) ------------------------

    def read_state_batched(self) -> dict[str, np.ndarray]:
        """Pack everything the interactive viewer needs into TWO contiguous
        Warp arrays, then issue a single .numpy() per packed buffer. Replaces
        seven separate stream-syncing .numpy() calls with two.

        Returns a dict with keys:
            positions          (n_b, 3) float32
            orientations       (n_b, 4) float32   xyzw
            angular_velocities (n_b, 3) float32
            lambdas            (n_c,)   float32
            active             (n_c,)   int32
            was_static         (n_c,)   int32
            c_type             (n_c,)   int32

        Falls back to the per-array readers if the solver hasn't flushed yet
        (caller hit it before the first step()).
        """
        n_b = len(self._x)
        # n_c = LIVE row count (static + dynamic from the GPU pool). After
        # gap-#1 the dynamic rows live in c_* GPU arrays — `self._rows` is
        # only the static prefix and would severely under-count contacts.
        n_c_static = len(self._rows)
        if self.n_active_rows is not None:
            n_c = int(self.n_active_rows.numpy()[0])
        else:
            n_c = n_c_static
        if n_b == 0 or self.x is None:
            return {
                "positions": np.array(self._x, dtype=np.float32).reshape(-1, 3),
                "orientations": np.array(self._q, dtype=np.float32).reshape(-1, 4),
                "angular_velocities": np.array(self._omega, dtype=np.float32).reshape(-1, 3),
                "lambdas": np.zeros(n_c, dtype=np.float32),
                "active": np.ones(n_c, dtype=np.int32),
                "was_static": np.zeros(n_c, dtype=np.int32),
                "c_type": np.zeros(n_c, dtype=np.int32),
                "n_rows": n_c,
            }
        dev = self.device
        # Reuse staging buffers across frames; size the row buffer to total
        # row capacity (not the per-frame n_c) so dynamic-contact churn
        # doesn't reallocate.
        n_cap_alloc = max(self._gpu_pool_n_capacity, n_c_static)
        if (self._viewer_pack_bodies is None
                or self._viewer_pack_bodies_n != n_b):
            self._viewer_pack_bodies = wp.zeros(n_b * 10, dtype=float,
                                                device=dev)
            self._viewer_pack_bodies_n = n_b
        if (self._viewer_pack_rows is None
                or self._viewer_pack_rows_n != n_cap_alloc):
            self._viewer_pack_rows = (wp.zeros(n_cap_alloc * 3, dtype=float,
                                               device=dev)
                                      if n_cap_alloc > 0 else None)
            self._viewer_pack_rows_n = n_cap_alloc
        wp.launch(
            K.viewer_pack_bodies_6dof, dim=n_b,
            inputs=[self.x, self.q, self.omega],
            outputs=[self._viewer_pack_bodies],
            device=dev,
        )
        if n_c > 0:
            wp.launch(
                K.viewer_pack_rows_6dof, dim=n_c,
                inputs=[self.c_lambda, self.c_active,
                        self.c_was_static, self.c_type],
                outputs=[self._viewer_pack_rows],
                device=dev,
            )
        bod = self._viewer_pack_bodies.numpy().reshape(n_b, 10)
        if n_c > 0:
            # Pack buffer is sized to n_cap_alloc; slice the live n_c prefix.
            row_full = self._viewer_pack_rows.numpy().reshape(n_cap_alloc, 3)
            row = row_full[:n_c]
            lam = row[:, 0].astype(np.float32, copy=True)
            act = row[:, 1].astype(np.int32, copy=False)
            ws_type = row[:, 2].astype(np.int32, copy=False)
            was = (ws_type // 16).astype(np.int32, copy=False)
            ctype = (ws_type % 16).astype(np.int32, copy=False)
        else:
            lam = np.zeros(0, dtype=np.float32)
            act = np.zeros(0, dtype=np.int32)
            was = np.zeros(0, dtype=np.int32)
            ctype = np.zeros(0, dtype=np.int32)
        return {
            "positions": bod[:, 0:3].astype(np.float32, copy=True),
            "orientations": bod[:, 3:7].astype(np.float32, copy=True),
            "angular_velocities": bod[:, 7:10].astype(np.float32, copy=True),
            "lambdas": lam,
            "active": act,
            "was_static": was,
            "c_type": ctype,
            "n_rows": n_c,
        }
