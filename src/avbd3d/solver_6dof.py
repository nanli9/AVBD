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
from .coloring import build_body_edges, color_summary, greedy_color

# Re-export the constraint type codes for callers / tests.
FLOOR_CONTACT_6DOF = 0
CONTACT_TANGENT_6DOF = 1
PIN_6DOF = 2
BOX_BOX_CONTACT_6DOF = 3


def _q_to_R(q_xyzw) -> np.ndarray:
    """Quaternion (xyzw) → 3×3 rotation matrix. Columns are the world-space
    images of the body-local basis vectors."""
    qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Duff et al. 2017 stable orthonormal basis perpendicular to unit n̂."""
    n = n / (np.linalg.norm(n) + 1e-20)
    sign = 1.0 if n[2] >= 0.0 else -1.0
    a = -1.0 / (sign + n[2])
    b_ = n[0] * n[1] * a
    t = np.array([1.0 + sign * n[0] * n[0] * a, sign * b_, -sign * n[0]],
                 dtype=np.float32)
    bvec = np.array([b_, sign + n[1] * n[1] * a, -n[1]], dtype=np.float32)
    return t, bvec


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


def _box_face_data(c: np.ndarray, R: np.ndarray, e: np.ndarray,
                   axis: int, sign: float):
    """Return (face_center, face_outward_normal, [4 vertices in world],
    [4 side planes: (point, outward_normal)]) for the box face along sign·R[:,axis]."""
    n_face = sign * R[:, axis]
    face_center = c + n_face * e[axis]
    other = [k for k in range(3) if k != axis]
    u = R[:, other[0]]
    v = R[:, other[1]]
    eu, ev = e[other[0]], e[other[1]]
    verts = [
        face_center + u * eu + v * ev,
        face_center - u * eu + v * ev,
        face_center - u * eu - v * ev,
        face_center + u * eu - v * ev,
    ]
    # Side planes — outward normals point AWAY from the face's center.
    side_planes = [
        (face_center + u * eu,  u),
        (face_center - u * eu, -u),
        (face_center + v * ev,  v),
        (face_center - v * ev, -v),
    ]
    return face_center, n_face, verts, side_planes


def _sh_clip(polygon: list, plane_pt: np.ndarray, plane_n: np.ndarray) -> list:
    """Sutherland-Hodgman in 3D: clip the polygon against a single half-space
    {q : (q − plane_pt) · plane_n ≤ 0} (keep the INSIDE side). Returns the
    clipped polygon as a list of 3D points."""
    if not polygon:
        return []
    out = []
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        da = float(np.dot(a - plane_pt, plane_n))
        db = float(np.dot(b - plane_pt, plane_n))
        if da <= 0.0:
            out.append(a)
            if db > 0.0:
                t = da / (da - db)
                out.append(a + t * (b - a))
        elif db <= 0.0:
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


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
        # Index of the first row in self._rows that came from the dynamic
        # contact pool — rows at index ≥ this are stripped+rebuilt every step.
        self._contact_pool_start: int | None = None
        # Persistent warm-start cache: key = (a, b, sat_axis, corner_id),
        # value = (λ_n, λ_t, λ_b, k_n, k_t, k_b). Lets stacked boxes carry
        # full augmented-Lagrangian state across frames so the contact penalty
        # doesn't relitigate from PENALTY_MIN every frame.
        self._contact_cache: dict[tuple, tuple[float, float, float, float, float, float]] = {}

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
        # See _rebuild_contact_pool for the per-step pipeline.
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
        # GPU-resident contact warm-start scratch (AVBD_PERFORMANCE_GAP §5).
        # Each substep's contact pool gets index arrays (idx_n/t/b) so
        # cache_restore_6dof + cache_collect_6dof can mutate c_lambda /
        # c_penalty / c_was_static in place without a full-array
        # GPU↔CPU round trip. Reused frame-to-frame; the helper
        # `_ensure_pool_buffers` grows them in 256-pair chunks when the
        # pool count goes up.
        self._pool_idx_n = None       # wp.array(int)
        self._pool_idx_t = None       # wp.array(int)
        self._pool_idx_b = None       # wp.array(int)
        self._pool_cache_valid = None # wp.array(int)
        self._pool_in_packed = None   # wp.array(float), shape (n_pool*8,)
        self._pool_out_packed = None  # wp.array(float), shape (n_pool*8,)
        self._pool_buf_cap = 0
        # Viewer staging buffers — built lazily on first read_batched() call,
        # shared across frames, reuploaded only when body/row counts change.
        self._viewer_pack_bodies = None
        self._viewer_pack_rows = None
        self._viewer_pack_bodies_n = 0
        self._viewer_pack_rows_n = 0

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
        """Turn on per-step OBB-OBB contact generation. Pairs of bodies are
        tested with 15-axis SAT every step and emit BOX_BOX_CONTACT_6DOF +
        CONTACT_TANGENT_6DOF rows for the active contacts (up to 4 contacts
        per pair from face clipping). Brute O(N²) broad phase — fine for ≤50
        bodies; AVBD paper §4 uses LBVH for scale."""
        self._self_collide = bool(enabled)
        self._self_friction = max(0.0, float(default_friction))
        if enabled and self._contact_pool_start is None:
            self._contact_pool_start = len(self._rows)

    def _emit_obb_pair(self, i: int, j: int, positions: np.ndarray,
                       quats: np.ndarray) -> None:
        """CPU fallback path: SAT + face-clip + emit. Used only by the legacy
        brute-force broadphase (kept for parity tests). The hot path now goes
        through `_emit_obb_pair_with_sat` after the Warp BVH broadphase + SAT
        kernels have already computed the separating axis and contact normal.
        """
        c_A, c_B = positions[i], positions[j]
        e_A = np.asarray(self._half_extents[i], dtype=np.float32)
        e_B = np.asarray(self._half_extents[j], dtype=np.float32)
        R_A = _q_to_R(quats[i])
        R_B = _q_to_R(quats[j])
        sat = _obb_sat(c_A, R_A, e_A, c_B, R_B, e_B)
        if sat is None:
            return
        sat_idx, n_hat, _overlap = sat
        self._emit_obb_pair_with_sat(i, j, positions, quats,
                                     sat_idx, n_hat, c_A, c_B,
                                     e_A, e_B, R_A, R_B)

    def _emit_obb_pair_with_sat(self, i: int, j: int,
                                positions: np.ndarray, quats: np.ndarray,
                                sat_idx: int, n_hat: np.ndarray,
                                c_A: np.ndarray, c_B: np.ndarray,
                                e_A: np.ndarray, e_B: np.ndarray,
                                R_A: np.ndarray, R_B: np.ndarray) -> None:
        """Face-clip + emit using precomputed SAT result. Pulled out of
        `_emit_obb_pair` so the Warp-side parallel SAT kernel can feed it
        directly."""

        # Identify reference vs incident body based on which axis won.
        if sat_idx < 3:
            ref_i, inc_i = i, j
            ref_c, ref_R, ref_e = c_A, R_A, e_A
            inc_c, inc_R, inc_e = c_B, R_B, e_B
            ref_axis = sat_idx
        elif sat_idx < 6:
            ref_i, inc_i = j, i
            ref_c, ref_R, ref_e = c_B, R_B, e_B
            inc_c, inc_R, inc_e = c_A, R_A, e_A
            ref_axis = sat_idx - 3
            # n_hat from SAT was oriented "B to A" with body indices (i, j) =
            # (A_input, B_input). Now we swapped which body is "ref"; we need
            # n_hat to point from new inc (=i) to new ref (=j). Flip.
            n_hat = -n_hat
        else:
            # Edge-edge fallback: single contact at the "deepest" body's
            # closest vertex projected onto the other body's surface. Crude
            # but stable. Proper edge-edge needs closest-segment-pair.
            self._emit_obb_edge_edge(i, j, c_A, R_A, e_A, c_B, R_B, e_B,
                                     n_hat, sat_idx, positions, quats)
            return

        # Reference face on ref body: pick the face along ref_axis whose
        # outward normal opposes n_hat (i.e., points TOWARD the incident).
        dot = float(np.dot(ref_R[:, ref_axis], n_hat))
        ref_sign = -1.0 if dot > 0 else 1.0
        ref_face_c, ref_face_n, _ref_verts, ref_side_planes = _box_face_data(
            ref_c, ref_R, ref_e, ref_axis, ref_sign)

        # Incident face on inc body: pick the inc local axis (and sign)
        # whose outward normal is most aligned with n_hat (points toward ref).
        best = (0, 1.0, -np.inf)
        for k in range(3):
            for sign in (-1.0, 1.0):
                d = float(np.dot(sign * inc_R[:, k], n_hat))
                if d > best[2]:
                    best = (k, sign, d)
        inc_axis, inc_sign, _ = best
        _inc_face_c, _inc_face_n, inc_verts, _ = _box_face_data(
            inc_c, inc_R, inc_e, inc_axis, inc_sign)

        # Clip incident-face polygon against the 4 side planes of ref face.
        polygon = list(inc_verts)
        for pp, pn in ref_side_planes:
            polygon = _sh_clip(polygon, pp, pn)
            if not polygon:
                return

        # Keep points within `margin` of the ref face plane (penetrating OR
        # slightly separated). Separated rows produce zero force (C > 0 →
        # f clamped to 0) but their λ persists in the warm-start cache, which
        # is what makes stacking stable across the brief separations that
        # post-stab + finalize_velocity introduce.
        margin = 0.005
        contacts = []
        for p in polygon:
            d = float(np.dot(p - ref_face_c, ref_face_n))
            if d < margin:
                contacts.append((p, -d))
        if not contacts:
            return
        contacts.sort(key=lambda x: -x[1])
        contacts = contacts[:4]  # up to 4 per pair, same as 3-DOF AABB version

        # Build tangent basis once per pair.
        t_hat, b_hat = _orthonormal_basis(n_hat)
        mu = self._self_friction
        if (self._friction[ref_i] > 0.0 or self._friction[inc_i] > 0.0):
            mu = math.sqrt(self._friction[ref_i] * self._friction[inc_i])

        for cid, (p_inc, _depth) in enumerate(contacts):
            # Contact point on REF body: project p_inc onto ref face plane.
            p_ref = p_inc - np.dot(p_inc - ref_face_c, ref_face_n) * ref_face_n
            # Body-local anchors — transform world offsets into each body's
            # rest frame. R^T = R^{-1} since R is orthonormal.
            off_ref_local = ref_R.T @ (p_ref - ref_c)
            off_inc_local = inc_R.T @ (p_inc - inc_c)

            normal_idx = len(self._rows)
            self._rows.append(_Row(
                type=BOX_BOX_CONTACT_6DOF,
                body_a=ref_i, body_b=inc_i,
                world_anchor=(float(n_hat[0]), float(n_hat[1]), float(n_hat[2])),
                off_a=(float(off_ref_local[0]), float(off_ref_local[1]), float(off_ref_local[2])),
                off_b=(float(off_inc_local[0]), float(off_inc_local[1]), float(off_inc_local[2])),
                rest=0.0,
                stiffness=math.inf,
                fmin=-math.inf, fmax=0.0,
            ))
            t_idx, b_idx = -1, -1
            if mu > 0.0:
                mu_s = mu * self.friction_static_mult
                t_idx = len(self._rows)
                self._rows.append(_Row(
                    type=CONTACT_TANGENT_6DOF,
                    body_a=ref_i, body_b=inc_i,
                    world_anchor=(float(t_hat[0]), float(t_hat[1]), float(t_hat[2])),
                    off_a=(float(off_ref_local[0]), float(off_ref_local[1]), float(off_ref_local[2])),
                    off_b=(float(off_inc_local[0]), float(off_inc_local[1]), float(off_inc_local[2])),
                    stiffness=math.inf, sibling=normal_idx, friction=mu,
                    friction_static=mu_s,
                ))
                b_idx = len(self._rows)
                self._rows.append(_Row(
                    type=CONTACT_TANGENT_6DOF,
                    body_a=ref_i, body_b=inc_i,
                    world_anchor=(float(b_hat[0]), float(b_hat[1]), float(b_hat[2])),
                    off_a=(float(off_ref_local[0]), float(off_ref_local[1]), float(off_ref_local[2])),
                    off_b=(float(off_inc_local[0]), float(off_inc_local[1]), float(off_inc_local[2])),
                    stiffness=math.inf, sibling=normal_idx, friction=mu,
                    friction_static=mu_s,
                ))
                self._rows[t_idx].partner = b_idx
                self._rows[b_idx].partner = t_idx
            # Cache key — quantize the body-local contact point on the
            # LOWER-INDEX body to a 5 mm grid. Using the lower-index body
            # (rather than "ref") keeps the key invariant to which side SAT
            # picked as ref on a tiebreak; without this, axis switches
            # between frames cache-miss every time and break stack stability.
            if ref_i < inc_i:
                off_key = off_ref_local
            else:
                off_key = off_inc_local
            qx = int(round(float(off_key[0]) * 200))
            qy = int(round(float(off_key[1]) * 200))
            qz = int(round(float(off_key[2]) * 200))
            cache_key = (min(ref_i, inc_i), max(ref_i, inc_i), qx, qy, qz)
            self._pool_pair_rows.append((cache_key, normal_idx, t_idx, b_idx))

    def _emit_obb_edge_edge(self, i, j, c_A, R_A, e_A, c_B, R_B, e_B,
                            n_hat, sat_idx, positions, quats) -> None:
        """Proper edge-edge OBB contact (Ericson §5.1.9 + §15.6.3).

        When SAT picks an edge×edge separating axis L = R_A[:,k_A] × R_B[:,k_B],
        the contact happens between one edge on A parallel to R_A[:,k_A] and
        one edge on B parallel to R_B[:,k_B]. The contact normal is n_hat
        (already oriented B→A by the SAT). The contact "point" is the
        midpoint of the closest-segment-pair on those two edges.

        Selecting WHICH parallel edge on each body: each box has 4 edges
        along a given direction (at the 4 corners of the perpendicular face).
        The contact edge is the one whose midpoint is closest to the OTHER
        body's centre — equivalently, whose perpendicular-axis offsets
        have the right sign to face the other body.
        """
        # Decode SAT index back to (k_A, k_B) edge axes (paired in row-major
        # 3×3 order). sat_idx ∈ [6, 15).
        eidx = sat_idx - 6
        k_A = eidx // 3
        k_B = eidx % 3
        eA_dir = R_A[:, k_A]
        eB_dir = R_B[:, k_B]
        # Degenerate: cross product near zero (edges nearly parallel) → skip,
        # the face-axis cases will already have captured the contact.
        cross_mag = float(np.linalg.norm(np.cross(eA_dir, eB_dir)))
        if cross_mag < 1.0e-4:
            return

        # Pick the contact edge on A: 4 candidates, indexed by sign pair
        # (s_b, s_c) on A's perpendicular axes (k_A+1)%3 and (k_A+2)%3.
        # Choose signs that put the edge midpoint on the side of A closest
        # to c_B.
        kA1 = (k_A + 1) % 3
        kA2 = (k_A + 2) % 3
        toB = c_B - c_A
        s_b_A = 1.0 if float(np.dot(toB, R_A[:, kA1])) >= 0 else -1.0
        s_c_A = 1.0 if float(np.dot(toB, R_A[:, kA2])) >= 0 else -1.0
        edge_A_mid = c_A + s_b_A * e_A[kA1] * R_A[:, kA1] + s_c_A * e_A[kA2] * R_A[:, kA2]
        # A's edge spans [-e_A[k_A], +e_A[k_A]] along eA_dir from edge_A_mid.
        P1 = edge_A_mid - e_A[k_A] * eA_dir
        Q1 = edge_A_mid + e_A[k_A] * eA_dir

        # Same for B (toward c_A).
        kB1 = (k_B + 1) % 3
        kB2 = (k_B + 2) % 3
        toA = c_A - c_B
        s_b_B = 1.0 if float(np.dot(toA, R_B[:, kB1])) >= 0 else -1.0
        s_c_B = 1.0 if float(np.dot(toA, R_B[:, kB2])) >= 0 else -1.0
        edge_B_mid = c_B + s_b_B * e_B[kB1] * R_B[:, kB1] + s_c_B * e_B[kB2] * R_B[:, kB2]
        P2 = edge_B_mid - e_B[k_B] * eB_dir
        Q2 = edge_B_mid + e_B[k_B] * eB_dir

        # Closest points on two segments (Ericson §5.1.9).
        d1 = Q1 - P1
        d2 = Q2 - P2
        r = P1 - P2
        a = float(np.dot(d1, d1))
        e = float(np.dot(d2, d2))
        f = float(np.dot(d2, r))
        eps = 1.0e-12
        if a <= eps and e <= eps:
            s_p, t_p = 0.0, 0.0
        elif a <= eps:
            s_p = 0.0
            t_p = float(np.clip(f / max(e, eps), 0.0, 1.0))
        elif e <= eps:
            t_p = 0.0
            c_ = float(np.dot(d1, r))
            s_p = float(np.clip(-c_ / a, 0.0, 1.0))
        else:
            c_ = float(np.dot(d1, r))
            b_ = float(np.dot(d1, d2))
            denom = a * e - b_ * b_
            if denom != 0.0:
                s_p = float(np.clip((b_ * f - c_ * e) / denom, 0.0, 1.0))
            else:
                s_p = 0.0
            t_p = (b_ * s_p + f) / e
            if t_p < 0.0:
                t_p = 0.0
                s_p = float(np.clip(-c_ / a, 0.0, 1.0))
            elif t_p > 1.0:
                t_p = 1.0
                s_p = float(np.clip((b_ - c_) / a, 0.0, 1.0))
        p_on_A = P1 + s_p * d1
        p_on_B = P2 + t_p * d2

        # Penetration check: the two closest points should be within `margin`
        # along n_hat (signed gap). n_hat points B→A so (p_on_A - p_on_B)·n_hat
        # > 0 means separated, ≤ 0 means penetrating.
        gap = float(np.dot(p_on_A - p_on_B, n_hat))
        if gap > 0.005:
            return

        off_a_local = R_A.T @ (p_on_A - c_A)
        off_b_local = R_B.T @ (p_on_B - c_B)
        t_hat, b_hat = _orthonormal_basis(n_hat)
        mu = self._self_friction
        if (self._friction[i] > 0.0 or self._friction[j] > 0.0):
            mu = math.sqrt(self._friction[i] * self._friction[j])

        normal_idx = len(self._rows)
        self._rows.append(_Row(
            type=BOX_BOX_CONTACT_6DOF,
            body_a=i, body_b=j,
            world_anchor=(float(n_hat[0]), float(n_hat[1]), float(n_hat[2])),
            off_a=(float(off_a_local[0]), float(off_a_local[1]), float(off_a_local[2])),
            off_b=(float(off_b_local[0]), float(off_b_local[1]), float(off_b_local[2])),
            rest=0.0, stiffness=math.inf,
            fmin=-math.inf, fmax=0.0,
        ))
        t_idx, b_idx = -1, -1
        if mu > 0.0:
            mu_s = mu * self.friction_static_mult
            t_idx = len(self._rows)
            self._rows.append(_Row(
                type=CONTACT_TANGENT_6DOF, body_a=i, body_b=j,
                world_anchor=(float(t_hat[0]), float(t_hat[1]), float(t_hat[2])),
                off_a=(float(off_a_local[0]), float(off_a_local[1]), float(off_a_local[2])),
                off_b=(float(off_b_local[0]), float(off_b_local[1]), float(off_b_local[2])),
                stiffness=math.inf, sibling=normal_idx, friction=mu,
                friction_static=mu_s,
            ))
            b_idx = len(self._rows)
            self._rows.append(_Row(
                type=CONTACT_TANGENT_6DOF, body_a=i, body_b=j,
                world_anchor=(float(b_hat[0]), float(b_hat[1]), float(b_hat[2])),
                off_a=(float(off_a_local[0]), float(off_a_local[1]), float(off_a_local[2])),
                off_b=(float(off_b_local[0]), float(off_b_local[1]), float(off_b_local[2])),
                stiffness=math.inf, sibling=normal_idx, friction=mu,
                friction_static=mu_s,
            ))
            self._rows[t_idx].partner = b_idx
            self._rows[b_idx].partner = t_idx
        # Cache key on lower-index body for tiebreak invariance.
        off_key = off_a_local if i < j else off_b_local
        qx = int(round(float(off_key[0]) * 200))
        qy = int(round(float(off_key[1]) * 200))
        qz = int(round(float(off_key[2]) * 200))
        cache_key = (min(i, j), max(i, j), qx, qy, qz)
        self._pool_pair_rows.append((cache_key, normal_idx, t_idx, b_idx))

    def _rebuild_contact_pool(self) -> None:
        """Strip last frame's BOX_BOX rows + their tangent siblings, run the
        SAT broad phase on every pair of rigid bodies, append fresh rows.
        Renumbers `sibling` indices on any surviving CONTACT_TANGENT_6DOF
        rows that pointed to BOX_BOX (none in MVP — floor tangents survive)."""
        if not self._self_collide:
            return
        kept: list[_Row] = []
        old_to_new: dict[int, int] = {}
        for old_idx, r in enumerate(self._rows):
            if r.type == BOX_BOX_CONTACT_6DOF:
                continue
            if (r.type == CONTACT_TANGENT_6DOF
                    and 0 <= r.sibling < len(self._rows)):
                sib = self._rows[r.sibling]
                if sib.type == BOX_BOX_CONTACT_6DOF:
                    continue
            old_to_new[old_idx] = len(kept)
            kept.append(r)
        for r in kept:
            if r.type == CONTACT_TANGENT_6DOF and r.sibling >= 0:
                r.sibling = old_to_new.get(r.sibling, -1)
            if r.type == CONTACT_TANGENT_6DOF and r.partner >= 0:
                r.partner = old_to_new.get(r.partner, -1)
        self._rows = kept
        self._contact_pool_start = len(self._rows)
        self._pool_pair_rows: list = []

        positions = (self.x.numpy().reshape(-1, 3) if self.x is not None
                     else np.array(self._x, dtype=np.float32).reshape(-1, 3))
        quats = (self.q.numpy().reshape(-1, 4) if self.q is not None
                 else np.array(self._q, dtype=np.float32).reshape(-1, 4))
        n = len(self._x)
        if n < 2:
            self._dirty = True
            return

        import time as _t
        t_bp0 = _t.perf_counter()
        self._warp_broadphase_emit_contacts(n, positions, quats)
        self.broadphase_ms = (_t.perf_counter() - t_bp0) * 1000.0
        self._dirty = True

    def _warp_broadphase_emit_contacts(
            self, n: int, positions: np.ndarray, quats: np.ndarray) -> None:
        """AVBD Alg 1 line 1 broadphase. Build per-body world AABBs on the
        Warp device, construct an LBVH over them, query for overlapping
        pairs, run 15-axis SAT in parallel on candidate pairs, then run the
        Python face-clip on confirmed-overlap pairs only.

        The legacy O(N²) Python sphere-test + per-pair SAT loop dominated
        ~56% of frame time in dense scenes (towers + dominoes); this path
        moves both the broadphase and the SAT inner loop to Warp where they
        execute in parallel across pairs."""
        dev = self.device

        # Margin matches the per-pair SAT margin used in _obb_sat (5 mm) —
        # bodies within this margin still emit a contact so warm-start λ
        # persists across brief separations during settle. The AABB margin
        # must be at least this large or the broadphase silently drops
        # those grazing pairs and we lose the persistence trick.
        margin = 0.005

        # 1. Half-extent buffer — only needs rebuild if body count changed.
        he_np = np.asarray(self._half_extents, dtype=np.float32).reshape(-1, 3)
        if (self._bp_half_extents is None
                or self._bp_half_extents.shape[0] != n):
            self._bp_half_extents = wp.array(he_np, dtype=wp.vec3, device=dev)
            self._bp_aabb_lo = wp.zeros(n, dtype=wp.vec3, device=dev)
            self._bp_aabb_hi = wp.zeros(n, dtype=wp.vec3, device=dev)
        else:
            # Half-extents are scene-static; only flush if user appended
            # bodies after the initial _flush. Cheaper to just copy now
            # than to track a dirty flag for this one buffer.
            self._bp_half_extents.assign(he_np)

        # 2. Pair-list buffer — generous fixed allocation. Realistic cap
        # for dense rigid stacks: each body sees ≲ 12 neighbors (corner
        # contacts on a cube grid), so 16*n is a safe upper bound. We
        # detect overflow and grow next frame if needed.
        cap_target = max(256, 16 * n)
        if self._bp_max_pairs < cap_target:
            self._bp_max_pairs = cap_target
            self._bp_pair_a = wp.zeros(cap_target, dtype=int, device=dev)
            self._bp_pair_b = wp.zeros(cap_target, dtype=int, device=dev)
            self._bp_pair_overlap = wp.zeros(cap_target, dtype=int, device=dev)
            self._bp_pair_sat_idx = wp.zeros(cap_target, dtype=int, device=dev)
            self._bp_pair_n_hat = wp.zeros(cap_target, dtype=wp.vec3, device=dev)
            self._bp_pair_depth = wp.zeros(cap_target, dtype=float, device=dev)
        if self._bp_pair_count is None:
            self._bp_pair_count = wp.zeros(1, dtype=int, device=dev)
        else:
            self._bp_pair_count.zero_()

        # 3. World AABBs on device — needs self.x / self.q which _flush()
        # populated upstream; we're in the post-_flush path of _step_one.
        wp.launch(
            K.compute_body_aabb_6dof, dim=n,
            inputs=[self.x, self.q, self._bp_half_extents, margin],
            outputs=[self._bp_aabb_lo, self._bp_aabb_hi],
            device=dev,
        )

        # 4. Build the BVH. AVBD §4 specifies LBVH (Lauterbach 2009) but
        # Warp 1.13 only ships the LBVH constructor for CUDA trees. On CPU
        # the closest equivalent is SAH (Surface Area Heuristic, top-down)
        # — same O(N log N) build, slightly different split quality, no
        # behavioral difference for the broadphase.
        constructor = "lbvh" if str(dev).startswith("cuda") else "sah"
        self._bp_bvh = wp.Bvh(self._bp_aabb_lo, self._bp_aabb_hi,
                              constructor=constructor)

        # 5. Broadphase pair generation — each body queries the tree.
        wp.launch(
            K.bvh_broadphase_pairs, dim=n,
            inputs=[self._bp_bvh.id, self._bp_aabb_lo, self._bp_aabb_hi,
                    self.mass, self._bp_pair_count,
                    self._bp_pair_a, self._bp_pair_b, self._bp_max_pairs],
            device=dev,
        )

        n_pairs = int(self._bp_pair_count.numpy()[0])
        if n_pairs == 0:
            return
        if n_pairs > self._bp_max_pairs:
            # Overflow — grow buffer and retry next frame. Conservatively
            # drop this frame's late pairs; the BVH will reissue them next
            # substep with the resized buffer.
            self._bp_max_pairs = max(2 * self._bp_max_pairs, n_pairs * 2)
            n_pairs = self._bp_max_pairs

        # 6. 15-axis SAT in parallel across candidate pairs.
        wp.launch(
            K.obb_sat_pairs, dim=n_pairs,
            inputs=[self.x, self.q, self._bp_half_extents,
                    self._bp_pair_a, self._bp_pair_b, n_pairs, margin],
            outputs=[self._bp_pair_overlap, self._bp_pair_sat_idx,
                     self._bp_pair_n_hat, self._bp_pair_depth],
            device=dev,
        )

        # 7. Readback + Python face-clip on confirmed overlaps. We pay one
        # GPU→CPU sync here for the SAT result arrays (4 small int/float
        # buffers, ~50 entries on a tower scene). Face-clip variable output
        # is awkward in Warp; keeping it on CPU is the residual cost. (A
        # full Warp port with fixed-size polygon buffers is left as a
        # follow-up — the broadphase + SAT move was the dominant share.)
        a_np = self._bp_pair_a.numpy()[:n_pairs]
        b_np = self._bp_pair_b.numpy()[:n_pairs]
        ov_np = self._bp_pair_overlap.numpy()[:n_pairs]
        si_np = self._bp_pair_sat_idx.numpy()[:n_pairs]
        nh_np = self._bp_pair_n_hat.numpy().reshape(-1, 3)[:n_pairs]

        for p in range(n_pairs):
            if ov_np[p] == 0:
                continue
            i = int(a_np[p])
            j = int(b_np[p])
            sat_idx = int(si_np[p])
            n_hat = nh_np[p].astype(np.float32)
            c_A, c_B = positions[i], positions[j]
            e_A = np.asarray(self._half_extents[i], dtype=np.float32)
            e_B = np.asarray(self._half_extents[j], dtype=np.float32)
            R_A = _q_to_R(quats[i])
            R_B = _q_to_R(quats[j])
            self._emit_obb_pair_with_sat(i, j, positions, quats,
                                         sat_idx, n_hat,
                                         c_A, c_B, e_A, e_B, R_A, R_B)

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

    # ---- Adjacency + Warp upload -------------------------------------------

    def _build_adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._x)
        counts = np.zeros(n, dtype=np.int32)
        for r in self._rows:
            counts[r.body_a] += 1
            # NOTE: for PIN rows we re-use body_b as the axis index — those
            # don't count as a second body in the adjacency.
            if r.type not in (PIN_6DOF,) and r.body_b >= 0:
                counts[r.body_b] += 1
        starts = np.zeros(n + 1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        indices = np.zeros(starts[-1] if n > 0 else 0, dtype=np.int32)
        cursor = starts[:-1].copy()
        for ci, r in enumerate(self._rows):
            indices[cursor[r.body_a]] = ci
            cursor[r.body_a] += 1
            if r.type not in (PIN_6DOF,) and r.body_b >= 0:
                indices[cursor[r.body_b]] = ci
                cursor[r.body_b] += 1
        return starts, indices

    def _flush(self) -> None:
        if not self._dirty:
            return
        n_b = len(self._x)
        n_c = len(self._rows)
        dev = self.device

        # Snapshot current state so we preserve already-simulated bodies and
        # constraint λ when bodies/rows are added mid-simulation.
        cur_x = self.x.numpy() if self.x is not None else None
        cur_q = self.q.numpy() if self.q is not None else None
        cur_v = self.v.numpy() if self.v is not None else None
        cur_w = self.omega.numpy() if self.omega is not None else None
        cur_prev_v = self.prev_v.numpy() if self.prev_v is not None else None
        cur_prev_w = self.prev_omega.numpy() if self.prev_omega is not None else None
        cur_lam = self.c_lambda.numpy() if self.c_lambda is not None else None
        cur_pen = self.c_penalty.numpy() if self.c_penalty is not None else None
        cur_act = self.c_active.numpy() if self.c_active is not None else None
        n_b_prev = 0 if cur_x is None else int(cur_x.shape[0])
        n_c_prev = 0 if cur_lam is None else int(cur_lam.shape[0])

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
        self.x_initial = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.q_initial = wp.zeros(n_b, dtype=wp.quat, device=dev)
        self.x_inertial = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.q_inertial = wp.zeros(n_b, dtype=wp.quat, device=dev)
        self.inv_inertia_world = wp.zeros(n_b, dtype=wp.mat33, device=dev)
        self.inertia_world = wp.zeros(n_b, dtype=wp.mat33, device=dev)

        # Adjacency uses body indices only.
        body_a_list = [r.body_a for r in self._rows]
        body_b_list = [r.body_b if r.type != PIN_6DOF else -1 for r in self._rows]
        adj = build_body_edges(n_b, body_a_list, body_b_list)
        color_np = greedy_color(adj) if n_b > 0 else np.zeros(0, dtype=np.int32)
        self.num_colors = int(color_np.max() + 1) if n_b > 0 else 0
        self.color_counts = color_summary(color_np)
        self.body_color = wp.array(color_np, dtype=int, device=dev)

        def f32(seq): return np.array(seq, dtype=np.float32) if n_c else np.zeros(0, np.float32)
        def i32(seq): return np.array(seq, dtype=np.int32) if n_c else np.zeros(0, np.int32)

        anchor_np = (np.array([r.world_anchor for r in self._rows], dtype=np.float32).reshape(-1, 3)
                     if n_c else np.zeros((0, 3), np.float32))
        off_a_np = (np.array([r.off_a for r in self._rows], dtype=np.float32).reshape(-1, 3)
                    if n_c else np.zeros((0, 3), np.float32))
        off_b_np = (np.array([r.off_b for r in self._rows], dtype=np.float32).reshape(-1, 3)
                    if n_c else np.zeros((0, 3), np.float32))

        self.c_type = wp.array(i32([r.type for r in self._rows]), dtype=int, device=dev)
        self.c_body_a = wp.array(i32([r.body_a for r in self._rows]), dtype=int, device=dev)
        self.c_body_b = wp.array(i32([r.body_b for r in self._rows]), dtype=int, device=dev)
        self.c_world_anchor = wp.array(anchor_np, dtype=wp.vec3, device=dev)
        self.c_off_a = wp.array(off_a_np, dtype=wp.vec3, device=dev)
        self.c_off_b = wp.array(off_b_np, dtype=wp.vec3, device=dev)
        self.c_rest = wp.array(f32([r.rest for r in self._rows]), dtype=float, device=dev)
        self.c_stiffness = wp.array(f32([r.stiffness for r in self._rows]), dtype=float, device=dev)

        lam_np = np.zeros(n_c, dtype=np.float32)
        pen_np = np.full(n_c, 1.0, dtype=np.float32)
        act_np = np.ones(n_c, dtype=np.int32)
        n_keep = min(n_c_prev, n_c)
        if n_keep > 0:
            lam_np[:n_keep] = cur_lam[:n_keep]
            if cur_pen is not None:
                pen_np[:n_keep] = cur_pen[:n_keep]
            if cur_act is not None:
                act_np[:n_keep] = cur_act[:n_keep]
        self.c_lambda = wp.array(lam_np, dtype=float, device=dev)
        self.c_penalty = wp.array(pen_np, dtype=float, device=dev)
        self.c_fmin = wp.array(f32([r.fmin for r in self._rows]), dtype=float, device=dev)
        self.c_fmax = wp.array(f32([r.fmax for r in self._rows]), dtype=float, device=dev)
        self.c_alpha_C0 = wp.zeros(n_c, dtype=float, device=dev)
        self.c_active = wp.array(act_np, dtype=int, device=dev)
        self.c_fracture = wp.array(f32([r.fracture for r in self._rows]), dtype=float, device=dev)
        self.c_sibling = wp.array(i32([r.sibling for r in self._rows]), dtype=int, device=dev)
        self.c_friction = wp.array(f32([r.friction for r in self._rows]), dtype=float, device=dev)
        self.c_friction_static = wp.array(
            f32([r.friction_static for r in self._rows]), dtype=float, device=dev)
        self.c_partner = wp.array(
            i32([r.partner for r in self._rows]), dtype=int, device=dev)
        # c_was_static persists across substeps via the contact cache + the
        # update_static_friction kernel. Seed to 0 on first build; subsequent
        # _flush() calls only happen when scene topology changes (new rows
        # appended) so we re-init to 0 there too — any in-flight contact will
        # be re-evaluated on the very next substep.
        self.c_was_static = wp.zeros(n_c, dtype=int, device=dev)

        starts, indices = self._build_adjacency()
        self.body_con_starts = wp.array(starts, dtype=int, device=dev)
        self.body_con_indices = wp.array(indices, dtype=int, device=dev)
        self._dirty = False

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

    def _ensure_pool_buffers(self, n_pool: int) -> None:
        """Reallocate GPU-resident cache scratch arrays when the dynamic
        contact pool grows. Buffers are shared across substeps and only
        resized when the active pair count exceeds the current capacity.
        Each pool pair owns 8 floats in the packed buffer — see the layout
        header above `cache_restore_6dof` in kernels_6dof.py.

        Also reuses the host-side numpy staging buffers so the per-substep
        upload pattern is (assign into preallocated wp.array) instead of
        (wp.array(...) → wp.copy → free)."""
        if n_pool <= self._pool_buf_cap:
            return
        cap = max(256, n_pool * 2)
        dev = self.device
        self._pool_idx_n = wp.zeros(cap, dtype=int, device=dev)
        self._pool_idx_t = wp.zeros(cap, dtype=int, device=dev)
        self._pool_idx_b = wp.zeros(cap, dtype=int, device=dev)
        self._pool_cache_valid = wp.zeros(cap, dtype=int, device=dev)
        self._pool_in_packed = wp.zeros(cap * 8, dtype=float, device=dev)
        self._pool_out_packed = wp.zeros(cap * 8, dtype=float, device=dev)
        # Host-side staging arrays — sized to capacity so we can reuse them
        # in subsequent substeps without re-allocating. wp.array.assign()
        # writes only the prefix that the source covers.
        self._pool_h_idx_n = np.empty(cap, dtype=np.int32)
        self._pool_h_idx_t = np.empty(cap, dtype=np.int32)
        self._pool_h_idx_b = np.empty(cap, dtype=np.int32)
        self._pool_h_valid = np.empty(cap, dtype=np.int32)
        self._pool_h_packed = np.empty(cap * 8, dtype=np.float32)
        self._pool_buf_cap = cap

    def _restore_cache_from_pool(self) -> None:
        """Build the per-pair index + cached-state buffers on the CPU, upload
        once, then run cache_restore_6dof to write λ/k/was_static into the
        live c_* arrays in place. Replaces three full-array GPU↔CPU round
        trips (c_lambda.numpy(), c_penalty.numpy(), c_was_static.numpy() +
        their wp.array re-creations) with one small upload + one launch.
        See AVBD_PERFORMANCE_GAP §5."""
        n_pool = len(self._pool_pair_rows)
        if n_pool == 0:
            return
        self._ensure_pool_buffers(n_pool)
        # Fill the staging numpy buffers (preallocated, no new allocations).
        idx_n = self._pool_h_idx_n
        idx_t = self._pool_h_idx_t
        idx_b = self._pool_h_idx_b
        valid = self._pool_h_valid
        packed = self._pool_h_packed
        valid[:n_pool] = 0
        packed[:n_pool * 8] = 0.0
        for p_idx, (pair, n_idx, t_idx, b_idx) in enumerate(self._pool_pair_rows):
            idx_n[p_idx] = n_idx
            idx_t[p_idx] = t_idx
            idx_b[p_idx] = b_idx
            cached = self._contact_cache.get(pair)
            if cached is None:
                continue
            if not all(math.isfinite(v) for v in cached[:6]):
                continue
            lam_n, lam_t, lam_b, k_n, k_t, k_b, was = cached
            base = p_idx * 8
            packed[base + 0] = lam_n
            packed[base + 1] = lam_t
            packed[base + 2] = lam_b
            packed[base + 3] = k_n
            packed[base + 4] = k_t
            packed[base + 5] = k_b
            packed[base + 7] = float(was)
            valid[p_idx] = 1
        # `assign()` writes the source's prefix into the preallocated wp.array
        # storage — no per-substep device-side allocation, no temp wp.array.
        self._pool_idx_n.assign(idx_n[:n_pool])
        self._pool_idx_t.assign(idx_t[:n_pool])
        self._pool_idx_b.assign(idx_b[:n_pool])
        self._pool_cache_valid.assign(valid[:n_pool])
        self._pool_in_packed.assign(packed[:n_pool * 8])
        wp.launch(
            K.cache_restore_6dof, dim=n_pool,
            inputs=[self._pool_idx_n, self._pool_idx_t, self._pool_idx_b,
                    self._pool_cache_valid, self._pool_in_packed,
                    self.c_lambda, self.c_penalty, self.c_was_static],
            device=self.device,
        )

    def _persist_cache_from_pool(self) -> None:
        """Inverse of _restore_cache_from_pool — runs cache_collect_6dof to
        gather the post-solve λ/k/active/was_static of every pool-pair row
        into one packed staging buffer, then does ONE .numpy() readback
        before rebuilding the Python contact_cache dict. Eliminates the
        four full-array .numpy() calls (lam, pen, act, was) the original
        loop did per substep."""
        n_pool = len(self._pool_pair_rows)
        if n_pool == 0:
            self._contact_cache = {}
            return
        dev = self.device
        wp.launch(
            K.cache_collect_6dof, dim=n_pool,
            inputs=[self._pool_idx_n, self._pool_idx_t, self._pool_idx_b,
                    self.c_lambda, self.c_penalty,
                    self.c_active, self.c_was_static],
            outputs=[self._pool_out_packed],
            device=dev,
        )
        arr = self._pool_out_packed.numpy()[:n_pool * 8].reshape(n_pool, 8)
        new_cache: dict[
            tuple, tuple[float, float, float, float, float, float, int]
        ] = {}
        for p_idx, (pair, _n_idx, _t_idx, _b_idx) in enumerate(self._pool_pair_rows):
            if arr[p_idx, 6] == 0.0:   # c_active[n_idx] == 0
                continue
            lam_n = float(arr[p_idx, 0])
            lam_t = float(arr[p_idx, 1])
            lam_b = float(arr[p_idx, 2])
            k_n = float(arr[p_idx, 3])
            k_t = float(arr[p_idx, 4])
            k_b = float(arr[p_idx, 5])
            was = int(arr[p_idx, 7])
            if not all(math.isfinite(v)
                       for v in (lam_n, lam_t, lam_b, k_n, k_t, k_b)):
                continue
            new_cache[pair] = (lam_n, lam_t, lam_b, k_n, k_t, k_b, was)
        self._contact_cache = new_cache

    def _step_one(self) -> None:
        # Generate dynamic OBB-OBB contacts from CURRENT body positions
        # BEFORE the upload — _rebuild_contact_pool mutates self._rows and
        # sets _dirty=True, then _flush() rebuilds Warp arrays + coloring.
        if self._self_collide:
            self._flush()  # make sure self.x / self.q reflect latest state
            self._rebuild_contact_pool()
        self._flush()
        # Seed λ + penalty for pool rows from the persistent cache so OBB
        # stacks carry their augmented-Lagrangian state across frames (without
        # this, every contact restarts from PENALTY_MIN every frame and a
        # stable stack looks like a soft spring during the transient). The
        # restore runs as a Warp kernel against preallocated GPU buffers so
        # the live c_lambda / c_penalty / c_was_static arrays stay resident
        # through the substep — see AVBD_PERFORMANCE_GAP §5.
        if self._self_collide and self._pool_pair_rows:
            self._restore_cache_from_pool()
        n_b = len(self._x)
        n_c = len(self._rows)
        if n_b == 0:
            return
        dev = self.device

        # 1. Inertial target + warm-started x⁰, q⁰; cache BOTH R·I_inv·R^T
        # AND R·I·R^T so primal_update_6dof never has to invert.
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

        if n_c > 0:
            # Fused substep prelude — warmstart_duals + update_static_friction
            # + cache_alpha_C0(main) in one launch. Each was previously ~95%
            # launch-overhead-bound (~18 μs Python launch vs <1 μs compute);
            # fusion cuts 2 launches per substep (= 16/step).
            # AVBD post-stabilize mode: main-iter α=1.0 (preserve initial
            # constraint error, only resist NEW motion → contact decelerates
            # the body correctly), then a separate cache_alpha_C0 launch
            # below uses α=0.0 for the final post-stab iter.
            main_alpha = 1.0 if self.post_stabilize else self.alpha
            wp.launch(
                K.substep_prelude_6dof, dim=n_c,
                inputs=[self.x_initial, self.q_initial,
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
        for it in range(total_iters):
            if self.post_stabilize and it == self.iterations and n_c > 0:
                wp.launch(
                    K.cache_alpha_C0_6dof, dim=n_c,
                    inputs=[self.x, self.q, self.x_initial, self.q_initial,
                            self.c_type, self.c_body_a, self.c_body_b,
                            self.c_world_anchor, self.c_off_a, self.c_off_b,
                            self.c_rest, self.c_active, 0.0],
                    outputs=[self.c_alpha_C0],
                    device=dev,
                )

            for color_id in range(self.num_colors):
                wp.launch(
                    K.primal_update_6dof, dim=n_b,
                    inputs=[self.x, self.q, self.mass,
                            self.inv_inertia_world, self.inertia_world,
                            self.x_inertial, self.q_inertial, self.body_color,
                            self.c_type, self.c_body_a, self.c_body_b,
                            self.c_world_anchor, self.c_off_a, self.c_off_b,
                            self.c_rest, self.c_stiffness,
                            self.c_lambda, self.c_penalty,
                            self.c_fmin, self.c_fmax,
                            self.c_alpha_C0, self.c_active,
                            self.c_sibling, self.c_friction,
                            self.c_friction_static, self.c_was_static,
                            self.body_con_starts, self.body_con_indices,
                            self.dt, color_id],
                    device=dev,
                )

            if n_c > 0 and it < self.iterations:
                wp.launch(
                    K.dual_update_6dof, dim=n_c,
                    inputs=[self.x, self.q,
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
                # Fused finalize + cap saves 1 launch per substep. Disabled
                # cap (max_*=inf or ≤0) is folded into the kernel via the
                # `sl > max_lin` test — wp.inf is never exceeded, so the
                # cap branch is a no-op then.
                if (self.max_linear_speed > 0.0
                        and math.isfinite(self.max_linear_speed)
                        and self.max_angular_speed > 0.0
                        and math.isfinite(self.max_angular_speed)):
                    max_lin = float(self.max_linear_speed)
                    max_ang = float(self.max_angular_speed)
                else:
                    max_lin = math.inf
                    max_ang = math.inf
                wp.launch(
                    K.finalize_and_cap_6dof, dim=n_b,
                    inputs=[self.x, self.q, self.x_initial, self.q_initial,
                            self.mass, self.dt, max_lin, max_ang],
                    outputs=[self.v, self.omega, self.prev_v, self.prev_omega],
                    device=dev,
                )

        # Persist contact-pool λ + penalty for the next step. Drop pairs
        # whose contact became inactive (e.g., separated or fractured).
        # The collect kernel packs everything into one float buffer so we
        # do a single .numpy() (vs four) — see AVBD_PERFORMANCE_GAP §5.
        if self._self_collide and self._pool_pair_rows:
            self._persist_cache_from_pool()

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
        n_c = len(self._rows)
        if n_b == 0 or self.x is None:
            return {
                "positions": np.array(self._x, dtype=np.float32).reshape(-1, 3),
                "orientations": np.array(self._q, dtype=np.float32).reshape(-1, 4),
                "angular_velocities": np.array(self._omega, dtype=np.float32).reshape(-1, 3),
                "lambdas": np.zeros(n_c, dtype=np.float32),
                "active": np.ones(n_c, dtype=np.int32),
                "was_static": np.zeros(n_c, dtype=np.int32),
                "c_type": np.zeros(n_c, dtype=np.int32),
            }
        dev = self.device
        # Reuse staging buffers across frames; reallocate only on count change.
        if (self._viewer_pack_bodies is None
                or self._viewer_pack_bodies_n != n_b):
            self._viewer_pack_bodies = wp.zeros(n_b * 10, dtype=float,
                                                device=dev)
            self._viewer_pack_bodies_n = n_b
        if (self._viewer_pack_rows is None
                or self._viewer_pack_rows_n != n_c):
            self._viewer_pack_rows = (wp.zeros(n_c * 3, dtype=float,
                                               device=dev)
                                      if n_c > 0 else None)
            self._viewer_pack_rows_n = n_c
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
            row = self._viewer_pack_rows.numpy().reshape(n_c, 3)
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
        }
