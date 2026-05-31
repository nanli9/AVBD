"""AVBD 3D particle solver.

Sequential Gauss-Seidel per-body update (one body per color class). Graph
coloring for true parallelism is a follow-up — kernels already accept a
`current_color` gate so adding coloring is just a different `body_color`
mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import warp as wp

from . import kernels as K
from .coloring import build_body_edges, color_summary, greedy_color
from .scene import Body, ConstraintHandle, Shape

# Constraint type codes (must match kernels.py)
PIN_X = 0
PIN_Y = 1
PIN_Z = 2
DISTANCE = 3
FLOOR_CONTACT = 4
SPHERE_CONTACT = 5
CONTACT_TANGENT = 6
SPHERE_BOX_CONTACT = 7
BOX_BOX_CONTACT = 8


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build an orthonormal (t̂, b̂) frame perpendicular to unit normal n̂.
    Uses the numerically-stable branch-on-sign method (Duff et al. 2017).
    """
    n = n / (np.linalg.norm(n) + 1e-20)
    sign = 1.0 if n[2] >= 0.0 else -1.0
    a = -1.0 / (sign + n[2])
    b_ = n[0] * n[1] * a
    t = np.array([1.0 + sign * n[0] * n[0] * a, sign * b_, -sign * n[0]],
                 dtype=np.float32)
    bvec = np.array([b_, sign + n[1] * n[1] * a, -n[1]], dtype=np.float32)
    return t, bvec


def _shape_collision_spheres(shape: Shape | None) -> list[tuple[tuple[float, float, float], float]]:
    """Return the body's collision geometry as a list of (offset_xyz, radius)
    sub-spheres in the body's local frame. The broad phase tests every pair
    of (sub-sphere on body i, sub-sphere on body j) for overlap, so this is
    how we approximate non-spherical bodies without a new constraint type.

    Sphere: 1 sub-sphere at the body centre, full radius.
    Cube  : 1 inscribed centre sphere (radius r = min(half_extents)) PLUS 8
            corner sub-spheres at (±α·r, ±α·r, ±α·r) with radius (1−α)·r and
            α tuned so the corner sub-spheres just touch the cube faces in
            their octant and don't false-trigger along the axis directions
            (which are handled by the centre sphere). α=0.85 gives ≤25 mm
            residual overlap in the worst-case (corner-direction) contact,
            and zero gap on face-on contact.
    Pillar: a STACK of inscribed-radius spheres along Y.

    True box-box and sphere-cylinder manifolds are still TODO; this multi-
    sphere approximation is the right cheap fix for the demo.
    """
    if shape is None:
        return [((0.0, 0.0, 0.0), 0.0)]
    if shape.kind == "sphere":
        return [((0.0, 0.0, 0.0), float(shape.size[0]))]
    if shape.kind == "cube":
        # Box-vs-anything-not-a-box now uses analytical SPHERE_BOX_CONTACT
        # in the broad phase (closest-point-on-AABB). The list returned here
        # is only consulted by the box-box fallback, which still uses the
        # multi-sub-sphere approximation (true box-box needs SAT/GJK). We
        # keep the 1+8 sub-spheres so box-box pairs land in the same place
        # they used to; sphere-vs-cube no longer touches this list.
        r = float(min(shape.size))
        out: list[tuple[tuple[float, float, float], float]] = [((0.0, 0.0, 0.0), r)]
        alpha = 0.85
        rho = (1.0 - alpha) * r
        s = alpha * r
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    out.append(((sx * s, sy * s, sz * s), rho))
        return out
    if shape.kind == "pillar":
        r = float(shape.size[0])
        h = float(shape.size[1])  # half-height
        if h <= r:
            return [((0.0, 0.0, 0.0), r)]
        n = max(2, int(math.ceil(h / r)))
        return [((0.0, -(h - r) + i * 2 * (h - r) / (n - 1), 0.0), r)
                for i in range(n)]
    return [((0.0, 0.0, 0.0), 0.0)]


def _shape_collision_radius(shape: Shape | None) -> float:
    """Maximum sub-sphere centre-to-edge distance for coarse broad-phase
    rejection: returns max over sub-spheres of (||offset|| + radius)."""
    spheres = _shape_collision_spheres(shape)
    if not spheres:
        return 0.0
    return max(math.sqrt(o[0]**2 + o[1]**2 + o[2]**2) + r for o, r in spheres)


@dataclass
class _ConstraintRow:
    type: int
    body_a: int
    body_b: int
    world_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rest: float = 0.0
    stiffness: float = math.inf
    fracture: float = math.inf
    fmin: float = -math.inf
    fmax: float = math.inf
    # For CONTACT_TANGENT rows only: index of the sibling normal row,
    # and the combined coefficient of friction μ = sqrt(μ_a · μ_b).
    sibling: int = -1
    friction: float = 0.0
    # For SPHERE_CONTACT rows: per-side sub-sphere offsets in body-local
    # frame. Adding (off_a / off_b) to the body centres gives the actual
    # contact-point centres used in C and J. Defaults are (0,0,0) so all
    # other constraint types are unaffected.
    off_a: tuple[float, float, float] = (0.0, 0.0, 0.0)
    off_b: tuple[float, float, float] = (0.0, 0.0, 0.0)


class Solver:
    """3D AVBD particle solver.

    Parameters mirror the 2D reference (savant117/avbd-demo2d).
    """

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
        max_speed: float = 30.0,
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
        # Defensive runaway guard — see kernels.cap_velocity docstring. Real-
        # world objects in a tabletop demo don't move 30+ m/s; clamping here
        # breaks the post-stab-snap → overshoot → bigger-collision feedback
        # loop without changing well-behaved frames.
        self.max_speed = float(max_speed)

        self._bodies_x: list[tuple[float, float, float]] = []
        self._bodies_v: list[tuple[float, float, float]] = []
        self._bodies_mass: list[float] = []
        self._bodies_radius: list[float] = []  # max sub-sphere radius (broad-phase coarse filter)
        # Per-body collision geometry as a list of (offset_xyz, radius) sub-
        # spheres in the body's local frame. Sphere → 1 entry; cube → 1+8 (only
        # used by box-box fallback); pillar → stacked.
        self._bodies_collision_spheres: list[list[tuple[tuple[float, float, float], float]]] = []
        # Per-body shape kind ("sphere" / "box" / "pillar" / "") so the broad
        # phase can dispatch to the analytical SPHERE_BOX_CONTACT path when a
        # box is involved. Half-extents stored only when meaningful (boxes).
        self._bodies_kind: list[str] = []
        self._bodies_half_extents: list[tuple[float, float, float]] = []
        self._bodies_collide: list[bool] = []  # whether the body participates in broad-phase
        self._bodies_friction: list[float] = []  # per-body μ (combined as sqrt(μ_a·μ_b))
        self._constraints: list[_ConstraintRow] = []
        self._dirty = True
        # Body-body contact: dynamic constraint pool rebuilt each frame.
        # Index range into self._constraints — slots from _contact_pool_start
        # onward are owned by the broad phase. AVBD paper uses LBVH (Sec.4);
        # for this demo a brute O(N²) sweep is fine.
        self._contact_pool_start: int | None = None
        self._enable_self_collision: bool = False
        self._contact_margin: float = 0.0  # extra distance to trigger contact
        # Default friction coefficient for sphere-sphere contacts when neither
        # body specifies one (the broad phase combines as sqrt(μ_a·μ_b)).
        self._default_friction: float = 0.0
        # Persistent λ + penalty cache for sphere-sphere contacts, keyed by
        # (body_a, body_b, sub_sphere_a, sub_sphere_b) so a multi-sub-sphere
        # pillar's separate contacts each warm-start independently. Value is
        # (λ_n, λ_t, λ_b, k_n, k_t, k_b). Persisting both λ AND the grown
        # penalty is what makes AVBD's augmented-Lagrangian advantage real
        # for stacked/resting cubes — without it the dynamic pool restarts
        # every frame from PENALTY_MIN and contact behaves like a soft spring
        # during the transient.
        self._contact_state_cache: dict[tuple[int, int, int, int], tuple[float, float, float, float, float, float]] = {}

        # Warp arrays — built lazily in _flush().
        self.x = self.v = self.prev_v = self.mass = None
        self.initial = self.inertial = self.body_color = None
        self.c_type = self.c_body_a = self.c_body_b = None
        self.c_world_anchor = self.c_rest = self.c_stiffness = None
        self.c_lambda = self.c_penalty = self.c_fmin = self.c_fmax = None
        self.c_alpha_C0 = self.c_active = self.c_fracture = None
        self.body_con_starts = self.body_con_indices = None

    # ---- Scene building -----------------------------------------------------

    def add_particle(
        self,
        position: tuple[float, float, float],
        mass: float,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shape: Shape | None = None,
        collide: bool = True,
        radius: float | None = None,
        friction: float | None = None,
    ) -> Body:
        idx = len(self._bodies_x)
        self._bodies_x.append(tuple(float(v) for v in position))
        self._bodies_v.append(tuple(float(v) for v in velocity))
        self._bodies_mass.append(float(mass))
        sh = shape or Shape()
        if radius is not None:
            spheres = [((0.0, 0.0, 0.0), float(radius))]
        else:
            spheres = _shape_collision_spheres(sh)
        self._bodies_collision_spheres.append(spheres)
        kind = sh.kind if radius is None else "sphere"
        self._bodies_kind.append(kind if kind in ("sphere", "cube", "pillar") else "")
        if sh.kind == "cube" and radius is None:
            hx, hy, hz = float(sh.size[0]), float(sh.size[1]), float(sh.size[2])
            self._bodies_half_extents.append((hx, hy, hz))
        else:
            self._bodies_half_extents.append((0.0, 0.0, 0.0))
        r_max = _shape_collision_radius(sh) if radius is None else float(radius)
        self._bodies_radius.append(r_max)
        self._bodies_collide.append(bool(collide) and r_max > 0.0)
        mu = float(friction) if friction is not None else self._default_friction
        self._bodies_friction.append(max(0.0, mu))
        self._dirty = True
        return Body(index=idx, shape=sh)

    def add_pin(
        self,
        body: Body,
        world_point: tuple[float, float, float],
        stiffness: float = math.inf,
        fracture: float = math.inf,
    ) -> ConstraintHandle:
        """Pin a particle to a world point along all three axes."""
        start = len(self._constraints)
        for axis in (PIN_X, PIN_Y, PIN_Z):
            self._constraints.append(
                _ConstraintRow(
                    type=axis,
                    body_a=body.index,
                    body_b=-1,
                    world_anchor=tuple(float(p) for p in world_point),
                    stiffness=stiffness,
                    fracture=fracture,
                )
            )
        self._dirty = True
        return ConstraintHandle(index=start, rows=3)

    def add_distance(
        self,
        body_a: Body,
        body_b: Body,
        rest: float,
        stiffness: float = math.inf,
        fracture: float = math.inf,
    ) -> ConstraintHandle:
        idx = len(self._constraints)
        self._constraints.append(
            _ConstraintRow(
                type=DISTANCE,
                body_a=body_a.index,
                body_b=body_b.index,
                rest=float(rest),
                stiffness=stiffness,
                fracture=fracture,
            )
        )
        self._dirty = True
        return ConstraintHandle(index=idx, rows=1)

    def enable_self_collision(
        self,
        enabled: bool = True,
        margin: float = 0.0,
        default_friction: float | None = None,
    ) -> None:
        """Turn on per-step broad-phase sphere-sphere contact generation.

        Each step() will scan all bodies whose `collide=True` flag was set in
        `add_particle`, generate one SPHERE_CONTACT constraint per overlapping
        pair, and discard them after the step. Connected pairs (sharing a
        distance constraint, e.g. chain links) are NOT collided against each
        other — the rest-length constraint already keeps them apart.

        `margin` adds slack to the contact radius (Sec.3.5 of VBD calls this
        the "thickness h" of the soft contact zone — we keep it 0 for hard
        contact). AVBD Sec.4 uses an LBVH; we use brute O(N²), fine for ≤50
        bodies. Algorithm 1, line 2 of AVBD calls for re-coloring every
        frame; we do that implicitly via the dirty flag in _flush().
        """
        self._enable_self_collision = bool(enabled)
        self._contact_margin = float(margin)
        if default_friction is not None:
            self._default_friction = max(0.0, float(default_friction))
            # Backfill bodies that were added before friction was configured.
            for k in range(len(self._bodies_friction)):
                if self._bodies_friction[k] == 0.0:
                    self._bodies_friction[k] = self._default_friction
        if enabled and self._contact_pool_start is None:
            # Reserve the boundary between fixed constraints and the dynamic
            # contact pool. Everything appended later will be a contact row.
            self._contact_pool_start = len(self._constraints)

    def _emit_sphere_box(
        self,
        sphere_idx: int,
        box_idx: int,
        sphere_subspheres: list[tuple[tuple[float, float, float], float]],
        box_half_extents: tuple[float, float, float],
        positions: np.ndarray,
        margin: float,
        mu: float,
    ) -> None:
        """Generate analytical SPHERE_BOX_CONTACT rows for each of the
        sphere body's sub-spheres against the box. A plain sphere body has
        just one sub-sphere at (0,0,0); a pillar produces several stacked
        along Y. Each row that's overlap-positive in the broad-phase coarse
        test gets:
          - 1 SPHERE_BOX_CONTACT row (normal)
          - 2 CONTACT_TANGENT rows (friction t̂, b̂), if μ > 0
        """
        hext = np.asarray(box_half_extents, dtype=np.float32)
        p_box = positions[box_idx]
        for si, (off_xyz, r_s) in enumerate(sphere_subspheres):
            off_arr = np.asarray(off_xyz, dtype=np.float32)
            p_sphere = positions[sphere_idx] + off_arr
            # Closest-point-on-AABB test — exact, same code path the kernel
            # uses, just in numpy. We allow `margin` of slack so a contact
            # that's about to engage gets a row this frame.
            rel = p_sphere - p_box
            clamped = np.clip(rel, -hext, hext)
            d_vec = rel - clamped
            d_len = float(np.linalg.norm(d_vec))
            # Penetration: d_len < r_s (or sphere centre inside box → d_len=0)
            if d_len > r_s + margin:
                continue
            # Build contact frame n̂ for friction (matches kernel's branch).
            if d_len > 1e-9:
                n_hat = d_vec / d_len
            else:
                # Sphere centre inside box — use smallest-face axis (must
                # match kernel exactly so t̂ / b̂ are consistent).
                inside = hext - np.abs(rel)
                axis = int(np.argmin(inside))
                n_hat = np.zeros(3, dtype=np.float32)
                n_hat[axis] = float(np.sign(rel[axis]) or 1.0)
            t_hat, b_hat = _orthonormal_basis(n_hat)
            normal_idx = len(self._constraints)
            self._constraints.append(
                _ConstraintRow(
                    type=SPHERE_BOX_CONTACT,
                    body_a=sphere_idx, body_b=box_idx,
                    world_anchor=(float(hext[0]), float(hext[1]), float(hext[2])),
                    rest=float(r_s),
                    stiffness=math.inf,
                    fmin=-math.inf, fmax=0.0,
                    off_a=(float(off_arr[0]), float(off_arr[1]), float(off_arr[2])),
                    off_b=(0.0, 0.0, 0.0),
                )
            )
            t_idx = -1
            b_idx = -1
            if mu > 0.0:
                t_idx = len(self._constraints)
                self._constraints.append(
                    _ConstraintRow(
                        type=CONTACT_TANGENT,
                        body_a=sphere_idx, body_b=box_idx,
                        world_anchor=(float(t_hat[0]), float(t_hat[1]), float(t_hat[2])),
                        stiffness=math.inf, sibling=normal_idx, friction=mu,
                    )
                )
                b_idx = len(self._constraints)
                self._constraints.append(
                    _ConstraintRow(
                        type=CONTACT_TANGENT,
                        body_a=sphere_idx, body_b=box_idx,
                        world_anchor=(float(b_hat[0]), float(b_hat[1]), float(b_hat[2])),
                        stiffness=math.inf, sibling=normal_idx, friction=mu,
                    )
                )
            # Cache key disambiguates per-sub-sphere on the sphere side; the
            # box contributes a single "virtual sub-sphere" idx 0.
            self._pool_pair_rows.append((
                (sphere_idx, box_idx, si, 0), normal_idx, t_idx, b_idx,
            ))

    def _emit_box_box(
        self,
        i: int,
        j: int,
        h_i_tuple: tuple[float, float, float],
        h_j_tuple: tuple[float, float, float],
        positions: np.ndarray,
        margin: float,
        mu: float,
    ) -> None:
        """3D box-box collision: SAT (3 axes for AABBs) + face-clip rectangle.

        For axis-aligned boxes, SAT degenerates to 3 face-axis tests; the 9
        edge-edge cross products vanish because all edges are axis-parallel.
        The smallest-overlap axis is the contact normal n̂ (per Erin Catto's
        SAT — see avbd-demo2d/source/collide.cpp:211-246). The two
        perpendicular axes define a 2D overlap rectangle whose 4 corners
        become the 4 contact points. Each contact is its own AVBD Eq. 15
        constraint with independent (n̂, t̂, b̂) basis and friction cone.
        """
        h_i = np.asarray(h_i_tuple, dtype=np.float32)
        h_j = np.asarray(h_j_tuple, dtype=np.float32)
        p_i = positions[i]
        p_j = positions[j]
        diff = p_i - p_j
        # Per-axis overlap; negative on any axis ⇒ AABBs are separated.
        overlap = (h_i + h_j) - np.abs(diff)
        if np.any(overlap < -margin):
            return
        # Smallest-overlap axis = contact-normal direction (deepest into
        # separation), matching the 2D demo's bestAxis selection.
        axis = int(np.argmin(overlap))
        # n̂ points from j to i along that axis.
        sgn = 1.0 if diff[axis] >= 0.0 else -1.0
        n_hat = np.zeros(3, dtype=np.float32)
        n_hat[axis] = sgn
        # 2D overlap rectangle in the plane perpendicular to `axis`.
        other = [k for k in range(3) if k != axis]
        rect_lo = np.array([
            max(p_i[other[0]] - h_i[other[0]], p_j[other[0]] - h_j[other[0]]),
            max(p_i[other[1]] - h_i[other[1]], p_j[other[1]] - h_j[other[1]]),
        ], dtype=np.float32)
        rect_hi = np.array([
            min(p_i[other[0]] + h_i[other[0]], p_j[other[0]] + h_j[other[0]]),
            min(p_i[other[1]] + h_i[other[1]], p_j[other[1]] + h_j[other[1]]),
        ], dtype=np.float32)
        # Degenerate overlap (point or line — happens at glancing contact).
        # Skip rather than emit zero-area contacts that would be unstable.
        if rect_hi[0] <= rect_lo[0] or rect_hi[1] <= rect_lo[1]:
            return
        # Contact points on each box's contact face (in world coords).
        # Box i's face along -n̂: at p_i[axis] - sgn·h_i[axis].
        # Box j's face along +n̂: at p_j[axis] + sgn·h_j[axis].
        face_i_axis = p_i[axis] - sgn * h_i[axis]
        face_j_axis = p_j[axis] + sgn * h_j[axis]
        # Friction tangent basis (built once per pair; same for all 4 corners).
        t_hat, b_hat = _orthonormal_basis(n_hat)
        # Emit one BOX_BOX_CONTACT per corner of the overlap rectangle.
        # Corner-id ∈ {0..3} so warm-starting across frames is stable.
        for corner_id, (cx, cy) in enumerate([
            (rect_lo[0], rect_lo[1]),
            (rect_hi[0], rect_lo[1]),
            (rect_lo[0], rect_hi[1]),
            (rect_hi[0], rect_hi[1]),
        ]):
            # Box i's contact point at this corner (world coords)
            world_i = np.zeros(3, dtype=np.float32)
            world_i[axis] = face_i_axis
            world_i[other[0]] = cx
            world_i[other[1]] = cy
            # Box j's contact point at the same (x,y) lateral position
            world_j = np.zeros(3, dtype=np.float32)
            world_j[axis] = face_j_axis
            world_j[other[0]] = cx
            world_j[other[1]] = cy
            off_a = world_i - p_i
            off_b = world_j - p_j
            normal_idx = len(self._constraints)
            self._constraints.append(
                _ConstraintRow(
                    type=BOX_BOX_CONTACT,
                    body_a=i, body_b=j,
                    world_anchor=(float(n_hat[0]), float(n_hat[1]), float(n_hat[2])),
                    rest=0.0,
                    stiffness=math.inf,
                    fmin=-math.inf, fmax=0.0,
                    off_a=(float(off_a[0]), float(off_a[1]), float(off_a[2])),
                    off_b=(float(off_b[0]), float(off_b[1]), float(off_b[2])),
                )
            )
            t_idx = -1
            b_idx = -1
            if mu > 0.0:
                t_idx = len(self._constraints)
                self._constraints.append(
                    _ConstraintRow(
                        type=CONTACT_TANGENT,
                        body_a=i, body_b=j,
                        world_anchor=(float(t_hat[0]), float(t_hat[1]), float(t_hat[2])),
                        stiffness=math.inf, sibling=normal_idx, friction=mu,
                    )
                )
                b_idx = len(self._constraints)
                self._constraints.append(
                    _ConstraintRow(
                        type=CONTACT_TANGENT,
                        body_a=i, body_b=j,
                        world_anchor=(float(b_hat[0]), float(b_hat[1]), float(b_hat[2])),
                        stiffness=math.inf, sibling=normal_idx, friction=mu,
                    )
                )
            # Cache key: (i, j, axis*4+corner_id, 0). Encodes both which face
            # pair (axis) and which corner of the clipped manifold, so warm-
            # starting across frames keeps each corner's λ independent — same
            # idea as the 2D demo's feature.value persistence (manifold.cpp:42).
            self._pool_pair_rows.append((
                (i, j, axis * 4 + corner_id, 0), normal_idx, t_idx, b_idx,
            ))

    def _rebuild_contact_pool(self) -> None:
        """Drop the previous frame's SPHERE_CONTACT rows and regenerate them
        from the current body positions. Brute O(N²) pairs.

        Skip pairs that already share a constraint (any type) so chain
        links / pinned anchors / etc. don't double up with a contact
        constraint on top of the joint that already binds them.
        """
        if not self._enable_self_collision:
            return
        # Strip ONLY SPHERE_CONTACT and the tangent rows we attached to them —
        # leave PIN/DISTANCE/FLOOR (and floor friction) alone. We can't filter
        # CONTACT_TANGENT purely by type because some are paired to FLOOR rows.
        # As we strip, build an old→new index map so we can renumber the
        # `sibling` field on surviving tangent rows (otherwise their sibling
        # would still point to the pre-strip layout and a later iteration
        # would index past the end of the rebuilt list).
        kept: list[_ConstraintRow] = []
        old_to_new: dict[int, int] = {}
        for old_idx, c in enumerate(self._constraints):
            if c.type == SPHERE_CONTACT:
                continue
            if c.type == CONTACT_TANGENT and 0 <= c.sibling < len(self._constraints):
                sib = self._constraints[c.sibling]
                if sib.type == SPHERE_CONTACT:
                    continue  # sphere-sphere friction — drop with the normal row
            old_to_new[old_idx] = len(kept)
            kept.append(c)
        # Renumber sibling indices for kept tangent rows.
        for c in kept:
            if c.type == CONTACT_TANGENT and c.sibling >= 0:
                c.sibling = old_to_new.get(c.sibling, -1)
        self._constraints = kept
        # The boundary moves as fixed constraints get added/removed, so we
        # always reset it to the current tail before appending.
        self._contact_pool_start = len(self._constraints)
        # Track ((body_a, body_b, sub_a, sub_b), normal_idx, t_idx, b_idx)
        # so we can seed λ from the persistent cache after _flush builds the
        # Warp arrays.
        self._pool_pair_rows: list[tuple[tuple[int, int, int, int], int, int, int]] = []

        # Build a set of "already constrained" pairs to skip in the broad phase.
        # CRITICAL: must respect c_active. A fractured distance constraint stays
        # in self._constraints with active=0 — without this check, the pair would
        # remain excluded forever and broken chain links would pass through each
        # other instead of colliding.
        act = self.c_active.numpy() if self.c_active is not None else None
        existing_pairs: set[tuple[int, int]] = set()
        for ci, c in enumerate(self._constraints):
            if c.body_b < 0:
                continue
            if act is not None and ci < len(act) and act[ci] == 0:
                continue
            a, b = (c.body_a, c.body_b) if c.body_a < c.body_b else (c.body_b, c.body_a)
            existing_pairs.add((a, b))

        positions = (
            self.x.numpy().reshape(-1, 3) if self.x is not None
            else np.array(self._bodies_x, dtype=np.float32).reshape(-1, 3)
        )
        n = len(self._bodies_x)
        radii = np.asarray(self._bodies_radius, dtype=np.float32)
        collide = np.asarray(self._bodies_collide, dtype=bool)
        margin = self._contact_margin
        for i in range(n):
            if not collide[i]:
                continue
            for j in range(i + 1, n):
                if not collide[j]:
                    continue
                if (i, j) in existing_pairs:
                    continue
                # Coarse reject via centre-distance vs. max-radius bound.
                spheres_i = self._bodies_collision_spheres[i]
                spheres_j = self._bodies_collision_spheres[j]
                coarse_rest = float(radii[i] + radii[j])
                d2 = float(np.sum((positions[i] - positions[j]) ** 2))
                if d2 > (coarse_rest + margin) ** 2:
                    continue
                # Dispatch by shape pair. The sphere/pillar ↔ box path uses
                # the analytical SPHERE_BOX_CONTACT (one row per
                # (sphere-sub-sphere on the non-box, the box) — exact contact
                # geometry, no sub-sphere approximation). Box ↔ box still
                # falls back to the sub-sphere multi-contact path because
                # true box-box needs SAT/GJK (TODO).
                ki = self._bodies_kind[i]
                kj = self._bodies_kind[j]
                mu = math.sqrt(self._bodies_friction[i] * self._bodies_friction[j])
                if ki == "cube" and kj != "cube":
                    # j is sphere or pillar — emit SPHERE_BOX_CONTACT per
                    # (j sub-sphere, box i). Sphere body = j, box body = i.
                    self._emit_sphere_box(
                        sphere_idx=j, box_idx=i,
                        sphere_subspheres=spheres_j,
                        box_half_extents=self._bodies_half_extents[i],
                        positions=positions, margin=margin, mu=mu,
                    )
                    continue
                if kj == "cube" and ki != "cube":
                    self._emit_sphere_box(
                        sphere_idx=i, box_idx=j,
                        sphere_subspheres=spheres_i,
                        box_half_extents=self._bodies_half_extents[j],
                        positions=positions, margin=margin, mu=mu,
                    )
                    continue
                if ki == "cube" and kj == "cube":
                    # SAT + face clip → up to 4 BOX_BOX_CONTACT rows per pair
                    # (avbd-demo2d's 2D analog clipped face manifolds to ≤2
                    # contacts; the 3D analog gives ≤4). Each contact is its
                    # own AVBD Eq. 15 constraint with its own (n̂, t̂, b̂).
                    self._emit_box_box(
                        i, j,
                        self._bodies_half_extents[i],
                        self._bodies_half_extents[j],
                        positions=positions, margin=margin, mu=mu,
                    )
                    continue
                # All remaining pairs (sphere-sphere, sphere-pillar, pillar-
                # pillar) → sub-sphere path.
                # Fine pass: iterate every (sub-sphere, sub-sphere) pair and
                # generate one SPHERE_CONTACT row per overlap. This is what
                # turns a stacked-sphere pillar into a proper extruded contact
                # surface and is the box-box fallback.
                for si, (off_i_xyz, r_i) in enumerate(spheres_i):
                    off_i_arr = np.asarray(off_i_xyz, dtype=np.float32)
                    for sj, (off_j_xyz, r_j) in enumerate(spheres_j):
                        off_j_arr = np.asarray(off_j_xyz, dtype=np.float32)
                        rest = float(r_i + r_j)
                        pi = positions[i] + off_i_arr
                        pj = positions[j] + off_j_arr
                        d_ij = pi - pj
                        d2 = float(np.dot(d_ij, d_ij))
                        if d2 > (rest + margin) ** 2:
                            continue
                        d_norm = float(math.sqrt(d2))
                        if d_norm > 1e-9:
                            n_hat = d_ij / d_norm
                        else:
                            n_hat = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        t_hat, b_hat = _orthonormal_basis(n_hat)
                        # 3D offsets stored in dedicated off_a/off_b fields
                        # (world_anchor is back to "unused" for SPHERE_CONTACT).
                        normal_idx = len(self._constraints)
                        self._constraints.append(
                            _ConstraintRow(
                                type=SPHERE_CONTACT,
                                body_a=i, body_b=j,
                                rest=rest,
                                stiffness=math.inf,
                                fmin=-math.inf, fmax=0.0,
                                off_a=(float(off_i_arr[0]), float(off_i_arr[1]), float(off_i_arr[2])),
                                off_b=(float(off_j_arr[0]), float(off_j_arr[1]), float(off_j_arr[2])),
                            )
                        )
                        t_idx = -1
                        b_idx = -1
                        if mu > 0.0:
                            t_idx = len(self._constraints)
                            self._constraints.append(
                                _ConstraintRow(
                                    type=CONTACT_TANGENT, body_a=i, body_b=j,
                                    world_anchor=(float(t_hat[0]), float(t_hat[1]), float(t_hat[2])),
                                    stiffness=math.inf, sibling=normal_idx, friction=mu,
                                )
                            )
                            b_idx = len(self._constraints)
                            self._constraints.append(
                                _ConstraintRow(
                                    type=CONTACT_TANGENT, body_a=i, body_b=j,
                                    world_anchor=(float(b_hat[0]), float(b_hat[1]), float(b_hat[2])),
                                    stiffness=math.inf, sibling=normal_idx, friction=mu,
                                )
                            )
                        # Cache key disambiguates sub-sphere pair so a tall
                        # pillar's separate contacts don't share λ.
                        self._pool_pair_rows.append((
                            (i, j, si, sj), normal_idx, t_idx, b_idx,
                        ))
        self._dirty = True

    def add_floor_contact(
        self,
        body: Body,
        floor_y: float = 0.0,
        stiffness: float = math.inf,
        friction: float | None = None,
    ) -> ConstraintHandle:
        """One-sided horizontal floor at y = floor_y. Only pushes body up.

        Convention matches the 2D reference's Manifold (fmin=-inf, fmax=0).
        Cost is O(1) per iteration regardless of penetration state — when the
        body is above the floor the f-clamp zeroes the force automatically.

        If `friction` (μ) > 0 — or the body itself has a non-zero friction
        coefficient — TWO additional `CONTACT_TANGENT` rows along x̂ and ẑ
        are also emitted. They use the AVBD per-row clamp |λ_t| ≤ μ|λ_n|,
        which is the square-cone approximation of the proper isotropic disk
        (Sec.3.3) — same simplification the avbd-demo2d uses in 2D.
        """
        idx = len(self._constraints)
        self._constraints.append(
            _ConstraintRow(
                type=FLOOR_CONTACT,
                body_a=body.index,
                body_b=-1,
                world_anchor=(0.0, float(floor_y), 0.0),
                stiffness=stiffness,
                fmin=-math.inf,
                fmax=0.0,
            )
        )
        rows = 1
        mu = float(friction) if friction is not None else self._bodies_friction[body.index]
        if mu > 0.0:
            for tvec in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
                self._constraints.append(
                    _ConstraintRow(
                        type=CONTACT_TANGENT,
                        body_a=body.index,
                        body_b=-1,
                        world_anchor=tvec,
                        stiffness=stiffness,
                        sibling=idx,
                        friction=mu,
                    )
                )
                rows += 1
        self._dirty = True
        return ConstraintHandle(index=idx, rows=rows)

    # --- runtime perturbations (for interactive demos) -----------------------

    def set_velocity(self, body: Body, v: tuple[float, float, float]) -> None:
        """Override a single body's velocity (e.g. to apply an impulse).
        Triggers a Warp-side copy."""
        self._flush()
        vs = self.v.numpy().copy()
        vs[body.index] = np.array(v, dtype=np.float32)
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    def add_impulse(self, body: Body, dv: tuple[float, float, float]) -> None:
        self._flush()
        vs = self.v.numpy().copy()
        vs[body.index] = vs[body.index] + np.array(dv, dtype=np.float32)
        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)

    def set_position(self, body: Body, p: tuple[float, float, float]) -> None:
        """Teleport a body. Useful for mouse-drag interactions."""
        self._flush()
        xs = self.x.numpy().copy()
        xs[body.index] = np.array(p, dtype=np.float32)
        self.x = wp.array(xs, dtype=wp.vec3, device=self.device)

    # ---- Adjacency + upload -------------------------------------------------

    def _build_adjacency(self) -> tuple[np.ndarray, np.ndarray]:
        n_bodies = len(self._bodies_x)
        counts = np.zeros(n_bodies, dtype=np.int32)
        for c in self._constraints:
            counts[c.body_a] += 1
            if c.body_b >= 0:
                counts[c.body_b] += 1
        starts = np.zeros(n_bodies + 1, dtype=np.int32)
        starts[1:] = np.cumsum(counts)
        indices = np.zeros(starts[-1] if n_bodies > 0 else 0, dtype=np.int32)
        cursor = starts[:-1].copy()
        for ci, c in enumerate(self._constraints):
            indices[cursor[c.body_a]] = ci
            cursor[c.body_a] += 1
            if c.body_b >= 0:
                indices[cursor[c.body_b]] = ci
                cursor[c.body_b] += 1
        return starts, indices

    def _flush(self) -> None:
        if not self._dirty:
            return
        n_b = len(self._bodies_x)
        n_c = len(self._constraints)
        dev = self.device

        # Snapshot the current Warp state so we can preserve already-simulated
        # bodies/constraints. Without this, _flush() (called when adding a
        # body or constraint mid-simulation) resets every body's position and
        # every constraint's λ to its initial value — effectively a scene reset.
        cur_x = self.x.numpy() if self.x is not None else None
        cur_v = self.v.numpy() if self.v is not None else None
        cur_prev_v = self.prev_v.numpy() if self.prev_v is not None else None
        cur_lam = self.c_lambda.numpy() if self.c_lambda is not None else None
        cur_pen = self.c_penalty.numpy() if self.c_penalty is not None else None
        cur_act = self.c_active.numpy() if self.c_active is not None else None
        n_b_prev = 0 if cur_x is None else int(cur_x.shape[0])
        n_c_prev = 0 if cur_lam is None else int(cur_lam.shape[0])

        x_np = np.array(self._bodies_x, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        v_np = np.array(self._bodies_v, dtype=np.float32).reshape(-1, 3) if n_b else np.zeros((0, 3), np.float32)
        m_np = np.array(self._bodies_mass, dtype=np.float32) if n_b else np.zeros(0, np.float32)
        prev_v_np = np.zeros((n_b, 3), dtype=np.float32)

        # Preserve current simulator state for already-existing bodies.
        if n_b_prev > 0 and n_b_prev <= n_b:
            x_np[:n_b_prev] = cur_x[:n_b_prev]
            v_np[:n_b_prev] = cur_v[:n_b_prev]
            if cur_prev_v is not None:
                prev_v_np[:n_b_prev] = cur_prev_v[:n_b_prev]

        self.x = wp.array(x_np, dtype=wp.vec3, device=dev)
        self.v = wp.array(v_np, dtype=wp.vec3, device=dev)
        self.prev_v = wp.array(prev_v_np, dtype=wp.vec3, device=dev)
        self.mass = wp.array(m_np, dtype=float, device=dev)
        self.initial = wp.zeros(n_b, dtype=wp.vec3, device=dev)
        self.inertial = wp.zeros(n_b, dtype=wp.vec3, device=dev)

        # Graph color the body adjacency so bodies with disjoint constraint
        # neighbourhoods update in parallel inside one launch.
        adj = build_body_edges(
            n_b,
            [c.body_a for c in self._constraints],
            [c.body_b for c in self._constraints],
        )
        color_np = greedy_color(adj) if n_b > 0 else np.zeros(0, dtype=np.int32)
        self.num_colors = int(color_np.max() + 1) if n_b > 0 else 0
        self.color_counts = color_summary(color_np)
        self.body_color = wp.array(color_np, dtype=int, device=dev)

        def f32(seq):
            return np.array(seq, dtype=np.float32) if n_c else np.zeros(0, np.float32)

        def i32(seq):
            return np.array(seq, dtype=np.int32) if n_c else np.zeros(0, np.int32)

        c_anchor = (
            np.array([c.world_anchor for c in self._constraints], dtype=np.float32).reshape(-1, 3)
            if n_c else np.zeros((0, 3), np.float32)
        )

        self.c_type = wp.array(i32([c.type for c in self._constraints]), dtype=int, device=dev)
        self.c_body_a = wp.array(i32([c.body_a for c in self._constraints]), dtype=int, device=dev)
        self.c_body_b = wp.array(i32([c.body_b for c in self._constraints]), dtype=int, device=dev)
        self.c_world_anchor = wp.array(c_anchor, dtype=wp.vec3, device=dev)
        self.c_rest = wp.array(f32([c.rest for c in self._constraints]), dtype=float, device=dev)
        self.c_stiffness = wp.array(f32([c.stiffness for c in self._constraints]), dtype=float, device=dev)

        # Preserve λ / penalty / active for already-existing FIXED constraints.
        # Dynamic-pool contact rows (indices ≥ _contact_pool_start) must get
        # fresh λ each frame — otherwise a stale λ from last frame's pair (a,b)
        # would leak into this frame's pair (c,d) sitting in the same slot.
        lam_np = np.zeros(n_c, dtype=np.float32)
        pen_np = np.full(n_c, 1.0, dtype=np.float32)
        act_np = np.ones(n_c, dtype=np.int32)
        preserve_upper = n_c
        if self._contact_pool_start is not None:
            preserve_upper = min(preserve_upper, self._contact_pool_start)
        n_keep = min(n_c_prev, preserve_upper)
        if n_keep > 0:
            lam_np[:n_keep] = cur_lam[:n_keep]
            if cur_pen is not None:
                pen_np[:n_keep] = cur_pen[:n_keep]
            if cur_act is not None:
                act_np[:n_keep] = cur_act[:n_keep]
        self.c_lambda = wp.array(lam_np, dtype=float, device=dev)
        self.c_penalty = wp.array(pen_np, dtype=float, device=dev)
        self.c_fmin = wp.array(f32([c.fmin for c in self._constraints]), dtype=float, device=dev)
        self.c_fmax = wp.array(f32([c.fmax for c in self._constraints]), dtype=float, device=dev)
        self.c_alpha_C0 = wp.zeros(n_c, dtype=float, device=dev)
        self.c_active = wp.array(act_np, dtype=int, device=dev)
        self.c_fracture = wp.array(f32([c.fracture for c in self._constraints]), dtype=float, device=dev)
        self.c_sibling = wp.array(i32([c.sibling for c in self._constraints]), dtype=int, device=dev)
        self.c_friction = wp.array(f32([c.friction for c in self._constraints]), dtype=float, device=dev)
        c_off_a_np = (np.array([c.off_a for c in self._constraints], dtype=np.float32).reshape(-1, 3)
                      if n_c else np.zeros((0, 3), np.float32))
        c_off_b_np = (np.array([c.off_b for c in self._constraints], dtype=np.float32).reshape(-1, 3)
                      if n_c else np.zeros((0, 3), np.float32))
        self.c_off_a = wp.array(c_off_a_np, dtype=wp.vec3, device=dev)
        self.c_off_b = wp.array(c_off_b_np, dtype=wp.vec3, device=dev)

        starts, indices = self._build_adjacency()
        self.body_con_starts = wp.array(starts, dtype=int, device=dev)
        self.body_con_indices = wp.array(indices, dtype=int, device=dev)
        self._dirty = False

    # ---- The step -----------------------------------------------------------

    def step(self) -> None:
        # Generate dynamic body-body contacts from the current positions
        # BEFORE flushing — _rebuild_contact_pool mutates self._constraints
        # and sets _dirty=True, then _flush rebuilds Warp arrays + coloring.
        if self._enable_self_collision:
            self._flush()  # ensure self.x reflects latest state for broad phase
            # If any body's position is non-finite (last step's λ blew up,
            # drag teleport to garbage, etc.), restore it to a safe initial
            # value BEFORE the broad phase squares the inf and crashes.
            if self.x is not None:
                xs = self.x.numpy()
                if not np.all(np.isfinite(xs)):
                    bad = ~np.all(np.isfinite(xs), axis=1)
                    n_bad = int(bad.sum())
                    if n_bad > 0:
                        # Restore from the static initial scene positions and
                        # zero the velocity / wipe the contact cache for these
                        # bodies — there's no sane warm-start after divergence.
                        safe = np.array(self._bodies_x, dtype=np.float32).reshape(-1, 3)
                        xs[bad] = safe[bad]
                        self.x = wp.array(xs, dtype=wp.vec3, device=self.device)
                        vs = self.v.numpy()
                        vs[bad] = 0.0
                        self.v = wp.array(vs, dtype=wp.vec3, device=self.device)
                        bad_set = set(int(i) for i in np.where(bad)[0])
                        self._contact_state_cache = {
                            k: v for k, v in self._contact_state_cache.items()
                            if k[0] not in bad_set and k[1] not in bad_set
                        }
                        import warnings
                        warnings.warn(
                            f"avbd3d: {n_bad} body position(s) were non-finite "
                            f"(solver divergence); reset to initial scene "
                            f"positions and zeroed velocity.",
                            RuntimeWarning, stacklevel=2,
                        )
            self._rebuild_contact_pool()
        self._flush()
        # Seed λ + penalty for pool rows from the persistent pair cache
        # (Fix C). _flush zeros λ and resets penalty for indices ≥
        # _contact_pool_start; we now restore them so the next AVBD step
        # picks up exactly where the previous frame left off.
        if self._enable_self_collision and self._pool_pair_rows:
            lam_np = self.c_lambda.numpy().copy()
            pen_np = self.c_penalty.numpy().copy()
            for pair, n_idx, t_idx, b_idx in self._pool_pair_rows:
                cached = self._contact_state_cache.get(pair)
                if cached is None:
                    continue
                # Defensive: skip any cache entry that contains non-finite
                # state — would otherwise propagate inf into the next solve.
                if not all(math.isfinite(v) for v in cached):
                    continue
                lam_n, lam_t, lam_b, k_n, k_t, k_b = cached
                lam_np[n_idx] = lam_n
                pen_np[n_idx] = k_n
                if t_idx >= 0:
                    lam_np[t_idx] = lam_t
                    pen_np[t_idx] = k_t
                if b_idx >= 0:
                    lam_np[b_idx] = lam_b
                    pen_np[b_idx] = k_b
            self.c_lambda = wp.array(lam_np, dtype=float, device=self.device)
            self.c_penalty = wp.array(pen_np, dtype=float, device=self.device)
        n_b = len(self._bodies_x)
        n_c = len(self._constraints)
        if n_b == 0:
            return
        dev = self.device

        # 1. Inertial target + adaptive warm-start
        wp.launch(
            K.predict_inertial,
            dim=n_b,
            inputs=[self.x, self.v, self.prev_v, self.mass, self.dt, wp.vec3(*self.gravity)],
            outputs=[self.initial, self.inertial, self.x],
            device=dev,
        )

        if n_c > 0:
            # 2. Warm-start λ, penalty (Eq. 19)
            wp.launch(
                K.warmstart_duals,
                dim=n_c,
                inputs=[
                    self.c_lambda, self.c_penalty, self.c_stiffness, self.c_type,
                    self.alpha, self.gamma, 1 if self.post_stabilize else 0,
                ],
                device=dev,
            )
            # 3. Cache α·C₀ (Eq. 18)
            wp.launch(
                K.cache_alpha_C0,
                dim=n_c,
                inputs=[
                    self.x, self.initial, self.c_type, self.c_body_a, self.c_body_b,
                    self.c_world_anchor, self.c_rest, self.c_active,
                    self.c_off_a, self.c_off_b,
                    self.alpha,
                ],
                outputs=[self.c_alpha_C0],
                device=dev,
            )

        total_iters = self.iterations + (1 if self.post_stabilize else 0)
        for it in range(total_iters):
            # Post-stabilization pass: drop the α·C₀ term so we drive C → 0 hard.
            if self.post_stabilize and it == self.iterations and n_c > 0:
                wp.launch(
                    K.cache_alpha_C0,
                    dim=n_c,
                    inputs=[
                        self.x, self.initial, self.c_type, self.c_body_a, self.c_body_b,
                        self.c_world_anchor, self.c_rest, self.c_active,
                        self.c_off_a, self.c_off_b,
                        0.0,
                    ],
                    outputs=[self.c_alpha_C0],
                    device=dev,
                )

            # Primal: one launch per color class. Bodies of the same color
            # share no constraint and update in parallel within the launch.
            for color_id in range(self.num_colors):
                wp.launch(
                    K.primal_update,
                    dim=n_b,
                    inputs=[
                        self.x, self.inertial, self.mass, self.body_color,
                        self.c_type, self.c_body_a, self.c_body_b,
                        self.c_world_anchor, self.c_rest, self.c_stiffness,
                        self.c_lambda, self.c_penalty, self.c_fmin, self.c_fmax,
                        self.c_alpha_C0, self.c_active,
                        self.c_sibling, self.c_friction,
                        self.c_off_a, self.c_off_b,
                        self.body_con_starts, self.body_con_indices,
                        self.dt, color_id,
                    ],
                    device=dev,
                )

            # Dual update — skipped during the extra stabilization iter.
            if n_c > 0 and it < self.iterations:
                wp.launch(
                    K.dual_update,
                    dim=n_c,
                    inputs=[
                        self.x, self.c_type, self.c_body_a, self.c_body_b,
                        self.c_world_anchor, self.c_rest, self.c_stiffness,
                        self.c_lambda, self.c_penalty, self.c_fmin, self.c_fmax,
                        self.c_alpha_C0, self.c_active, self.c_fracture,
                        self.c_sibling, self.c_friction,
                        self.c_off_a, self.c_off_b,
                        self.beta,
                    ],
                    device=dev,
                )

            if it == self.iterations - 1:
                wp.launch(
                    K.finalize_velocity,
                    dim=n_b,
                    inputs=[self.x, self.initial, self.mass, self.dt],
                    outputs=[self.v, self.prev_v],
                    device=dev,
                )
                if self.max_speed > 0.0 and math.isfinite(self.max_speed):
                    wp.launch(
                        K.cap_velocity,
                        dim=n_b,
                        inputs=[self.v, self.max_speed],
                        device=dev,
                    )

        # Fix C tail: persist λ AND grown penalty for surviving sphere-sphere
        # contacts. Pairs whose contact has separated drop from the cache;
        # surviving pairs carry full augmented-Lagrangian state into the next
        # frame so AVBD doesn't have to relitigate penalty growth from
        # PENALTY_MIN every frame. Keys now include sub-sphere indices so the
        # stacked pillar's separate contacts get independent warm-start.
        if self._enable_self_collision and self._pool_pair_rows:
            lam_np = self.c_lambda.numpy()
            pen_np = self.c_penalty.numpy()
            act_np = self.c_active.numpy()
            new_cache: dict[tuple[int, int, int, int], tuple[float, float, float, float, float, float]] = {}
            for pair, n_idx, t_idx, b_idx in self._pool_pair_rows:
                if act_np[n_idx] == 0:
                    continue  # contact broken — drop from cache
                lam_n = float(lam_np[n_idx])
                k_n = float(pen_np[n_idx])
                lam_t = float(lam_np[t_idx]) if t_idx >= 0 else 0.0
                k_t = float(pen_np[t_idx]) if t_idx >= 0 else 1.0
                lam_b = float(lam_np[b_idx]) if b_idx >= 0 else 0.0
                k_b = float(pen_np[b_idx]) if b_idx >= 0 else 1.0
                # Don't cache non-finite state (would taint next frame).
                if not all(math.isfinite(v) for v in (lam_n, lam_t, lam_b, k_n, k_t, k_b)):
                    continue
                new_cache[pair] = (lam_n, lam_t, lam_b, k_n, k_t, k_b)
            self._contact_state_cache = new_cache

    # ---- Read-back ----------------------------------------------------------

    def positions(self) -> np.ndarray:
        if self.x is None:
            return np.array(self._bodies_x, dtype=np.float32).reshape(-1, 3)
        return self.x.numpy().reshape(-1, 3)

    def velocities(self) -> np.ndarray:
        if self.v is None:
            return np.array(self._bodies_v, dtype=np.float32).reshape(-1, 3)
        return self.v.numpy().reshape(-1, 3)

    def lambdas(self) -> np.ndarray:
        if self.c_lambda is None:
            return np.zeros(len(self._constraints), dtype=np.float32)
        return self.c_lambda.numpy()

    def active(self) -> np.ndarray:
        if self.c_active is None:
            return np.ones(len(self._constraints), dtype=np.int32)
        return self.c_active.numpy()
