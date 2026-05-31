"""Warp kernels for AVBD 3D particle solver.

Equation references are to the AVBD SIGGRAPH 2025 paper unless otherwise noted.
The 2D reference implementation we follow is
https://github.com/savant117/avbd-demo2d (solver.cpp).

For this first cut we use ONE row per constraint (scalar C). Multi-row
constraints (e.g. pin = 3 rows, full rigid joint = 6 rows) are represented as
multiple scalar constraints sharing the same body pair. This keeps Warp arrays
flat (one float per row, not a MAX_ROWS-sized struct) and matches how AVBD
treats each row independently anyway.

Constraint types:
    0 = PIN       : body A's centre pinned to world point `world_anchor`. Three
                    scalar constraints per pin (one per axis).
    1 = DISTANCE  : ||x_a - x_b|| - rest = 0. One scalar constraint.
    4 = FLOOR     : one-sided y ≥ floor_y (push-only).
    5 = SPHERE-SPHERE CONTACT : one-sided ||p_a-p_b|| ≥ r_a+r_b. Specialisation
                    of VBD Eq. (12) (Chen et al. 2024a §3.5) and the normal row
                    of AVBD Eq. (15) (Giles et al. 2025 §3.3). n̂ is held
                    constant during a step (VBD §3.5: "we simplify the gradient
                    and Hessian by not differentiating through n̂").
"""

import warp as wp

# Constraint type codes
PIN_X = wp.constant(0)
PIN_Y = wp.constant(1)
PIN_Z = wp.constant(2)
DISTANCE = wp.constant(3)
# One-sided contact between a body and an infinite horizontal floor at y = world_anchor[1].
# C = y - floor_y (positive above floor). fmin=-inf, fmax=0 → only pushes up.
FLOOR_CONTACT = wp.constant(4)
# One-sided sphere-sphere contact. C = ||p_a - p_b|| - rest where rest = r_a + r_b.
# fmin=-inf, fmax=0 (matches FLOOR_CONTACT sign convention: AVBD Sec.3.3's
# λ_n^min=0, λ_n^max=∞ is the sign-flipped equivalent — the demo2d reference
# uses the same fmax=0 convention).
SPHERE_CONTACT = wp.constant(5)
# Tangent friction row paired with a SPHERE_CONTACT or FLOOR_CONTACT sibling.
# Per AVBD Sec.3.3 Eq.(15): contact constraint is C = [n̂  t̂  b̂]^T (r_a − r_b);
# the n̂ row is the normal contact (handled by SPHERE_CONTACT / FLOOR_CONTACT)
# and the t̂ / b̂ rows are tangent friction. The friction cone in the paper is
# isotropic: ||(λ_t, λ_b)|| ≤ μ λ_n. We implement the per-row BOX clamp
# |λ_t| ≤ μ |λ_n| instead (a SQUARE-cone approximation that matches what
# avbd-demo2d does in 2D), so the friction limit can be at most √2× wider than
# the disk in the worst diagonal direction — fine for a visual demo.
#
# For a CONTACT_TANGENT row j, we store:
#   c_world_anchor[j] = tangent unit vector (t̂ for the first tangent row,
#                       b̂ for the second; both cached at contact creation and
#                       held constant during the step per VBD §3.5).
#   c_sibling[j]      = index of the sibling SPHERE_CONTACT / FLOOR_CONTACT row.
#                       The bound at clamp time is μ * |c_lambda[sibling]|.
#   c_friction[j]     = μ (combined coefficient of friction for this pair).
# C = t̂ · (x_a − x_b) for sphere-sphere; C = t̂ · x_a for floor (body_b = −1).
CONTACT_TANGENT = wp.constant(6)
# Analytical sphere ↔ axis-aligned box contact. body_a = sphere body,
# body_b = box body. c_rest = sphere radius. c_world_anchor = box half-
# extents (vec3). c_off_a may shift the "sphere" centre so a pillar's
# stacked sub-spheres can collide with a box too. We use the standard
# closest-point-on-AABB formula:
#   p_closest = box.centre + clamp(rel, ±half_extents)
#   d_vec    = rel - clamp(...)          (zero when sphere centre inside box)
#   C        = ||d_vec|| - r_sphere
# When the sphere centre is inside the box, we push it out along the axis
# of smallest face distance (standard "deepest face" fallback). One-sided
# clamp `fmin=-inf, fmax=0` matches FLOOR_CONTACT / SPHERE_CONTACT.
SPHERE_BOX_CONTACT = wp.constant(7)
# Box ↔ box contact (AVBD Sec. 3.3 Eq. 15 directly):
#   C_n = n̂ · (r_a − r_b)
# where r_a, r_b are the world-space contact points on the two boxes. The
# broad phase (CPU side) runs AABB SAT — for axis-aligned boxes that's just
# 3 axes — picks the smallest-overlap axis as the contact normal n̂, then
# clips the overlap rectangle on the other two axes and emits ONE BOX_BOX
# row per corner of the clipped rectangle (up to 4 contacts per pair). This
# is the 3D analog of what avbd-demo2d/source/collide.cpp does in 2D (face
# SAT + Sutherland-Hodgman → up to 2 contacts).
#
# Storage:
#   c_world_anchor = n̂  (contact normal, constant during a step)
#   c_off_a        = r_a − x[body_a]   (contact point in body_a's frame)
#   c_off_b        = r_b − x[body_b]   (contact point in body_b's frame)
#   c_rest         = 0
# One-sided clamp `fmin=-inf, fmax=0` (push-only) matches every other
# normal-direction contact in this solver.
BOX_BOX_CONTACT = wp.constant(8)

# AVBD penalty clamps (paper §3.3, "we clamp penalty to [k_min, k_max]").
# Two floors — one for the per-row clamp on normal / pin / distance constraints,
# a SEPARATE much smaller one for friction tangent rows.
#
# Normal contact: k_min = 1e6.
# AVBD's α·C₀ stabilisation makes the iteration's stabilised constraint value
# C̃ → 0 near steady state, which makes Eq. 16's growth term β·|C̃| → 0 as
# well — penalty effectively *stops* growing once C̃ converges. That's fine
# for constraints that saturate during a long transient (the floor contact
# under a free-falling body grows k to PENALTY_MAX during the impact arc),
# but devastating for one-sided contacts that *appear* with the body already
# deep inside the other (sphere-sphere broad phase). With low k, the post-
# stabilisation pass (raw C, not C̃) can only nudge the body a few mm per
# frame and cubes visibly clip into each other. k_min=1e6 keeps even brand-
# new contacts stiff enough that one post-stab pass resolves ~10 mm of
# penetration. SOFT constraints (finite stiffness) still cap penalty at the
# material value via warmstart_duals' `min(p, stiffness)`, so springs and
# cohesive interfaces are not affected by this knob.
#
# Friction tangent: k_min = 1.0 (demo2d default).
# A Coulomb tangent row is force-bounded (|λ_t| ≤ μ|λ_n|), so the *force*
# magnitude is correct regardless of k. What k changes is how much the
# primal solve lets the body "lag" the inertial target tangentially under
# that bounded force: max lag = μ|λ_n| / (M/dt² + k). For sliding to
# decelerate at μg per frame (Coulomb prediction), lag must equal μg·dt²,
# which holds only when k ≪ M/dt². So tangent k MUST stay small.
PENALTY_MIN = wp.constant(1.0e6)
PENALTY_MIN_TANGENT = wp.constant(1.0)
PENALTY_MAX = wp.constant(1.0e9)


# -----------------------------------------------------------------------------
# Inertial target + adaptive warm-start (VBD Eq. 2 + adaptive accel weighting)
# -----------------------------------------------------------------------------
@wp.kernel
def predict_inertial(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    prev_v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    dt: float,
    gravity: wp.vec3,
    # outputs
    initial: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    x_warm: wp.array(dtype=wp.vec3),
):
    """For each body, save x⁻, compute inertial target y, and warm-start x⁰.

    Adaptive accel weighting mirrors the 2D reference (solver.cpp lines 138–146):
    if the body is already accelerating along gravity (e.g. free fall), include
    the full gravity term in the warm-start; if it's resting on a constraint,
    don't.
    """
    i = wp.tid()
    m = mass[i]
    initial[i] = x[i]
    if m <= 0.0:
        # static / kinematic body — no inertial update
        inertial[i] = x[i]
        x_warm[i] = x[i]
        return

    g_dt2 = gravity * (dt * dt)
    # Eq. 2: y = x + v·dt + g·dt²
    inertial[i] = x[i] + v[i] * dt + g_dt2

    # Adaptive warm-start (VBD §4.2): weight the gravity contribution by how
    # much the body's recent acceleration aligns with gravity.
    accel = (v[i] - prev_v[i]) / dt
    g_norm = wp.length(gravity)
    if g_norm > 0.0:
        g_hat = gravity / g_norm
        accel_ext = wp.dot(accel, g_hat)
        w = wp.clamp(accel_ext / g_norm, 0.0, 1.0)
    else:
        w = 0.0
    x_warm[i] = x[i] + v[i] * dt + g_dt2 * w


# -----------------------------------------------------------------------------
# Warm-start dual variables and penalty (AVBD Eq. 19; reference solver.cpp 109–117)
# -----------------------------------------------------------------------------
@wp.kernel
def warmstart_duals(
    lam: wp.array(dtype=float),
    pen: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    c_type: wp.array(dtype=int),
    alpha: float,
    gamma: float,
    post_stabilize: int,
):
    j = wp.tid()
    # Tangent friction rows clamp to a much smaller floor (see PENALTY_MIN_TANGENT
    # comment) so Coulomb deceleration matches μg.
    k_floor = PENALTY_MIN
    if c_type[j] == CONTACT_TANGENT:
        k_floor = PENALTY_MIN_TANGENT
    # Penalty decays each frame
    p = wp.clamp(pen[j] * gamma, k_floor, PENALTY_MAX)
    # Lambda only decays if we are not relying on post-stabilization
    if post_stabilize == 0:
        lam[j] = lam[j] * alpha * gamma
    # Clamp penalty to material stiffness for soft constraints
    s = stiffness[j]
    if not wp.isnan(s) and s < wp.inf:
        p = wp.min(p, s)
    pen[j] = p


# -----------------------------------------------------------------------------
# Constraint evaluation (per row): C, J, geometric Hessian factor
# -----------------------------------------------------------------------------
@wp.func
def eval_pin_axis(
    body_pos: wp.vec3,
    world_anchor: wp.vec3,
    axis: int,
) -> float:
    # C = (body_pos - world_anchor)[axis]
    if axis == 0:
        return body_pos[0] - world_anchor[0]
    if axis == 1:
        return body_pos[1] - world_anchor[1]
    return body_pos[2] - world_anchor[2]


@wp.func
def axis_basis(axis: int) -> wp.vec3:
    if axis == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if axis == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def eval_distance(pa: wp.vec3, pb: wp.vec3, rest: float) -> float:
    return wp.length(pa - pb) - rest


@wp.func
def sphere_box_C_and_normal(
    p_sphere: wp.vec3,
    p_box: wp.vec3,
    hext: wp.vec3,
    r_sphere: float,
):
    """Analytical sphere ↔ axis-aligned-box contact.

    Returns (C, n̂) where C is the signed gap (negative = penetration) and
    n̂ points FROM the box surface OUTWARD toward the sphere centre — i.e.
    the Jacobian for the sphere body is +n̂ and the Jacobian for the box
    body is -n̂.
    """
    rel = p_sphere - p_box
    cx = wp.clamp(rel[0], -hext[0], hext[0])
    cy = wp.clamp(rel[1], -hext[1], hext[1])
    cz = wp.clamp(rel[2], -hext[2], hext[2])
    d_vec = wp.vec3(rel[0] - cx, rel[1] - cy, rel[2] - cz)
    d_len = wp.length(d_vec)
    if d_len > 1.0e-9:
        # Sphere centre is OUTSIDE the box (or exactly on the surface).
        n_hat = d_vec / d_len
        C = d_len - r_sphere
    else:
        # Sphere centre is strictly INSIDE the box. Push out along the
        # smallest-face-distance axis (standard "deepest face" fallback).
        dx = hext[0] - wp.abs(rel[0])
        dy = hext[1] - wp.abs(rel[1])
        dz = hext[2] - wp.abs(rel[2])
        if dx <= dy and dx <= dz:
            sgn = float(1.0)
            if rel[0] < 0.0:
                sgn = -1.0
            n_hat = wp.vec3(sgn, 0.0, 0.0)
            C = -(dx + r_sphere)
        elif dy <= dz:
            sgn = float(1.0)
            if rel[1] < 0.0:
                sgn = -1.0
            n_hat = wp.vec3(0.0, sgn, 0.0)
            C = -(dy + r_sphere)
        else:
            sgn = float(1.0)
            if rel[2] < 0.0:
                sgn = -1.0
            n_hat = wp.vec3(0.0, 0.0, sgn)
            C = -(dz + r_sphere)
    return C, n_hat


@wp.func
def distance_jac(pa: wp.vec3, pb: wp.vec3) -> wp.vec3:
    """Jacobian of C = ||pa - pb|| - rest w.r.t. pa.
    Jacobian w.r.t. pb is the negative of this.
    """
    d = pa - pb
    n = wp.length(d)
    if n > 1.0e-12:
        return d / n
    return wp.vec3(0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Primal update: per-body 3x3 SPD solve (AVBD Eqs. 4, 13, 17)
# -----------------------------------------------------------------------------
@wp.kernel
def primal_update(
    # body state
    x: wp.array(dtype=wp.vec3),
    inertial: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    body_color: wp.array(dtype=int),
    # constraints
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_stiffness: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),  # α · C₀ for stabilization (Eq. 18)
    c_active: wp.array(dtype=int),
    c_sibling: wp.array(dtype=int),     # tangent → normal row (or -1)
    c_friction: wp.array(dtype=float),  # μ for CONTACT_TANGENT rows
    c_off_a: wp.array(dtype=wp.vec3),   # body_a sub-sphere offset (SPHERE_CONTACT)
    c_off_b: wp.array(dtype=wp.vec3),   # body_b sub-sphere offset (SPHERE_CONTACT)
    # body → constraint adjacency (CSR)
    body_con_starts: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
    # params
    dt: float,
    current_color: int,
):
    """Per-body local solve. Launch with dim = num_bodies; only bodies whose
    color matches `current_color` actually update (others early-out).

    Until graph coloring is added we launch with current_color=-1 and dim=1
    inside a Python loop that walks bodies sequentially (Gauss-Seidel).
    """
    i = wp.tid()
    if current_color != -1 and body_color[i] != current_color:
        return
    m = mass[i]
    if m <= 0.0:
        return

    # Build LHS = M/dt² + Σ JᵀkJ + G,  RHS = M/dt² (x - y) + Σ J·f
    inv_dt2 = 1.0 / (dt * dt)
    M_over_dt2 = m * inv_dt2
    lhs = wp.mat33(
        M_over_dt2, 0.0, 0.0,
        0.0, M_over_dt2, 0.0,
        0.0, 0.0, M_over_dt2,
    )
    rhs = (x[i] - inertial[i]) * M_over_dt2

    start = body_con_starts[i]
    end = body_con_starts[i + 1]
    for k in range(start, end):
        cj = body_con_indices[k]
        if c_active[cj] == 0:
            continue

        t = c_type[cj]
        # Per-row Jacobian J = ∂C/∂x_i and constraint value C
        J = wp.vec3(0.0, 0.0, 0.0)
        C = 0.0
        if t == PIN_X or t == PIN_Y or t == PIN_Z:
            axis = t  # 0/1/2
            anchor = c_world_anchor[cj]
            J = axis_basis(axis)
            C = eval_pin_axis(x[i], anchor, axis)
        elif t == DISTANCE:
            # Pick "the other body" (works whether i is body_a or body_b)
            other_i = c_body_a[cj]
            if c_body_a[cj] == i:
                other_i = c_body_b[cj]
            p_i = x[i]
            p_other = x[other_i]
            # ∂|p_i - p_other|/∂p_i = (p_i - p_other) / |p_i - p_other|
            J = distance_jac(p_i, p_other)
            C = eval_distance(p_i, p_other, c_rest[cj])
        elif t == FLOOR_CONTACT:
            anchor = c_world_anchor[cj]
            J = wp.vec3(0.0, 1.0, 0.0)
            C = x[i][1] - anchor[1]
        elif t == SPHERE_CONTACT:
            # Specialisation of VBD Eq. (12) / AVBD Eq. (15) normal row, with
            # per-side 3D sub-sphere offsets so a cube can collide as 1 inscribed
            # centre sphere + 8 corner sub-spheres, and a pillar as a Y-stack.
            # Sphere/cube/pillar centre sub-spheres have offset (0,0,0); cube
            # corners have (±α·r, ±α·r, ±α·r); pillar sub-spheres have (0, y, 0).
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            pa_off = x[ba] + c_off_a[cj]
            pb_off = x[bb] + c_off_b[cj]
            if ba == i:
                p_i_eff = pa_off
                p_other = pb_off
            else:
                p_i_eff = pb_off
                p_other = pa_off
            J = distance_jac(p_i_eff, p_other)
            C = eval_distance(p_i_eff, p_other, c_rest[cj])
        elif t == SPHERE_BOX_CONTACT:
            # Analytical sphere ↔ AABB contact. body_a is the sphere (with
            # optional sub-sphere offset c_off_a, e.g. for a pillar's stack),
            # body_b is the box. world_anchor holds the box half-extents.
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            p_sphere = x[ba] + c_off_a[cj]
            p_box = x[bb]
            hext = c_world_anchor[cj]
            r_sphere = c_rest[cj]
            C_val, n_hat = sphere_box_C_and_normal(p_sphere, p_box, hext, r_sphere)
            C = C_val
            if ba == i:
                J = n_hat
            else:
                J = -n_hat
        elif t == BOX_BOX_CONTACT:
            # AVBD Eq. 15 normal row, directly: C = n̂ · (r_a − r_b). The
            # contact normal n̂ and per-side contact points (in body-local
            # frame) were resolved by the CPU-side SAT + face clip and live
            # in c_world_anchor / c_off_a / c_off_b. n̂ is held constant
            # during the AVBD step (consistent with VBD §3.5).
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            r_a = x[ba] + c_off_a[cj]
            r_b = x[bb] + c_off_b[cj]
            n_hat = c_world_anchor[cj]
            C = wp.dot(n_hat, r_a - r_b)
            if ba == i:
                J = n_hat
            else:
                J = -n_hat
        elif t == CONTACT_TANGENT:
            # AVBD Eq. (15) tangent row. C = t̂ · (x_a − x_b) (or t̂ · x_a for
            # floor friction, body_b = -1). t̂ is held constant during the step.
            tangent = c_world_anchor[cj]
            J = tangent
            # For sphere-sphere pairs: when i is body_a, J = +t̂; body_b → -t̂.
            other_i = c_body_b[cj]
            if c_body_b[cj] >= 0:
                if c_body_a[cj] == i:
                    # i is body_a, other = body_b → already correct (+t̂)
                    C = wp.dot(tangent, x[c_body_a[cj]] - x[c_body_b[cj]])
                else:
                    # i is body_b, flip Jacobian
                    J = -tangent
                    C = wp.dot(tangent, x[c_body_a[cj]] - x[c_body_b[cj]])
            else:
                # Floor friction: C = t̂ · x_a (drives the contact point to
                # stop sliding in the tangent direction).
                C = wp.dot(tangent, x[c_body_a[cj]])

        # Stabilized C (Eq. 18) for hard constraints: subtract α·C₀
        s = c_stiffness[cj]
        if s >= wp.inf:
            C = C - c_alpha_C0[cj]

        # f = clamp(k·C + λ, fmin, fmax)  (Sec 3.2)
        # For hard constraints λ is used; for soft constraints we use λ=0 in the
        # force, since the spring is the constraint itself (paper Sec 3.4).
        lam_eff = c_lambda[cj]
        if not (s >= wp.inf):
            lam_eff = 0.0
        # For tangent friction rows the clamp uses the dynamic friction-cone
        # bound ±μ|λ_n| (computed each iteration from the SIBLING normal row's
        # current λ), not the static c_fmin/c_fmax. AVBD Sec.3.3.
        if t == CONTACT_TANGENT:
            sib = c_sibling[cj]
            mu = c_friction[cj]
            bound = mu * wp.abs(c_lambda[sib])
            f = wp.clamp(c_penalty[cj] * C + lam_eff, -bound, bound)
        else:
            f = wp.clamp(c_penalty[cj] * C + lam_eff, c_fmin[cj], c_fmax[cj])

        # Geometric stiffness G (Sec 3.5): for our scalar constraints H = 0
        # (Jacobian is constant w.r.t. linear position). PIN and DISTANCE have
        # zero second derivative w.r.t. a single endpoint in the linearised
        # form we use, so G = 0. We omit it.
        # (Will need to revisit when adding rigid-body 6-DOF constraints.)

        # Accumulate (Eq. 13) and (Eq. 17)
        rhs = rhs + J * f
        # Outer product J Jᵀ * k
        k_p = c_penalty[cj]
        lhs = lhs + wp.mat33(
            J[0]*J[0]*k_p, J[0]*J[1]*k_p, J[0]*J[2]*k_p,
            J[1]*J[0]*k_p, J[1]*J[1]*k_p, J[1]*J[2]*k_p,
            J[2]*J[0]*k_p, J[2]*J[1]*k_p, J[2]*J[2]*k_p,
        )

    # Solve 3×3 SPD via inverse (cheap for 3×3, numerically OK since lhs is SPD)
    inv = wp.inverse(lhs)
    dx = inv * rhs
    x[i] = x[i] - dx


# -----------------------------------------------------------------------------
# Dual update (AVBD Eq. 11 + penalty growth Eq. 16; reference solver.cpp 203–227)
# -----------------------------------------------------------------------------
@wp.kernel
def dual_update(
    x: wp.array(dtype=wp.vec3),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_stiffness: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_fracture: wp.array(dtype=float),
    c_sibling: wp.array(dtype=int),
    c_friction: wp.array(dtype=float),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    beta: float,
):
    j = wp.tid()
    if c_active[j] == 0:
        return

    t = c_type[j]
    C = 0.0
    if t == PIN_X or t == PIN_Y or t == PIN_Z:
        axis = t
        anchor = c_world_anchor[j]
        C = eval_pin_axis(x[c_body_a[j]], anchor, axis)
    elif t == DISTANCE:
        C = eval_distance(x[c_body_a[j]], x[c_body_b[j]], c_rest[j])
    elif t == FLOOR_CONTACT:
        C = x[c_body_a[j]][1] - c_world_anchor[j][1]
    elif t == SPHERE_CONTACT:
        pa_off = x[c_body_a[j]] + c_off_a[j]
        pb_off = x[c_body_b[j]] + c_off_b[j]
        C = eval_distance(pa_off, pb_off, c_rest[j])
    elif t == SPHERE_BOX_CONTACT:
        p_sphere = x[c_body_a[j]] + c_off_a[j]
        p_box = x[c_body_b[j]]
        hext = c_world_anchor[j]
        C_val, _n = sphere_box_C_and_normal(p_sphere, p_box, hext, c_rest[j])
        C = C_val
    elif t == BOX_BOX_CONTACT:
        r_a = x[c_body_a[j]] + c_off_a[j]
        r_b = x[c_body_b[j]] + c_off_b[j]
        C = wp.dot(c_world_anchor[j], r_a - r_b)
    elif t == CONTACT_TANGENT:
        tangent = c_world_anchor[j]
        if c_body_b[j] >= 0:
            C = wp.dot(tangent, x[c_body_a[j]] - x[c_body_b[j]])
        else:
            C = wp.dot(tangent, x[c_body_a[j]])

    s = c_stiffness[j]
    if s >= wp.inf:
        C = C - c_alpha_C0[j]

    lam_eff = c_lambda[j]
    if not (s >= wp.inf):
        lam_eff = 0.0

    # Eq. 11: λ ← clamp(k·C + λ_prev, fmin, fmax)
    # For tangent friction rows, the clamp bound is ±μ|λ_n| from the sibling
    # normal row (AVBD Sec.3.3). For all other rows, use the static c_fmin/c_fmax.
    lam_min = c_fmin[j]
    lam_max = c_fmax[j]
    if t == CONTACT_TANGENT:
        bound = c_friction[j] * wp.abs(c_lambda[c_sibling[j]])
        lam_min = -bound
        lam_max = bound
    new_lam = wp.clamp(c_penalty[j] * C + lam_eff, lam_min, lam_max)
    c_lambda[j] = new_lam

    # Fracture: disable constraint if |λ| ≥ threshold (this is the AVBD-native
    # impulse-criterion fracture; see I3D 2018 for the energy-to-KE conversion).
    if wp.abs(new_lam) >= c_fracture[j]:
        c_active[j] = 0
        c_lambda[j] = 0.0
        c_penalty[j] = 0.0
        return

    # Penalty growth (Eq. 16): only within force bounds, clamped to material
    if new_lam > lam_min and new_lam < lam_max:
        upper = wp.min(PENALTY_MAX, s)
        c_penalty[j] = wp.min(c_penalty[j] + beta * wp.abs(C), upper)


# -----------------------------------------------------------------------------
# Velocity finalise (BDF1)
# -----------------------------------------------------------------------------
@wp.kernel
def finalize_velocity(
    x: wp.array(dtype=wp.vec3),
    initial: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    dt: float,
    v: wp.array(dtype=wp.vec3),
    prev_v: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    prev_v[i] = v[i]
    if mass[i] > 0.0:
        v[i] = (x[i] - initial[i]) / dt


@wp.kernel
def cap_velocity(v: wp.array(dtype=wp.vec3), max_speed: float):
    """Defensive runaway guard. Real-world objects don't move 30+ m/s in a
    tabletop demo; if the post-stabilization pass snapped a body back from a
    deep penetration, finalize_velocity = (x − initial)/dt can synthesize a
    very large v that the NEXT frame's predict_inertial then projects out by
    v·dt — overshooting through other bodies and feeding the next collision
    with even more energy. Clamping the magnitude here breaks the loop while
    keeping the direction (so a body skidding fast still skids, just slower)."""
    i = wp.tid()
    speed = wp.length(v[i])
    if speed > max_speed:
        v[i] = v[i] * (max_speed / speed)


# -----------------------------------------------------------------------------
# C₀ caching for hard-constraint stabilisation (Eq. 18: C̃ = C - α·C₀)
# -----------------------------------------------------------------------------
@wp.kernel
def cache_alpha_C0(
    x: wp.array(dtype=wp.vec3),
    initial: wp.array(dtype=wp.vec3),  # pre-warm-start positions (x at frame start)
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    alpha: float,
    c_alpha_C0: wp.array(dtype=float),
):
    j = wp.tid()
    if c_active[j] == 0:
        c_alpha_C0[j] = 0.0
        return
    t = c_type[j]
    C0 = 0.0
    if t == PIN_X or t == PIN_Y or t == PIN_Z:
        C0 = eval_pin_axis(x[c_body_a[j]], c_world_anchor[j], t)
    elif t == DISTANCE:
        C0 = eval_distance(x[c_body_a[j]], x[c_body_b[j]], c_rest[j])
    elif t == FLOOR_CONTACT:
        C0 = x[c_body_a[j]][1] - c_world_anchor[j][1]
    elif t == SPHERE_CONTACT:
        pa_off = x[c_body_a[j]] + c_off_a[j]
        pb_off = x[c_body_b[j]] + c_off_b[j]
        C0 = eval_distance(pa_off, pb_off, c_rest[j])
    elif t == SPHERE_BOX_CONTACT:
        p_sphere = x[c_body_a[j]] + c_off_a[j]
        p_box = x[c_body_b[j]]
        hext = c_world_anchor[j]
        C0_val, _n = sphere_box_C_and_normal(p_sphere, p_box, hext, c_rest[j])
        C0 = C0_val
    elif t == BOX_BOX_CONTACT:
        r_a = x[c_body_a[j]] + c_off_a[j]
        r_b = x[c_body_b[j]] + c_off_b[j]
        C0 = wp.dot(c_world_anchor[j], r_a - r_b)
    elif t == CONTACT_TANGENT:
        # C0 = t̂ · (x_a − x_b) at FRAME START (NOT post-warm-start) so the
        # iteration's C − C0 measures the full v·dt slide that friction must
        # oppose. Read `initial[a]` instead of `x[a]`. For tangent friction we
        # also use α = 1.0 (not the global α=0.99): the rest-anchor moves WITH
        # the body each frame, otherwise α<1 would behave like a soft spring
        # pulling sliding bodies back toward their original contact point.
        tangent = c_world_anchor[j]
        if c_body_b[j] >= 0:
            C0 = wp.dot(tangent, initial[c_body_a[j]] - initial[c_body_b[j]])
        else:
            C0 = wp.dot(tangent, initial[c_body_a[j]])
        c_alpha_C0[j] = C0
        return
    c_alpha_C0[j] = alpha * C0
