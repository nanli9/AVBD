"""Warp kernels for 6-DOF AVBD rigid-body solver.

Generalizes the 3-DOF particle kernels in `kernels.py` to full 6-DOF rigid
bodies: each body now has position (vec3), orientation (quat), linear velocity
(vec3), angular velocity (vec3), mass (scalar), and body-local inverse inertia
tensor (mat33).

Per-body local solve becomes a 6×6 SPD system. We solve it via a 2-block
Schur complement using Warp's `wp.inverse` on 3×3 matrices (Warp has no
mat66 type as of 1.13). The system has the form
    [ A   B^T ] [ Δx ]   [ r_x ]
    [ B   D   ] [ Δθ ] = [ r_θ ]
where A is 3×3 linear, D is 3×3 angular, and B is 3×3 angular-linear coupling.

Math references (verified against the 2D demo + AVBD paper):
  - AVBD Eq. 2 (inertial target) — extended to orientation: q_inertial =
    exp_q(ω·dt/2) ⊗ q_old (world-frame angular velocity, left composition).
  - AVBD Eq. 4 / 13 / 17 (primal solve) — 6-DOF analogue: M_world = block_diag(
    m·I3, R·I_local·R^T); per-row J ∈ R^6 = (J_lin, J_ang) where J_lin =
    ±∂C/∂x and J_ang = ±(R·off) × ∂C/∂x (cross product gives the torque arm,
    same pattern as 2D demo manifold.cpp:77).
  - AVBD Eq. 19 (warm-start) — λ and penalty preserved as before; orientation
    has no separate dual.
  - VBD §3.5 / AVBD §4 (truncated Taylor for contacts) — n̂ and contact frame
    are held constant during a single step. We DO recompute J each iteration
    using the current quaternion (simpler than truncated-Taylor in 3D and
    follows the 3-DOF solver's pattern).
  - BDF1 velocity update: v = (x − x_init)/dt; ω = axis_angle(q_init^{-1} ⊗ q)/dt.
"""

import warp as wp

# This solver is forward-only (VBD/AVBD is not autodiff-based — nothing in the
# package builds a wp.Tape or reads gradients). Disable adjoint codegen for the
# whole module: it halves JIT/compile work, and — critically — lets us use
# `@wp.func_native` (the __shfl_down_sync reduction, which has no adjoint
# snippet) without breaking the module's backward compilation. With backward
# enabled, Warp tries to differentiate the func_native call and fails to find
# the adjoint symbol, which cascades to every kernel's _backward in the module.
wp.set_module_options({"enable_backward": False})

# Constraint type codes — must match solver_6dof.py.
# Start above the 3-DOF codes to make co-existence easier in mixed scenes.
FLOOR_CONTACT_6DOF = wp.constant(0)     # C = y(x + R·off_a) − floor_y, fmax=0 (push-up)
CONTACT_TANGENT_6DOF = wp.constant(1)   # friction tangent row paired with sibling normal
PIN_6DOF = wp.constant(2)               # body-local point pinned to world point (3 rows, one per axis)
BOX_BOX_CONTACT_6DOF = wp.constant(3)   # C = n̂·(r_a − r_b) with body-local anchors (placeholder)

# Same penalty floors as the 3-DOF kernels (see kernels.py PENALTY_MIN docstring).
PENALTY_MIN = wp.constant(1.0e6)
PENALTY_MIN_TANGENT = wp.constant(1.0)
PENALTY_MAX = wp.constant(1.0e9)


# -----------------------------------------------------------------------------
# Quaternion helpers
# -----------------------------------------------------------------------------
# Warp's wp.quat is (x, y, z, w) — XYZW order. The identity is wp.quat(0,0,0,1).
# Quaternion multiplication is wp.mul(q1, q2) or q1 * q2. Vector rotation is
# wp.quat_rotate(q, v). To-matrix is wp.quat_to_matrix(q).


@wp.func
def quat_from_rotvec(rv: wp.vec3) -> wp.quat:
    """Build a unit quaternion from a rotation vector (axis * angle).
    Uses the small-angle stable form sin(θ/2)/θ · rv (Taylor for tiny θ)."""
    theta = wp.length(rv)
    if theta < 1.0e-9:
        # Small-angle limit: sin(θ/2)/θ ≈ 1/2 (1 − θ²/24); we just use 1/2.
        return wp.quat(0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0)
    half = 0.5 * theta
    s = wp.sin(half) / theta
    return wp.quat(s * rv[0], s * rv[1], s * rv[2], wp.cos(half))


@wp.func
def quat_to_rotvec(q: wp.quat) -> wp.vec3:
    """Inverse of quat_from_rotvec — extract a rotation vector (axis * angle)
    from a unit quaternion. Picks the equivalent rotation in [−π, π] so
    BDF1-style v = Δθ/dt stays well-behaved across the q ↔ −q ambiguity."""
    # Sign-canonicalize to the hemisphere with w ≥ 0 (chooses the rotation
    # whose absolute angle is ≤ π).
    qw = q[3]
    qx = q[0]
    qy = q[1]
    qz = q[2]
    if qw < 0.0:
        qw = -qw
        qx = -qx
        qy = -qy
        qz = -qz
    qv = wp.vec3(qx, qy, qz)
    qv_len = wp.length(qv)
    if qv_len < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    angle = 2.0 * wp.atan2(qv_len, qw)
    return qv * (angle / qv_len)


@wp.func
def skew(v: wp.vec3) -> wp.mat33:
    """Skew-symmetric matrix S(v) such that S(v) · u = v × u."""
    return wp.mat33(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )


@wp.func
def outer3(a: wp.vec3, b: wp.vec3) -> wp.mat33:
    return wp.mat33(
        a[0]*b[0], a[0]*b[1], a[0]*b[2],
        a[1]*b[0], a[1]*b[1], a[1]*b[2],
        a[2]*b[0], a[2]*b[1], a[2]*b[2],
    )


# -----------------------------------------------------------------------------
# Predict inertial target + warm-start  (AVBD Eq. 2 extended to 6-DOF)
# -----------------------------------------------------------------------------
@wp.kernel
def predict_inertial_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    prev_v: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inv_inertia_local: wp.array(dtype=wp.mat33),
    inertia_local: wp.array(dtype=wp.mat33),
    dt: float,
    gravity: wp.vec3,
    # outputs
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    x_inertial: wp.array(dtype=wp.vec3),
    q_inertial: wp.array(dtype=wp.quat),
    inv_inertia_world: wp.array(dtype=wp.mat33),
    inertia_world: wp.array(dtype=wp.mat33),
    x_warm: wp.array(dtype=wp.vec3),
    q_warm: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    m = mass[i]
    # Save initial state for BDF1 velocity finalize.
    x_initial[i] = x[i]
    q_initial[i] = q[i]
    # Cache R · I_local^{-1} · R^T AND R · I_local · R^T — both used by the
    # primal solve LHS. The previous version stored only the inverse and the
    # primal called wp.inverse() on it every iteration. Storing both removes
    # one 3×3 inverse per body per primal iteration (~10% of primal compute
    # at small body counts; was wasted work since R · I^-1 · R^T inverts
    # exactly to R · I · R^T).
    R = wp.quat_to_matrix(q[i])
    Rt = wp.transpose(R)
    inv_inertia_world[i] = R * inv_inertia_local[i] * Rt
    inertia_world[i] = R * inertia_local[i] * Rt

    if m <= 0.0:
        # Static / kinematic body — no inertial prediction.
        x_inertial[i] = x[i]
        q_inertial[i] = q[i]
        x_warm[i] = x[i]
        q_warm[i] = q[i]
        return

    # Linear inertial target — same as 3-DOF (Eq. 2).
    g_dt2 = gravity * (dt * dt)
    x_inertial[i] = x[i] + v[i] * dt + g_dt2
    # Angular inertial target — body would freely spin at current ω. World-
    # frame composition: q_new = exp_q(ω·dt/2) ⊗ q_old (left-multiply).
    dq = quat_from_rotvec(omega[i] * dt)
    q_inertial[i] = dq * q[i]

    # Adaptive warm-start gravity weighting (VBD §4.2) — only linear since
    # gravity is a pure linear force.
    accel = (v[i] - prev_v[i]) / dt
    g_norm = wp.length(gravity)
    w = float(0.0)
    if g_norm > 0.0:
        g_hat = gravity / g_norm
        accel_ext = wp.dot(accel, g_hat)
        w = wp.clamp(accel_ext / g_norm, 0.0, 1.0)
    x_warm[i] = x[i] + v[i] * dt + g_dt2 * w
    # Angular warm-start: same full ω·dt rotation; no gravity component.
    q_warm[i] = dq * q[i]


# -----------------------------------------------------------------------------
# Warm-start dual variables and penalty (AVBD Eq. 19)  — same as 3-DOF
# -----------------------------------------------------------------------------
@wp.kernel
def warmstart_duals_6dof(
    lam: wp.array(dtype=float),
    pen: wp.array(dtype=float),
    stiffness: wp.array(dtype=float),
    c_type: wp.array(dtype=int),
    alpha: float,
    gamma: float,
    post_stabilize: int,
):
    j = wp.tid()
    k_floor = PENALTY_MIN
    if c_type[j] == CONTACT_TANGENT_6DOF:
        k_floor = PENALTY_MIN_TANGENT
    p = wp.clamp(pen[j] * gamma, k_floor, PENALTY_MAX)
    if post_stabilize == 0:
        lam[j] = lam[j] * alpha * gamma
    s = stiffness[j]
    if not wp.isnan(s) and s < wp.inf:
        p = wp.min(p, s)
    pen[j] = p


# -----------------------------------------------------------------------------
# Constraint evaluation helpers
# -----------------------------------------------------------------------------
@wp.func
def eval_floor_C(x: wp.vec3, q: wp.quat, off: wp.vec3, floor_y: float) -> float:
    """C = (x + R·off)[1] − floor_y. Positive above floor."""
    r_world = x + wp.quat_rotate(q, off)
    return r_world[1] - floor_y


@wp.func
def floor_J(q: wp.quat, off: wp.vec3):
    """Floor contact Jacobian for body. Linear = (0,1,0); angular = (R·off) × (0,1,0)."""
    r = wp.quat_rotate(q, off)
    n = wp.vec3(0.0, 1.0, 0.0)
    j_lin = n
    j_ang = wp.cross(r, n)
    return j_lin, j_ang


@wp.func
def tangent_J(q: wp.quat, off: wp.vec3, tangent: wp.vec3):
    r = wp.quat_rotate(q, off)
    return tangent, wp.cross(r, tangent)


@wp.func
def pin_axis_J(q: wp.quat, off: wp.vec3, axis: int):
    """PIN axis Jacobian — pin body's local point `off` to world along `axis`."""
    r = wp.quat_rotate(q, off)
    if axis == 0:
        j_lin = wp.vec3(1.0, 0.0, 0.0)
    elif axis == 1:
        j_lin = wp.vec3(0.0, 1.0, 0.0)
    else:
        j_lin = wp.vec3(0.0, 0.0, 1.0)
    j_ang = wp.cross(r, j_lin)
    return j_lin, j_ang


@wp.func
def eval_pin_axis(x: wp.vec3, q: wp.quat, off: wp.vec3, anchor: wp.vec3, axis: int) -> float:
    r_world = x + wp.quat_rotate(q, off)
    if axis == 0:
        return r_world[0] - anchor[0]
    if axis == 1:
        return r_world[1] - anchor[1]
    return r_world[2] - anchor[2]


@wp.func
def eval_box_box_C(
    x_a: wp.vec3, q_a: wp.quat, off_a: wp.vec3,
    x_b: wp.vec3, q_b: wp.quat, off_b: wp.vec3,
    n: wp.vec3,
) -> float:
    """BOX_BOX_CONTACT_6DOF constraint value: C = n̂ · (r_a − r_b) where
    r = x + R·off is the contact point in world space. n̂ is held constant
    for the step (cached by the CPU-side SAT at frame start)."""
    r_a = x_a + wp.quat_rotate(q_a, off_a)
    r_b = x_b + wp.quat_rotate(q_b, off_b)
    return wp.dot(n, r_a - r_b)


@wp.func
def geom_stiffness_diag(n: wp.vec3, r: wp.vec3) -> wp.vec3:
    """Diagonal column-norm approximation of ∂²C/∂θ² for any C = n̂ · (x + R·off).
    Returns vec3 of column norms (g₀, g₁, g₂) used by AVBD Eq 17:
        G̃ = diag(||G_col_c||),   G_col_c = column c of (∂²C/∂θ²)

    Derivation: for C(θ) = n̂ · R(θ)·off, with world-frame δθ:
        δ²C = n̂ · (δθ × (δθ × r))
            = (n̂·δθ)(δθ·r) − (n̂·r)|δθ|²
    so the Hessian is
        H[i,c] = ½(n[i]·r[c] + r[i]·n[c]) − (n̂·r)·δ_{ic}.

    Multiplied by |λ⁺| outside, this becomes G_ij from Eq 17; the column-
    norm diagonal is symmetric positive-definite by construction (Sec 3.5,
    paragraph 2) so adding it to D never breaks the SPD guarantee on the
    angular block — that's why the paper says "always use" the approximate
    Hessian for rigid bodies. The earlier per-pin-only L1 estimate
    `(|r_j| + |r_k|)/2` over-estimates by up to √2 and was wrong for two
    of the three columns.
    """
    nr = wp.dot(n, r)
    # Column 0
    h00 = n[0] * r[0] - nr
    h10 = 0.5 * (n[1] * r[0] + r[1] * n[0])
    h20 = 0.5 * (n[2] * r[0] + r[2] * n[0])
    g0 = wp.sqrt(h00 * h00 + h10 * h10 + h20 * h20)
    # Column 1
    h01 = 0.5 * (n[0] * r[1] + r[0] * n[1])
    h11 = n[1] * r[1] - nr
    h21 = 0.5 * (n[2] * r[1] + r[2] * n[1])
    g1 = wp.sqrt(h01 * h01 + h11 * h11 + h21 * h21)
    # Column 2
    h02 = 0.5 * (n[0] * r[2] + r[0] * n[2])
    h12 = 0.5 * (n[1] * r[2] + r[1] * n[2])
    h22 = n[2] * r[2] - nr
    g2 = wp.sqrt(h02 * h02 + h12 * h12 + h22 * h22)
    return wp.vec3(g0, g1, g2)


# -----------------------------------------------------------------------------
# Primal update — 6×6 local SPD solve per body, Schur-complement on 3×3 blocks
# -----------------------------------------------------------------------------
@wp.kernel
def primal_update_6dof(
    # state
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    inv_inertia_world: wp.array(dtype=wp.mat33),
    inertia_world: wp.array(dtype=wp.mat33),
    x_inertial: wp.array(dtype=wp.vec3),
    q_inertial: wp.array(dtype=wp.quat),
    # constraints
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_stiffness: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_sibling: wp.array(dtype=int),
    c_friction: wp.array(dtype=float),
    c_friction_static: wp.array(dtype=float),
    c_was_static: wp.array(dtype=int),
    # adjacency
    body_con_starts: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
    # Device-side color partition (round 3 §A). `color_bodies` is the
    # body-id permutation bucket-sorted by color; `color_starts[c..c+1]`
    # is the slice for color `c`. Launched at fixed dim=n_b for every
    # color; threads past this color's slice early-return.
    color_starts: wp.array(dtype=int),
    color_bodies: wp.array(dtype=int),
    color_id: int,
    dt: float,
):
    tid = wp.tid()
    base = color_starts[color_id]
    end = color_starts[color_id + 1]
    if tid >= end - base:
        return
    i = color_bodies[base + tid]
    m = mass[i]
    if m <= 0.0:
        return

    inv_dt2 = 1.0 / (dt * dt)

    # M_world block view (we need M_world, not its inverse, on the LHS).
    # Both inv_inertia_world[i] = R · I^{-1} · R^T and inertia_world[i] =
    # R · I · R^T are precomputed once per substep in predict_inertial_6dof,
    # eliminating a per-iteration wp.inverse(mat33) call (3 mul-acc savings).
    I_world = inertia_world[i]
    # Linear A_lin = m/dt² · I3; angular D_ang = I_world/dt²; coupling B = 0
    # initially (mass matrix has no linear-angular coupling at the CoM).
    A = wp.mat33(m*inv_dt2, 0.0, 0.0, 0.0, m*inv_dt2, 0.0, 0.0, 0.0, m*inv_dt2)
    D = I_world * inv_dt2
    B = wp.mat33()  # zeros

    # RHS — linear part: M_lin/dt² · (x − x_inertial)
    r_lin = (x[i] - x_inertial[i]) * (m * inv_dt2)
    # RHS — angular part: M_ang/dt² · Δθ_inertial, with Δθ in WORLD frame.
    # World-frame convention is essential for consistency with the primal
    # update (q ← exp_q(-Δθ) ⊗ q is a left-multiply = world-frame rotation)
    # and with the J_ang = r × e_axis Jacobian we use. Extract via:
    #   q_current = exp(Δθ_world) ⊗ q_inertial  ⇒  Δq_world = q ⊗ q_inertial⁻¹
    # The earlier q_inertial⁻¹ ⊗ q form gives body-frame Δθ, which mixes
    # frames with I_world and causes angular-momentum drift on a corner-
    # pinned body (it spins itself up as the convention mismatch leaks
    # energy into rotation each step).
    dq_iner = wp.mul(q[i], wp.quat_inverse(q_inertial[i]))
    dtheta_iner = quat_to_rotvec(dq_iner)
    r_ang = I_world * (dtheta_iner * inv_dt2)

    # Cache q[i] once per body — vec3-build of (q[i], q[i], q[i], q[i]) was
    # otherwise emitted on every quat_rotate. Same for x[i].
    qi = q[i]
    xi = x[i]
    start = body_con_starts[i]
    end = body_con_starts[i + 1]
    for k in range(start, end):
        cj = body_con_indices[k]
        if c_active[cj] == 0:
            continue
        # Hoist all per-constraint reads to single registers — Warp's CPU
        # backend re-loads the array slot on every reference otherwise.
        t = c_type[cj]
        s = c_stiffness[cj]
        k_p = c_penalty[cj]
        off_a = c_off_a[cj]
        anchor = c_world_anchor[cj]   # n̂ for BOX_BOX / TANGENT; floor pos / pin pos otherwise

        j_lin = wp.vec3(0.0, 0.0, 0.0)
        j_ang = wp.vec3(0.0, 0.0, 0.0)
        C = float(0.0)
        # `r_self_w` = R(q[i]) · off (the body-LOCAL anchor on `i` rotated to
        # world). Computed at most once per constraint and reused by the J
        # calc, the C eval, and the G column-norm. Was being recomputed up to
        # 4× per BOX_BOX iteration (eval_box_box_C + J + G).
        r_self_w = wp.vec3(0.0, 0.0, 0.0)
        n_for_G = wp.vec3(0.0, 0.0, 0.0)
        have_G = False

        if t == FLOOR_CONTACT_6DOF:
            r_self_w = wp.quat_rotate(qi, off_a)
            n_hat = wp.vec3(0.0, 1.0, 0.0)
            j_lin = n_hat
            j_ang = wp.cross(r_self_w, n_hat)
            C = (xi[1] + r_self_w[1]) - anchor[1]   # floor_y = anchor.y
            n_for_G = n_hat
            have_G = True
        elif t == CONTACT_TANGENT_6DOF:
            tangent = anchor
            bb = c_body_b[cj]
            if bb < 0:
                # Floor friction: single body, anchor on `i`.
                r_self_w = wp.quat_rotate(qi, off_a)
                j_lin = tangent
                j_ang = wp.cross(r_self_w, tangent)
                C = wp.dot(tangent, xi + r_self_w)
            else:
                ba = c_body_a[cj]
                off_b = c_off_b[cj]
                # Both r_a_w and r_b_w needed for C; pick whichever is "self"
                # for J + G.
                if ba == i:
                    r_self_w = wp.quat_rotate(qi, off_a)
                    r_other_w = wp.quat_rotate(q[bb], off_b)
                    C = wp.dot(tangent,
                               (xi + r_self_w) - (x[bb] + r_other_w))
                    j_lin = tangent
                    j_ang = wp.cross(r_self_w, tangent)
                else:
                    r_other_w = wp.quat_rotate(q[ba], off_a)
                    r_self_w = wp.quat_rotate(qi, off_b)
                    C = wp.dot(tangent,
                               (x[ba] + r_other_w) - (xi + r_self_w))
                    j_lin = -tangent
                    j_ang = -wp.cross(r_self_w, tangent)
            n_for_G = tangent
            have_G = True
        elif t == PIN_6DOF:
            axis = c_body_b[cj]
            r_self_w = wp.quat_rotate(qi, off_a)
            world_anchor_pt = xi + r_self_w
            if axis == 0:
                n_hat = wp.vec3(1.0, 0.0, 0.0)
                C = world_anchor_pt[0] - anchor[0]
            elif axis == 1:
                n_hat = wp.vec3(0.0, 1.0, 0.0)
                C = world_anchor_pt[1] - anchor[1]
            else:
                n_hat = wp.vec3(0.0, 0.0, 1.0)
                C = world_anchor_pt[2] - anchor[2]
            j_lin = n_hat
            j_ang = wp.cross(r_self_w, n_hat)
            n_for_G = n_hat
            have_G = True
        elif t == BOX_BOX_CONTACT_6DOF:
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            n_hat = anchor   # stored constant during step
            off_b = c_off_b[cj]
            if ba == i:
                r_self_w = wp.quat_rotate(qi, off_a)
                r_other_w = wp.quat_rotate(q[bb], off_b)
                C = wp.dot(n_hat,
                           (xi + r_self_w) - (x[bb] + r_other_w))
                j_lin = n_hat
                j_ang = wp.cross(r_self_w, n_hat)
            else:
                r_other_w = wp.quat_rotate(q[ba], off_a)
                r_self_w = wp.quat_rotate(qi, off_b)
                C = wp.dot(n_hat,
                           (x[ba] + r_other_w) - (xi + r_self_w))
                j_lin = -n_hat
                j_ang = -wp.cross(r_self_w, n_hat)
            n_for_G = n_hat
            have_G = True

        # Eq. 18 stabilized C for hard constraints.
        hard = s >= wp.inf
        if hard:
            C = C - c_alpha_C0[cj]

        lam_eff = c_lambda[cj]
        if not hard:
            lam_eff = 0.0

        # Force magnitude — clamp differs for tangent (friction cone) vs others.
        # lam_plus is the un-clamped force, used both for the actual clamp and
        # for Eq.14's Hessian rescaling test below.
        lam_plus = k_p * C + lam_eff
        if t == CONTACT_TANGENT_6DOF:
            sib = c_sibling[cj]
            mu = c_friction[cj]
            if c_was_static[sib] != 0:
                mu = c_friction_static[cj]
            bound = mu * wp.abs(c_lambda[sib])
            f_lo = -bound
            f_hi = bound
        else:
            f_lo = c_fmin[cj]
            f_hi = c_fmax[cj]
        f = wp.clamp(lam_plus, f_lo, f_hi)

        # AVBD Eq.14 Hessian rescaling (Sec.3.2). When the unclamped force
        # lam_plus = k·C + λ falls outside [f_lo, f_hi], the actual delta-x
        # produced by the linearised k will overshoot what the clamped force
        # warrants. Replace k by k̃ = |bound − lam_plus| / |C| for the LHS
        # only — this is the stiffness that makes a one-step linear model
        # land exactly on the bound. RHS still uses the clamped f, so the
        # force magnitude is unchanged. Skips when |C| ≈ 0 (no rescale
        # information) or when in-bounds.
        k_for_lhs = k_p
        abs_C = wp.abs(C)
        if abs_C > 1.0e-12:
            if lam_plus < f_lo:
                k_for_lhs = wp.abs(f_lo - lam_plus) / abs_C
            elif lam_plus > f_hi:
                k_for_lhs = wp.abs(f_hi - lam_plus) / abs_C

        # LHS accumulation (J·k·J^T outer products split into A/B/D 3×3).
        A = A + outer3(j_lin, j_lin) * k_for_lhs
        B = B + outer3(j_ang, j_lin) * k_for_lhs
        D = D + outer3(j_ang, j_ang) * k_for_lhs

        # G column-norm diagonal (AVBD Eq 17 + Sec 3.5). r_self_w is the body-
        # local anchor on `i` after rotation — already computed above, no
        # extra quat_rotate here.
        f_mag = wp.abs(f)
        if f_mag > 0.0 and have_G:
            g_diag = geom_stiffness_diag(n_for_G, r_self_w) * f_mag
            D = D + wp.mat33(g_diag[0], 0.0, 0.0,
                             0.0, g_diag[1], 0.0,
                             0.0, 0.0, g_diag[2])
        r_lin = r_lin + j_lin * f
        r_ang = r_ang + j_ang * f

    # Schur-complement solve:
    #   [ A  B^T ] [Δx]   [r_lin]
    #   [ B  D   ] [Δθ] = [r_ang]
    # ⇒ S = D − B · A⁻¹ · B^T;  Δθ = S⁻¹ · (r_ang − B · A⁻¹ · r_lin)
    #   Δx = A⁻¹ · (r_lin − B^T · Δθ)
    A_inv = wp.inverse(A)
    BAinv = B * A_inv
    rhs_theta = r_ang - BAinv * r_lin
    S = D - BAinv * wp.transpose(B)
    S_inv = wp.inverse(S)
    d_theta = S_inv * rhs_theta
    d_x = A_inv * (r_lin - wp.transpose(B) * d_theta)

    # Apply update — translation: x ← x − Δx; rotation: q ← exp_q(−Δθ/2) ⊗ q.
    x[i] = x[i] - d_x
    dq = quat_from_rotvec(-d_theta)
    q[i] = wp.normalize(dq * q[i])


# -----------------------------------------------------------------------------
# Warp-per-body primal update (perf: same AVBD math, cooperative reduction)
# -----------------------------------------------------------------------------
# `primal_update_6dof` runs ONE thread per body. On large scenes colored
# Gauss-Seidel launches only ~n_bodies/n_colors threads per color (e.g. 221 of
# 414 → 0.5% of the GPU), and each thread serially loops ~32 incident
# constraints — latency-bound at <1% occupancy (see avbd-stress-profile).
#
# This pair splits the SAME update across a *group* of G lanes per body:
#   1. primal_accumulate_6dof: dim=(n_b, G). Lane `l` strides the body's
#      constraint list (k = start+l, start+l+G, …), evaluates the IDENTICAL
#      per-constraint contribution as the serial kernel into local registers,
#      then ONE atomic_add per block folds the lane partials into per-body
#      scratch (A/B/D Hessian blocks + r_lin/r_ang gradient). The inertial
#      terms are NOT added here — they're per-body, added once in the solve.
#   2. primal_solve_6dof: dim=n_b. Per body, add the inertial init, run the
#      identical Schur-complement solve, apply x/q, and zero the scratch for
#      the next iteration.
# Math is byte-for-byte the serial kernel's except the constraint sum is
# reduced in atomic (i.e. nondeterministic) order — a parallelization detail,
# not a solver-math change; the reorder is far below the existing GPU-atomic
# noise floor. The two kernels are plain launches (no tiles), so they
# graph-capture exactly like the serial path.
#
# The reduction `join` has two flavours, selected by Solver6DOF._primal_shuffle:
#   - atomic   (primal_accumulate_6dof):  wp.atomic_add into scratch. Portable,
#     but G lanes contend on the same per-body addresses (hurts at high G).
#   - shuffle  (primal_accumulate_shuffle_6dof): a wp.func_native __shfl_down_
#     sync reduction folds the lane partials register-to-register (no atomics,
#     no contention, deterministic within the warp). Lane 0 then STORES the
#     complete sum (no pre-zero needed). Requires a 1-D launch padded to a
#     multiple of 32 with NO early-return before the shuffle (see below).
#
# Warp-shuffle reduction of a vec3 across `width` consecutive lanes (width|32).
# Lane 0 of each sub-group ends with the sum. The full 0xffffffff mask is valid
# ONLY because the launch pads the grid to a multiple of 32 and every lane
# reaches this call (no divergent early-return before it).
# NOTE the `#ifdef __CUDA_ARCH__` guard: __shfl_down_sync is a CUDA-only
# intrinsic, but Warp compiles every kernel in this module for the CPU backend
# too (and tests run on device="cpu"). The CPU branch is a no-op — the
# warp-per-body shuffle path is gated to CUDA in the solver, so it's never
# actually launched on CPU; it only has to *compile*.
_WARP_SUM_VEC3_SNIPPET = """
#ifdef __CUDA_ARCH__
    wp::vec3 r = value;
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        r[0] += __shfl_down_sync(0xffffffffu, r[0], offset, width);
        r[1] += __shfl_down_sync(0xffffffffu, r[1], offset, width);
        r[2] += __shfl_down_sync(0xffffffffu, r[2], offset, width);
    }
    return r;
#else
    return value;  // CPU: warp-per-body is gated off (G=1); never reached
#endif
"""


@wp.func_native(_WARP_SUM_VEC3_SNIPPET)
def warp_sum_vec3(value: wp.vec3, width: int) -> wp.vec3:
    ...


@wp.kernel
def primal_accumulate_6dof(
    # state (read-only here; the solve kernel applies the update)
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    # constraints
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_stiffness: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_sibling: wp.array(dtype=int),
    c_friction: wp.array(dtype=float),
    c_friction_static: wp.array(dtype=float),
    c_was_static: wp.array(dtype=int),
    # adjacency + color partition
    body_con_starts: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
    color_starts: wp.array(dtype=int),
    color_bodies: wp.array(dtype=int),
    color_id: int,
    group_size: int,
    # per-body reduction scratch (atomically accumulated; zeroed by the solve)
    scratch_A: wp.array(dtype=wp.mat33),
    scratch_B: wp.array(dtype=wp.mat33),
    scratch_D: wp.array(dtype=wp.mat33),
    scratch_rlin: wp.array(dtype=wp.vec3),
    scratch_rang: wp.array(dtype=wp.vec3),
):
    slot, lane = wp.tid()
    base = color_starts[color_id]
    end_c = color_starts[color_id + 1]
    if slot >= end_c - base:
        return
    i = color_bodies[base + slot]
    m = mass[i]
    if m <= 0.0:
        return

    qi = q[i]
    xi = x[i]
    start = body_con_starts[i]
    end = body_con_starts[i + 1]

    # Local lane partials — constraint contributions only (no inertial term).
    A = wp.mat33()
    B = wp.mat33()
    D = wp.mat33()
    r_lin = wp.vec3(0.0, 0.0, 0.0)
    r_ang = wp.vec3(0.0, 0.0, 0.0)

    for k in range(start + lane, end, group_size):
        cj = body_con_indices[k]
        if c_active[cj] == 0:
            continue
        t = c_type[cj]
        s = c_stiffness[cj]
        k_p = c_penalty[cj]
        off_a = c_off_a[cj]
        anchor = c_world_anchor[cj]   # n̂ for BOX_BOX / TANGENT; floor / pin pos otherwise

        j_lin = wp.vec3(0.0, 0.0, 0.0)
        j_ang = wp.vec3(0.0, 0.0, 0.0)
        C = float(0.0)
        r_self_w = wp.vec3(0.0, 0.0, 0.0)
        n_for_G = wp.vec3(0.0, 0.0, 0.0)
        have_G = False

        if t == FLOOR_CONTACT_6DOF:
            r_self_w = wp.quat_rotate(qi, off_a)
            n_hat = wp.vec3(0.0, 1.0, 0.0)
            j_lin = n_hat
            j_ang = wp.cross(r_self_w, n_hat)
            C = (xi[1] + r_self_w[1]) - anchor[1]
            n_for_G = n_hat
            have_G = True
        elif t == CONTACT_TANGENT_6DOF:
            tangent = anchor
            bb = c_body_b[cj]
            if bb < 0:
                r_self_w = wp.quat_rotate(qi, off_a)
                j_lin = tangent
                j_ang = wp.cross(r_self_w, tangent)
                C = wp.dot(tangent, xi + r_self_w)
            else:
                ba = c_body_a[cj]
                off_b = c_off_b[cj]
                if ba == i:
                    r_self_w = wp.quat_rotate(qi, off_a)
                    r_other_w = wp.quat_rotate(q[bb], off_b)
                    C = wp.dot(tangent,
                               (xi + r_self_w) - (x[bb] + r_other_w))
                    j_lin = tangent
                    j_ang = wp.cross(r_self_w, tangent)
                else:
                    r_other_w = wp.quat_rotate(q[ba], off_a)
                    r_self_w = wp.quat_rotate(qi, off_b)
                    C = wp.dot(tangent,
                               (x[ba] + r_other_w) - (xi + r_self_w))
                    j_lin = -tangent
                    j_ang = -wp.cross(r_self_w, tangent)
            n_for_G = tangent
            have_G = True
        elif t == PIN_6DOF:
            axis = c_body_b[cj]
            r_self_w = wp.quat_rotate(qi, off_a)
            world_anchor_pt = xi + r_self_w
            if axis == 0:
                n_hat = wp.vec3(1.0, 0.0, 0.0)
                C = world_anchor_pt[0] - anchor[0]
            elif axis == 1:
                n_hat = wp.vec3(0.0, 1.0, 0.0)
                C = world_anchor_pt[1] - anchor[1]
            else:
                n_hat = wp.vec3(0.0, 0.0, 1.0)
                C = world_anchor_pt[2] - anchor[2]
            j_lin = n_hat
            j_ang = wp.cross(r_self_w, n_hat)
            n_for_G = n_hat
            have_G = True
        elif t == BOX_BOX_CONTACT_6DOF:
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            n_hat = anchor   # stored constant during step
            off_b = c_off_b[cj]
            if ba == i:
                r_self_w = wp.quat_rotate(qi, off_a)
                r_other_w = wp.quat_rotate(q[bb], off_b)
                C = wp.dot(n_hat,
                           (xi + r_self_w) - (x[bb] + r_other_w))
                j_lin = n_hat
                j_ang = wp.cross(r_self_w, n_hat)
            else:
                r_other_w = wp.quat_rotate(q[ba], off_a)
                r_self_w = wp.quat_rotate(qi, off_b)
                C = wp.dot(n_hat,
                           (x[ba] + r_other_w) - (xi + r_self_w))
                j_lin = -n_hat
                j_ang = -wp.cross(r_self_w, n_hat)
            n_for_G = n_hat
            have_G = True

        hard = s >= wp.inf
        if hard:
            C = C - c_alpha_C0[cj]

        lam_eff = c_lambda[cj]
        if not hard:
            lam_eff = 0.0

        lam_plus = k_p * C + lam_eff
        if t == CONTACT_TANGENT_6DOF:
            sib = c_sibling[cj]
            mu = c_friction[cj]
            if c_was_static[sib] != 0:
                mu = c_friction_static[cj]
            bound = mu * wp.abs(c_lambda[sib])
            f_lo = -bound
            f_hi = bound
        else:
            f_lo = c_fmin[cj]
            f_hi = c_fmax[cj]
        f = wp.clamp(lam_plus, f_lo, f_hi)

        k_for_lhs = k_p
        abs_C = wp.abs(C)
        if abs_C > 1.0e-12:
            if lam_plus < f_lo:
                k_for_lhs = wp.abs(f_lo - lam_plus) / abs_C
            elif lam_plus > f_hi:
                k_for_lhs = wp.abs(f_hi - lam_plus) / abs_C

        A = A + outer3(j_lin, j_lin) * k_for_lhs
        B = B + outer3(j_ang, j_lin) * k_for_lhs
        D = D + outer3(j_ang, j_ang) * k_for_lhs

        f_mag = wp.abs(f)
        if f_mag > 0.0 and have_G:
            g_diag = geom_stiffness_diag(n_for_G, r_self_w) * f_mag
            D = D + wp.mat33(g_diag[0], 0.0, 0.0,
                             0.0, g_diag[1], 0.0,
                             0.0, 0.0, g_diag[2])
        r_lin = r_lin + j_lin * f
        r_ang = r_ang + j_ang * f

    # Fold this lane's partial into the body's reduction scratch. One atomic
    # set per lane (G-way contention per body) — the per-constraint serial
    # work above is what we parallelized; this is the join.
    wp.atomic_add(scratch_A, i, A)
    wp.atomic_add(scratch_B, i, B)
    wp.atomic_add(scratch_D, i, D)
    wp.atomic_add(scratch_rlin, i, r_lin)
    wp.atomic_add(scratch_rang, i, r_ang)


@wp.kernel
def primal_accumulate_shuffle_6dof(
    # state (read-only here; the solve kernel applies the update)
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    # constraints
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_stiffness: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_sibling: wp.array(dtype=int),
    c_friction: wp.array(dtype=float),
    c_friction_static: wp.array(dtype=float),
    c_was_static: wp.array(dtype=int),
    # adjacency + color partition
    body_con_starts: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
    color_starts: wp.array(dtype=int),
    color_bodies: wp.array(dtype=int),
    color_id: int,
    group_size: int,
    scratch_A: wp.array(dtype=wp.mat33),
    scratch_B: wp.array(dtype=wp.mat33),
    scratch_D: wp.array(dtype=wp.mat33),
    scratch_rlin: wp.array(dtype=wp.vec3),
    scratch_rang: wp.array(dtype=wp.vec3),
):
    # 1-D launch padded to a multiple of 32 (see solver). `slot` is the body's
    # index in the color slice; the `group_size` lanes of a body are
    # consecutive tids => one width-G warp sub-group. CRITICAL: do NOT return
    # before warp_sum_vec3 — every lane in the warp must reach the shuffle or
    # the full-mask __shfl_down_sync is undefined. Padding/static/empty lanes
    # carry a zero partial and simply don't write.
    tid = wp.tid()
    slot = tid // group_size
    lane = tid % group_size
    base = color_starts[color_id]
    end_c = color_starts[color_id + 1]
    i = int(0)
    active = int(0)
    if slot < end_c - base:
        i = color_bodies[base + slot]
        if mass[i] > 0.0:
            active = 1

    A = wp.mat33()
    B = wp.mat33()
    D = wp.mat33()
    r_lin = wp.vec3(0.0, 0.0, 0.0)
    r_ang = wp.vec3(0.0, 0.0, 0.0)

    if active == 1:
        qi = q[i]
        xi = x[i]
        start = body_con_starts[i]
        end = body_con_starts[i + 1]
        for k in range(start + lane, end, group_size):
            cj = body_con_indices[k]
            if c_active[cj] == 0:
                continue
            t = c_type[cj]
            s = c_stiffness[cj]
            k_p = c_penalty[cj]
            off_a = c_off_a[cj]
            anchor = c_world_anchor[cj]

            j_lin = wp.vec3(0.0, 0.0, 0.0)
            j_ang = wp.vec3(0.0, 0.0, 0.0)
            C = float(0.0)
            r_self_w = wp.vec3(0.0, 0.0, 0.0)
            n_for_G = wp.vec3(0.0, 0.0, 0.0)
            have_G = False

            if t == FLOOR_CONTACT_6DOF:
                r_self_w = wp.quat_rotate(qi, off_a)
                n_hat = wp.vec3(0.0, 1.0, 0.0)
                j_lin = n_hat
                j_ang = wp.cross(r_self_w, n_hat)
                C = (xi[1] + r_self_w[1]) - anchor[1]
                n_for_G = n_hat
                have_G = True
            elif t == CONTACT_TANGENT_6DOF:
                tangent = anchor
                bb = c_body_b[cj]
                if bb < 0:
                    r_self_w = wp.quat_rotate(qi, off_a)
                    j_lin = tangent
                    j_ang = wp.cross(r_self_w, tangent)
                    C = wp.dot(tangent, xi + r_self_w)
                else:
                    ba = c_body_a[cj]
                    off_b = c_off_b[cj]
                    if ba == i:
                        r_self_w = wp.quat_rotate(qi, off_a)
                        r_other_w = wp.quat_rotate(q[bb], off_b)
                        C = wp.dot(tangent,
                                   (xi + r_self_w) - (x[bb] + r_other_w))
                        j_lin = tangent
                        j_ang = wp.cross(r_self_w, tangent)
                    else:
                        r_other_w = wp.quat_rotate(q[ba], off_a)
                        r_self_w = wp.quat_rotate(qi, off_b)
                        C = wp.dot(tangent,
                                   (x[ba] + r_other_w) - (xi + r_self_w))
                        j_lin = -tangent
                        j_ang = -wp.cross(r_self_w, tangent)
                n_for_G = tangent
                have_G = True
            elif t == PIN_6DOF:
                axis = c_body_b[cj]
                r_self_w = wp.quat_rotate(qi, off_a)
                world_anchor_pt = xi + r_self_w
                if axis == 0:
                    n_hat = wp.vec3(1.0, 0.0, 0.0)
                    C = world_anchor_pt[0] - anchor[0]
                elif axis == 1:
                    n_hat = wp.vec3(0.0, 1.0, 0.0)
                    C = world_anchor_pt[1] - anchor[1]
                else:
                    n_hat = wp.vec3(0.0, 0.0, 1.0)
                    C = world_anchor_pt[2] - anchor[2]
                j_lin = n_hat
                j_ang = wp.cross(r_self_w, n_hat)
                n_for_G = n_hat
                have_G = True
            elif t == BOX_BOX_CONTACT_6DOF:
                ba = c_body_a[cj]
                bb = c_body_b[cj]
                n_hat = anchor
                off_b = c_off_b[cj]
                if ba == i:
                    r_self_w = wp.quat_rotate(qi, off_a)
                    r_other_w = wp.quat_rotate(q[bb], off_b)
                    C = wp.dot(n_hat,
                               (xi + r_self_w) - (x[bb] + r_other_w))
                    j_lin = n_hat
                    j_ang = wp.cross(r_self_w, n_hat)
                else:
                    r_other_w = wp.quat_rotate(q[ba], off_a)
                    r_self_w = wp.quat_rotate(qi, off_b)
                    C = wp.dot(n_hat,
                               (x[ba] + r_other_w) - (xi + r_self_w))
                    j_lin = -n_hat
                    j_ang = -wp.cross(r_self_w, n_hat)
                n_for_G = n_hat
                have_G = True

            hard = s >= wp.inf
            if hard:
                C = C - c_alpha_C0[cj]

            lam_eff = c_lambda[cj]
            if not hard:
                lam_eff = 0.0

            lam_plus = k_p * C + lam_eff
            if t == CONTACT_TANGENT_6DOF:
                sib = c_sibling[cj]
                mu = c_friction[cj]
                if c_was_static[sib] != 0:
                    mu = c_friction_static[cj]
                bound = mu * wp.abs(c_lambda[sib])
                f_lo = -bound
                f_hi = bound
            else:
                f_lo = c_fmin[cj]
                f_hi = c_fmax[cj]
            f = wp.clamp(lam_plus, f_lo, f_hi)

            k_for_lhs = k_p
            abs_C = wp.abs(C)
            if abs_C > 1.0e-12:
                if lam_plus < f_lo:
                    k_for_lhs = wp.abs(f_lo - lam_plus) / abs_C
                elif lam_plus > f_hi:
                    k_for_lhs = wp.abs(f_hi - lam_plus) / abs_C

            A = A + outer3(j_lin, j_lin) * k_for_lhs
            B = B + outer3(j_ang, j_lin) * k_for_lhs
            D = D + outer3(j_ang, j_ang) * k_for_lhs

            f_mag = wp.abs(f)
            if f_mag > 0.0 and have_G:
                g_diag = geom_stiffness_diag(n_for_G, r_self_w) * f_mag
                D = D + wp.mat33(g_diag[0], 0.0, 0.0,
                                 0.0, g_diag[1], 0.0,
                                 0.0, 0.0, g_diag[2])
            r_lin = r_lin + j_lin * f
            r_ang = r_ang + j_ang * f

    # Cooperative join via warp shuffle — UNCONDITIONAL so every warp lane
    # participates (the full-mask shuffle requires it). Reduce each 3×3 block
    # row-by-row as a vec3, plus the two gradient vec3s. Inactive lanes feed
    # zeros, so a sub-group's leader gets the body's full constraint sum.
    a0 = warp_sum_vec3(wp.vec3(A[0, 0], A[0, 1], A[0, 2]), group_size)
    a1 = warp_sum_vec3(wp.vec3(A[1, 0], A[1, 1], A[1, 2]), group_size)
    a2 = warp_sum_vec3(wp.vec3(A[2, 0], A[2, 1], A[2, 2]), group_size)
    b0 = warp_sum_vec3(wp.vec3(B[0, 0], B[0, 1], B[0, 2]), group_size)
    b1 = warp_sum_vec3(wp.vec3(B[1, 0], B[1, 1], B[1, 2]), group_size)
    b2 = warp_sum_vec3(wp.vec3(B[2, 0], B[2, 1], B[2, 2]), group_size)
    d0 = warp_sum_vec3(wp.vec3(D[0, 0], D[0, 1], D[0, 2]), group_size)
    d1 = warp_sum_vec3(wp.vec3(D[1, 0], D[1, 1], D[1, 2]), group_size)
    d2 = warp_sum_vec3(wp.vec3(D[2, 0], D[2, 1], D[2, 2]), group_size)
    rl = warp_sum_vec3(r_lin, group_size)
    ra = warp_sum_vec3(r_ang, group_size)

    # Sub-group leader stores the complete sum (overwrite — primal_solve_6dof
    # still zeros scratch afterward, harmless here).
    if active == 1 and lane == 0:
        scratch_A[i] = wp.mat33(a0[0], a0[1], a0[2],
                                a1[0], a1[1], a1[2],
                                a2[0], a2[1], a2[2])
        scratch_B[i] = wp.mat33(b0[0], b0[1], b0[2],
                                b1[0], b1[1], b1[2],
                                b2[0], b2[1], b2[2])
        scratch_D[i] = wp.mat33(d0[0], d0[1], d0[2],
                                d1[0], d1[1], d1[2],
                                d2[0], d2[1], d2[2])
        scratch_rlin[i] = rl
        scratch_rang[i] = ra


@wp.kernel
def primal_solve_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    inertia_world: wp.array(dtype=wp.mat33),
    x_inertial: wp.array(dtype=wp.vec3),
    q_inertial: wp.array(dtype=wp.quat),
    color_starts: wp.array(dtype=int),
    color_bodies: wp.array(dtype=int),
    color_id: int,
    dt: float,
    scratch_A: wp.array(dtype=wp.mat33),
    scratch_B: wp.array(dtype=wp.mat33),
    scratch_D: wp.array(dtype=wp.mat33),
    scratch_rlin: wp.array(dtype=wp.vec3),
    scratch_rang: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    base = color_starts[color_id]
    end_c = color_starts[color_id + 1]
    if tid >= end_c - base:
        return
    i = color_bodies[base + tid]
    m = mass[i]
    if m <= 0.0:
        return

    inv_dt2 = 1.0 / (dt * dt)
    I_world = inertia_world[i]

    # Inertial init (identical to the serial kernel) + the constraint sum that
    # primal_accumulate_6dof folded into scratch.
    A = wp.mat33(m * inv_dt2, 0.0, 0.0,
                 0.0, m * inv_dt2, 0.0,
                 0.0, 0.0, m * inv_dt2) + scratch_A[i]
    D = I_world * inv_dt2 + scratch_D[i]
    B = scratch_B[i]

    r_lin = (x[i] - x_inertial[i]) * (m * inv_dt2) + scratch_rlin[i]
    dq_iner = wp.mul(q[i], wp.quat_inverse(q_inertial[i]))
    dtheta_iner = quat_to_rotvec(dq_iner)
    r_ang = I_world * (dtheta_iner * inv_dt2) + scratch_rang[i]

    # Schur-complement solve (verbatim from primal_update_6dof).
    A_inv = wp.inverse(A)
    BAinv = B * A_inv
    rhs_theta = r_ang - BAinv * r_lin
    S = D - BAinv * wp.transpose(B)
    S_inv = wp.inverse(S)
    d_theta = S_inv * rhs_theta
    d_x = A_inv * (r_lin - wp.transpose(B) * d_theta)

    x[i] = x[i] - d_x
    dq = quat_from_rotvec(-d_theta)
    q[i] = wp.normalize(dq * q[i])

    # Reset scratch for the next iteration's accumulate (initial state is zero;
    # this restores it without a separate clear launch).
    scratch_A[i] = wp.mat33()
    scratch_B[i] = wp.mat33()
    scratch_D[i] = wp.mat33()
    scratch_rlin[i] = wp.vec3(0.0, 0.0, 0.0)
    scratch_rang[i] = wp.vec3(0.0, 0.0, 0.0)


# -----------------------------------------------------------------------------
# Dual update (AVBD Eq. 11 + Eq. 16) — same structure as 3-DOF
# -----------------------------------------------------------------------------
@wp.kernel
def dual_update_6dof(
    # Device-side row count — host launches at dim=_gpu_pool_n_capacity.
    n_active_rows: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
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
    c_friction_static: wp.array(dtype=float),
    c_was_static: wp.array(dtype=int),
    beta: float,
):
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    if c_active[j] == 0:
        return
    t = c_type[j]
    C = float(0.0)
    if t == FLOOR_CONTACT_6DOF:
        floor_y = c_world_anchor[j][1]
        C = eval_floor_C(x[c_body_a[j]], q[c_body_a[j]], c_off_a[j], floor_y)
    elif t == CONTACT_TANGENT_6DOF:
        tangent = c_world_anchor[j]
        bb = c_body_b[j]
        if bb < 0:
            r_world = x[c_body_a[j]] + wp.quat_rotate(q[c_body_a[j]], c_off_a[j])
            C = wp.dot(tangent, r_world)
        else:
            r_a = x[c_body_a[j]] + wp.quat_rotate(q[c_body_a[j]], c_off_a[j])
            r_b = x[bb] + wp.quat_rotate(q[bb], c_off_b[j])
            C = wp.dot(tangent, r_a - r_b)
    elif t == PIN_6DOF:
        axis = c_body_b[j]
        C = eval_pin_axis(x[c_body_a[j]], q[c_body_a[j]], c_off_a[j],
                          c_world_anchor[j], axis)
    elif t == BOX_BOX_CONTACT_6DOF:
        n_hat = c_world_anchor[j]
        C = eval_box_box_C(x[c_body_a[j]], q[c_body_a[j]], c_off_a[j],
                           x[c_body_b[j]], q[c_body_b[j]], c_off_b[j], n_hat)

    s = c_stiffness[j]
    if s >= wp.inf:
        C = C - c_alpha_C0[j]
    lam_eff = c_lambda[j]
    if not (s >= wp.inf):
        lam_eff = 0.0

    lam_min = c_fmin[j]
    lam_max = c_fmax[j]
    is_tangent = (t == CONTACT_TANGENT_6DOF)
    sib = -1
    used_static = False
    if is_tangent:
        sib = c_sibling[j]
        mu = c_friction[j]
        if c_was_static[sib] != 0:
            mu = c_friction_static[j]
            used_static = True
        bound = mu * wp.abs(c_lambda[sib])
        lam_min = -bound
        lam_max = bound
    new_lam = wp.clamp(c_penalty[j] * C + lam_eff, lam_min, lam_max)
    c_lambda[j] = new_lam

    if wp.abs(new_lam) >= c_fracture[j]:
        c_active[j] = 0
        c_lambda[j] = 0.0
        c_penalty[j] = 0.0
        return

    if new_lam > lam_min and new_lam < lam_max:
        upper = wp.min(PENALTY_MAX, s)
        c_penalty[j] = wp.min(c_penalty[j] + beta * wp.abs(C), upper)
    elif is_tangent and used_static:
        # Static-friction clamp activated → ||λ_tb|| would exceed μ_s·|λ_n|.
        # AVBD Sec 3.3: "we immediately switch to dynamic friction using
        # μ = μ_d." Write c_was_static = 0 on the SHARED normal-row slot so
        # both tangent partners pick up the dynamic μ on subsequent iters /
        # frames. Single-writer here (dual is the only one that writes
        # c_was_static during the iter loop); partner-tangent dual launches
        # may race but they all want to write 0 → idempotent.
        c_was_static[sib] = 0


# -----------------------------------------------------------------------------
# Fused substep prelude — combines warmstart_duals + update_static_friction +
# main-pass cache_alpha_C0 into a single dim=n_c launch.
# -----------------------------------------------------------------------------
# Profiling showed each of the three small kernels was ~95% launch-overhead
# (~18 μs Python launch vs ~0.01 μs/constraint compute). Fusing into one
# kernel saves 2 launches per substep × 8 substeps = 16 launches per step
# (~320 μs at 20 μs launch overhead each).
#
# Ordering inside the kernel matches the unfused sequence:
#   1. warmstart: decay λ by α·γ (or 0 for post-stab), grow k toward γ·k
#   2. static friction: read decayed λ_t, λ_b, compare to μ_s·|λ_n|
#   3. cache α·C0 for main iter (post-stab α=0 still uses the separate
#      cache_alpha_C0_6dof launch — see _step_one).
@wp.kernel
def substep_prelude_6dof(
    # Device-side row count — host launches at dim=_gpu_pool_n_capacity.
    n_active_rows: wp.array(dtype=int),
    # state (read-only)
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    # constraint structure
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_stiffness: wp.array(dtype=float),
    c_sibling: wp.array(dtype=int),
    c_partner: wp.array(dtype=int),
    c_friction_static: wp.array(dtype=float),
    # in/out
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_was_static: wp.array(dtype=int),
    c_alpha_C0: wp.array(dtype=float),
    # scalars
    alpha: float,
    gamma: float,
    post_stabilize: int,
    main_alpha: float,
):
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    # --- 1. warmstart decay (Eq 19) ---
    k_floor = PENALTY_MIN
    if c_type[j] == CONTACT_TANGENT_6DOF:
        k_floor = PENALTY_MIN_TANGENT
    p = wp.clamp(c_penalty[j] * gamma, k_floor, PENALTY_MAX)
    if post_stabilize == 0:
        c_lambda[j] = c_lambda[j] * alpha * gamma
    s = c_stiffness[j]
    if not wp.isnan(s) and s < wp.inf:
        p = wp.min(p, s)
    c_penalty[j] = p

    # --- 2. static-friction check (Sec 3.3) — only on TANGENT rows ---
    if c_type[j] == CONTACT_TANGENT_6DOF and c_active[j] != 0:
        sib = c_sibling[j]
        if sib >= 0 and c_active[sib] != 0:
            partner = c_partner[j]
            # Process each (t,b) pair once on the lower-index tangent.
            if partner < 0 or partner >= j:
                lam_n = wp.abs(c_lambda[sib])
                if lam_n < 1.0e-9:
                    c_was_static[sib] = 0
                else:
                    lam_t = c_lambda[j]
                    lam_b = float(0.0)
                    if partner >= 0:
                        lam_b = c_lambda[partner]
                    lam_tb = wp.sqrt(lam_t * lam_t + lam_b * lam_b)
                    mu_s = c_friction_static[j]
                    if lam_tb <= mu_s * lam_n:
                        c_was_static[sib] = 1
                    else:
                        c_was_static[sib] = 0

    # --- 3. cache α·C0 for main iter (Eq 18) ---
    if c_active[j] == 0:
        c_alpha_C0[j] = 0.0
        return
    t = c_type[j]
    C0 = float(0.0)
    unilateral = False
    if t == FLOOR_CONTACT_6DOF:
        floor_y = c_world_anchor[j][1]
        C0 = eval_floor_C(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                          c_off_a[j], floor_y)
        unilateral = True
    elif t == CONTACT_TANGENT_6DOF:
        tangent = c_world_anchor[j]
        bb = c_body_b[j]
        if bb < 0:
            r_world = x_initial[c_body_a[j]] + wp.quat_rotate(
                q_initial[c_body_a[j]], c_off_a[j])
            c_alpha_C0[j] = wp.dot(tangent, r_world)
        else:
            r_a = x_initial[c_body_a[j]] + wp.quat_rotate(
                q_initial[c_body_a[j]], c_off_a[j])
            r_b = x_initial[bb] + wp.quat_rotate(q_initial[bb], c_off_b[j])
            c_alpha_C0[j] = wp.dot(tangent, r_a - r_b)
        return
    elif t == PIN_6DOF:
        axis = c_body_b[j]
        C0 = eval_pin_axis(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                           c_off_a[j], c_world_anchor[j], axis)
    elif t == BOX_BOX_CONTACT_6DOF:
        n_hat = c_world_anchor[j]
        C0 = eval_box_box_C(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                            c_off_a[j],
                            x_initial[c_body_b[j]], q_initial[c_body_b[j]],
                            c_off_b[j], n_hat)
        unilateral = True
    if unilateral and C0 > 0.0:
        c_alpha_C0[j] = 0.0
    else:
        c_alpha_C0[j] = main_alpha * C0


# -----------------------------------------------------------------------------
# Static-friction state update (AVBD Sec 3.3) — runs once per substep at start
# -----------------------------------------------------------------------------
# For each NORMAL contact row N with at least one tangent partner-pair (t,b)
# pointing back at it via c_sibling, this kernel reads the previous frame's
# (λ_t, λ_b) from those tangent rows and writes:
#   c_was_static[N] = 1  if ||λ_tb|| ≤ μ_s · |λ_n|  (within bound → static OK)
#   c_was_static[N] = 0  otherwise                  (was sliding → dynamic)
#
# The kernel is launched per-row, gated to TANGENT rows that own the
# "lower" half of a partner pair (c_partner[j] > j), so each pair is
# evaluated exactly once. If the row has no partner (1-D floor friction in
# a single tangent direction — pre-OBB code), it falls back to checking
# |λ_t| alone.
@wp.kernel
def update_static_friction_6dof(
    c_type: wp.array(dtype=int),
    c_lambda: wp.array(dtype=float),
    c_sibling: wp.array(dtype=int),
    c_partner: wp.array(dtype=int),
    c_friction_static: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_was_static: wp.array(dtype=int),
):
    j = wp.tid()
    if c_type[j] != CONTACT_TANGENT_6DOF:
        return
    if c_active[j] == 0:
        return
    sib = c_sibling[j]
    if sib < 0:
        return
    if c_active[sib] == 0:
        c_was_static[sib] = 0
        return
    partner = c_partner[j]
    # Process each (t,b) pair once on the lower-index tangent.
    if partner >= 0 and partner < j:
        return
    lam_n = wp.abs(c_lambda[sib])
    if lam_n < 1.0e-9:
        c_was_static[sib] = 0
        return
    lam_t = c_lambda[j]
    lam_b = float(0.0)
    if partner >= 0:
        lam_b = c_lambda[partner]
    lam_tb = wp.sqrt(lam_t * lam_t + lam_b * lam_b)
    mu_s = c_friction_static[j]
    if lam_tb <= mu_s * lam_n:
        c_was_static[sib] = 1
    else:
        c_was_static[sib] = 0


# -----------------------------------------------------------------------------
# α·C₀ cache for hard constraints (Eq. 18) — uses POSITION AT FRAME START
# -----------------------------------------------------------------------------
@wp.kernel
def cache_alpha_C0_6dof(
    # Device-side row count — host launches at dim=_gpu_pool_n_capacity.
    n_active_rows: wp.array(dtype=int),
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    alpha: float,
    c_alpha_C0: wp.array(dtype=float),
):
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    if c_active[j] == 0:
        c_alpha_C0[j] = 0.0
        return
    t = c_type[j]
    C0 = float(0.0)
    unilateral = False
    # C0 must be evaluated at the FRAME-START position (x_initial), not at
    # the warm-started position (x). The 2D AVBD reference computes contact
    # C0 in Manifold::initialize() which runs before warmstart_bodies — so
    # C0 reflects the pre-gravity-predict body state. If we instead use
    # x = x_warm here, a body falling fast warm-starts to a position well
    # below the floor, C0 becomes deeply negative, and α·C0 stabilization
    # then "pins" the body to that underground position (it settles at
    # y≈0 instead of y=h). Tangent rows already use x_initial.
    if t == FLOOR_CONTACT_6DOF:
        floor_y = c_world_anchor[j][1]
        C0 = eval_floor_C(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                          c_off_a[j], floor_y)
        unilateral = True
    elif t == CONTACT_TANGENT_6DOF:
        # Tangent C₀ uses pre-warm-start state (initial), with α=1 implicit
        # (see 3-DOF kernels.py cache_alpha_C0 comment for why).
        tangent = c_world_anchor[j]
        bb = c_body_b[j]
        if bb < 0:
            r_world = x_initial[c_body_a[j]] + wp.quat_rotate(
                q_initial[c_body_a[j]], c_off_a[j])
            c_alpha_C0[j] = wp.dot(tangent, r_world)
        else:
            r_a = x_initial[c_body_a[j]] + wp.quat_rotate(
                q_initial[c_body_a[j]], c_off_a[j])
            r_b = x_initial[bb] + wp.quat_rotate(q_initial[bb], c_off_b[j])
            c_alpha_C0[j] = wp.dot(tangent, r_a - r_b)
        return
    elif t == PIN_6DOF:
        axis = c_body_b[j]
        C0 = eval_pin_axis(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                           c_off_a[j], c_world_anchor[j], axis)
    elif t == BOX_BOX_CONTACT_6DOF:
        n_hat = c_world_anchor[j]
        C0 = eval_box_box_C(x_initial[c_body_a[j]], q_initial[c_body_a[j]],
                            c_off_a[j],
                            x_initial[c_body_b[j]], q_initial[c_body_b[j]],
                            c_off_b[j], n_hat)
        unilateral = True
    # Unilateral contacts (floor, box-box) use always-on rows rather than the
    # 2D ref's dynamic manifold creation. To match the ref's "no force when
    # separated" behavior we clip C0 to ≤0: when C0>0 (body sits ABOVE floor
    # / above contact plane at start of substep) we set c_alpha_C0=0 so the
    # constraint is raw and the unilateral fmax=0 clamp on f leaves force=0.
    # When C0≤0 (actual penetration), full α·C0 stabilization applies and the
    # main-iter solve preserves the initial penetration so velocity properly
    # decelerates; the post-stab iter (α=0) then lifts the body up.
    if unilateral and C0 > 0.0:
        c_alpha_C0[j] = 0.0
    else:
        c_alpha_C0[j] = alpha * C0


# -----------------------------------------------------------------------------
# Velocity finalise (BDF1) — linear and angular
# -----------------------------------------------------------------------------
@wp.kernel
def finalize_velocity_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    dt: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    prev_v: wp.array(dtype=wp.vec3),
    prev_omega: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    prev_v[i] = v[i]
    prev_omega[i] = omega[i]
    if mass[i] > 0.0:
        v[i] = (x[i] - x_initial[i]) / dt
        # Angular: ω in WORLD frame.
        #   q_current = exp(ω·dt)_world ⊗ q_initial  ⇒  Δq_world = q ⊗ q_initial⁻¹
        # This matches the world-frame convention used by predict_inertial
        # (q_inertial = exp_q(ω·dt) ⊗ q is a left-multiply) and the primal
        # update (q ← exp_q(-Δθ) ⊗ q is also a left-multiply, world-frame).
        # The earlier form (q_initial⁻¹ ⊗ q) extracts BODY-frame ω, which
        # gets reinterpreted as world-frame in the next predict — for an
        # asymmetric inertia or a pinned body this convention mismatch
        # leaks energy into rotation each step.
        dq = wp.mul(q[i], wp.quat_inverse(q_initial[i]))
        omega[i] = quat_to_rotvec(dq) / dt


# -----------------------------------------------------------------------------
# Velocity cap — clamps |v| AND |ω| independently
# -----------------------------------------------------------------------------
@wp.kernel
def cap_velocity_6dof(
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    max_lin: float,
    max_ang: float,
):
    i = wp.tid()
    s = wp.length(v[i])
    if s > max_lin:
        v[i] = v[i] * (max_lin / s)
    sa = wp.length(omega[i])
    if sa > max_ang:
        omega[i] = omega[i] * (max_ang / sa)


# -----------------------------------------------------------------------------
# Fused finalize_velocity + cap — saves 1 launch per substep (~15 μs CPU).
# When max_lin/max_ang are set to wp.inf the cap branch is a no-op.
# -----------------------------------------------------------------------------
@wp.kernel
def finalize_and_cap_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    mass: wp.array(dtype=float),
    dt: float,
    max_lin: float,
    max_ang: float,
    v: wp.array(dtype=wp.vec3),
    omega: wp.array(dtype=wp.vec3),
    prev_v: wp.array(dtype=wp.vec3),
    prev_omega: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    prev_v[i] = v[i]
    prev_omega[i] = omega[i]
    if mass[i] > 0.0:
        new_v = (x[i] - x_initial[i]) / dt
        dq = wp.mul(q[i], wp.quat_inverse(q_initial[i]))
        new_w = quat_to_rotvec(dq) / dt
        # Apply velocity cap inline.
        sl = wp.length(new_v)
        if sl > max_lin:
            new_v = new_v * (max_lin / sl)
        sa = wp.length(new_w)
        if sa > max_ang:
            new_w = new_w * (max_ang / sa)
        v[i] = new_v
        omega[i] = new_w


# =============================================================================
# Broadphase + parallel SAT  (AVBD Alg 1 line 1 — LBVH broadphase)
# =============================================================================
# Pipeline:
#   1. compute_body_aabb_6dof   — per body, world-axis AABB from OBB+R
#   2. wp.Bvh(...).rebuild()    — host-side LBVH build over those AABBs
#   3. bvh_broadphase_pairs     — per body, query BVH, atomic-append candidate
#                                 (i, j) pairs to a fixed-size buffer
#   4. obb_sat_pairs            — per candidate pair, 15-axis SAT, write
#                                 (overlap, sat_idx, n_hat, depth)
#   5. CPU                      — for each overlapping pair, run Python
#                                 Sutherland-Hodgman face-clip to emit up
#                                 to 4 BOX_BOX_CONTACT_6DOF rows + tangents.
#
# Steps 1–4 replace the previous Python O(N²) loop. Face-clip stays in Python
# because variable-length polygon output is awkward in Warp; it now runs on
# ~25 confirmed pairs per substep (instead of all O(N²) sphere-passing pairs).

@wp.kernel
def compute_body_aabb_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    half_extents: wp.array(dtype=wp.vec3),
    margin: float,
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
):
    """World-axis AABB of an OBB. Each AABB half-extent along world axis k is
    Σ_j |R[k,j]| · he[j] — the standard OBB→AABB projection. ``margin``
    inflates the box on every side so the broadphase keeps grazing pairs
    (the warm-start cache wants their λ to persist across brief separations)."""
    i = wp.tid()
    c = x[i]
    he = half_extents[i]
    R = wp.quat_to_matrix(q[i])
    ex = wp.abs(R[0, 0]) * he[0] + wp.abs(R[0, 1]) * he[1] + wp.abs(R[0, 2]) * he[2] + margin
    ey = wp.abs(R[1, 0]) * he[0] + wp.abs(R[1, 1]) * he[1] + wp.abs(R[1, 2]) * he[2] + margin
    ez = wp.abs(R[2, 0]) * he[0] + wp.abs(R[2, 1]) * he[1] + wp.abs(R[2, 2]) * he[2] + margin
    aabb_lo[i] = c - wp.vec3(ex, ey, ez)
    aabb_hi[i] = c + wp.vec3(ex, ey, ez)


@wp.kernel
def bvh_broadphase_pairs(
    bvh_id: wp.uint64,
    aabb_lo: wp.array(dtype=wp.vec3),
    aabb_hi: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    pair_count: wp.array(dtype=int),   # atomic counter, length 1
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    max_pairs: int,
):
    """Per body i, query the BVH for AABB overlaps. Emits each ordered pair
    (i, j) with i < j exactly once into pair_a/pair_b via wp.atomic_add on
    pair_count[0]. Pairs where both bodies are static are skipped (they were
    already pre-pinned by their floor rows).

    When pair_count[0] exceeds max_pairs the extras are simply dropped — the
    Python caller checks and grows the buffer next frame."""
    i = wp.tid()
    lo = aabb_lo[i]
    hi = aabb_hi[i]
    m_i = mass[i]
    query = wp.bvh_query_aabb(bvh_id, lo, hi)
    j = int(0)
    while wp.bvh_query_next(query, j):
        if j <= i:
            continue
        if m_i <= 0.0 and mass[j] <= 0.0:
            continue
        slot = wp.atomic_add(pair_count, 0, 1)
        if slot < max_pairs:
            pair_a[slot] = i
            pair_b[slot] = j


@wp.func
def obb_proj_radius3(L: wp.vec3, c0: wp.vec3, c1: wp.vec3, c2: wp.vec3, he: wp.vec3) -> float:
    """Projection radius of an OBB onto axis L (|L|=1). Standard SAT term:
    r = Σ_k he[k] · |L · R[:,k]|.

    Takes R's columns as pre-extracted vec3s so callers that loop over
    multiple SAT axes don't rebuild them 15 times per pair."""
    return he[0] * wp.abs(wp.dot(L, c0)) + he[1] * wp.abs(wp.dot(L, c1)) + he[2] * wp.abs(wp.dot(L, c2))


@wp.kernel
def obb_sat_pairs(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    half_extents: wp.array(dtype=wp.vec3),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    # Device-side pair count (was `n_pairs: int`). The host launches this at
    # dim=_bp_max_pairs and the runtime count comes from broadphase via
    # `pair_count[0]`. Required for CUDA-graph capture: kernel arguments
    # cannot depend on per-substep host scalars.
    pair_count: wp.array(dtype=int),
    margin: float,
    # outputs (one entry per pair)
    pair_overlap: wp.array(dtype=int),    # 1 if within margin, 0 otherwise
    pair_sat_idx: wp.array(dtype=int),    # 0–14 (3 face A, 3 face B, 9 edge×edge)
    pair_n_hat: wp.array(dtype=wp.vec3),  # unit normal from B to A
    pair_depth: wp.array(dtype=float),    # min overlap along best axis
):
    """Parallel 15-axis SAT over candidate pairs. Mirrors the CPU _obb_sat
    in solver_6dof.py but vectorized across pairs. Output is fed into the
    Python face-clip pipeline for the actual contact-point emission."""
    p = wp.tid()
    if p >= pair_count[0]:
        return
    i = pair_a[p]
    j = pair_b[p]

    c_A = x[i]
    c_B = x[j]
    e_A = half_extents[i]
    e_B = half_extents[j]
    R_A = wp.quat_to_matrix(q[i])
    R_B = wp.quat_to_matrix(q[j])
    t = c_B - c_A

    # Pre-extract each rotation matrix's columns once. The SAT loop uses
    # each column up to 15 times (in obb_proj_radius and the edge-edge
    # cross-product loop); rebuilding the vec3 from R[i,k] on every call
    # was 45 redundant mat33-element loads per pair.
    A0 = wp.vec3(R_A[0, 0], R_A[1, 0], R_A[2, 0])
    A1 = wp.vec3(R_A[0, 1], R_A[1, 1], R_A[2, 1])
    A2 = wp.vec3(R_A[0, 2], R_A[1, 2], R_A[2, 2])
    B0 = wp.vec3(R_B[0, 0], R_B[1, 0], R_B[2, 0])
    B1 = wp.vec3(R_B[0, 1], R_B[1, 1], R_B[2, 1])
    B2 = wp.vec3(R_B[0, 2], R_B[1, 2], R_B[2, 2])

    eps = 1.0e-6
    best_overlap = 1.0e20
    best_idx = -1
    best_axis = wp.vec3(0.0, 0.0, 0.0)

    # 3 face axes from A — L is A's own column k, so projection radius rA = e_A[k].
    for k in range(3):
        if k == 0:
            L = A0
        elif k == 1:
            L = A1
        else:
            L = A2
        rA = e_A[k]
        rB = obb_proj_radius3(L, B0, B1, B2, e_B)
        t_dot_L = wp.dot(t, L)
        sep = wp.abs(t_dot_L)
        ov = rA + rB - sep
        if ov < -margin:
            pair_overlap[p] = 0
            return
        if ov < best_overlap:
            best_overlap = ov
            best_idx = k
            if t_dot_L > 0.0:
                best_axis = -L
            else:
                best_axis = L

    # 3 face axes from B
    for k in range(3):
        if k == 0:
            L = B0
        elif k == 1:
            L = B1
        else:
            L = B2
        rA = obb_proj_radius3(L, A0, A1, A2, e_A)
        rB = e_B[k]
        t_dot_L = wp.dot(t, L)
        sep = wp.abs(t_dot_L)
        ov = rA + rB - sep
        if ov < -margin:
            pair_overlap[p] = 0
            return
        if ov < best_overlap:
            best_overlap = ov
            best_idx = 3 + k
            if t_dot_L > 0.0:
                best_axis = -L
            else:
                best_axis = L

    # 9 edge × edge cross products
    for i_e in range(3):
        if i_e == 0:
            A_col = A0
        elif i_e == 1:
            A_col = A1
        else:
            A_col = A2
        for j_e in range(3):
            if j_e == 0:
                B_col = B0
            elif j_e == 1:
                B_col = B1
            else:
                B_col = B2
            L = wp.cross(A_col, B_col)
            n_len = wp.length(L)
            if n_len < eps:
                continue   # parallel edges — degenerate axis
            L = L / n_len
            rA = obb_proj_radius3(L, A0, A1, A2, e_A)
            rB = obb_proj_radius3(L, B0, B1, B2, e_B)
            t_dot_L = wp.dot(t, L)
            sep = wp.abs(t_dot_L)
            ov = rA + rB - sep
            if ov < 0.0:
                pair_overlap[p] = 0
                return
            if ov < best_overlap:
                best_overlap = ov
                best_idx = 6 + 3 * i_e + j_e
                if t_dot_L > 0.0:
                    best_axis = -L
                else:
                    best_axis = L

    pair_overlap[p] = 1
    pair_sat_idx[p] = best_idx
    pair_n_hat[p] = best_axis
    pair_depth[p] = best_overlap


# =============================================================================
# Viewer-side fused readback — packs every per-frame stat the viewer reads
# into one staging buffer (closes AVBD_PERFORMANCE_GAP §6).
# =============================================================================
# Per-frame the interactive viewer asks for: positions, orientations, angular
# velocities, lambdas, active flags, c_was_static, c_type. Done naively that's
# 7 separate .numpy() calls — each one a stream sync on CUDA. This kernel
# fuses everything into one float array so the viewer issues 2 transfers
# (one for the per-body block, one for the per-row stats block).
#
# Per-body layout (10 floats):
#     [0..2] position
#     [3..6] orientation (xyzw)
#     [7..9] angular velocity
# Per-row stats layout (3 floats):
#     [0] lambda
#     [1] float(active)
#     [2] float(was_static * 16 + c_type)   # 1 byte was, 1 byte type
# was_static * 16 fits because c_type ∈ {0,1,2,3} and was_static ∈ {0,1}.


@wp.kernel
def viewer_pack_bodies_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=float),
):
    """Pack (position, orientation, angular_velocity) into 10 contiguous
    floats per body for a single viewer .numpy() readback."""
    i = wp.tid()
    base = i * 10
    xi = x[i]
    qi = q[i]
    wi = omega[i]
    out[base + 0] = xi[0]
    out[base + 1] = xi[1]
    out[base + 2] = xi[2]
    out[base + 3] = qi[0]
    out[base + 4] = qi[1]
    out[base + 5] = qi[2]
    out[base + 6] = qi[3]
    out[base + 7] = wi[0]
    out[base + 8] = wi[1]
    out[base + 9] = wi[2]


# =============================================================================
# OBB contact manifold — GPU face-clip + edge-edge closest-segment
# (closes AVBD_PERFORMANCE_GAP §3 hot inner loop).
# =============================================================================
# Sutherland-Hodgman face-clip + tangent-basis + body-local offset computation
# in one kernel. Output is up to 4 contacts per pair in fixed-size buffers;
# `gpu_pool_emit_rows` reads those outputs and atomically writes the BOX_BOX +
# tangent constraint rows directly into the c_* arrays — no CPU readback,
# no Python row construction (AVBD_PERFORMANCE_GAP §1 / §2).
#
# Layout per pair p:
#   out_contact_count[p]            in {0, 1, 2, 3, 4}
#   out_ref_is_a[p]                 1 if "reference" body == pair_a[p], else 0
#                                   (always 1 for edge-edge contacts)
#   out_n_hat[p], out_t_hat[p], out_b_hat[p]  contact-frame world axes
#   out_off_ref[p*4 + c]            body-local anchor on ref body for contact c
#   out_off_inc[p*4 + c]            body-local anchor on inc body for contact c
#
# Scratch per pair: 16 vec3 slots in `poly_scratch` (two 8-vert polygons used
# as the SH double-buffer) + 8 floats / 8 ints in `depth_scratch` /
# `idx_scratch` for the top-4 selection sort.
#
# Edge-edge contact (sat_idx >= 6) emits a single contact via Ericson §5.1.9
# closest-segment-pair, identical math to _emit_obb_edge_edge.

vec4i = wp.types.vector(length=4, dtype=int)


@wp.func
def orthonormal_basis_3d(n: wp.vec3) -> wp.vec3:
    """Duff 2017 stable orthonormal basis perp to n̂. Returns t̂; the third
    basis vector b̂ = n̂ × t̂. Matches the Python `_orthonormal_basis` in
    solver_6dof.py byte-for-byte so kernel-emitted tangent rows have the
    same (t, b) frame as the Python reference path."""
    sign = wp.where(n[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[2])
    b_ = n[0] * n[1] * a
    t = wp.vec3(1.0 + sign * n[0] * n[0] * a, sign * b_, -sign * n[0])
    return t


@wp.kernel
def obb_contact_manifold_6dof(
    x: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quat),
    half_extents: wp.array(dtype=wp.vec3),
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    pair_overlap: wp.array(dtype=int),
    pair_sat_idx: wp.array(dtype=int),
    pair_n_hat: wp.array(dtype=wp.vec3),
    # Device-side pair count (was `n_pairs: int`); see obb_sat_pairs above.
    pair_count: wp.array(dtype=int),
    margin: float,
    # scratch: 16 vec3 slots per pair for SH polygon double-buffer
    poly_scratch: wp.array(dtype=wp.vec3),
    # outputs (per pair, 1 each)
    out_contact_count: wp.array(dtype=int),
    out_ref_is_a: wp.array(dtype=int),
    out_n_hat: wp.array(dtype=wp.vec3),
    out_t_hat: wp.array(dtype=wp.vec3),
    out_b_hat: wp.array(dtype=wp.vec3),
    # outputs (per pair × 4 contact slots)
    out_off_ref: wp.array(dtype=wp.vec3),
    out_off_inc: wp.array(dtype=wp.vec3),
):
    p = wp.tid()
    if p >= pair_count[0]:
        return
    out_contact_count[p] = 0
    if pair_overlap[p] == 0:
        return
    sat_idx = pair_sat_idx[p]
    i = pair_a[p]
    j = pair_b[p]
    c_A = x[i]
    c_B = x[j]
    e_A = half_extents[i]
    e_B = half_extents[j]
    R_A = wp.quat_to_matrix(q[i])
    R_B = wp.quat_to_matrix(q[j])
    n_hat_in = pair_n_hat[p]

    # ------------------------- Edge-edge case (sat_idx ≥ 6) -------------------
    if sat_idx >= 6:
        eidx = sat_idx - 6
        k_A = eidx // 3
        k_B = eidx % 3
        eA_dir = wp.vec3(R_A[0, k_A], R_A[1, k_A], R_A[2, k_A])
        eB_dir = wp.vec3(R_B[0, k_B], R_B[1, k_B], R_B[2, k_B])
        cross_mag = wp.length(wp.cross(eA_dir, eB_dir))
        if cross_mag < 1.0e-4:
            return    # parallel edges — face-axis case already handled
        # Sign-select the contact edge on each body (away from the other COM).
        kA1 = (k_A + 1) % 3
        kA2 = (k_A + 2) % 3
        A1c = wp.vec3(R_A[0, kA1], R_A[1, kA1], R_A[2, kA1])
        A2c = wp.vec3(R_A[0, kA2], R_A[1, kA2], R_A[2, kA2])
        toB = c_B - c_A
        s_b_A = wp.where(wp.dot(toB, A1c) >= 0.0, 1.0, -1.0)
        s_c_A = wp.where(wp.dot(toB, A2c) >= 0.0, 1.0, -1.0)
        edge_A_mid = c_A + A1c * (s_b_A * e_A[kA1]) + A2c * (s_c_A * e_A[kA2])
        P1 = edge_A_mid - eA_dir * e_A[k_A]
        Q1 = edge_A_mid + eA_dir * e_A[k_A]
        kB1 = (k_B + 1) % 3
        kB2 = (k_B + 2) % 3
        B1c = wp.vec3(R_B[0, kB1], R_B[1, kB1], R_B[2, kB1])
        B2c = wp.vec3(R_B[0, kB2], R_B[1, kB2], R_B[2, kB2])
        toA = c_A - c_B
        s_b_B = wp.where(wp.dot(toA, B1c) >= 0.0, 1.0, -1.0)
        s_c_B = wp.where(wp.dot(toA, B2c) >= 0.0, 1.0, -1.0)
        edge_B_mid = c_B + B1c * (s_b_B * e_B[kB1]) + B2c * (s_c_B * e_B[kB2])
        P2 = edge_B_mid - eB_dir * e_B[k_B]
        Q2 = edge_B_mid + eB_dir * e_B[k_B]
        # Closest-segment-pair (Ericson §5.1.9) — same branch structure as
        # the Python reference.
        d1 = Q1 - P1
        d2 = Q2 - P2
        r = P1 - P2
        a = wp.dot(d1, d1)
        e_val = wp.dot(d2, d2)
        f = wp.dot(d2, r)
        eps = 1.0e-12
        s_p = float(0.0)
        t_p = float(0.0)
        if a <= eps and e_val <= eps:
            s_p = 0.0
            t_p = 0.0
        elif a <= eps:
            s_p = 0.0
            t_p = wp.clamp(f / wp.max(e_val, eps), 0.0, 1.0)
        elif e_val <= eps:
            t_p = 0.0
            c_ = wp.dot(d1, r)
            s_p = wp.clamp(-c_ / a, 0.0, 1.0)
        else:
            c_ = wp.dot(d1, r)
            b_ = wp.dot(d1, d2)
            denom = a * e_val - b_ * b_
            if denom != 0.0:
                s_p = wp.clamp((b_ * f - c_ * e_val) / denom, 0.0, 1.0)
            else:
                s_p = 0.0
            t_p = (b_ * s_p + f) / e_val
            if t_p < 0.0:
                t_p = 0.0
                s_p = wp.clamp(-c_ / a, 0.0, 1.0)
            elif t_p > 1.0:
                t_p = 1.0
                s_p = wp.clamp((b_ - c_) / a, 0.0, 1.0)
        p_on_A = P1 + d1 * s_p
        p_on_B = P2 + d2 * t_p
        gap = wp.dot(p_on_A - p_on_B, n_hat_in)
        if gap > margin:
            return
        Rt_A = wp.transpose(R_A)
        Rt_B = wp.transpose(R_B)
        off_a_local = Rt_A * (p_on_A - c_A)
        off_b_local = Rt_B * (p_on_B - c_B)
        t_hat = orthonormal_basis_3d(n_hat_in)
        b_hat = wp.cross(n_hat_in, t_hat)
        out_contact_count[p] = 1
        out_ref_is_a[p] = 1
        out_n_hat[p] = n_hat_in
        out_t_hat[p] = t_hat
        out_b_hat[p] = b_hat
        out_off_ref[p * 4 + 0] = off_a_local
        out_off_inc[p * 4 + 0] = off_b_local
        return

    # ------------------------- Face-face case (sat_idx < 6) -------------------
    ref_is_a_int = wp.where(sat_idx < 3, 1, 0)
    out_ref_is_a[p] = ref_is_a_int
    # Select ref/inc geometry. Note: when ref=B (sat_idx ∈ [3,6)), we flip
    # n_hat because pair_n_hat was returned "B→A" by obb_sat_pairs and we
    # need it to point "inc→ref" for the new mapping.
    ref_axis = int(0)
    n_hat_used = wp.vec3(0.0, 0.0, 0.0)
    ref_c = wp.vec3(0.0, 0.0, 0.0)
    R_ref = wp.mat33()
    e_ref = wp.vec3(0.0, 0.0, 0.0)
    inc_c = wp.vec3(0.0, 0.0, 0.0)
    R_inc = wp.mat33()
    e_inc = wp.vec3(0.0, 0.0, 0.0)
    if sat_idx < 3:
        ref_axis = sat_idx
        n_hat_used = n_hat_in
        ref_c = c_A
        R_ref = R_A
        e_ref = e_A
        inc_c = c_B
        R_inc = R_B
        e_inc = e_B
    else:
        ref_axis = sat_idx - 3
        n_hat_used = -n_hat_in
        ref_c = c_B
        R_ref = R_B
        e_ref = e_B
        inc_c = c_A
        R_inc = R_A
        e_inc = e_A

    # Reference face: outward normal opposes n_hat_used.
    ref_col_axis = wp.vec3(R_ref[0, ref_axis], R_ref[1, ref_axis], R_ref[2, ref_axis])
    ref_sign = wp.where(wp.dot(ref_col_axis, n_hat_used) > 0.0, -1.0, 1.0)
    ref_face_n = ref_col_axis * ref_sign
    ref_face_c = ref_c + ref_face_n * e_ref[ref_axis]
    ax1 = (ref_axis + 1) % 3
    ax2 = (ref_axis + 2) % 3
    u_ref = wp.vec3(R_ref[0, ax1], R_ref[1, ax1], R_ref[2, ax1])
    v_ref = wp.vec3(R_ref[0, ax2], R_ref[1, ax2], R_ref[2, ax2])
    eu = e_ref[ax1]
    ev = e_ref[ax2]

    # Incident face: max-dot axis × sign on inc body.
    inc_axis = int(0)
    inc_sign = float(1.0)
    best_dot = float(-1.0e20)
    for k in range(3):
        col = wp.vec3(R_inc[0, k], R_inc[1, k], R_inc[2, k])
        d_pos = wp.dot(col, n_hat_used)
        if d_pos > best_dot:
            best_dot = d_pos
            inc_axis = k
            inc_sign = 1.0
        d_neg = -d_pos
        if d_neg > best_dot:
            best_dot = d_neg
            inc_axis = k
            inc_sign = -1.0
    inc_col_axis = wp.vec3(R_inc[0, inc_axis], R_inc[1, inc_axis], R_inc[2, inc_axis])
    inc_face_n = inc_col_axis * inc_sign
    inc_face_c = inc_c + inc_face_n * e_inc[inc_axis]
    inc_ax1 = (inc_axis + 1) % 3
    inc_ax2 = (inc_axis + 2) % 3
    inc_u = wp.vec3(R_inc[0, inc_ax1], R_inc[1, inc_ax1], R_inc[2, inc_ax1])
    inc_v = wp.vec3(R_inc[0, inc_ax2], R_inc[1, inc_ax2], R_inc[2, inc_ax2])
    inc_eu = e_inc[inc_ax1]
    inc_ev = e_inc[inc_ax2]

    # Sutherland-Hodgman clip — double-buffered polygon in poly_scratch.
    # Each pair owns 16 vec3 slots: [p*16, p*16+8) and [p*16+8, p*16+16).
    cur_base = p * 16
    nxt_base = p * 16 + 8
    # Initial polygon: 4 incident-face verts, same winding order as Python.
    poly_scratch[cur_base + 0] = inc_face_c + inc_u * inc_eu + inc_v * inc_ev
    poly_scratch[cur_base + 1] = inc_face_c - inc_u * inc_eu + inc_v * inc_ev
    poly_scratch[cur_base + 2] = inc_face_c - inc_u * inc_eu - inc_v * inc_ev
    poly_scratch[cur_base + 3] = inc_face_c + inc_u * inc_eu - inc_v * inc_ev
    poly_a_len = int(4)

    for plane_idx in range(4):
        plane_pt = wp.vec3(0.0, 0.0, 0.0)
        plane_n = wp.vec3(0.0, 0.0, 0.0)
        if plane_idx == 0:
            plane_pt = ref_face_c + u_ref * eu
            plane_n = u_ref
        elif plane_idx == 1:
            plane_pt = ref_face_c - u_ref * eu
            plane_n = -u_ref
        elif plane_idx == 2:
            plane_pt = ref_face_c + v_ref * ev
            plane_n = v_ref
        else:
            plane_pt = ref_face_c - v_ref * ev
            plane_n = -v_ref
        if poly_a_len == 0:
            return
        out_len = int(0)
        for i_v in range(poly_a_len):
            a_pt = poly_scratch[cur_base + i_v]
            nxt_i = (i_v + 1) % poly_a_len
            b_pt = poly_scratch[cur_base + nxt_i]
            da = wp.dot(a_pt - plane_pt, plane_n)
            db = wp.dot(b_pt - plane_pt, plane_n)
            if da <= 0.0:
                if out_len < 8:
                    poly_scratch[nxt_base + out_len] = a_pt
                    out_len = out_len + 1
                if db > 0.0:
                    t = da / (da - db)
                    if out_len < 8:
                        poly_scratch[nxt_base + out_len] = a_pt + (b_pt - a_pt) * t
                        out_len = out_len + 1
            else:
                if db <= 0.0:
                    t = da / (da - db)
                    if out_len < 8:
                        poly_scratch[nxt_base + out_len] = a_pt + (b_pt - a_pt) * t
                        out_len = out_len + 1
        # Swap polygon buffers.
        tmp = cur_base
        cur_base = nxt_base
        nxt_base = tmp
        poly_a_len = out_len

    if poly_a_len == 0:
        return

    # Top-4 by depth (descending). depth = -d where d = (pt - ref_face_c)·ref_face_n.
    # Reject if d ≥ margin (separated by more than the warm-start gap). Maintain
    # a 4-slot "best so far" set in two parallel vec4s (float depths + int indices).
    best_d = wp.vec4(-1.0e30, -1.0e30, -1.0e30, -1.0e30)
    best_i = vec4i(-1, -1, -1, -1)
    n_best = int(0)
    for i_v in range(poly_a_len):
        pt = poly_scratch[cur_base + i_v]
        d_signed = wp.dot(pt - ref_face_c, ref_face_n)
        if d_signed < margin:
            depth = -d_signed
            if n_best < 4:
                best_d[n_best] = depth
                best_i[n_best] = i_v
                n_best = n_best + 1
            else:
                min_pos = int(0)
                min_val = best_d[0]
                for k in range(1, 4):
                    if best_d[k] < min_val:
                        min_val = best_d[k]
                        min_pos = k
                if depth > min_val:
                    best_d[min_pos] = depth
                    best_i[min_pos] = i_v
    if n_best == 0:
        return

    # Sort the n_best slots by depth descending — selection sort, n ≤ 4.
    for i_s in range(n_best - 1):
        max_pos = i_s
        for k in range(i_s + 1, n_best):
            if best_d[k] > best_d[max_pos]:
                max_pos = k
        if max_pos != i_s:
            tmp_d = best_d[i_s]
            best_d[i_s] = best_d[max_pos]
            best_d[max_pos] = tmp_d
            tmp_i = best_i[i_s]
            best_i[i_s] = best_i[max_pos]
            best_i[max_pos] = tmp_i

    # Emit contacts.
    out_n_hat[p] = n_hat_used
    t_hat = orthonormal_basis_3d(n_hat_used)
    b_hat = wp.cross(n_hat_used, t_hat)
    out_t_hat[p] = t_hat
    out_b_hat[p] = b_hat
    out_contact_count[p] = n_best
    Rt_ref = wp.transpose(R_ref)
    Rt_inc = wp.transpose(R_inc)
    for c in range(n_best):
        i_v = best_i[c]
        p_inc = poly_scratch[cur_base + i_v]
        d_signed = wp.dot(p_inc - ref_face_c, ref_face_n)
        p_ref = p_inc - ref_face_n * d_signed
        out_off_ref[p * 4 + c] = Rt_ref * (p_ref - ref_c)
        out_off_inc[p * 4 + c] = Rt_inc * (p_inc - inc_c)


@wp.kernel
def viewer_pack_rows_6dof(
    c_lambda: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_was_static: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    """Pack (lambda, active, was_static*16+type) into 3 contiguous floats per
    constraint row for the viewer HUD. `was_static*16 + type` packs 5 bits
    into one float — well within IEEE-754 exact integer range."""
    j = wp.tid()
    base = j * 3
    out[base + 0] = c_lambda[j]
    out[base + 1] = float(c_active[j])
    out[base + 2] = float(c_was_static[j] * 16 + c_type[j])


# =============================================================================
# GPU-resident dynamic constraint pool (AVBD_PERFORMANCE_GAP §1 follow-up).
# =============================================================================
# Below this point are the kernels that let the per-substep contact pipeline
# stay GPU-side: row emission, pair-slot hash lookup/store, CSR adjacency
# rebuild, and active-row reset. They replace the per-substep CPU row append
# + _flush() rebuild path. See solver_6dof.py:Solver6DOF for the new flow.
#
# Layout conventions:
#   - c_* arrays are pre-allocated at capacity = N_static + N_dyn_capacity.
#     Static rows live in [0, N_static); dynamic BOX_BOX + tangent rows live
#     in the dynamic region above. Each dynamic contact owns a 3-row block
#     (normal + 2 tangents) claimed via wp.atomic_add on n_active_rows.
#   - Pair hash uses open-addressing linear probe. Key = encode(a, b, c_idx)
#     with a < b and c_idx ∈ [0,4). Empty slot = -1. Capacity should be at
#     least 2× the expected number of (pair, contact) entries.
#   - Hash state per slot: 8 floats — λ_n, λ_t, λ_b, k_n, k_t, k_b, was_t, was_b.

HASH_EMPTY = wp.constant(-1)
GPU_POOL_C_PER_PAIR = wp.constant(4)
# Upper bound on colors emitted by the device-side Jones-Plassmann
# coloring. The primal-update loop unrolls to this many launches per
# iteration; empty colors no-op via a device-side bounds check. Sized to
# cover dense pile-ups (typical contact graphs need ≤16 colors).
MAX_COLORS = wp.constant(32)
# Jones-Plassmann rounds per substep. Each round colors one independent
# set; the loop is bounded by ~log(n_b) for typical graphs.
JP_ROUNDS = wp.constant(24)


@wp.func
def encode_pair_key(a: int, b: int, c_idx: int) -> int:
    # a ≤ b enforced by caller. (a, b) ∈ [0, 16384), c_idx ∈ [0, 4).
    # Fits int32 up to ~1.07e9. For scenes with n_b > 16383, swap this for
    # a wider key or a true 64-bit hash table; the avbd3d prototype tops
    # out well below that today.
    return (a * 16384 + b) * 4 + c_idx


@wp.func
def hash_pair_key(key: int, cap: int) -> int:
    # Bit-mix then mask to int31 so the modulo result is always nonnegative.
    h = key * 73856093
    h = h ^ (h >> 16)
    h = h & 0x7fffffff
    return h % cap


@wp.kernel
def gpu_pool_reset_and_csr_zero(
    n_active_rows: wp.array(dtype=int),
    pool_count: wp.array(dtype=int),
    bp_pair_count: wp.array(dtype=int),
    body_con_counts: wp.array(dtype=int),
    n_static: int,
):
    """Fused reset + CSR zero. Launched at dim=n_bodies once per substep.
    Thread 0 resets the substep-scoped atomic counters; every thread also
    zeros its per-body constraint count. Replaces the dim=1 gpu_pool_reset
    plus the dim=n_b gpu_csr_zero_counts (one launch instead of two)."""
    tid = wp.tid()
    if tid == 0:
        n_active_rows[0] = n_static
        pool_count[0] = 0
        bp_pair_count[0] = 0
    body_con_counts[tid] = 0


@wp.kernel
def gpu_csr_count(
    n_active_rows: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    body_con_counts: wp.array(dtype=int),
):
    """Atomic histogram pass — counts how many active rows touch each body.
    PIN_6DOF re-uses c_body_b as an axis index, so we exclude it from the
    second-body increment."""
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    ba = c_body_a[j]
    if ba >= 0:
        wp.atomic_add(body_con_counts, ba, 1)
    t = c_type[j]
    if t != PIN_6DOF:
        bb = c_body_b[j]
        if bb >= 0:
            wp.atomic_add(body_con_counts, bb, 1)


@wp.kernel
def gpu_csr_starts_from_counts(
    body_con_counts: wp.array(dtype=int),
    body_con_starts: wp.array(dtype=int),
    n_bodies: int,
):
    """Single-thread inclusive scan into body_con_starts[1..n]. Tiny serial
    scan — n_bodies is small in practice (≤ 10⁵). Replace with a parallel
    scan if profiling shows it dominates."""
    if wp.tid() != 0:
        return
    body_con_starts[0] = 0
    acc = int(0)
    for i in range(n_bodies):
        acc = acc + body_con_counts[i]
        body_con_starts[i + 1] = acc


@wp.kernel
def gpu_csr_scatter(
    n_active_rows: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    body_con_cursor: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
):
    """For each active row, atomically claim a slot in body_con_indices for
    body_a (and body_b if applicable) using a per-body cursor that starts
    at body_con_starts[i]."""
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    ba = c_body_a[j]
    if ba >= 0:
        slot = wp.atomic_add(body_con_cursor, ba, 1)
        body_con_indices[slot] = j
    t = c_type[j]
    if t != PIN_6DOF:
        bb = c_body_b[j]
        if bb >= 0:
            slot = wp.atomic_add(body_con_cursor, bb, 1)
            body_con_indices[slot] = j


@wp.kernel
def gpu_pool_emit_rows(
    # Device-side pair count (was `n_pairs: int`); host launches at
    # dim=_bp_max_pairs.
    pair_count: wp.array(dtype=int),
    # row + pool capacity (host-side; emit drops contacts past these)
    n_cap_total: int,
    pool_cap: int,
    pair_a: wp.array(dtype=int),
    pair_b: wp.array(dtype=int),
    mf_contact_count: wp.array(dtype=int),
    mf_ref_is_a: wp.array(dtype=int),
    mf_n_hat: wp.array(dtype=wp.vec3),
    mf_t_hat: wp.array(dtype=wp.vec3),
    mf_b_hat: wp.array(dtype=wp.vec3),
    mf_off_ref: wp.array(dtype=wp.vec3),
    mf_off_inc: wp.array(dtype=wp.vec3),
    body_friction: wp.array(dtype=float),
    default_mu: float,
    friction_static_mult: float,
    # pool/output state
    n_active_rows: wp.array(dtype=int),
    pool_count: wp.array(dtype=int),
    pool_idx_n: wp.array(dtype=int),
    pool_idx_t: wp.array(dtype=int),
    pool_idx_b: wp.array(dtype=int),
    pool_idx_c: wp.array(dtype=int),
    # pair hash for warm-start lookup (read-only)
    hash_cap: int,
    hash_keys: wp.array(dtype=int),
    hash_state: wp.array(dtype=float),
    # constraint arrays (written into the dynamic region)
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_world_anchor: wp.array(dtype=wp.vec3),
    c_off_a: wp.array(dtype=wp.vec3),
    c_off_b: wp.array(dtype=wp.vec3),
    c_rest: wp.array(dtype=float),
    c_stiffness: wp.array(dtype=float),
    c_fmin: wp.array(dtype=float),
    c_fmax: wp.array(dtype=float),
    c_alpha_C0: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_fracture: wp.array(dtype=float),
    c_sibling: wp.array(dtype=int),
    c_partner: wp.array(dtype=int),
    c_friction: wp.array(dtype=float),
    c_friction_static: wp.array(dtype=float),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_was_static: wp.array(dtype=int),
):
    """For each broadphase pair with a contact manifold, atomically claim a
    contiguous block of 3 rows in the dynamic region of the c_* arrays per
    contact point (normal + 2 tangents) and write the full row records
    directly. Looks up (a, b, c_idx) in the pair hash to seed warm-start
    λ/k/was_static for stable stacks across substeps.

    Replaces the legacy CPU readback + Python _Row append loop. No data
    leaves the GPU during emission."""
    p = wp.tid()
    if p >= pair_count[0]:
        return
    n_contacts = mf_contact_count[p]
    if n_contacts == 0:
        return

    a_in = pair_a[p]
    b_in = pair_b[p]
    if mf_ref_is_a[p] == 1:
        ref_i = a_in
        inc_i = b_in
    else:
        ref_i = b_in
        inc_i = a_in
    n_hat = mf_n_hat[p]
    t_hat = mf_t_hat[p]
    b_hat = mf_b_hat[p]

    mu_ref = body_friction[ref_i]
    mu_inc = body_friction[inc_i]
    mu = default_mu
    if mu_ref > 0.0 or mu_inc > 0.0:
        mu = wp.sqrt(mu_ref * mu_inc)
    mu_s = mu * friction_static_mult

    # Canonical pair order for hashing — invariant to ref/inc selection.
    ka = ref_i
    kb = inc_i
    if kb < ka:
        ka = inc_i
        kb = ref_i

    for c in range(n_contacts):
        off_ref = mf_off_ref[p * 4 + c]
        off_inc = mf_off_inc[p * 4 + c]
        rows_per_contact = int(3)
        if mu <= 0.0:
            rows_per_contact = 1
        row_base = wp.atomic_add(n_active_rows, 0, rows_per_contact)
        # Capacity guard — host grows the pool next substep on overflow.
        # The reservation stays in n_active_rows so the host can detect it
        # (it is clamped against n_cap_total on readback).
        if row_base + rows_per_contact > n_cap_total:
            return

        # ----- Normal row -----
        n_idx = row_base
        c_type[n_idx] = BOX_BOX_CONTACT_6DOF
        c_body_a[n_idx] = ref_i
        c_body_b[n_idx] = inc_i
        c_world_anchor[n_idx] = n_hat
        c_off_a[n_idx] = off_ref
        c_off_b[n_idx] = off_inc
        c_rest[n_idx] = 0.0
        c_stiffness[n_idx] = wp.inf
        c_fmin[n_idx] = -wp.inf
        c_fmax[n_idx] = 0.0
        c_alpha_C0[n_idx] = 0.0
        c_active[n_idx] = 1
        c_fracture[n_idx] = wp.inf
        c_sibling[n_idx] = -1
        c_partner[n_idx] = -1
        c_friction[n_idx] = 0.0
        c_friction_static[n_idx] = 0.0

        # ----- Tangent rows (if friction) -----
        t_idx = -1
        b_idx = -1
        if mu > 0.0:
            t_idx = row_base + 1
            b_idx = row_base + 2
            c_type[t_idx] = CONTACT_TANGENT_6DOF
            c_body_a[t_idx] = ref_i
            c_body_b[t_idx] = inc_i
            c_world_anchor[t_idx] = t_hat
            c_off_a[t_idx] = off_ref
            c_off_b[t_idx] = off_inc
            c_rest[t_idx] = 0.0
            c_stiffness[t_idx] = wp.inf
            c_fmin[t_idx] = -wp.inf
            c_fmax[t_idx] = wp.inf
            c_alpha_C0[t_idx] = 0.0
            c_active[t_idx] = 1
            c_fracture[t_idx] = wp.inf
            c_sibling[t_idx] = n_idx
            c_partner[t_idx] = b_idx
            c_friction[t_idx] = mu
            c_friction_static[t_idx] = mu_s

            c_type[b_idx] = CONTACT_TANGENT_6DOF
            c_body_a[b_idx] = ref_i
            c_body_b[b_idx] = inc_i
            c_world_anchor[b_idx] = b_hat
            c_off_a[b_idx] = off_ref
            c_off_b[b_idx] = off_inc
            c_rest[b_idx] = 0.0
            c_stiffness[b_idx] = wp.inf
            c_fmin[b_idx] = -wp.inf
            c_fmax[b_idx] = wp.inf
            c_alpha_C0[b_idx] = 0.0
            c_active[b_idx] = 1
            c_fracture[b_idx] = wp.inf
            c_sibling[b_idx] = n_idx
            c_partner[b_idx] = t_idx
            c_friction[b_idx] = mu
            c_friction_static[b_idx] = mu_s

        # ----- Warm-start lookup -----
        # Defaults if no cache hit: λ=0, penalty=PENALTY_MIN floors, was_static=0.
        lam_n = float(0.0)
        lam_t = float(0.0)
        lam_b = float(0.0)
        k_n = float(PENALTY_MIN)
        k_t = float(PENALTY_MIN_TANGENT)
        k_b = float(PENALTY_MIN_TANGENT)
        was_t = int(0)
        was_b = int(0)
        key = encode_pair_key(ka, kb, c)
        h = hash_pair_key(key, hash_cap)
        for probe in range(hash_cap):
            slot = (h + probe) % hash_cap
            stored_key = hash_keys[slot]
            if stored_key == key:
                base = slot * 8
                lam_n = hash_state[base + 0]
                lam_t = hash_state[base + 1]
                lam_b = hash_state[base + 2]
                k_n = hash_state[base + 3]
                k_t = hash_state[base + 4]
                k_b = hash_state[base + 5]
                was_t = int(hash_state[base + 6])
                was_b = int(hash_state[base + 7])
                break
            if stored_key == HASH_EMPTY:
                break

        c_lambda[n_idx] = lam_n
        c_penalty[n_idx] = k_n
        c_was_static[n_idx] = 0
        if t_idx >= 0:
            c_lambda[t_idx] = lam_t
            c_penalty[t_idx] = k_t
            c_was_static[t_idx] = was_t
        if b_idx >= 0:
            c_lambda[b_idx] = lam_b
            c_penalty[b_idx] = k_b
            c_was_static[b_idx] = was_b

        # ----- Pool entry (for hash_collect after solve) -----
        pool_id = wp.atomic_add(pool_count, 0, 1)
        if pool_id >= pool_cap:
            continue
        pool_idx_n[pool_id] = n_idx
        pool_idx_t[pool_id] = t_idx
        pool_idx_b[pool_id] = b_idx
        pool_idx_c[pool_id] = c


@wp.kernel
def gpu_pool_hash_clear(
    hash_keys: wp.array(dtype=int),
):
    """Mark every slot empty before a hash_collect pass."""
    hash_keys[wp.tid()] = HASH_EMPTY


@wp.kernel
def gpu_pool_hash_collect(
    pool_count: wp.array(dtype=int),
    pool_idx_n: wp.array(dtype=int),
    pool_idx_t: wp.array(dtype=int),
    pool_idx_b: wp.array(dtype=int),
    pool_idx_c: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    c_lambda: wp.array(dtype=float),
    c_penalty: wp.array(dtype=float),
    c_active: wp.array(dtype=int),
    c_was_static: wp.array(dtype=int),
    hash_cap: int,
    hash_keys: wp.array(dtype=int),
    hash_state: wp.array(dtype=float),
):
    """For each pool entry, read the current λ/k/was_static of its 3 rows and
    upsert into the pair hash table keyed on (a, b, c_idx). On a fresh
    post-solve this rebuilds the warm-start cache from in-place GPU state —
    no .numpy() readback, no Python dict. c_idx was recorded by
    gpu_pool_emit_rows in emission order so the mapping is stable from one
    substep to the next for stable poses (manifold SH-clip is deterministic)."""
    p = wp.tid()
    if p >= pool_count[0]:
        return
    n_idx = pool_idx_n[p]
    if c_active[n_idx] == 0:
        return
    a_raw = c_body_a[n_idx]
    b_raw = c_body_b[n_idx]
    ka = a_raw
    kb = b_raw
    if kb < ka:
        ka = b_raw
        kb = a_raw
    c_idx = pool_idx_c[p]

    lam_n = c_lambda[n_idx]
    k_n = c_penalty[n_idx]
    lam_t = float(0.0)
    lam_b = float(0.0)
    k_t = float(PENALTY_MIN_TANGENT)
    k_b = float(PENALTY_MIN_TANGENT)
    was_t = int(0)
    was_b = int(0)
    t_idx = pool_idx_t[p]
    if t_idx >= 0:
        lam_t = c_lambda[t_idx]
        k_t = c_penalty[t_idx]
        was_t = c_was_static[t_idx]
    b_idx = pool_idx_b[p]
    if b_idx >= 0:
        lam_b = c_lambda[b_idx]
        k_b = c_penalty[b_idx]
        was_b = c_was_static[b_idx]

    key = encode_pair_key(ka, kb, c_idx)
    h = hash_pair_key(key, hash_cap)
    for probe in range(hash_cap):
        slot = (h + probe) % hash_cap
        prev = wp.atomic_cas(hash_keys, slot, HASH_EMPTY, key)
        if prev == HASH_EMPTY or prev == key:
            base = slot * 8
            hash_state[base + 0] = lam_n
            hash_state[base + 1] = lam_t
            hash_state[base + 2] = lam_b
            hash_state[base + 3] = k_n
            hash_state[base + 4] = k_t
            hash_state[base + 5] = k_b
            hash_state[base + 6] = float(was_t)
            hash_state[base + 7] = float(was_b)
            return


# =========================================================================
# Device-side body coloring (Jones–Plassmann parallel coloring).
# =========================================================================
# Rationale: round-1 coloring was static (initial AABB graph) and could
# race once moving bodies collided. Round-2 lifted that by recoloring
# each substep, but did so on the host with three full .numpy()
# transfers — breaking GPU-residency. Round-3 rebuilds the body-body
# adjacency CSR from the live `c_body_a`/`c_body_b` set and runs
# Jones–Plassmann device-side. No per-substep readbacks.
#
# Per substep the host launches, in order:
#   1. gpu_body_adj_reset           — zero counts/starts, mark all uncolored
#   2. gpu_body_adj_count           — atomic histogram (per active row)
#   3. gpu_csr_starts_from_counts   — single-thread serial scan (reused)
#   4. gpu_body_adj_scatter         — atomic scatter of neighbor body ids
#   5. gpu_color_round × JP_ROUNDS  — one Jones–Plassmann round each
#   6. gpu_color_counts_bin         — bincount body_color → color_counts
#   7. gpu_csr_starts_from_counts   — prefix sum → color_starts (reused)
#   8. gpu_color_bodies_scatter     — bucket-sort body ids into color_bodies
# All of these are fixed-dim launches that capture cleanly into a CUDA
# graph; the priorities are uploaded once at _flush so the per-substep
# coloring is deterministic and graph-replay-safe.


@wp.kernel
def gpu_body_adj_reset(
    body_neighbor_counts: wp.array(dtype=int),
    body_neighbor_cursor: wp.array(dtype=int),
    body_color: wp.array(dtype=int),
    color_counts: wp.array(dtype=int),
):
    """Zero per-body neighbor counts (input to the atomic histogram),
    zero the per-body scatter cursor, and mark every body uncolored
    (sentinel = -1). Launched at dim=max(n_b, MAX_COLORS) so the same
    kernel also zeros the color_counts histogram in the same pass.

    Threads 0..MAX_COLORS-1 zero the per-color count. Threads
    0..n_b-1 zero the per-body fields. Bounds-check both."""
    tid = wp.tid()
    if tid < color_counts.shape[0]:
        color_counts[tid] = 0
    if tid < body_neighbor_counts.shape[0]:
        body_neighbor_counts[tid] = 0
        body_neighbor_cursor[tid] = 0
        body_color[tid] = -1


@wp.kernel
def gpu_body_adj_count(
    n_active_rows: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    body_neighbor_counts: wp.array(dtype=int),
):
    """For every active row that names two distinct bodies, atomically
    bump both bodies' neighbor counts. PIN_6DOF excluded (its c_body_b
    is an axis id, not a body). May over-count when a body pair has
    multiple contacts; that's fine — duplicates don't break the
    coloring round (color-used masks are idempotent under duplicates)."""
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    if c_type[j] == PIN_6DOF:
        return
    a = c_body_a[j]
    b = c_body_b[j]
    if a < 0 or b < 0 or a == b:
        return
    wp.atomic_add(body_neighbor_counts, a, 1)
    wp.atomic_add(body_neighbor_counts, b, 1)


@wp.kernel
def gpu_body_adj_scatter(
    n_active_rows: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    body_neighbor_starts: wp.array(dtype=int),
    body_neighbor_cursor: wp.array(dtype=int),
    body_neighbor_indices: wp.array(dtype=int),
):
    """Symmetric scatter: each active row contributes (a, b) into a's
    neighbor list and (b, a) into b's. Cursor is per-body atomic so
    threads don't trample each other."""
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    if c_type[j] == PIN_6DOF:
        return
    a = c_body_a[j]
    b = c_body_b[j]
    if a < 0 or b < 0 or a == b:
        return
    slot_a = wp.atomic_add(body_neighbor_cursor, a, 1)
    body_neighbor_indices[body_neighbor_starts[a] + slot_a] = b
    slot_b = wp.atomic_add(body_neighbor_cursor, b, 1)
    body_neighbor_indices[body_neighbor_starts[b] + slot_b] = a


@wp.kernel
def gpu_color_round(
    body_priority: wp.array(dtype=float),
    body_neighbor_starts: wp.array(dtype=int),
    body_neighbor_counts: wp.array(dtype=int),
    body_neighbor_indices: wp.array(dtype=int),
    body_color: wp.array(dtype=int),
):
    """One Jones–Plassmann round.

    Each thread = one body. If we are still uncolored AND no uncolored
    neighbor has a higher priority than us, we win this round: pick the
    smallest color not used by any already-colored neighbor.

    Convergence: with random priorities and bounded max degree d, a
    body's expected number of rounds to be colored is O(log n / log(d))
    — typically ≤ 8 for contact graphs we see. JP_ROUNDS=24 is a
    generous worst-case bound; uncolored bodies after that fall through
    with color 0 (a final-pass fallback in gpu_color_finalize)."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    if body_color[i] != -1:
        return
    my_pri = body_priority[i]
    start = body_neighbor_starts[i]
    end = start + body_neighbor_counts[i]
    # Defer this round if any uncolored neighbor has a higher priority.
    for k in range(start, end):
        nb = body_neighbor_indices[k]
        if body_color[nb] == -1 and body_priority[nb] >= my_pri:
            # Tie-break by index so the (very unlikely) priority tie
            # doesn't deadlock the round.
            if body_priority[nb] > my_pri or nb < i:
                return
    # We're the winner — pick the smallest color not used by any
    # *already-colored* neighbor. Bitmask over [0, MAX_COLORS).
    used_lo = wp.uint32(0)   # bits 0..31 — covers MAX_COLORS ≤ 32
    for k in range(start, end):
        nb = body_neighbor_indices[k]
        c = body_color[nb]
        if c >= 0 and c < int(MAX_COLORS):
            used_lo = used_lo | (wp.uint32(1) << wp.uint32(c))
    # Scan for lowest unset bit in used_lo.
    chosen = int(MAX_COLORS) - 1
    for c in range(int(MAX_COLORS)):
        if (used_lo & (wp.uint32(1) << wp.uint32(c))) == wp.uint32(0):
            chosen = c
            break
    body_color[i] = chosen


@wp.kernel
def gpu_color_finalize(
    body_color: wp.array(dtype=int),
):
    """Catch any body that wasn't reached in JP_ROUNDS rounds — pin it
    to color 0. Only matters for pathological graphs (e.g. a very dense
    clique with many priority ties); the safety net keeps the partition
    well-defined even in those cases. Such a body becomes a
    constraint-correctness risk only if it shares an edge with another
    color-0 body, which is unlikely-but-possible — log if measured."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    if body_color[i] == -1:
        body_color[i] = 0


@wp.kernel
def gpu_color_counts_bin(
    body_color: wp.array(dtype=int),
    color_counts: wp.array(dtype=int),
):
    """Histogram body_color → color_counts via atomic_add."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    c = body_color[i]
    if c >= 0 and c < color_counts.shape[0]:
        wp.atomic_add(color_counts, c, 1)


@wp.kernel
def gpu_color_bodies_scatter(
    body_color: wp.array(dtype=int),
    color_starts: wp.array(dtype=int),
    color_cursor: wp.array(dtype=int),
    color_bodies: wp.array(dtype=int),
):
    """Bucket-sort: each body atomically claims a slot in its color's
    range of color_bodies. After this, primal_update can iterate the
    per-color slice via color_bodies[color_starts[c] + tid]."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    c = body_color[i]
    if c < 0 or c >= color_cursor.shape[0]:
        return
    slot = wp.atomic_add(color_cursor, c, 1)
    color_bodies[color_starts[c] + slot] = i


@wp.kernel
def gpu_max_color(
    body_color: wp.array(dtype=int),
    out_max_color: wp.array(dtype=int),   # shape [1], pre-zeroed
):
    """Reduce max(body_color) into out_max_color[0] (pre-zeroed by the
    caller). The *achieved* color count is out_max_color[0] + 1: colors
    are 0-based and both Jones–Plassmann and the speculative-greedy
    coloring assign a contiguous, gap-free range starting at 0, so the
    highest used color + 1 is exactly the number of non-empty colors.

    This is read back once per substep to bound the per-color primal loop
    (no point launching the empty tail up to MAX_COLORS) and to key the
    CUDA-graph signature so a changed count forces a recapture."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    c = body_color[i]
    if c >= 0:
        wp.atomic_max(out_max_color, 0, c)


# ---- Speculative ("Jacobi") greedy coloring (A2) ------------------------
# An alternative to Jones–Plassmann, switchable at runtime. Each round colors
# *every* uncolored body at once (Jacobi sweep) by first-fit, then un-colors
# the loser of any same-color adjacency. Converges in O(few) rounds for the
# sparse contact graphs we see and packs colors more tightly than JP, so the
# achieved color count (and thus the primal serialization chain) is usually
# smaller. Produces a valid (conflict-free) partition exactly like JP; only
# the assignment strategy differs — the AVBD solve that consumes it is
# unchanged. Shares the adjacency CSR and the bincount/scatter tail.
@wp.kernel
def gpu_color_spec_assign(
    body_neighbor_starts: wp.array(dtype=int),
    body_neighbor_counts: wp.array(dtype=int),
    body_neighbor_indices: wp.array(dtype=int),
    body_color: wp.array(dtype=int),
):
    """Assign phase: every still-uncolored body picks (in parallel, from
    the same snapshot) the smallest color not used by an already-colored
    neighbor. Two adjacent uncolored bodies may collide on a color this
    round — gpu_color_spec_resolve un-colors the lower-priority one."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    if body_color[i] != -1:
        return
    start = body_neighbor_starts[i]
    end = start + body_neighbor_counts[i]
    used_lo = wp.uint32(0)   # bits 0..31 — covers MAX_COLORS ≤ 32
    for k in range(start, end):
        nb = body_neighbor_indices[k]
        c = body_color[nb]
        if c >= 0 and c < int(MAX_COLORS):
            used_lo = used_lo | (wp.uint32(1) << wp.uint32(c))
    chosen = int(MAX_COLORS) - 1
    for c in range(int(MAX_COLORS)):
        if (used_lo & (wp.uint32(1) << wp.uint32(c))) == wp.uint32(0):
            chosen = c
            break
    body_color[i] = chosen


@wp.kernel
def gpu_color_spec_resolve(
    body_priority: wp.array(dtype=float),
    body_neighbor_starts: wp.array(dtype=int),
    body_neighbor_counts: wp.array(dtype=int),
    body_neighbor_indices: wp.array(dtype=int),
    body_color: wp.array(dtype=int),        # post-assign snapshot (in)
    body_color_next: wp.array(dtype=int),   # survivor coloring (out)
):
    """Conflict-resolution phase: reads the post-assign snapshot and
    writes survivors into `body_color_next` (double-buffered to avoid an
    in-place read/write race). A body keeps its color unless it shares it
    with a higher-priority neighbor (tie-break by lower index), in which
    case it is un-colored (-1) to retry next round. Exactly one body of
    each conflicting pair loses, so the survivors are conflict-free."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    ci = body_color[i]
    body_color_next[i] = ci
    if ci < 0:
        return
    my_pri = body_priority[i]
    start = body_neighbor_starts[i]
    end = start + body_neighbor_counts[i]
    for k in range(start, end):
        nb = body_neighbor_indices[k]
        if body_color[nb] == ci:
            if body_priority[nb] > my_pri or (body_priority[nb] == my_pri
                                              and nb < i):
                body_color_next[i] = -1
                return


@wp.kernel
def gpu_count_uncolored(
    body_color: wp.array(dtype=int),
    out_count: wp.array(dtype=int),   # shape [1], pre-zeroed
):
    """Count bodies still uncolored (body_color == -1) into out_count[0].
    Drives the speculative-coloring convergence loop: stop once zero."""
    i = wp.tid()
    if i >= body_color.shape[0]:
        return
    if body_color[i] == -1:
        wp.atomic_add(out_count, 0, 1)


@wp.kernel
def gpu_count_color_conflicts(
    n_active_rows: wp.array(dtype=int),
    c_type: wp.array(dtype=int),
    c_body_a: wp.array(dtype=int),
    c_body_b: wp.array(dtype=int),
    body_color: wp.array(dtype=int),
    out_conflicts: wp.array(dtype=int),   # shape [1], pre-zeroed
):
    """Diagnostic: count active body-body rows whose two bodies share a
    color (a coloring race in primal_update). Must be 0 for any valid
    coloring. Mirrors the edge definition in gpu_body_adj_count (skips
    PIN rows, whose c_body_b is an axis id, and self/invalid pairs)."""
    j = wp.tid()
    if j >= n_active_rows[0]:
        return
    if c_type[j] == PIN_6DOF:
        return
    a = c_body_a[j]
    b = c_body_b[j]
    if a < 0 or b < 0 or a == b:
        return
    if body_color[a] == body_color[b]:
        wp.atomic_add(out_conflicts, 0, 1)
