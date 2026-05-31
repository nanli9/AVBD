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
    dt: float,
    gravity: wp.vec3,
    # outputs
    x_initial: wp.array(dtype=wp.vec3),
    q_initial: wp.array(dtype=wp.quat),
    x_inertial: wp.array(dtype=wp.vec3),
    q_inertial: wp.array(dtype=wp.quat),
    inv_inertia_world: wp.array(dtype=wp.mat33),
    x_warm: wp.array(dtype=wp.vec3),
    q_warm: wp.array(dtype=wp.quat),
):
    i = wp.tid()
    m = mass[i]
    # Save initial state for BDF1 velocity finalize.
    x_initial[i] = x[i]
    q_initial[i] = q[i]
    # Cache R · I_local^{-1} · R^T — used by primal solve LHS. Identity for
    # static bodies; arbitrary mat33 otherwise.
    R = wp.quat_to_matrix(q[i])
    inv_inertia_world[i] = R * inv_inertia_local[i] * wp.transpose(R)

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
    x_inertial: wp.array(dtype=wp.vec3),
    q_inertial: wp.array(dtype=wp.quat),
    body_color: wp.array(dtype=int),
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
    # adjacency
    body_con_starts: wp.array(dtype=int),
    body_con_indices: wp.array(dtype=int),
    dt: float,
    current_color: int,
):
    i = wp.tid()
    if current_color != -1 and body_color[i] != current_color:
        return
    m = mass[i]
    if m <= 0.0:
        return

    inv_dt2 = 1.0 / (dt * dt)

    # M_world block-inverse view (we need M_world, not its inverse, on the LHS).
    # inv_inertia_world[i] = R · I_local^{-1} · R^T. The angular block of M is its
    # matrix inverse. Rather than invert again, build I_world by inverting back:
    I_world = wp.inverse(inv_inertia_world[i])
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

    start = body_con_starts[i]
    end = body_con_starts[i + 1]
    for k in range(start, end):
        cj = body_con_indices[k]
        if c_active[cj] == 0:
            continue
        t = c_type[cj]

        j_lin = wp.vec3(0.0, 0.0, 0.0)
        j_ang = wp.vec3(0.0, 0.0, 0.0)
        C = float(0.0)
        if t == FLOOR_CONTACT_6DOF:
            floor_y = c_world_anchor[cj][1]
            j_lin, j_ang = floor_J(q[i], c_off_a[cj])
            C = eval_floor_C(x[i], q[i], c_off_a[cj], floor_y)
        elif t == CONTACT_TANGENT_6DOF:
            tangent = c_world_anchor[cj]
            # Sibling-of-floor (single body) vs. sibling-of-box-box (two bodies)
            # — c_body_b[cj] = -1 marks the floor-friction case.
            bb = c_body_b[cj]
            if bb < 0:
                # Floor friction: C = t̂ · (x_a + R_a · off_a).
                j_lin, j_ang = tangent_J(q[i], c_off_a[cj], tangent)
                r_world = x[i] + wp.quat_rotate(q[i], c_off_a[cj])
                C = wp.dot(tangent, r_world)
            else:
                # Box-box friction: C = t̂ · (r_a − r_b). Per-body Jacobian
                # flips sign on the linear component and uses each body's own
                # body-local anchor for the torque arm.
                ba = c_body_a[cj]
                r_a = x[ba] + wp.quat_rotate(q[ba], c_off_a[cj])
                r_b = x[bb] + wp.quat_rotate(q[bb], c_off_b[cj])
                C = wp.dot(tangent, r_a - r_b)
                if ba == i:
                    j_lin, j_ang = tangent_J(q[i], c_off_a[cj], tangent)
                else:
                    j_lin = -tangent
                    j_ang = -wp.cross(wp.quat_rotate(q[i], c_off_b[cj]), tangent)
        elif t == PIN_6DOF:
            axis = c_body_b[cj]  # we encode the axis index here for PIN rows
            j_lin, j_ang = pin_axis_J(q[i], c_off_a[cj], axis)
            C = eval_pin_axis(x[i], q[i], c_off_a[cj], c_world_anchor[cj], axis)
        elif t == BOX_BOX_CONTACT_6DOF:
            # AVBD Eq. 15 normal row with body-local anchors:
            #   C = n̂ · (x_a + R_a·off_a − x_b − R_b·off_b)
            # n̂ is stored in c_world_anchor[cj] (held constant across the step,
            # cached at frame start by the CPU-side SAT).
            ba = c_body_a[cj]
            bb = c_body_b[cj]
            n_hat = c_world_anchor[cj]
            C = eval_box_box_C(x[ba], q[ba], c_off_a[cj],
                               x[bb], q[bb], c_off_b[cj], n_hat)
            if ba == i:
                j_lin = n_hat
                j_ang = wp.cross(wp.quat_rotate(q[i], c_off_a[cj]), n_hat)
            else:
                j_lin = -n_hat
                j_ang = -wp.cross(wp.quat_rotate(q[i], c_off_b[cj]), n_hat)

        # Eq. 18 stabilized C for hard constraints.
        s = c_stiffness[cj]
        if s >= wp.inf:
            C = C - c_alpha_C0[cj]

        lam_eff = c_lambda[cj]
        if not (s >= wp.inf):
            lam_eff = 0.0

        # Force magnitude — clamp differs for tangent (friction cone) vs others.
        if t == CONTACT_TANGENT_6DOF:
            sib = c_sibling[cj]
            mu = c_friction[cj]
            bound = mu * wp.abs(c_lambda[sib])
            f = wp.clamp(c_penalty[cj] * C + lam_eff, -bound, bound)
        else:
            f = wp.clamp(c_penalty[cj] * C + lam_eff, c_fmin[cj], c_fmax[cj])

        # Accumulate into LHS / RHS (6×6 outer J·k·J^T → split into the three
        # 3×3 blocks A, B, D).
        k_p = c_penalty[cj]
        A = A + outer3(j_lin, j_lin) * k_p
        B = B + outer3(j_ang, j_lin) * k_p
        D = D + outer3(j_ang, j_ang) * k_p
        # Geometric stiffness (AVBD Eq. 17, 2D ref solver.cpp:188).
        # For pin/floor/box-box rows, C is linear in x but quadratic-in-θ
        # via R(q)·off. The second derivative ∂²C/∂θ² has magnitude ~|r|·|f|
        # where r = R·off is the body-local offset rotated to world. Without
        # this term the iteration is Gauss-Newton — it ignores the curvature
        # of C in θ, so large angular updates overshoot, fail to converge,
        # and pump energy into rotation each frame (a pinned cube spins up
        # to the angular-speed cap within ~0.5 s). We use the 2D ref's
        # diagonal-lumped approximation: add |r|·|f|·I to the D block.
        # Skip tangent rows (the friction force already saturates at μ·|λ_n|
        # so |f| stays bounded and G would add unnecessary stiffness).
        # Geometric stiffness G — AVBD Eq. 17 / 2D ref solver.cpp:188.
        # ONLY applied to PIN_6DOF. The 2D reference deliberately discards
        # the second-order term for contact manifolds (manifold.cpp:75
        # comment: "we discard the second order term, since it is
        # insignificant for contacts"). For joints/pins it is essential —
        # without it large angular motion produces a poor linearization
        # and a free-rotation pin spins itself up to the cap within ~1 s.
        # For contacts (floor / box-box) adding G actually injects energy
        # via over-stiffening of the angular block under corner-contact
        # with body spin; matches what we observed empirically.
        if t == PIN_6DOF:
            r_off = wp.quat_rotate(q[i], c_off_a[cj])
            # Per-axis lumped diagonal of |f|·H_θθ. For pin row k
            # (e_axis = e_k): H_ij = (1/2)(δ_ik off[j] + δ_jk off[i])
            #                       − off[k] δ_ij. Column norms below.
            f_mag = wp.abs(f)
            gx = (wp.abs(r_off[1]) + wp.abs(r_off[2])) * 0.5 * f_mag
            gy = (wp.abs(r_off[0]) + wp.abs(r_off[2])) * 0.5 * f_mag
            gz = (wp.abs(r_off[0]) + wp.abs(r_off[1])) * 0.5 * f_mag
            D = D + wp.mat33(gx, 0.0, 0.0,
                             0.0, gy, 0.0,
                             0.0, 0.0, gz)
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
# Dual update (AVBD Eq. 11 + Eq. 16) — same structure as 3-DOF
# -----------------------------------------------------------------------------
@wp.kernel
def dual_update_6dof(
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
    beta: float,
):
    j = wp.tid()
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
    if t == CONTACT_TANGENT_6DOF:
        bound = c_friction[j] * wp.abs(c_lambda[c_sibling[j]])
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


# -----------------------------------------------------------------------------
# α·C₀ cache for hard constraints (Eq. 18) — uses POSITION AT FRAME START
# -----------------------------------------------------------------------------
@wp.kernel
def cache_alpha_C0_6dof(
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
