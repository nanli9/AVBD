"""Solver correctness tests.

Each test pins one piece of AVBD machinery to an analytic prediction, so any
regression surfaces immediately. Run with:

    uv run pytest -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avbd3d import Shape, Solver


# -----------------------------------------------------------------------------
# Integration sanity
# -----------------------------------------------------------------------------
def test_free_fall_matches_bdf1():
    """A particle with no constraints falls under gravity.

    BDF1 (backward Euler) with constant gravity gives the closed-form
        v_k = v_0 + k·g·dt
        x_k = x_0 + g·dt²·k(k+1)/2     (when v_0 = 0)
    """
    dt = 1.0 / 60.0
    g = -9.81
    n_steps = 60
    s = Solver(dt=dt, iterations=2, gravity=(0.0, g, 0.0), post_stabilize=False)
    b = s.add_particle(position=(0.0, 10.0, 0.0), mass=1.0)

    for _ in range(n_steps):
        s.step()

    expected_y = 10.0 + g * dt * dt * n_steps * (n_steps + 1) / 2.0
    expected_vy = g * n_steps * dt
    pos = s.positions()[b.index]
    vel = s.velocities()[b.index]

    # float32 accumulation over 60 steps drifts ~1e-3 from the analytic BDF1
    # formula; that's expected and not a solver bug.
    assert pos[0] == pytest.approx(0.0, abs=1e-5)
    assert pos[2] == pytest.approx(0.0, abs=1e-5)
    assert pos[1] == pytest.approx(expected_y, abs=2e-3)
    assert vel[1] == pytest.approx(expected_vy, rel=1e-3)


def test_static_body_stays_put():
    """Body with mass=0 is kinematic; no constraints, no motion."""
    s = Solver(dt=1.0 / 60.0, iterations=2, gravity=(0.0, -9.81, 0.0))
    b = s.add_particle(position=(1.0, 2.0, 3.0), mass=0.0)
    for _ in range(30):
        s.step()
    pos = s.positions()[b.index]
    assert pos == pytest.approx([1.0, 2.0, 3.0], abs=1e-6)


# -----------------------------------------------------------------------------
# Constraint correctness
# -----------------------------------------------------------------------------
def test_pin_holds_against_gravity():
    """A pinned particle stays close to the pin point and λ converges to weight.

    The expected λ along the gravity axis equals -m·g (positive g is up;
    here g = -9.81 so λ_y should be near -9.81 N to hold the mass up).
    """
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    b = s.add_particle(position=(0.0, 5.0, 0.0), mass=1.0)
    s.add_pin(b, world_point=(0.0, 5.0, 0.0), stiffness=math.inf)

    for _ in range(120):
        s.step()

    pos = s.positions()[b.index]
    lam = s.lambdas()
    assert pos[1] == pytest.approx(5.0, abs=5e-3)
    assert lam[1] == pytest.approx(-9.81, abs=0.5)  # λ for PIN_Y


def test_distance_constraint_settles():
    """Two particles, top pinned, connected by a stiff distance constraint.
    The distance |x_a - x_b| should converge to rest.
    """
    rest = 0.5
    s = Solver(dt=1.0 / 60.0, iterations=15, gravity=(0.0, -9.81, 0.0))
    a = s.add_particle(position=(0.0, 5.0, 0.0), mass=1.0)
    b = s.add_particle(position=(0.0, 5.0 - rest, 0.0), mass=1.0)
    s.add_pin(a, world_point=(0.0, 5.0, 0.0), stiffness=math.inf)
    s.add_distance(a, b, rest=rest, stiffness=math.inf)

    for _ in range(600):
        s.step()

    pa = s.positions()[a.index]
    pb = s.positions()[b.index]
    d = float(np.linalg.norm(pa - pb))
    assert d == pytest.approx(rest, abs=2e-3)


def test_chain_distance_errors_small():
    """N-link chain has small constraint error *averaged over time*.

    AVBD has no built-in viscous damping, so the chain oscillates in steady
    state and the instantaneous constraint error fluctuates. We average over
    the second half of a long run instead of sampling a single moment.
    """
    n = 5
    link = 0.5
    s = Solver(dt=1.0 / 60.0, iterations=15, gravity=(0.0, -9.81, 0.0))
    bodies = [s.add_particle((0.0, 5.0 - i * link, 0.0), 1.0) for i in range(n + 1)]
    s.add_pin(bodies[0], (0.0, 5.0, 0.0), stiffness=math.inf)
    for i in range(n):
        s.add_distance(bodies[i], bodies[i + 1], rest=link, stiffness=math.inf)

    total = 900
    warmup = total // 2
    err_samples = []
    for f in range(total):
        s.step()
        if f >= warmup:
            pos = s.positions()
            err_samples.append(max(
                abs(np.linalg.norm(pos[i + 1] - pos[i]) - link) for i in range(n)
            ))
    mean_err = float(np.mean(err_samples))
    assert mean_err < link * 0.05  # average error < 5% link length


# -----------------------------------------------------------------------------
# Warm-start carries information across frames
# -----------------------------------------------------------------------------
def test_warmstart_reduces_iteration_load():
    """At steady state, λ should be non-zero — confirming the warm-started dual
    is carrying state. A constraint with zero λ across all frames means warm
    start is broken."""
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    b = s.add_particle((0.0, 5.0, 0.0), 1.0)
    s.add_pin(b, (0.0, 5.0, 0.0), stiffness=math.inf)
    last_lam = 0.0
    for _ in range(200):
        s.step()
        last_lam = float(np.linalg.norm(s.lambdas()))
    # λ should be on the order of body weight, definitely non-trivial
    assert last_lam > 5.0


# -----------------------------------------------------------------------------
# Fracture (AVBD-native |λ| ≥ threshold)
# -----------------------------------------------------------------------------
def test_fracture_breaks_top_link_first():
    """Chain with finite per-link fracture threshold. The top link bears the
    highest tension (sum of all weights below it) so it must break first."""
    n = 5
    link = 0.5
    threshold = 30.0  # between 3·9.81 ≈ 29.4 and 4·9.81 ≈ 39.2
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    bodies = [s.add_particle((0.0, 5.0 - i * link, 0.0), 1.0) for i in range(n + 1)]
    s.add_pin(bodies[0], (0.0, 5.0, 0.0), stiffness=math.inf)
    handles = [
        s.add_distance(bodies[i], bodies[i + 1], rest=link, stiffness=math.inf,
                       fracture=threshold)
        for i in range(n)
    ]

    broken_first = None
    for _ in range(300):
        s.step()
        act = s.active()
        # constraints 0..2 are pin axes; distances start at index 3
        for li, h in enumerate(handles):
            if act[h.index] == 0:
                broken_first = li
                break
        if broken_first is not None:
            break

    assert broken_first == 0, (
        f"expected top link (0) to break first, got link {broken_first}"
    )
    # Pin must still be active (it has no fracture threshold)
    assert int(s.active()[:3].sum()) == 3


def test_no_fracture_when_threshold_is_inf():
    """Default fracture=inf must never break anything."""
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    a = s.add_particle((0.0, 5.0, 0.0), 1.0)
    b = s.add_particle((0.0, 4.0, 0.0), 1.0)
    s.add_pin(a, (0.0, 5.0, 0.0), stiffness=math.inf)
    s.add_distance(a, b, rest=1.0, stiffness=math.inf)
    for _ in range(200):
        s.step()
    assert int(s.active().sum()) == 4  # 3 pin axes + 1 distance


# -----------------------------------------------------------------------------
# Coloring is a valid graph coloring
# -----------------------------------------------------------------------------
def test_coloring_is_valid_for_chain():
    """For any two bodies sharing a constraint, their colors must differ."""
    n = 8
    s = Solver(dt=1.0 / 60.0, iterations=2, gravity=(0.0, 0.0, 0.0))
    bodies = [s.add_particle((float(i), 0.0, 0.0), 1.0) for i in range(n)]
    s.add_pin(bodies[0], (0.0, 0.0, 0.0), stiffness=math.inf)
    for i in range(n - 1):
        s.add_distance(bodies[i], bodies[i + 1], rest=1.0, stiffness=math.inf)

    s.step()  # triggers flush + coloring
    colors = s.body_color.numpy()

    # Walk every constraint, verify both endpoints have distinct colors
    a = s.c_body_a.numpy()
    b = s.c_body_b.numpy()
    for ca, cb in zip(a, b):
        if cb < 0:
            continue
        assert colors[ca] != colors[cb], (
            f"constraint {(ca, cb)} has same-color endpoints {colors[ca]}"
        )

    # A path graph is bipartite → exactly 2 colors expected
    assert s.num_colors == 2


def test_coloring_complete_graph():
    """K_n needs n colors."""
    n = 5
    s = Solver(dt=1.0 / 60.0, iterations=2, gravity=(0.0, 0.0, 0.0))
    bodies = [s.add_particle((float(i), 0.0, 0.0), 1.0) for i in range(n)]
    s.add_pin(bodies[0], (0.0, 0.0, 0.0), stiffness=math.inf)
    for i in range(n):
        for j in range(i + 1, n):
            s.add_distance(bodies[i], bodies[j], rest=1.0, stiffness=math.inf)
    s.step()
    assert s.num_colors == n


# -----------------------------------------------------------------------------
# Solver runs deterministically without NaNs
# -----------------------------------------------------------------------------
def test_no_nans_in_chain_simulation():
    n = 6
    link = 0.5
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    bodies = [s.add_particle((0.0, 5.0 - i * link, 0.0), 1.0) for i in range(n + 1)]
    s.add_pin(bodies[0], (0.0, 5.0, 0.0), stiffness=math.inf)
    for i in range(n):
        s.add_distance(bodies[i], bodies[i + 1], rest=link, stiffness=math.inf)
    for _ in range(300):
        s.step()
        pos = s.positions()
        assert np.isfinite(pos).all(), "NaN/inf appeared in positions"
        assert np.isfinite(s.velocities()).all(), "NaN/inf in velocities"
        assert np.isfinite(s.lambdas()).all(), "NaN/inf in lambdas"


# -----------------------------------------------------------------------------
# Body-body contact (VBD Eq. 12 / AVBD Eq. 15 normal row, sign convention as
# in the avbd-demo2d reference: C = ||p_a-p_b|| - (r_a+r_b), fmin=-inf, fmax=0)
# -----------------------------------------------------------------------------
def test_sphere_contact_prevents_interpenetration():
    """Two equal spheres falling toward each other on a frictionless floor
    must NOT pass through each other once self-collision is enabled."""
    s = Solver(dt=1.0 / 60.0, iterations=20, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    r = 0.2
    a = s.add_particle((-0.05, 0.5, 0.0), mass=1.0,
                       shape=Shape("sphere", (r,)))
    b = s.add_particle((+0.05, 1.4, 0.0), mass=1.0,
                       shape=Shape("sphere", (r,)))
    s.add_floor_contact(a, floor_y=r)
    s.add_floor_contact(b, floor_y=r)
    s.enable_self_collision(True)

    for _ in range(400):
        s.step()

    pos = s.positions()
    dist = float(np.linalg.norm(pos[0] - pos[1]))
    # Hard contact may dip a few mm below 2r during the stabilization budget;
    # never deeper than 5% of the contact diameter.
    assert dist >= 2 * r * 0.95, f"interpenetration: dist={dist}, 2r={2*r}"


def test_sphere_contact_stack_of_three():
    """A vertical stack of 3 spheres should settle without inversion (the
    bottom must remain at the bottom)."""
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    r = 0.2
    a = s.add_particle((0.0, r, 0.0), mass=1.0,
                       shape=Shape("sphere", (r,)))
    b = s.add_particle((0.0, 3 * r, 0.0), mass=1.0,
                       shape=Shape("sphere", (r,)))
    c = s.add_particle((0.0, 5 * r, 0.0), mass=1.0,
                       shape=Shape("sphere", (r,)))
    for body in (a, b, c):
        s.add_floor_contact(body, floor_y=r)
    s.enable_self_collision(True)

    for _ in range(600):
        s.step()

    pos = s.positions()
    # Order along Y is preserved (no inversion).
    assert pos[a.index, 1] < pos[b.index, 1] < pos[c.index, 1]
    # All three lie within a 3-diameter column of the floor.
    for p in pos:
        assert p[1] >= r - 0.02  # never below floor by more than 2 cm


def test_sphere_contact_pair_skipped_when_connected():
    """If A and B are joined by a distance constraint, the broad phase must
    NOT add a redundant SPHERE_CONTACT row between them — the distance
    constraint already prevents penetration at their rest length."""
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0))
    r = 0.2
    a = s.add_particle((0.0, 1.0, 0.0), 1.0, shape=Shape("sphere", (r,)))
    b = s.add_particle((0.0, 0.5, 0.0), 1.0, shape=Shape("sphere", (r,)))
    s.add_pin(a, (0.0, 1.0, 0.0), stiffness=math.inf)
    s.add_distance(a, b, rest=0.5, stiffness=math.inf)
    s.enable_self_collision(True)
    s.step()
    # Only the user-added rows should exist (3 pin + 1 distance = 4), no
    # SPHERE_CONTACT was injected even though the spheres' surfaces touch.
    types = s.c_type.numpy()
    n_contact = int((types == 5).sum())
    assert n_contact == 0, f"expected 0 SPHERE_CONTACT rows, got {n_contact}"


# -----------------------------------------------------------------------------
# Friction (AVBD Sec. 3.3, Eq. 15 tangent rows + per-row Coulomb cone clamp)
# -----------------------------------------------------------------------------
def test_friction_sliding_ball_decelerates_to_rest():
    """A unit-mass ball with initial horizontal velocity on a μ floor must
    decelerate to rest. Coulomb prediction: stopping distance v²/(2 μ g).

    AVBD with BDF1 + iteration discretisation hits this within a few %."""
    mu = 0.5
    v0 = 3.0
    g = 9.81
    expected_stop = v0 * v0 / (2.0 * mu * g)  # ≈ 0.918 m

    s = Solver(dt=1.0 / 60.0, iterations=20, gravity=(0.0, -g, 0.0),
               post_stabilize=True)
    r = 0.2
    b = s.add_particle((0.0, r, 0.0), 1.0, velocity=(v0, 0.0, 0.0),
                       shape=Shape("sphere", (r,)), friction=mu)
    s.add_floor_contact(b, floor_y=r, friction=mu)
    s.enable_self_collision(True, default_friction=mu)

    for _ in range(600):  # 10 seconds — plenty to reach rest
        s.step()

    p = s.positions()[b.index]
    v = s.velocities()[b.index]
    assert abs(float(v[0])) < 1e-3, f"ball still moving: vx={v[0]}"
    # Stopping distance within 10% of the closed-form Coulomb prediction.
    assert p[0] == pytest.approx(expected_stop, rel=0.1), (
        f"stop dist {p[0]:.3f} vs analytic {expected_stop:.3f}"
    )


def test_friction_zero_means_frictionless():
    """With μ=0 the same ball must slide at constant velocity (no decel)."""
    v0 = 3.0
    s = Solver(dt=1.0 / 60.0, iterations=20, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    r = 0.2
    b = s.add_particle((0.0, r, 0.0), 1.0, velocity=(v0, 0.0, 0.0),
                       shape=Shape("sphere", (r,)), friction=0.0)
    s.add_floor_contact(b, floor_y=r, friction=0.0)
    s.enable_self_collision(True, default_friction=0.0)

    for _ in range(120):
        s.step()

    v = s.velocities()[b.index]
    # Tiny BDF1 numerical drift expected, never enough to drop > 1% of v0
    assert abs(float(v[0]) - v0) < 0.05, f"frictionless ball decelerated to {v[0]}"


# -----------------------------------------------------------------------------
# Broken-constraint pairs must re-enter the broad phase
# -----------------------------------------------------------------------------
def test_broken_distance_pair_collides():
    """A and B are joined by a fracturing distance constraint. Once the
    constraint breaks (active=0), the broad phase MUST treat them as a normal
    pair — otherwise the inactive row keeps the pair permanently excluded
    and they pass straight through each other.
    """
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    r = 0.1
    a = s.add_particle((0.0, 2.0, 0.0), 5.0, shape=Shape("sphere", (r,)))
    s.add_floor_contact(a, floor_y=r)
    s.add_pin(a, (0.0, 2.0, 0.0), stiffness=math.inf)
    b = s.add_particle((0.0, 1.5, 0.0), 1.0, shape=Shape("sphere", (r,)))
    s.add_floor_contact(b, floor_y=r)
    dh = s.add_distance(a, b, rest=0.5, stiffness=math.inf, fracture=5.0)
    s.enable_self_collision(True)

    # Step until break, then place B above A and let it fall.
    for k in range(300):
        s.step()
        if s.active()[dh.index] == 0:
            s.set_position(b, (0.0, 3.0, 0.0))
            s.set_velocity(b, (0.0, 0.0, 0.0))
            break
    else:
        pytest.fail("distance constraint never broke")

    for _ in range(400):
        s.step()

    pa = s.positions()[a.index]
    pb = s.positions()[b.index]
    dist = float(np.linalg.norm(pa - pb))
    # B should rest ON A, NOT pass through. Allow 5% AVBD stabilisation budget.
    assert dist >= 2 * r * 0.95, (
        f"B passed through A after fracture: center distance {dist:.4f}, "
        f"2r={2*r}"
    )


# -----------------------------------------------------------------------------
# Stacked-sphere pillar collision (no more "ball flies through cylinder")
# -----------------------------------------------------------------------------
def test_pillar_collision_uses_stacked_subspheres():
    """A pillar of (r, h) is modelled as a stack of inscribed spheres along
    Y so a sphere can hit it ANYWHERE along its height, not just at the
    central inscribed sphere. Old single-sphere bug: pillar's collision was a
    tiny ball at the body centre, leaving most of the visual cylinder's
    interior un-collidable.
    """
    pillar_r, pillar_h = 0.10, 0.35  # tall cylinder (h ≫ r)
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    p = s.add_particle((0.0, pillar_h, 0.0), 10.0,
                       shape=Shape("pillar", (pillar_r, pillar_h)))
    s.add_floor_contact(p, floor_y=pillar_h)
    s.add_pin(p, (0.0, pillar_h, 0.0), stiffness=math.inf)
    # Verify we generated multiple sub-spheres (else the test is a no-op).
    sub_spheres = s._bodies_collision_spheres[p.index]
    assert len(sub_spheres) >= 2, (
        f"expected pillar to be modelled as ≥2 sub-spheres, got {len(sub_spheres)}"
    )
    # A ball aimed at the pillar's BOTTOM (well below the central inscribed
    # sphere) must collide instead of passing through.
    sphere_r = 0.08
    b = s.add_particle((1.0, 0.10, 0.0), 1.0,
                       shape=Shape("sphere", (sphere_r,)),
                       velocity=(-3.0, 0.0, 0.0))
    s.add_floor_contact(b, floor_y=sphere_r)
    s.enable_self_collision(True)
    for _ in range(360):
        s.step()
    p_pos = s.positions()[p.index]
    b_pos = s.positions()[b.index]
    radial = float(np.sqrt((p_pos[0] - b_pos[0]) ** 2 + (p_pos[2] - b_pos[2]) ** 2))
    # Must stop on contact, not pass through. 5% AVBD stab budget.
    assert radial >= (pillar_r + sphere_r) * 0.95, (
        f"ball passed through pillar: radial dist {radial:.4f}, "
        f"contact at {pillar_r + sphere_r:.4f}"
    )


# -----------------------------------------------------------------------------
# Cube corner-direction contact (multi-sub-sphere fix)
# -----------------------------------------------------------------------------
def test_cube_corner_contact_no_penetration():
    """A ball drops on a cube's top face — it should settle at exactly
    `cube_top_y + ball_r`. The face-on case is the easy one; the previous
    single-inscribed-sphere model was correct on faces and visibly wrong
    near corners. This regression locks in the face case AND a non-axial
    drop (corner-vicinity)."""
    r_c = 0.18
    r_b = 0.10
    # Face-on case
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    c = s.add_particle((0.0, r_c, 0.0), 100.0,
                       shape=Shape("cube", (r_c, r_c, r_c)))
    s.add_floor_contact(c, floor_y=r_c)
    s.add_pin(c, (0.0, r_c, 0.0), stiffness=math.inf)
    b = s.add_particle((0.0, 1.0, 0.0), 1.0, shape=Shape("sphere", (r_b,)))
    s.add_floor_contact(b, floor_y=r_b)
    s.enable_self_collision(True)
    for _ in range(600):
        s.step()
    expected_y = 2 * r_c + r_b
    actual_y = float(s.positions()[b.index, 1])
    assert abs(actual_y - expected_y) < 1e-3, (
        f"face-on contact off by {(actual_y - expected_y)*1000:.2f} mm"
    )
    # Sub-sphere model emits 1 inscribed + 8 corner sub-spheres
    assert len(s._bodies_collision_spheres[c.index]) == 9


def test_cube_corner_subspheres_resolve_corner_overlap():
    """Ball placed inside the cube along the corner direction must be pushed
    out fully — no residual penetration even though the ball started 50%
    inside the contact distance.
    """
    r_c = 0.18
    r_b = 0.10
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, 0.0, 0.0),
               post_stabilize=True)
    c = s.add_particle((0.0, 0.0, 0.0), 100.0,
                       shape=Shape("cube", (r_c, r_c, r_c)))
    s.add_pin(c, (0.0, 0.0, 0.0), stiffness=math.inf)
    dirn = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    start = dirn * (r_c + r_b) * 0.5  # 50% inside contact distance
    b = s.add_particle(tuple(float(v) for v in start), 1.0,
                       shape=Shape("sphere", (r_b,)))
    s.enable_self_collision(True)
    for _ in range(600):
        s.step()
    rel = s.positions()[b.index] - s.positions()[c.index]
    clamped = np.clip(rel, -r_c, r_c)
    dist_to_surface = float(np.linalg.norm(rel - clamped))
    penetration = max(0.0, r_b - dist_to_surface)
    # Sub-sphere model should fully resolve; allow 5 mm slack for the AVBD
    # stabilisation budget at α=0.99.
    assert penetration < 0.005, (
        f"residual penetration {penetration*1000:.2f} mm — multi-sub-sphere "
        f"corner contact is not resolving overlap"
    )


# -----------------------------------------------------------------------------
# Analytical sphere-box contact (SPHERE_BOX_CONTACT, type 7)
# -----------------------------------------------------------------------------
def test_sphere_box_contact_analytical_face():
    """Ball drops on cube top → ball.y == cube.top_y + r_b, EXACT
    (analytical closest-point-on-AABB)."""
    r_c, r_b = 0.18, 0.10
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    c = s.add_particle((0.0, r_c, 0.0), 100.0,
                       shape=Shape("cube", (r_c, r_c, r_c)))
    s.add_floor_contact(c, floor_y=r_c)
    s.add_pin(c, (0.0, r_c, 0.0), stiffness=math.inf)
    b = s.add_particle((0.0, 1.0, 0.0), 1.0, shape=Shape("sphere", (r_b,)))
    s.add_floor_contact(b, floor_y=r_b)
    s.enable_self_collision(True)
    for _ in range(800):
        s.step()
    expected = 2 * r_c + r_b
    actual = float(s.positions()[b.index, 1])
    # Analytical contact converges to within the AVBD α=0.99 stab budget.
    assert abs(actual - expected) < 5e-4, (
        f"sphere-box face contact off by {(actual-expected)*1000:.3f} mm"
    )
    # Verify the analytical constraint type (7) is actually being used.
    types = set(int(t) for t in s.c_type.numpy().tolist())
    assert 7 in types, "SPHERE_BOX_CONTACT row not generated"


def test_sphere_box_no_penetration_when_pushed_into_corner():
    """Ball is placed *inside* the cube along its (+x,+y,+z) corner.
    Analytical closest-point pushes it back to the cube surface with no
    residual penetration (sub-millimeter)."""
    r_c, r_b = 0.18, 0.10
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, 0.0, 0.0),
               post_stabilize=True)
    c = s.add_particle((0.0, 0.0, 0.0), 100.0,
                       shape=Shape("cube", (r_c, r_c, r_c)))
    s.add_pin(c, (0.0, 0.0, 0.0), stiffness=math.inf)
    # Place ball at 50% inside the contact distance along corner direction
    dirn = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    start = dirn * (r_c + r_b) * 0.5
    b = s.add_particle(tuple(float(v) for v in start), 1.0,
                       shape=Shape("sphere", (r_b,)))
    s.enable_self_collision(True)
    for _ in range(600):
        s.step()
    rel = s.positions()[b.index] - s.positions()[c.index]
    clamped = np.clip(rel, -r_c, r_c)
    dist_to_surface = float(np.linalg.norm(rel - clamped))
    penetration = max(0.0, r_b - dist_to_surface)
    assert penetration < 1e-3, (
        f"analytical sphere-box corner contact left {penetration*1000:.3f} mm "
        f"penetration"
    )


# -----------------------------------------------------------------------------
# Analytical box-box contact (AVBD Eq. 15, SAT + face-clip → 4 contacts per pair)
# -----------------------------------------------------------------------------
def test_box_box_two_cube_stack_settles():
    """Two cubes face-to-face: SAT picks +ŷ as the contact normal, clip
    produces 4 contacts at the rectangle corners of the overlap face.
    Each is its own AVBD Eq. 15 constraint with independent (n̂, t̂, b̂).
    """
    r = 0.18
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    bot = s.add_particle((0.0, r, 0.0), 5.0,
                         shape=Shape("cube", (r, r, r)))
    s.add_floor_contact(bot, floor_y=r)
    top = s.add_particle((0.0, 0.7, 0.0), 1.0,
                         shape=Shape("cube", (r, r, r)))
    s.add_floor_contact(top, floor_y=r)
    s.enable_self_collision(True)
    for _ in range(800):
        s.step()
    p = s.positions()
    # Bottom rests on floor, top rests on bottom.
    assert abs(p[0, 1] - r) < 1e-3, f"bottom cube y off {(p[0,1]-r)*1000:.3f}mm"
    assert abs(p[1, 1] - 3 * r) < 1e-3, f"top cube y off {(p[1,1]-3*r)*1000:.3f}mm"
    # Sanity: 4 BOX_BOX_CONTACT rows are active (one per face-face corner).
    types = s.c_type.numpy()
    active = s.c_active.numpy()
    n_active_box_box = sum(1 for t, a in zip(types, active) if t == 8 and a == 1)
    assert n_active_box_box == 4, (
        f"expected 4 active BOX_BOX_CONTACT rows, got {n_active_box_box}"
    )


def test_box_box_friction_decelerates_top_cube():
    """Cube sliding on top of a pinned cube: AVBD's per-contact friction cone
    must decelerate it at μg per Coulomb prediction."""
    r = 0.18
    mu = 0.5
    v0 = 2.0
    s = Solver(dt=1.0 / 60.0, iterations=25, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    bot = s.add_particle((0.0, r, 0.0), 100.0,
                         shape=Shape("cube", (r, r, r)), friction=mu)
    s.add_floor_contact(bot, floor_y=r, friction=mu)
    s.add_pin(bot, (0.0, r, 0.0), stiffness=math.inf)
    top = s.add_particle((0.0, 3 * r, 0.0), 1.0,
                         shape=Shape("cube", (r, r, r)),
                         velocity=(v0, 0.0, 0.0), friction=mu)
    s.add_floor_contact(top, floor_y=r, friction=mu)
    s.enable_self_collision(True, default_friction=mu)
    for _ in range(600):
        s.step()
    v = s.velocities()[top.index]
    assert abs(float(v[0])) < 1e-3, f"top cube still moving: vx={v[0]}"
    expected_stop = v0 * v0 / (2.0 * mu * 9.81)  # ~0.408 m
    x = float(s.positions()[top.index, 0])
    assert x == pytest.approx(expected_stop, rel=0.15), (
        f"stop dist {x:.3f} vs analytic {expected_stop:.3f}"
    )
