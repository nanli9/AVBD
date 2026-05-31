"""Tests for the 6-DOF rigid-body Solver6DOF.

Covers:
  - Free fall preserves linear momentum (constant a = -g).
  - Free spin in zero-g preserves angular velocity (and orientation
    integrates correctly: q(t) = exp_q(ω·t/2) ⊗ q_0).
  - Box dropped on floor settles at y = half-extent.
  - Spinning box on floor: friction torque brings ω to zero.
  - Tilted box dropped on floor: settles flat without runaway.
  - Box-corner pin holds against gravity.
"""

import math
import numpy as np
import pytest

from avbd3d import Solver6DOF


def test_free_fall_constant_acceleration():
    """No constraints, only gravity. v_y after t seconds should equal -g·t."""
    s = Solver6DOF(dt=1/60, iterations=5)
    b = s.add_box((0., 5., 0.), (0.5, 0.5, 0.5), mass=1.0)
    for _ in range(60):  # 1 sec
        s.step()
    vy = s.velocities()[0][1]
    assert vy == pytest.approx(-9.81, rel=1e-3), \
        f"after 1 sec free fall expected v_y ≈ -9.81, got {vy}"


def test_free_spin_preserves_angular_velocity():
    """Zero-g spinning box: ω should stay constant; q should integrate to
    the analytic axis-angle rotation."""
    s = Solver6DOF(dt=1/60, iterations=5, gravity=(0., 0., 0.))
    b = s.add_box((0., 0., 0.), (0.5, 0.5, 0.5), mass=1.0,
                  angular_velocity=(0., 0., 2.0))
    for _ in range(60):  # 1 sec → 2 rad about z
        s.step()
    w = s.angular_velocities()[0]
    assert np.linalg.norm(w - np.array([0., 0., 2.0])) < 1e-3, \
        f"ω drift in zero-g: {w}"
    q = s.orientations()[0]  # xyzw
    # exp_q(ω·t/2) ⊗ q₀ = (sin(1.0)·0, sin(1.0)·0, sin(1.0)·1, cos(1.0))
    expected = np.array([0., 0., math.sin(1.0), math.cos(1.0)], dtype=np.float32)
    # Allow either sign of q (quaternions are double cover of SO(3)).
    err = min(np.linalg.norm(q - expected), np.linalg.norm(q + expected))
    assert err < 5e-3, f"orientation drift: got {q}, expected {expected}"


def test_box_drops_on_floor_settles_at_half_extent():
    h = 0.3
    s = Solver6DOF(dt=1/60, iterations=20)
    b = s.add_box((0., 2.0, 0.), (h, h, h), mass=1.0, friction=0.5)
    s.add_floor_contact_box(b, floor_y=0.0, friction=0.5)
    for _ in range(300):  # 5 sec
        s.step()
    y = s.positions()[0][1]
    vy = s.velocities()[0][1]
    assert abs(y - h) < 0.01, f"box should sit at y≈{h}, got {y}"
    assert abs(vy) < 0.1, f"box should be at rest, got v_y={vy}"


def test_spinning_box_on_floor_friction_brakes_omega():
    """Box spinning about world-y while sitting on the floor — friction at the
    contact points produces a torque that brakes the spin to zero."""
    s = Solver6DOF(dt=1/60, iterations=25)
    h = 0.3
    b = s.add_box((0., h + 0.15, 0.), (h, h, h), mass=1.0, friction=0.8,
                  angular_velocity=(0., 5.0, 0.))
    s.add_floor_contact_box(b, friction=0.8)
    for _ in range(300):  # 5 sec
        s.step()
    w = s.angular_velocities()[0]
    assert np.linalg.norm(w) < 0.1, f"friction should brake spin, |ω|={np.linalg.norm(w)}"
    # Should also be resting on the floor.
    y = s.positions()[0][1]
    assert abs(y - h) < 0.02, f"box should be at y≈{h}, got {y}"


def test_tilted_box_drops_and_settles():
    """Box released at a 30° tilt should land, may slide a bit, but should
    end up at rest with ω ≈ 0 and bottom face on floor. Needs substeps for
    stiff corner-on-floor contact to converge — AVBD paper Fig.6 uses 5
    substeps for similar stiff scenarios."""
    s = Solver6DOF(dt=1/60, iterations=15, substeps=8)
    h = 0.3
    ang = math.radians(30)
    q0 = (math.sin(ang/2), 0., 0., math.cos(ang/2))
    b = s.add_box((0., 1.5, 0.), (h, h, h), mass=1.0, friction=0.5,
                  orientation=q0)
    s.add_floor_contact_box(b, friction=0.5)
    for _ in range(300):  # 5 sec
        s.step()
    w = s.angular_velocities()[0]
    v = s.velocities()[0]
    assert np.linalg.norm(w) < 0.05, f"should be at rest, |ω|={np.linalg.norm(w)}"
    assert np.linalg.norm(v) < 0.05, f"should be at rest, |v|={np.linalg.norm(v)}"
    # y should be at least near the half-extent (resting on floor).
    y = s.positions()[0][1]
    assert h - 0.02 < y < h + 0.02, f"box should sit at y≈{h}, got {y}"


def test_box_corner_pin_holds_against_gravity():
    """Pin a body-local corner to a world point. The box should hang from
    that corner without falling."""
    s = Solver6DOF(dt=1/60, iterations=30)
    h = 0.3
    # Pin one TOP corner at the world point (h, 2.0, h).
    world_pin = (h, 2.0, h)
    # Place the box so that the corner (h, h, h) in body-local frame already
    # coincides with the world pin: world_corner = pos + R·local. With R = I,
    # local = (h,h,h), so pos = world_pin - (h,h,h) = (0, 2-h, 0).
    b = s.add_box((0., 2.0 - h, 0.), (h, h, h), mass=1.0)
    s.add_pin_corner(b, body_local=(h, h, h), world_point=world_pin)
    for _ in range(180):  # 3 sec
        s.step()
    # The pinned corner should still be at the pin point.
    pos = s.positions()[0]
    q = s.orientations()[0]
    # Compute world-space pinned corner = pos + R(q) * local. Inline
    # quaternion (xyzw) → rotation matrix to avoid a scipy dependency.
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R_world = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)
    world_corner = pos + R_world @ np.array([h, h, h], dtype=np.float32)
    err = np.linalg.norm(world_corner - np.array(world_pin))
    assert err < 0.02, f"pinned corner drifted: {world_corner} vs {world_pin}, err={err}"


def test_two_cubes_stack_axis_aligned():
    """Drop one cube onto another. Both should rest with proper stacking
    geometry (bot at y=h, top at y=3h, gap=0). Needs substeps to converge."""
    h = 0.20
    s = Solver6DOF(dt=1/60, iterations=15, substeps=8)
    bot = s.add_box((0., 0.20, 0.), (h, h, h), mass=1.0, friction=0.5)
    top = s.add_box((0., 0.55, 0.), (h, h, h), mass=1.0, friction=0.5)
    for b in (bot, top):
        s.add_floor_contact_box(b, friction=0.5)
    s.enable_self_collision(True, default_friction=0.5)
    for _ in range(300):  # 5 sec
        s.step()
    ys = s.positions()[:, 1]
    assert abs(ys[0] - h) < 0.01, f"bot.y should be ~{h}, got {ys[0]}"
    assert abs(ys[1] - 3*h) < 0.02, f"top.y should be ~{3*h}, got {ys[1]}"
    gap = ys[1] - ys[0] - 2*h
    assert abs(gap) < 0.01, f"gap should be ~0, got {gap}"
    vys = np.abs(s.velocities()[:, 1])
    assert vys.max() < 0.05, f"both cubes should be at rest, max|vy|={vys.max()}"


def test_three_cubes_stack():
    """3-cube tower needs more substeps to settle without collapsing."""
    h = 0.20
    s = Solver6DOF(dt=1/60, iterations=15, substeps=8)
    b0 = s.add_box((0., 0.20, 0.), (h, h, h), mass=1.0, friction=0.5)
    b1 = s.add_box((0., 0.62, 0.), (h, h, h), mass=1.0, friction=0.5)
    b2 = s.add_box((0., 1.04, 0.), (h, h, h), mass=1.0, friction=0.5)
    for b in (b0, b1, b2):
        s.add_floor_contact_box(b, friction=0.5)
    s.enable_self_collision(True, default_friction=0.5)
    for _ in range(360):  # 6 sec
        s.step()
    ys = s.positions()[:, 1]
    # All three should be stacked: y = h, 3h, 5h.
    expected = [h, 3*h, 5*h]
    for i, e in enumerate(expected):
        assert abs(ys[i] - e) < 0.03, f"cube {i}: y={ys[i]} expected ~{e}"


def test_cube_on_floor_friction_brakes_translation():
    """Cube sliding on floor (not on another cube) should stop at the
    Coulomb predicted distance d = v² / (2 μ g)."""
    h = 0.20
    v0 = 2.0
    mu = 0.4
    s = Solver6DOF(dt=1/60, iterations=15, substeps=8)
    b = s.add_box((0., h, 0.), (h, h, h), mass=1.0, friction=mu,
                  velocity=(v0, 0., 0.))
    s.add_floor_contact_box(b, friction=mu)
    init_x = s.positions()[0][0]
    for _ in range(240):  # 4 sec
        s.step()
    distance = s.positions()[0][0] - init_x
    expected = v0 * v0 / (2 * mu * 9.81)
    assert abs(distance - expected) / expected < 0.10, \
        f"slide distance {distance} should be near {expected}"


def test_obb_sat_detects_axis_aligned_overlap():
    """SAT directly: two overlapping AABBs should report a face axis."""
    from avbd3d.solver_6dof import _obb_sat
    c_A = np.array([0., 0., 0.], dtype=np.float32)
    c_B = np.array([0., 0.30, 0.], dtype=np.float32)  # Δy=0.30; h=0.2 each
    e = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    result = _obb_sat(c_A, R, e, c_B, R, e)
    assert result is not None, "SAT should detect overlap"
    axis_idx, n_hat, overlap = result
    # Smallest overlap is the y-axis (axis_idx = 1, A's face axis).
    # Other face axes overlap by 0.4; y-axis overlaps by 0.1.
    assert abs(overlap - 0.1) < 1e-4, f"overlap should be 0.1, got {overlap}"
    # n_hat points from B (above) to A (below) = -y.
    assert abs(n_hat[1] + 1.0) < 1e-4, f"n_hat should be (0,-1,0), got {n_hat}"


def test_obb_sat_separated_returns_none():
    """Cubes separated by more than `margin` along any axis should return None."""
    from avbd3d.solver_6dof import _obb_sat
    c_A = np.array([0., 0., 0.], dtype=np.float32)
    c_B = np.array([0., 0.50, 0.], dtype=np.float32)  # gap 0.10 (h=0.2 each)
    e = np.array([0.2, 0.2, 0.2], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    result = _obb_sat(c_A, R, e, c_B, R, e)
    assert result is None, f"SAT should return None for separated cubes, got {result}"


def test_static_body_does_not_move():
    """mass=0 body should be skipped by predict_inertial + primal_update."""
    s = Solver6DOF(dt=1/60, iterations=5)
    b = s.add_box((0.5, 1.0, 0.5), (0.3, 0.3, 0.3), mass=0.0)
    init_x = s.positions()[0].copy()
    init_q = s.orientations()[0].copy()
    for _ in range(60):
        s.step()
    assert np.allclose(s.positions()[0], init_x), "mass=0 body moved"
    assert np.allclose(s.orientations()[0], init_q), "mass=0 body rotated"
