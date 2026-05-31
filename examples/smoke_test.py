"""Smoke test: one particle falling under gravity, no constraints.

Verifies that predict_inertial + finalize_velocity compile and run, and that a
free particle integrates gravity correctly (Δh = 0.5 g t² over N steps).
"""

import numpy as np

from avbd3d import Solver


def main():
    dt = 1.0 / 60.0
    solver = Solver(dt=dt, iterations=2, gravity=(0.0, -9.81, 0.0), post_stabilize=False)
    b = solver.add_particle(position=(0.0, 10.0, 0.0), mass=1.0)
    print(f"start position: {solver.positions()[b.index]}")

    n_steps = 60
    for _ in range(n_steps):
        solver.step()

    x = solver.positions()[b.index]
    v = solver.velocities()[b.index]
    t = n_steps * dt
    expected_y = 10.0 - 0.5 * 9.81 * t * t
    expected_vy = -9.81 * t
    print(f"after {n_steps} steps (t={t:.3f}s):")
    print(f"  position = {x}")
    print(f"  velocity = {v}")
    print(f"  expected y ≈ {expected_y:.4f}   (BDF1 will be slightly damped)")
    print(f"  expected vy ≈ {expected_vy:.4f}")


if __name__ == "__main__":
    main()
