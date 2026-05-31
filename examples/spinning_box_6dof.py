"""Headless 6-DOF demo: spinning box falls and lands on the floor.

Demonstrates the new Solver6DOF — full SE(3) rigid body with quaternion
orientation, angular velocity, body-local inertia tensor, and 8-corner
floor contacts. Friction torque brakes the spin.

Run:
    uv run python examples/spinning_box_6dof.py
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from avbd3d import Solver6DOF


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=300, help="number of dt steps")
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--mass", type=float, default=1.0)
    p.add_argument("--size", type=float, default=0.3, help="half-extent of the box")
    p.add_argument("--start-y", type=float, default=1.2)
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--spin-y", type=float, default=3.0,
                   help="initial angular velocity about world y (rad/s)")
    p.add_argument("--tilt-deg", type=float, default=10.0,
                   help="initial tilt about world x (degrees)")
    args = p.parse_args()

    s = Solver6DOF(dt=1/60, iterations=args.iters)
    h = args.size
    ang = math.radians(args.tilt_deg)
    q0 = (math.sin(ang / 2), 0.0, 0.0, math.cos(ang / 2))
    b = s.add_box(
        position=(0.0, args.start_y, 0.0),
        half_extents=(h, h, h),
        mass=args.mass,
        orientation=q0,
        angular_velocity=(0.0, args.spin_y, 0.0),
        friction=args.friction,
    )
    s.add_floor_contact_box(b, floor_y=0.0, friction=args.friction)

    print(f"Simulating {args.frames} frames "
          f"({args.frames / 60.0:.2f}s sim time)…")
    print(f"{'t':>6}  {'y':>8}  {'|v|':>8}  {'|ω|':>8}  {'q (xyzw)':>40}")

    t0 = time.perf_counter()
    for frame in range(args.frames):
        s.step()
        if frame % 30 == 0 or frame == args.frames - 1:
            pos = s.positions()[0]
            v = s.velocities()[0]
            w = s.angular_velocities()[0]
            q = s.orientations()[0]
            print(f"{frame/60.0:>6.2f}  {pos[1]:>8.4f}  "
                  f"{float(np.linalg.norm(v)):>8.4f}  "
                  f"{float(np.linalg.norm(w)):>8.4f}  "
                  f"[{q[0]:>+6.3f} {q[1]:>+6.3f} {q[2]:>+6.3f} {q[3]:>+6.3f}]")
    wall = time.perf_counter() - t0
    print(f"\nwall time: {wall*1000:.0f} ms "
          f"({args.frames / wall:.0f} Hz solver throughput, "
          f"{1000.0 * wall / args.frames:.2f} ms/step)")
    print(f"final |ω| = {float(np.linalg.norm(s.angular_velocities()[0])):.4f} rad/s "
          f"(friction torque braked the spin)")


if __name__ == "__main__":
    main()
