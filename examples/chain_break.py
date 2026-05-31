"""Chain with breakable distance constraints.

Same chain as hanging_chain.py, but each distance constraint has a finite
fracture threshold. When the dual variable |λ_i| exceeds the threshold
(which is AVBD's native impulse-magnitude criterion, mirroring I3D 2018
Section 3.4), the constraint is disabled and the chain breaks.

This is the FIRST hook for the fracture pipeline described in the project
roadmap. The next step is to replace the impulse criterion with the
Griffith energy criterion E_i ≥ G_c·ΔA_i and add the I3D Δv impulse
transfer on break (Gap 3 in the audit).
"""

import argparse

import numpy as np

from avbd3d import Solver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--threshold", type=float, default=30.0,
                   help="|λ| above this breaks the constraint")
    args = p.parse_args()

    n_links = args.n
    link = 0.5
    top = (0.0, 5.0, 0.0)

    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0), post_stabilize=True)

    bodies = []
    for i in range(n_links + 1):
        pos = (top[0], top[1] - i * link, top[2])
        bodies.append(s.add_particle(position=pos, mass=1.0))
    s.add_pin(bodies[0], world_point=top, stiffness=float("inf"))
    dist_handles = []
    for i in range(n_links):
        dist_handles.append(
            s.add_distance(bodies[i], bodies[i + 1], rest=link,
                           stiffness=float("inf"), fracture=args.threshold)
        )
    # Floor at y=0 so broken segments don't fall forever.
    for b in bodies:
        s.add_floor_contact(b, floor_y=0.0)

    print(f"Chain: {n_links + 1} bodies, fracture threshold |λ| ≥ {args.threshold} N")
    print(f"Expected steady-state link tensions (top to bottom):")
    for i in range(n_links):
        n_below = n_links - i
        print(f"  link {i}: ~{n_below * 9.81:.1f} N")
    print()

    active_history = []
    for f in range(args.frames):
        s.step()
        act = s.active().copy()
        if active_history and not np.array_equal(act, active_history[-1]):
            broken = np.where((act == 0) & (active_history[-1] == 1))[0]
            for bk in broken:
                # which distance constraint? (recall: first 3 are PIN_X/Y/Z)
                link_idx = bk - 3
                print(f"  frame {f:3d} (t={f * s.dt:.3f}s): link {link_idx} broke (|λ|≥{args.threshold})")
        active_history.append(act)

    final = s.positions()
    print(f"\nFinal state after {args.frames} frames:")
    print(f"  active constraints: {int(s.active().sum())}/{len(s.active())}")
    print(f"  body positions (y):")
    for i, b in enumerate(bodies):
        print(f"    body {i}: y={final[i, 1]:+.3f}")


if __name__ == "__main__":
    main()
