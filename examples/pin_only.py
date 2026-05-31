"""Single body pinned by 3 PIN_X/Y/Z constraints under gravity.

Expected: body stays at the pin world point. λ should converge to total
weight (m·g) along the gravity axis.
"""

import numpy as np

from avbd3d import Solver


def main():
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0), post_stabilize=True)
    b = s.add_particle(position=(0.0, 5.0, 0.0), mass=1.0)
    s.add_pin(b, world_point=(0.0, 5.0, 0.0), stiffness=float("inf"))

    print(f"start: pos={s.positions()[0]}, λ={s.lambdas()}")
    for f in range(60):
        s.step()
        if f % 10 == 0 or f < 3:
            x = s.positions()[0]
            lam = s.lambdas()
            print(f"f={f:3d}  x={x[0]:+.6f} y={x[1]:+.6f} z={x[2]:+.6f}  "
                  f"λ=[{lam[0]:+.4f}, {lam[1]:+.4f}, {lam[2]:+.4f}]")
    print(f"expected: y=5.0, λ_y ≈ +9.81 (force needed to hold body up)")


if __name__ == "__main__":
    main()
