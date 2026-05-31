"""Hanging chain demo: N particles connected by distance constraints, with the
top particle pinned to a world point. Under gravity, the chain should hang
nearly vertical at steady state.

This exercises:
  - AVBD primal/dual loop with multiple constraints per body
  - Warm-started λ across frames
  - Hard constraint stabilization (α·C₀)
  - Pin (3-row) and Distance (1-row) constraint types

Usage:
    uv run python examples/hanging_chain.py            # text output
    uv run python examples/hanging_chain.py --plot     # matplotlib viz
"""

import argparse
import time

import numpy as np

from avbd3d import Solver


def build_chain(n_links: int, link_len: float, top: tuple[float, float, float]):
    solver = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0), post_stabilize=True)
    bodies = []
    # Top particle is pinned (static-like via pin constraint, not zero mass)
    for i in range(n_links + 1):
        pos = (top[0], top[1] - i * link_len, top[2])
        b = solver.add_particle(position=pos, mass=1.0)
        bodies.append(b)
    solver.add_pin(bodies[0], world_point=top, stiffness=float("inf"))
    for i in range(n_links):
        solver.add_distance(bodies[i], bodies[i + 1], rest=link_len, stiffness=float("inf"))
    return solver, bodies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="number of links")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--link", type=float, default=0.5, help="link rest length")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    solver, bodies = build_chain(args.n, args.link, top=(0.0, 5.0, 0.0))

    # Trace every body's y position and the pin's lambda magnitude
    n_b = args.n + 1
    pos_history = np.zeros((args.frames + 1, n_b, 3), dtype=np.float32)
    pin_lam_history = np.zeros(args.frames + 1, dtype=np.float32)
    dist_lam_history = np.zeros((args.frames + 1, args.n), dtype=np.float32)
    pos_history[0] = solver.positions()

    t0 = time.perf_counter()
    for f in range(args.frames):
        solver.step()
        pos_history[f + 1] = solver.positions()
        lams = solver.lambdas()
        # pin = constraints 0..2 (PIN_X/Y/Z), distances = 3..3+n-1
        pin_lam_history[f + 1] = float(np.linalg.norm(lams[:3]))
        dist_lam_history[f + 1] = lams[3 : 3 + args.n]
    dt = time.perf_counter() - t0

    final = solver.positions()
    print(f"Simulated {args.frames} frames of {n_b} particles in {dt*1000:.1f} ms ({args.frames/dt:.1f} fps)")
    print(f"Final positions (showing first 5):")
    for i in range(min(5, n_b)):
        print(f"  body {i}: x={final[i, 0]:+.4f}  y={final[i, 1]:+.4f}  z={final[i, 2]:+.4f}")
    print(f"Pin |λ| at end: {pin_lam_history[-1]:.2f}  (≈ total weight {n_b * 9.81:.2f} N)")
    print(f"Distance |λ| at end (first 5): {dist_lam_history[-1, :5]}")

    # Sanity check: chain should hang almost straight down at rest
    expected_chain_lengths = []
    for i in range(args.n):
        d = np.linalg.norm(final[i + 1] - final[i])
        expected_chain_lengths.append(d)
    max_err = max(abs(d - args.link) for d in expected_chain_lengths)
    print(f"Max distance-constraint error after settle: {max_err:.5f} m  (rest={args.link})")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # Chain shape at last frame
        ax = axes[0]
        ax.plot(final[:, 0], final[:, 1], "o-")
        ax.set_aspect("equal")
        ax.set_title("Final chain shape (XY)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        # Y trace of bottom particle
        ax = axes[1]
        t = np.arange(args.frames + 1) * solver.dt
        ax.plot(t, pos_history[:, -1, 1], label=f"body {n_b - 1} (bottom)")
        ax.plot(t, pos_history[:, 0, 1], label="body 0 (pinned)")
        ax.set_title("Y position over time")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)

        # Pin lambda trace
        ax = axes[2]
        ax.plot(t, pin_lam_history, label="|pin λ|")
        ax.axhline(n_b * 9.81, color="k", linestyle="--", label="expected weight")
        ax.set_title("Pin λ (cohesive traction) over time")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("|λ|")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        out = "hanging_chain.png"
        plt.savefig(out, dpi=120)
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
