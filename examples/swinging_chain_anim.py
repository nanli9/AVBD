"""Animated 3D demo: a chain pinned at the top, kicked into a swing, then
loaded with a heavy weight that tears the bottom link.

Saves to swinging_chain.gif using matplotlib's PillowWriter (no ffmpeg
dependency). Run with:

    uv run python examples/swinging_chain_anim.py
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from avbd3d import Solver


def build_scene(n_links: int, link: float, top: tuple[float, float, float],
                impulse: tuple[float, float, float],
                heavy_mass: float, fracture_threshold: float) -> tuple[Solver, list]:
    """Chain with first body pinned, an initial sideways impulse on every body,
    and a heavy bob at the bottom that will eventually snap the lowest link.
    """
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    bodies = []
    for i in range(n_links + 1):
        pos = (top[0], top[1] - i * link, top[2])
        # heavy mass on the very last body
        m = heavy_mass if i == n_links else 1.0
        # initial sideways kick scaled down with depth so the chain starts in
        # a smooth swing rather than a discontinuous jerk
        v = (impulse[0] * (1.0 - 0.5 * i / n_links),
             impulse[1],
             impulse[2] * (1.0 - 0.5 * i / n_links))
        bodies.append(s.add_particle(position=pos, mass=m, velocity=v))
    s.add_pin(bodies[0], world_point=top, stiffness=math.inf)
    for i in range(n_links):
        s.add_distance(bodies[i], bodies[i + 1], rest=link, stiffness=math.inf,
                       fracture=fracture_threshold)
    return s, bodies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--link", type=float, default=0.4)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", type=str, default="swinging_chain.gif")
    p.add_argument("--impulse", type=float, default=4.0)
    p.add_argument("--heavy-mass", type=float, default=12.0,
                   help="bottom particle mass (tears the chain when large)")
    p.add_argument("--threshold", type=float, default=70.0,
                   help="distance |λ| above this breaks the link")
    args = p.parse_args()

    top = (0.0, 4.0, 0.0)
    solver, bodies = build_scene(
        n_links=args.n,
        link=args.link,
        top=top,
        impulse=(args.impulse, 0.0, 0.5 * args.impulse),
        heavy_mass=args.heavy_mass,
        fracture_threshold=args.threshold,
    )

    n_b = args.n + 1
    pos_hist = np.zeros((args.frames + 1, n_b, 3), dtype=np.float32)
    active_hist = np.zeros((args.frames + 1, len(solver._constraints)), dtype=np.int32)
    pos_hist[0] = solver.positions()
    active_hist[0] = solver.active()

    for f in range(args.frames):
        solver.step()
        pos_hist[f + 1] = solver.positions()
        active_hist[f + 1] = solver.active()

    n_break_events = int(((active_hist[:-1] == 1) & (active_hist[1:] == 0)).any(axis=1).sum())
    print(f"simulated {args.frames} frames; {n_break_events} break event(s)")

    # ------------------------------------------------------------------
    # Build the animation. Two-panel layout: 3D scene + λ trace below.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_lam = fig.add_subplot(1, 2, 2)

    # 3D extents
    rng_xz = 1.2 * (args.n * args.link)
    ax3d.set_xlim(-rng_xz, rng_xz)
    ax3d.set_zlim(-rng_xz, rng_xz)
    ax3d.set_ylim(top[1] - args.n * args.link - 3.0, top[1] + 0.5)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y (up)")
    ax3d.set_zlabel("z")
    ax3d.set_title("Swinging chain (3D)")

    # Initial 3D artists
    bones_intact, = ax3d.plot([], [], [], "o-", color="steelblue", lw=2, markersize=6,
                              label="intact link")
    bones_broken, = ax3d.plot([], [], [], "o-", color="lightgray", lw=1, markersize=5,
                              label="broken (free fall)")
    pin_pt, = ax3d.plot([top[0]], [top[1]], [top[2]], "rs", markersize=10, label="pin")
    ax3d.legend(loc="upper right", fontsize=8)

    # λ panel
    t_axis = np.arange(args.frames + 1) * solver.dt
    n_dist = len(solver._constraints) - 3
    # λ traces over time
    lam_hist = np.zeros((args.frames + 1, n_dist), dtype=np.float32)
    # We didn't capture λ per frame above; capture it now in a second pass.
    # For brevity, just plot |λ| from a re-simulated copy. Cheaper alt is to
    # capture during the main loop; we do that.

    # Re-run to capture λ, since the original loop only captured positions.
    solver2, _ = build_scene(
        n_links=args.n, link=args.link, top=top,
        impulse=(args.impulse, 0.0, 0.5 * args.impulse),
        heavy_mass=args.heavy_mass, fracture_threshold=args.threshold,
    )
    for f in range(args.frames):
        solver2.step()
        lams = solver2.lambdas()
        # distance constraints start at index 3 (after the 3-axis pin)
        lam_hist[f + 1] = lams[3:3 + n_dist]

    ax_lam.set_xlim(0, t_axis[-1])
    ax_lam.set_ylim(0, max(args.threshold * 1.3, float(np.abs(lam_hist).max()) * 1.1 + 1e-3))
    ax_lam.axhline(args.threshold, color="k", linestyle="--", lw=1, label=f"fracture |λ|={args.threshold}")
    ax_lam.set_xlabel("t (s)")
    ax_lam.set_ylabel("|λ_link|  (interface traction)")
    ax_lam.set_title("Per-link |λ| over time")
    lam_lines = [ax_lam.plot([], [], lw=1.2, label=f"link {i}")[0] for i in range(n_dist)]
    ax_lam.legend(loc="upper right", fontsize=7, ncols=2)

    # Time text
    time_text = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes, fontsize=9)
    n_break_text = ax3d.text2D(0.02, 0.88, "", transform=ax3d.transAxes, fontsize=9, color="firebrick")

    def update(frame: int):
        pos = pos_hist[frame]
        act = active_hist[frame]
        # split chain into intact-prefix and broken-suffix based on first broken link
        # link i connects body i to body i+1 → index in constraints is 3+i
        first_broken = -1
        for i in range(n_dist):
            if act[3 + i] == 0:
                first_broken = i
                break
        if first_broken < 0:
            intact = pos
            broken = np.zeros((0, 3), dtype=np.float32)
        else:
            intact = pos[: first_broken + 1]
            broken = pos[first_broken + 1:]

        bones_intact.set_data(intact[:, 0], intact[:, 1])
        bones_intact.set_3d_properties(intact[:, 2])
        if len(broken) > 0:
            bones_broken.set_data(broken[:, 0], broken[:, 1])
            bones_broken.set_3d_properties(broken[:, 2])
        else:
            bones_broken.set_data([], [])
            bones_broken.set_3d_properties([])

        # λ traces — plot up through current frame
        for i, line in enumerate(lam_lines):
            line.set_data(t_axis[: frame + 1], np.abs(lam_hist[: frame + 1, i]))

        time_text.set_text(f"t = {frame * solver.dt:.2f} s")
        n_broken = int((active_hist[0, 3:3+n_dist] == 1).sum() - act[3:3+n_dist].sum())
        n_break_text.set_text(f"broken: {n_broken}/{n_dist}" if n_broken else "")

        return [bones_intact, bones_broken, pin_pt, time_text, n_break_text, *lam_lines]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=args.frames + 1,
        interval=1000 / args.fps,
        blit=False,
    )
    print(f"writing {args.out} ...")
    ani.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    print(f"saved {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
