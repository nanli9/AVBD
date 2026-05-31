"""Live interactive 3D demo with keyboard controls.

This opens a matplotlib window (NOT headless) and steps the AVBD solver in real
time. Use the keyboard to perturb the scene:

    SPACE    apply a random sideways impulse to every body
    K        kick a single random body harder
    B        lower the fracture threshold of every link by 30% (eventually breaks)
    H        raise threshold back up
    R        reset the scene
    + / -    add / remove an iteration to the solver
    ESC      quit

The window's title bar shows the active constraint count and current
threshold. To use this script you need a GUI backend (matplotlib's default
on macOS / TkAgg). Run with:

    uv run python examples/interactive_demo.py
"""

from __future__ import annotations

import argparse
import math
import sys

import matplotlib

# Pick a GUI backend if available (skip Agg). matplotlib auto-selects on most
# systems; this just makes sure we don't accidentally land in headless mode.
for backend in ("MacOSX", "TkAgg", "Qt5Agg", "Qt6Agg"):
    try:
        matplotlib.use(backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from avbd3d import Solver


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
def build_chain(n_links: int, link: float, top: tuple[float, float, float],
                heavy_mass: float, threshold: float):
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0.0, -9.81, 0.0),
               post_stabilize=True)
    bodies = []
    for i in range(n_links + 1):
        pos = (top[0], top[1] - i * link, top[2])
        m = heavy_mass if i == n_links else 1.0
        bodies.append(s.add_particle(position=pos, mass=m))
    s.add_pin(bodies[0], world_point=top, stiffness=math.inf)
    handles = [
        s.add_distance(bodies[i], bodies[i + 1], rest=link, stiffness=math.inf,
                       fracture=threshold)
        for i in range(n_links)
    ]
    return s, bodies, handles


# ---------------------------------------------------------------------------
# Live driver
# ---------------------------------------------------------------------------
class LiveDriver:
    def __init__(self, args):
        self.args = args
        self.top = (0.0, args.top_y, 0.0)
        self.reset()

        # --- figure ---
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        rng = 1.5 * (args.n * args.link)
        self.ax.set_xlim(-rng, rng)
        self.ax.set_zlim(-rng, rng)
        self.ax.set_ylim(self.top[1] - args.n * args.link - 4.0, self.top[1] + 1.0)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y (up)")
        self.ax.set_zlabel("z")
        self.bones_intact, = self.ax.plot([], [], [], "o-", color="steelblue", lw=2.0, ms=6)
        self.bones_broken, = self.ax.plot([], [], [], "o-", color="lightgray", lw=1.0, ms=4)
        self.pin_pt, = self.ax.plot([self.top[0]], [self.top[1]], [self.top[2]],
                                    "rs", markersize=10)
        self.hud = self.ax.text2D(0.02, 0.95, "", transform=self.ax.transAxes,
                                  fontsize=9, family="monospace")
        self.help = self.ax.text2D(0.02, 0.02,
                                   "[space] kick all   [k] kick one   [b] lower threshold   "
                                   "[h] raise threshold   [r] reset   [+/-] iters   [esc] quit",
                                   transform=self.ax.transAxes, fontsize=7, color="gray")

        # Event hooks
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.timer = self.fig.canvas.new_timer(interval=int(1000 * self.solver.dt))
        self.timer.add_callback(self.tick)

    # ----- scene control -----
    def reset(self):
        self.solver, self.bodies, self.handles = build_chain(
            n_links=self.args.n, link=self.args.link, top=self.top,
            heavy_mass=self.args.heavy_mass, threshold=self.args.threshold,
        )
        self.threshold = float(self.args.threshold)
        self.frame = 0

    def kick_all(self, magnitude: float = 4.0):
        # Inject sideways velocity on every non-pinned body by stepping with an
        # impulse target. Simplest: read positions, write velocities directly.
        v = self.solver.velocities().copy()
        n_b = v.shape[0]
        # Skip body 0 (effectively held by pin).
        rng = np.random.default_rng()
        v[1:, 0] += magnitude * rng.uniform(-1.0, 1.0, size=n_b - 1)
        v[1:, 2] += magnitude * rng.uniform(-1.0, 1.0, size=n_b - 1)
        import warp as wp
        self.solver.v = wp.array(v, dtype=wp.vec3, device=self.solver.device)

    def kick_one(self, magnitude: float = 8.0):
        v = self.solver.velocities().copy()
        n_b = v.shape[0]
        if n_b <= 1:
            return
        i = int(np.random.default_rng().integers(1, n_b))
        v[i, 0] += magnitude * float(np.random.default_rng().uniform(-1, 1))
        v[i, 2] += magnitude * float(np.random.default_rng().uniform(-1, 1))
        import warp as wp
        self.solver.v = wp.array(v, dtype=wp.vec3, device=self.solver.device)

    def scale_threshold(self, factor: float):
        new_thr = max(1.0, self.threshold * factor)
        import warp as wp
        n_c = len(self.solver._constraints)
        fracs = self.solver.c_fracture.numpy().copy()
        # Pin (indices 0..2) keeps inf. Distances start at index 3.
        for i in range(3, n_c):
            if np.isfinite(fracs[i]):
                fracs[i] = new_thr
        self.solver.c_fracture = wp.array(fracs, dtype=float, device=self.solver.device)
        self.threshold = new_thr

    # ----- event handlers -----
    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "escape":
            self.timer.stop()
            plt.close(self.fig)
        elif k == " ":
            self.kick_all()
        elif k == "k":
            self.kick_one()
        elif k == "b":
            self.scale_threshold(0.7)
        elif k == "h":
            self.scale_threshold(1.0 / 0.7)
        elif k == "r":
            self.reset()
        elif k in ("+", "="):
            self.solver.iterations += 1
        elif k == "-":
            self.solver.iterations = max(1, self.solver.iterations - 1)

    # ----- main loop tick -----
    def tick(self):
        self.solver.step()
        self.frame += 1
        pos = self.solver.positions()
        act = self.solver.active()
        n_dist = len(self.handles)
        first_broken = -1
        for i in range(n_dist):
            if act[3 + i] == 0:
                first_broken = i
                break
        if first_broken < 0:
            intact, broken = pos, np.zeros((0, 3), dtype=np.float32)
        else:
            intact = pos[: first_broken + 1]
            broken = pos[first_broken + 1:]
        self.bones_intact.set_data(intact[:, 0], intact[:, 1])
        self.bones_intact.set_3d_properties(intact[:, 2])
        self.bones_broken.set_data(broken[:, 0], broken[:, 1])
        self.bones_broken.set_3d_properties(broken[:, 2])

        n_broken = int(n_dist - act[3 : 3 + n_dist].sum())
        self.hud.set_text(
            f"frame {self.frame:5d}  t={self.frame * self.solver.dt:5.2f}s  "
            f"iters={self.solver.iterations:2d}  "
            f"|λ|max={float(np.abs(self.solver.lambdas()).max()):6.1f}  "
            f"thr={self.threshold:6.1f}  "
            f"broken {n_broken}/{n_dist}"
        )
        self.fig.canvas.draw_idle()

    def run(self):
        self.timer.start()
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8, help="number of chain links")
    p.add_argument("--link", type=float, default=0.4)
    p.add_argument("--top-y", type=float, default=4.0)
    p.add_argument("--heavy-mass", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=200.0,
                   help="initial fracture threshold (very high → no break unless you press 'b')")
    args = p.parse_args()

    if matplotlib.get_backend().lower() == "agg":
        print("WARNING: matplotlib is running in headless 'agg' backend; no window will appear.")
        print("Install a GUI backend (e.g. Tk: 'brew install python-tk' on macOS) and re-run.")
        sys.exit(1)

    print(f"interactive demo running on matplotlib backend = {matplotlib.get_backend()}")
    LiveDriver(args).run()


if __name__ == "__main__":
    main()
