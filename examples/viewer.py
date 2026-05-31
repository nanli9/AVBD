"""Interactive 3D AVBD viewer (viser, browser-based).

Opens a viser server and a scene with:

  - A ground plane at y=0 (with a real one-sided contact constraint).
  - Sphere / cube / pillar primitives for each body (the solver still treats
    them as point masses; rotations are next-iteration work).
  - A pinned anchor rendered as a red pillar.
  - **Transform-control gizmos** on every dynamic body so you can DRAG them
    in 3D and the simulator follows in real time.
  - GUI panel: pause, reset, kick all, drop a fresh cube, fracture threshold
    slider, gravity / iterations sliders.

Run:

    uv run python examples/viewer.py
    # open http://localhost:8080 in a browser (URL also printed on stdout)

Drag a body's gizmo to pull it around; release to let it fall back. Lower the
threshold slider to break the chain. Drop boxes on the chain to load it.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import viser
from viser import transforms as vt

from avbd3d import Body, Shape, Solver


# -----------------------------------------------------------------------------
# Scene description (separate from solver — viewer reads this each tick).
# -----------------------------------------------------------------------------
@dataclass
class ViewerBody:
    body: Body
    handle: object  # viser scene handle (Icosphere / Box / Cylinder)
    tc: object | None  # transform-control gizmo, if interactive
    bottom_offset: float = 0.0  # distance from body centre to its bottom (so the
                                # visual primitive doesn't clip through the floor)


def _bottom_offset(shape: Shape) -> float:
    """Distance from body centre to its visual bottom. We raise the floor by
    this amount per body so the primitive sits cleanly on y=0."""
    if shape.kind == "sphere":
        return float(shape.size[0])
    if shape.kind == "cube":
        return float(shape.size[1])  # half-extent on Y
    if shape.kind == "pillar":
        return float(shape.size[1])  # half-height on Y
    return 0.0


def build_scene(args) -> tuple[Solver, list[ViewerBody], list]:
    """Build a chain + a couple of free cubes/spheres so the user has things to
    play with. Returns solver, list of viewer bodies, list of distance handles."""
    s = Solver(
        dt=1.0 / 60.0,
        iterations=int(args.iterations),
        gravity=(0.0, -9.81, 0.0),
        post_stabilize=True,
        device="cpu",
    )

    bodies: list[ViewerBody] = []
    link = args.link
    top = (0.0, args.top_y, 0.0)

    def _add(pos, mass, shape, collide=True):
        bo = _bottom_offset(shape)
        b = s.add_particle(pos, mass=mass, shape=shape, collide=collide,
                           friction=args.friction)
        # Each body's floor is shifted up by its bottom offset → visual sits on y=0
        s.add_floor_contact(b, floor_y=bo, friction=args.friction)
        bodies.append(ViewerBody(body=b, handle=None, tc=None, bottom_offset=bo))
        return b

    # Pinned anchor (red sphere). Mass = 0 marks it as a TRUE static body
    # (AVBD 2D demo convention) — predict_inertial and primal_update both
    # early-out on m ≤ 0, so x[anchor] never updates from any source. This
    # is the right way to model "fixed point in world space"; the PIN
    # constraint below is redundant in that regime but harmless.
    # collide=False also removes it from the body-body broad phase so a
    # dropped cube can't land on top of the anchor (which was causing the
    # red ball to visibly jitter under contact forces with mass=1).
    anchor = _add(top, 0.0, Shape("sphere", (0.12,), color=(0.9, 0.2, 0.2)),
                  collide=False)
    s.add_pin(anchor, world_point=top, stiffness=math.inf)

    # Chain links: blue spheres. The FIRST link (anchor → chain[1]) is the
    # "hook" — we mark it unbreakable so the chain stays attached to the
    # anchor; only the rope below the hook can fracture.
    prev = anchor
    for i in range(1, args.n + 1):
        pos = (top[0], top[1] - i * link, top[2])
        b = _add(pos, 1.0, Shape("sphere", (0.10,), color=(0.3, 0.5, 0.9)))
        frac = math.inf if i == 1 else args.threshold
        s.add_distance(prev, b, rest=link, stiffness=math.inf, fracture=frac)
        prev = b

    # Heavy cube at the bottom (it's still a point mass; the cube is visual).
    bottom = _add(
        (top[0], top[1] - (args.n + 1) * link, top[2]),
        args.heavy_mass,
        Shape("cube", (0.18, 0.18, 0.18), color=(0.7, 0.4, 0.1)),
    )
    s.add_distance(prev, bottom, rest=link, stiffness=math.inf, fracture=args.threshold)

    # A couple of free objects to drop on the chain.
    _add((1.8, top[1] + 0.5, 0.4), 1.5,
         Shape("sphere", (0.18,), color=(0.95, 0.85, 0.2)))
    _add((-1.6, top[1] - 1.2, -0.4), 2.0,
         Shape("pillar", (0.10, 0.35), color=(0.55, 0.85, 0.45)))

    # Turn on dynamic sphere-sphere contacts BEFORE returning so step() picks
    # up overlapping pairs (chain links connected by a distance constraint are
    # auto-skipped by the broad phase). See VBD Eq.(12) / AVBD Eq.(15) for the
    # contact constraint and Eq.(15) tangent rows for friction.
    s.enable_self_collision(True, default_friction=args.friction)
    dist_constraints = [c for c in s._constraints if c.type == 3]  # DISTANCE
    return s, bodies, dist_constraints


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------
class Viewer:
    def __init__(self, args):
        self.args = args
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        # World convention: +Y is up, gravity points in -Y. Tell the browser
        # explicitly so the camera puts +Y up on the user's monitor (otherwise
        # viser's default puts +Z up, which makes things look like they're
        # falling "into the screen").
        try:
            self.server.scene.set_up_direction("+y")
        except Exception:
            pass
        # Aim the initial camera roughly toward the chain from a sensible angle.
        try:
            self.server.scene.set_global_scene_node_visibility(True)  # type: ignore
        except Exception:
            pass

        self.solver, self.bodies, self.dist_rows = build_scene(args)

        # --- ground plane (a thin box that looks like a floor) ---
        self.server.scene.add_box(
            "/ground",
            dimensions=(8.0, 0.05, 8.0),
            position=(0.0, -0.025, 0.0),
            color=(0.85, 0.85, 0.85),
        )
        self.server.scene.add_grid(
            "/grid",
            width=8.0, height=8.0,
            cell_size=0.5,
            plane="xz",
        )

        # --- body primitives ---
        for i, vb in enumerate(self.bodies):
            self._add_body_primitive(vb, name=f"/bodies/{i}")

        # --- chain link rendering: one line-segments node, updated each tick ---
        self._dist_pairs = [
            (c.body_a, c.body_b) for c in self.solver._constraints if c.type == 3
        ]
        self._link_handle = self.server.scene.add_line_segments(
            "/links",
            points=self._compute_link_points(),
            colors=np.array([60, 90, 200], dtype=np.uint8),
            line_width=4.0,
        )

        # --- transform-control gizmos for DRAG ---
        # Gizmos are HIDDEN by default — they create visual clutter
        # (axis arrows + drift artifacts). The user toggles them on via the
        # "drag mode" checkbox below. When toggled on we snap every gizmo
        # to its body's current position; when off we hide them.
        for i, vb in enumerate(self.bodies):
            if i == 0:
                continue  # skip the pinned anchor
            tc = self.server.scene.add_transform_controls(
                f"/drag/{i}",
                position=tuple(self.solver.positions()[vb.body.index]),
                scale=args.gizmo_scale,
                disable_axes=False,
                disable_sliders=True,
                disable_rotations=True,
                visible=False,
            )
            vb.tc = tc
            self._wire_drag(vb, i)

        # --- GUI panel ---
        with self.server.gui.add_folder("Simulation"):
            self.gui_pause = self.server.gui.add_checkbox("pause", initial_value=False)
            self.gui_iters = self.server.gui.add_slider("iterations", 1, 30,
                                                        step=1, initial_value=args.iterations)
            self.gui_gravity = self.server.gui.add_slider("gravity (m/s²)", -30.0, 0.0,
                                                          step=0.5, initial_value=-9.81)
            self.gui_threshold = self.server.gui.add_slider("fracture threshold",
                                                            1.0, 500.0, step=1.0,
                                                            initial_value=args.threshold)
            self.gui_friction = self.server.gui.add_slider("friction μ",
                                                            0.0, 1.0, step=0.01,
                                                            initial_value=args.friction,
                                                            hint="Coulomb friction coefficient for floor + body-body "
                                                                 "contacts (AVBD Sec. 3.3 cone clamp |λ_t| ≤ μ|λ_n|)")
            self.gui_drag_mode = self.server.gui.add_checkbox(
                "drag mode (show handles)", initial_value=False,
                hint="When on, every body sprouts an XYZ gizmo you can drag in 3D. "
                     "Off by default to keep the scene clean.")
        with self.server.gui.add_folder("Actions"):
            self.gui_kick = self.server.gui.add_button("kick all bodies (random sideways)")
            self.gui_drop = self.server.gui.add_button("drop a fresh cube")
            self.gui_snap = self.server.gui.add_button("snap drag handles to bodies")
            self.gui_reset = self.server.gui.add_button("reset scene")
        with self.server.gui.add_folder("Status"):
            self.gui_frame = self.server.gui.add_text("frame", initial_value="0")
            self.gui_time = self.server.gui.add_text("t (s)", initial_value="0.0")
            self.gui_active = self.server.gui.add_text("active links",
                                                       initial_value=str(len(self.dist_rows)))
            self.gui_maxlam = self.server.gui.add_text("max |λ|", initial_value="0.0")
        with self.server.gui.add_folder("Performance"):
            # The whole solver runs on NVIDIA Warp (kernels.py — every AVBD
            # equation is a wp.kernel). On Apple Silicon Warp falls back to
            # CPU; flip `--device cuda:0` on an NVIDIA box for GPU.
            self.gui_perf_device = self.server.gui.add_text(
                "device", initial_value=self.solver.device,
                hint="Warp execution device. 'cpu' here because Apple Silicon "
                     "has no CUDA; passes through to wp.launch unchanged.")
            self.gui_perf_bodies = self.server.gui.add_text(
                "bodies", initial_value=str(len(self.solver._bodies_x)))
            self.gui_perf_constraints = self.server.gui.add_text(
                "constraints", initial_value="0",
                hint="Total rows in the AVBD constraint pool, including the "
                     "dynamic body-body contacts re-emitted every step.")
            self.gui_perf_colors = self.server.gui.add_text(
                "graph colors", initial_value="0",
                hint="Welsh–Powell coloring of the body-adjacency graph. "
                     "primal_update kernel launches once per color; bodies "
                     "sharing a color update fully in parallel inside that "
                     "launch.")
            self.gui_perf_step_ms = self.server.gui.add_text(
                "step time", initial_value="—",
                hint="Wall-clock time for one solver.step() call (the AVBD "
                     "iteration loop + broad phase). Lower is better.")
            self.gui_perf_capacity = self.server.gui.add_text(
                "solver capacity", initial_value="—",
                hint="1 / step_time. The maximum sustained Hz the solver "
                     "could deliver if rendering took zero time. NOT the "
                     "screen frame rate.")
            self.gui_perf_wall = self.server.gui.add_text(
                "wall tick", initial_value="—",
                hint="Actual viewer tick rate (one frame = solver.step() + "
                     "scene update + idle wait). Target is 1/dt = 60 Hz.")

        self.gui_kick.on_click(lambda _: self._kick_all())
        self.gui_drop.on_click(lambda _: self._drop_cube())
        self.gui_snap.on_click(lambda _: self._snap_gizmos())
        self.gui_reset.on_click(lambda _: self._reset())
        self.gui_threshold.on_update(self._threshold_changed)
        self.gui_iters.on_update(self._iters_changed)
        self.gui_gravity.on_update(self._gravity_changed)
        self.gui_drag_mode.on_update(self._drag_mode_changed)
        self.gui_friction.on_update(self._friction_changed)

        # Dragging state per body
        self._drag_targets: dict[int, np.ndarray] = {}
        # When True, on_update callbacks from programmatic gizmo moves are
        # ignored (otherwise calling vb.tc.position = ... would feed itself
        # back as a "user drag" and lock the body in place).
        self._gizmo_suppress = False
        # True while any gizmo is being yanked around — used to (a) keep
        # gizmos pinned to bodies in tick(), (b) temporarily suppress
        # fracture so the chain doesn't snap mid-drag.
        self._fracture_suppressed = False
        self._frame = 0
        self._t0 = time.perf_counter()
        # Rolling windows for solver step time and wall tick time, so the
        # status panel shows a smoothed reading rather than per-frame noise.
        self._step_ms_window: list[float] = []
        self._wall_tick_ms_window: list[float] = []
        self._last_tick_t = time.perf_counter()

    def _compute_link_points(self) -> np.ndarray:
        """Return (N, 2, 3) array of link endpoints in world space, with
        broken links collapsed to a degenerate (origin → origin) segment so
        they vanish from the render."""
        pos = self.solver.positions()
        act = self.solver.active()
        # walk constraints again (matches the order of self._dist_pairs).
        dist_indices = [i for i, c in enumerate(self.solver._constraints) if c.type == 3]
        out = np.zeros((max(1, len(self._dist_pairs)), 2, 3), dtype=np.float32)
        for k, ((a, b), ci) in enumerate(zip(self._dist_pairs, dist_indices)):
            if act[ci] == 0:
                continue
            out[k, 0] = pos[a]
            out[k, 1] = pos[b]
        return out

    def _add_body_primitive(self, vb: ViewerBody, name: str):
        sh = vb.body.shape
        # Flush so a freshly-added body has a slot in the Warp position array.
        self.solver._flush()
        pos = tuple(self.solver.positions()[vb.body.index])
        if sh.kind == "sphere":
            vb.handle = self.server.scene.add_icosphere(
                name, radius=float(sh.size[0]), color=sh.color,
                position=pos,
            )
        elif sh.kind == "cube":
            ex = sh.size
            vb.handle = self.server.scene.add_box(
                name, dimensions=(2 * ex[0], 2 * ex[1], 2 * ex[2]),
                color=sh.color, position=pos,
            )
        elif sh.kind == "pillar":
            r, hh = sh.size
            # viser doesn't have a primitive cylinder via Scene API in 1.0.29 in
            # all builds; fall back to a thin box of the same bounding extent.
            try:
                vb.handle = self.server.scene.add_mesh_simple(
                    name,
                    vertices=_cylinder_mesh(r, hh)[0],
                    faces=_cylinder_mesh(r, hh)[1],
                    color=sh.color, position=pos,
                )
            except Exception:
                vb.handle = self.server.scene.add_box(
                    name, dimensions=(2 * r, 2 * hh, 2 * r),
                    color=sh.color, position=pos,
                )
        else:
            vb.handle = self.server.scene.add_icosphere(name, radius=0.1, color=sh.color, position=pos)

    def _wire_drag(self, vb: ViewerBody, view_idx: int):
        body_idx = vb.body.index

        @vb.tc.on_update
        def _on_drag(_evt):  # noqa: ANN001
            # Ignore programmatic moves (snap, per-tick follow). Only a true
            # user mouse drag should populate _drag_targets.
            if self._gizmo_suppress:
                return
            # Reject moves where the gizmo is already on the body (epsilon
            # check guards against floating-point round-trips).
            body_p = self.solver.positions()[body_idx]
            new_p = np.asarray(vb.tc.position, dtype=np.float32)
            if np.linalg.norm(new_p - body_p) < 1e-4:
                return
            self._drag_targets[body_idx] = new_p

    # ----- GUI callbacks --------------------------------------------------
    def _snap_gizmos(self):
        """One-shot teleport each gizmo back to its body's current position."""
        pos = self.solver.positions()
        self._gizmo_suppress = True
        try:
            with self.server.atomic():
                for vb in self.bodies:
                    if vb.tc is not None:
                        p = pos[vb.body.index]
                        vb.tc.position = (float(p[0]), float(p[1]), float(p[2]))
        finally:
            self._gizmo_suppress = False

    def _set_all_fractures(self, val: float):
        """Override every breakable distance constraint's fracture threshold.
        `inf` disables fracture entirely (use while dragging)."""
        import warp as wp
        fracs = self.solver.c_fracture.numpy().copy()
        for i, c in enumerate(self.solver._constraints):
            if c.type == 3 and np.isfinite(c.fracture):
                fracs[i] = val
        self.solver.c_fracture = wp.array(fracs, dtype=float, device=self.solver.device)

    def _kick_all(self):
        rng = np.random.default_rng()
        for vb in self.bodies[1:]:
            dv = (float(rng.uniform(-3.5, 3.5)),
                  float(rng.uniform(0.5, 3.0)),
                  float(rng.uniform(-3.5, 3.5)))
            self.solver.add_impulse(vb.body, dv)

    def _drop_cube(self):
        rng = np.random.default_rng()
        # Rejection-sample an XZ that isn't on top of an existing body
        # (point-mass solver lacks body-body contact, so spawning inside
        # something would leave it visually clipped forever).
        existing = self.solver.positions()
        target_x = target_z = 0.0
        for _ in range(20):
            cx = float(rng.uniform(-2.0, 2.0))
            cz = float(rng.uniform(-2.0, 2.0))
            ok = True
            for p in existing:
                if (p[0] - cx) ** 2 + (p[2] - cz) ** 2 < 0.35 ** 2:
                    ok = False
                    break
            if ok:
                target_x, target_z = cx, cz
                break
        # No initial velocity; spawn ~1 m above the existing scene so impact
        # velocity stays modest (a higher drop means the cube has more KE to
        # discharge through one AVBD frame's contact, which the inscribed
        # sphere + penalty-min combo handles well but never perfectly).
        pos = (target_x, self.args.top_y + 0.6, target_z)
        vel = (0.0, 0.0, 0.0)
        shape = Shape("cube", (0.12, 0.12, 0.12), color=(0.2, 0.85, 0.85))
        bo = _bottom_offset(shape)
        mu = float(self.gui_friction.value)
        new_b = self.solver.add_particle(pos, mass=1.0, velocity=vel,
                                         shape=shape, collide=True,
                                         friction=mu)
        self.solver.add_floor_contact(new_b, floor_y=bo, friction=mu)
        # Flush so positions array is sized for the new body before we read it.
        self.solver._flush()
        vb = ViewerBody(body=new_b, handle=None, tc=None, bottom_offset=bo)
        idx = len(self.bodies)
        self.bodies.append(vb)
        self._add_body_primitive(vb, f"/bodies/{idx}")
        tc = self.server.scene.add_transform_controls(
            f"/drag/{idx}", position=pos, scale=self.args.gizmo_scale,
            disable_sliders=True, disable_rotations=True,
            visible=bool(self.gui_drag_mode.value),
        )
        vb.tc = tc
        self._wire_drag(vb, idx)

    def _reset(self):
        # Remove all scene nodes and rebuild from scratch.
        for vb in self.bodies:
            if vb.handle is not None:
                try:
                    vb.handle.remove()
                except Exception:
                    pass
            if vb.tc is not None:
                try:
                    vb.tc.remove()
                except Exception:
                    pass
        try:
            self._link_handle.remove()
        except Exception:
            pass
        self._drag_targets.clear()
        self.solver, self.bodies, self.dist_rows = build_scene(self.args)
        for i, vb in enumerate(self.bodies):
            self._add_body_primitive(vb, f"/bodies/{i}")
        # rebuild link line-segments
        self._dist_pairs = [
            (c.body_a, c.body_b) for c in self.solver._constraints if c.type == 3
        ]
        self._link_handle = self.server.scene.add_line_segments(
            "/links",
            points=self._compute_link_points(),
            colors=np.array([60, 90, 200], dtype=np.uint8),
            line_width=4.0,
        )
        for i, vb in enumerate(self.bodies):
            if i == 0:
                continue
            tc = self.server.scene.add_transform_controls(
                f"/drag/{i}",
                position=tuple(self.solver.positions()[vb.body.index]),
                scale=self.args.gizmo_scale,
                disable_sliders=True, disable_rotations=True,
                visible=bool(self.gui_drag_mode.value),
            )
            vb.tc = tc
            self._wire_drag(vb, i)
        # Re-apply GUI-controlled threshold / iters / gravity
        self._threshold_changed(None)
        self._iters_changed(None)
        self._gravity_changed(None)
        self._frame = 0

    def _threshold_changed(self, _evt):
        import warp as wp
        new_thr = float(self.gui_threshold.value)
        fracs = self.solver.c_fracture.numpy().copy()
        for i, c in enumerate(self.solver._constraints):
            if c.type == 3 and np.isfinite(c.fracture):
                fracs[i] = new_thr
        self.solver.c_fracture = wp.array(fracs, dtype=float, device=self.solver.device)

    def _iters_changed(self, _evt):
        self.solver.iterations = int(self.gui_iters.value)

    def _gravity_changed(self, _evt):
        g = float(self.gui_gravity.value)
        self.solver.gravity = (0.0, g, 0.0)

    def _friction_changed(self, _evt):
        """Update μ for all FLOOR_CONTACT tangent rows and for newly-generated
        sphere-sphere contacts. The dynamic pool is rebuilt every step, so a
        slider change shows up in the next frame."""
        import warp as wp
        mu = float(self.gui_friction.value)
        # 1) Update each body's stored friction (used by future broad-phase
        #    pairs and by add_floor_contact rows added after this point).
        for i in range(len(self.solver._bodies_friction)):
            self.solver._bodies_friction[i] = mu
        self.solver._default_friction = mu
        # 2) Rewrite the static tangent rows already in the constraint list
        #    (floor friction for existing bodies) and push to the Warp array.
        fric = self.solver.c_friction.numpy().copy()
        for i, c in enumerate(self.solver._constraints):
            if c.type == 6:  # CONTACT_TANGENT
                c.friction = mu
                fric[i] = mu
        self.solver.c_friction = wp.array(fric, dtype=float, device=self.solver.device)

    def _drag_mode_changed(self, _evt):
        show = bool(self.gui_drag_mode.value)
        if show:
            # Snap every gizmo to its body, then make it visible.
            self._snap_gizmos()
        with self.server.atomic():
            for vb in self.bodies:
                if vb.tc is not None:
                    vb.tc.visible = show

    # ----- main tick ------------------------------------------------------
    def tick(self):
        if self.gui_pause.value:
            return

        # --- drag handling ---
        # While the user is yanking a body, the distance constraint to its
        # neighbour develops a huge C (= ‖p_neighbour − p_dragged‖ − rest),
        # which makes |λ| explode and trip the fracture threshold. So while
        # ANY body is being dragged, suppress fracture entirely; restore the
        # slider value on release.
        dragging = len(self._drag_targets) > 0
        if dragging and not self._fracture_suppressed:
            self._set_all_fractures(float("inf"))
            self._fracture_suppressed = True
        elif not dragging and self._fracture_suppressed:
            self._set_all_fractures(float(self.gui_threshold.value))
            self._fracture_suppressed = False

        for body_idx, target in list(self._drag_targets.items()):
            self.solver.set_position(self.bodies_by_idx(body_idx).body, tuple(target))
            self.solver.set_velocity(self.bodies_by_idx(body_idx).body, (0.0, 0.0, 0.0))
        self._drag_targets.clear()

        t0 = time.perf_counter()
        self.solver.step()
        dt = time.perf_counter() - t0

        # Batch every scene mutation into one websocket frame so 60 Hz worth
        # of position updates arrives as one packet instead of dozens — this
        # is what kept the camera rotation from flickering before.
        pos = self.solver.positions()
        drag_mode = bool(self.gui_drag_mode.value)
        self._gizmo_suppress = True
        try:
            with self.server.atomic():
                for vb in self.bodies:
                    p_solver = pos[vb.body.index]
                    p_render = (float(p_solver[0]), float(p_solver[1]), float(p_solver[2]))
                    # Defensive: a scene reset (or drop_cube rebuild) can
                    # remove a handle between ticks; the next position write
                    # would otherwise crash the entire tick. Skip removed
                    # handles and clear the stale reference.
                    if vb.handle is not None:
                        try:
                            vb.handle.position = p_render
                        except RuntimeError:
                            vb.handle = None
                    if drag_mode and vb.tc is not None:
                        try:
                            vb.tc.position = p_render
                        except RuntimeError:
                            vb.tc = None
                try:
                    self._link_handle.points = self._compute_link_points()
                except RuntimeError:
                    pass
        finally:
            self._gizmo_suppress = False

        # HUD
        self._frame += 1
        self.gui_frame.value = str(self._frame)
        self.gui_time.value = f"{self._frame * self.solver.dt:.2f}"
        act = self.solver.active()
        n_alive = sum(1 for c, a in zip(self.solver._constraints, act)
                      if c.type == 3 and a == 1)
        self.gui_active.value = f"{n_alive}/{len(self.dist_rows)}"
        max_lam = float(np.abs(self.solver.lambdas()).max()) if len(self.solver._constraints) else 0.0
        self.gui_maxlam.value = f"{max_lam:.1f}"

        # Performance metrics. Solver-step time is the AVBD work proper;
        # wall-tick time also includes the scene-update payload and the
        # idle wait that pads us to dt. Both rolling-averaged.
        self._step_ms_window.append(dt * 1000.0)
        if len(self._step_ms_window) > 30:
            self._step_ms_window.pop(0)
        now = time.perf_counter()
        tick_dt = now - self._last_tick_t
        self._last_tick_t = now
        self._wall_tick_ms_window.append(tick_dt * 1000.0)
        if len(self._wall_tick_ms_window) > 30:
            self._wall_tick_ms_window.pop(0)
        step_ms = float(np.mean(self._step_ms_window))
        wall_ms = float(np.mean(self._wall_tick_ms_window))
        self.gui_perf_step_ms.value = f"{step_ms:.2f} ms"
        self.gui_perf_capacity.value = f"{1000.0/max(step_ms, 1e-3):.0f} Hz"
        self.gui_perf_wall.value = f"{1000.0/max(wall_ms, 1e-3):.0f} Hz"
        # Body and constraint counts are cheap; refresh every tick.
        self.gui_perf_bodies.value = str(len(self.solver._bodies_x))
        n_total = len(self.solver._constraints)
        n_active = int(act.sum()) if len(act) else 0
        self.gui_perf_constraints.value = f"{n_active}/{n_total} active"
        n_colors = int(self.solver.num_colors)
        n_bodies = len(self.solver._bodies_x)
        per_color = (n_bodies / max(n_colors, 1)) if n_colors else 0.0
        self.gui_perf_colors.value = (
            f"{n_colors}  (~{per_color:.1f} bodies/color)"
        )

    def bodies_by_idx(self, body_idx: int) -> ViewerBody:
        for vb in self.bodies:
            if vb.body.index == body_idx:
                return vb
        raise KeyError(body_idx)

    def run(self):
        target_dt = self.solver.dt
        print(f"\nviser server running. open the URL above in a browser to interact.\n")
        try:
            while True:
                t = time.perf_counter()
                self.tick()
                spent = time.perf_counter() - t
                if spent < target_dt:
                    time.sleep(target_dt - spent)
        except KeyboardInterrupt:
            print("\nstopping...")


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _cylinder_mesh(radius: float, half_height: float, sides: int = 24):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False, dtype=np.float32)
    cs, sn = np.cos(angles), np.sin(angles)
    top = np.stack([radius * cs, np.full_like(cs, half_height), radius * sn], axis=1)
    bot = np.stack([radius * cs, np.full_like(cs, -half_height), radius * sn], axis=1)
    centre_top = np.array([[0.0, half_height, 0.0]], dtype=np.float32)
    centre_bot = np.array([[0.0, -half_height, 0.0]], dtype=np.float32)
    verts = np.vstack([top, bot, centre_top, centre_bot]).astype(np.float32)
    n = sides
    ct = 2 * n
    cb = 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        # side quads (two triangles)
        faces.append([i, j, n + i])
        faces.append([j, n + j, n + i])
        # top fan
        faces.append([ct, j, i])
        # bottom fan
        faces.append([cb, n + i, n + j])
    return verts, np.array(faces, dtype=np.int32)


# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6, help="chain links")
    p.add_argument("--link", type=float, default=0.4)
    p.add_argument("--top-y", type=float, default=3.5)
    p.add_argument("--heavy-mass", type=float, default=4.0)
    p.add_argument("--threshold", type=float, default=120.0)
    p.add_argument("--friction", type=float, default=0.5,
                   help="Coulomb μ for floor and body-body contact "
                        "(AVBD §3.3 friction-cone clamp |λ_t| ≤ μ|λ_n|)")
    p.add_argument("--iterations", type=int, default=25,
                   help="AVBD iterations per step; more = less penetration but slower")
    p.add_argument("--gizmo-scale", type=float, default=0.18,
                   help="size of the drag handles (smaller = less visual clutter)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    Viewer(args).run()


if __name__ == "__main__":
    main()
