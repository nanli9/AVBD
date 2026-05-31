"""Interactive 3D AVBD viewer — 6-DOF rigid body edition (viser, browser-based).

Drives `Solver6DOF`. Each body is a full rigid box with orientation; the floor
is the only collision target right now (body-body OBB-OBB contact is the next
milestone). The pinned box hangs from a single body-local corner.

Run:

    uv run python examples/viewer.py
    # open http://localhost:8080 in a browser (URL also printed on stdout)

Drag a body's gizmo to teleport it; release to let it fall. The orientation
gizmo is rotation-locked — only translation is interactive. The particle
chain demo lives in examples/viewer_particles.py.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass

import numpy as np
import viser

from avbd3d import Solver6DOF, RigidBody


# -----------------------------------------------------------------------------
# Scene description
# -----------------------------------------------------------------------------
@dataclass
class ViewerBox:
    body: RigidBody
    handle: object  # viser scene Box handle
    tc: object | None
    color: tuple[float, float, float]
    is_static: bool = False  # mass=0; never moves


def warp_q_to_viser_wxyz(q_xyzw: np.ndarray) -> tuple[float, float, float, float]:
    """Warp wp.quat is XYZW; viser's add_box `wxyz` is W-first. Convert."""
    qx, qy, qz, qw = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
    return (qw, qx, qy, qz)


def random_orientation(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Uniform random unit quaternion (XYZW) — Marsaglia's method."""
    while True:
        s1 = 2.0 * rng.random() - 1.0
        s2 = 2.0 * rng.random() - 1.0
        d1 = s1 * s1 + s2 * s2
        if d1 < 1.0:
            break
    while True:
        s3 = 2.0 * rng.random() - 1.0
        s4 = 2.0 * rng.random() - 1.0
        d2 = s3 * s3 + s4 * s4
        if d2 < 1.0:
            break
    s = math.sqrt((1.0 - d1) / d2)
    return (float(s1), float(s2), float(s3 * s), float(s4 * s))


def build_scene(args) -> tuple[Solver6DOF, list[ViewerBox], list[int]]:
    """Build a 6-DOF scene: pinned anchor + a grid of cube towers + a row
    of standing domino slabs. Stresses the OBB-OBB contact + persistent
    augmented-Lagrangian warm-start that AVBD relies on for stable stacks.
    Returns solver, list of viewer boxes, list of pin-row indices."""
    s = Solver6DOF(
        dt=1.0 / 60.0,
        iterations=int(args.iterations),
        gravity=(0.0, -9.81, 0.0),
        post_stabilize=True,
        device=args.device,
        substeps=int(args.substeps),
        friction_static_mult=float(args.static_mult),
    )
    s.enable_self_collision(True, default_friction=args.friction)
    boxes: list[ViewerBox] = []
    rng = np.random.default_rng(seed=args.seed)

    def col(saturation=0.7, value=0.85):
        h = float(rng.random())
        # cheap HSV→RGB
        i = int(h * 6); f = h * 6 - i
        p = value * (1 - saturation)
        q_v = value * (1 - f * saturation); t = value * (1 - (1 - f) * saturation)
        return [(value, t, p), (q_v, value, p), (p, value, t),
                (p, q_v, value), (t, p, value), (value, p, q_v)][i % 6]

    # --- 1. Pinned anchor box (off to the side so it doesn't hit towers) ----
    h_anchor = (0.16, 0.16, 0.16)
    anchor_world_pin = (-2.0, args.top_y, -2.0)
    pos_anchor = (
        anchor_world_pin[0] - h_anchor[0],
        anchor_world_pin[1] - h_anchor[1],
        anchor_world_pin[2] - h_anchor[2],
    )
    anchor = s.add_box(pos_anchor, h_anchor, mass=1.0, friction=args.friction)
    s.add_pin_corner(anchor, body_local=h_anchor, world_point=anchor_world_pin)
    s.add_floor_contact_box(anchor, friction=args.friction)
    boxes.append(ViewerBox(body=anchor, handle=None, tc=None,
                           color=(0.9, 0.25, 0.25)))

    # --- 2. Grid of cube towers ---------------------------------------------
    # 3×3 layout, each tower is `tower_height` cubes tall. Cube half-extent
    # h=0.12 → full cube 24 cm. Tower spacing 0.55 m gives ~7 cm gap between
    # towers, enough to keep them from crosstalking on the first substep but
    # tight enough to look dense.
    h_cube = 0.12
    tower_height = 3
    tower_spacing = 0.55
    grid_n = 3
    grid_origin = -(grid_n - 1) * tower_spacing * 0.5  # centered on origin
    for ix in range(grid_n):
        for iz in range(grid_n):
            cx = grid_origin + ix * tower_spacing
            cz = grid_origin + iz * tower_spacing
            tower_color = col(saturation=0.55, value=0.92)
            for k in range(tower_height):
                # Tiny random xz offset (≤1 mm) so perfectly-aligned faces
                # don't pin the SAT to a tiebreak axis on every frame.
                jitter = rng.uniform(-1e-3, 1e-3, size=2)
                cy = h_cube + 2.0 * h_cube * k  # y = h, 3h, 5h, …
                b = s.add_box((cx + float(jitter[0]), cy, cz + float(jitter[1])),
                              (h_cube, h_cube, h_cube),
                              mass=1.0, friction=args.friction)
                s.add_floor_contact_box(b, friction=args.friction)
                # Slight per-level darkening so layers read visually.
                shade = 1.0 - 0.06 * k
                shaded = tuple(min(1.0, c * shade) for c in tower_color)
                boxes.append(ViewerBox(body=b, handle=None, tc=None,
                                       color=shaded))

    # --- 3. Row of standing domino slabs ------------------------------------
    # Thin in x (4 cm), tall in y (24 cm), medium in z (10 cm), spaced just
    # over their height in x so toppling one cascades into the next.
    domino_he = (0.025, 0.12, 0.06)  # half-extents → 5 × 24 × 12 cm slab
    domino_spacing = 0.10            # x-gap between adjacent slabs
    domino_count = 6
    domino_z = 1.8                   # in front of the tower grid
    domino_x0 = -(domino_count - 1) * domino_spacing * 0.5
    for i in range(domino_count):
        cx = domino_x0 + i * domino_spacing
        cy = domino_he[1]            # bottom face on floor
        b = s.add_box((cx, cy, domino_z), domino_he,
                      mass=0.5, friction=args.friction)
        s.add_floor_contact_box(b, friction=args.friction)
        # Alternating cool-warm palette so the row reads as dominoes.
        c = (0.95, 0.55, 0.25) if i % 2 == 0 else (0.25, 0.55, 0.95)
        boxes.append(ViewerBox(body=b, handle=None, tc=None, color=c))

    pin_indices: list[int] = []
    return s, boxes, pin_indices


# -----------------------------------------------------------------------------
# Viewer
# -----------------------------------------------------------------------------
class Viewer:
    def __init__(self, args):
        self.args = args
        # Solver mutation lock — create FIRST so any setup path that ends
        # up calling _add_box_primitive (which acquires the lock) finds it.
        # RLock so re-entrant calls (drop_box → _add_box_primitive) don't
        # self-deadlock. See drop_box / tick comment for why we need this.
        self._solver_lock = threading.RLock()

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+y")
        except Exception:
            pass

        self.solver, self.boxes, self.pin_rows = build_scene(args)

        # ground plane
        self.server.scene.add_box(
            "/ground",
            dimensions=(8.0, 0.05, 8.0),
            position=(0.0, -0.025, 0.0),
            color=(0.85, 0.85, 0.85),
        )
        self.server.scene.add_grid(
            "/grid", width=8.0, height=8.0, cell_size=0.5, plane="xz",
        )

        # body primitives
        for i, vb in enumerate(self.boxes):
            self._add_box_primitive(vb, name=f"/bodies/{i}")

        # transform controls — hidden by default (toggle via "drag mode")
        for i, vb in enumerate(self.boxes):
            if i == 0:
                continue  # pinned anchor isn't draggable
            tc = self.server.scene.add_transform_controls(
                f"/drag/{i}",
                position=tuple(self.solver.positions()[vb.body.index]),
                scale=self._gizmo_scale_for(vb.body),
                line_width=4.0,
                disable_axes=False,
                disable_sliders=False,   # plane handles for 2-axis drag
                disable_rotations=True,  # translation-only
                visible=False,
            )
            vb.tc = tc
            self._wire_drag(vb)

        # GUI panel
        with self.server.gui.add_folder("Simulation"):
            self.gui_pause = self.server.gui.add_checkbox("pause", initial_value=False)
            self.gui_iters = self.server.gui.add_slider("iterations", 1, 40,
                                                       step=1, initial_value=args.iterations)
            self.gui_gravity = self.server.gui.add_slider("gravity (m/s²)", -30.0, 0.0,
                                                         step=0.5, initial_value=-9.81)
            self.gui_friction = self.server.gui.add_slider(
                "kinetic μ_d", 0.0, 1.0,
                step=0.01, initial_value=args.friction,
                hint="Coulomb dynamic / kinetic friction coefficient. "
                     "AVBD Sec 3.3: applies when the previous step's "
                     "||λ_tb|| exceeded μ_s·|λ_n| (i.e. the contact was "
                     "sliding).")
            self.gui_static_mult = self.server.gui.add_slider(
                "static mult μ_s/μ_d", 1.0, 3.0,
                step=0.05, initial_value=args.static_mult,
                hint="μ_s = this × μ_d. AVBD Sec 3.3 static/dynamic "
                     "switch: when on a previous step ||λ_tb|| ≤ μ_s·|λ_n|, "
                     "the solver uses μ_s (stiction); if the cone clamp "
                     "fires it switches back to μ_d. Standard dry-steel "
                     "ratio is ~1.5.")
            self.gui_drag_mode = self.server.gui.add_checkbox(
                "drag mode (show handles)", initial_value=False,
                hint="When on, every free box sprouts an XYZ gizmo you can "
                     "drag in 3D to teleport (orientation locked).")
        with self.server.gui.add_folder("Actions"):
            self.gui_drop = self.server.gui.add_button("drop a fresh box")
            self.gui_kick = self.server.gui.add_button("kick all (random impulse)")
            self.gui_snap = self.server.gui.add_button("snap drag handles to bodies")
            self.gui_reset = self.server.gui.add_button("reset scene")
        with self.server.gui.add_folder("Status"):
            self.gui_frame = self.server.gui.add_text("frame", initial_value="0")
            self.gui_time = self.server.gui.add_text("t (s)", initial_value="0.0")
            self.gui_maxw = self.server.gui.add_text("max |ω| (rad/s)", initial_value="0.0")
            self.gui_maxlam = self.server.gui.add_text("max |λ|", initial_value="0.0")
        with self.server.gui.add_folder("Performance"):
            self.gui_perf_device = self.server.gui.add_text(
                "device", initial_value=self.solver.device,
                hint="Warp execution device. 'cpu' here because Apple Silicon "
                     "has no CUDA; pass --device cuda:0 on an NVIDIA host.")
            self.gui_perf_bodies = self.server.gui.add_text(
                "bodies", initial_value=str(len(self.boxes)))
            self.gui_perf_constraints = self.server.gui.add_text(
                "constraints", initial_value="0",
                hint="Total rows in the AVBD constraint pool — per-body floor "
                     "rows (8 corners × 3 = 24 per box) + pins + friction.")
            self.gui_perf_colors = self.server.gui.add_text(
                "graph colors", initial_value="0",
                hint="Welsh-Powell coloring of body adjacency. primal_update_6dof "
                     "launches once per color; same-color bodies update in parallel.")
            self.gui_perf_step_ms = self.server.gui.add_text("step time", initial_value="—")
            self.gui_perf_broadphase = self.server.gui.add_text(
                "broadphase", initial_value="—",
                hint="Wall-time of the Warp-side LBVH/SAH broadphase + 15-axis "
                     "OBB-OBB SAT kernels per step. Was 56% of step time in "
                     "the Python brute-force implementation; ≤5% with the "
                     "BVH path (AVBD Alg 1 line 1).")
            self.gui_perf_static_n = self.server.gui.add_text(
                "static contacts", initial_value="—",
                hint="Count of NORMAL contact rows whose tangent pair was "
                     "within μ_s·|λ_n| on the previous step (stiction). The "
                     "remaining contacts use kinetic μ_d. AVBD Sec 3.3.")
            self.gui_perf_capacity = self.server.gui.add_text(
                "solver capacity", initial_value="—",
                hint="1 / step_time. The max sustained Hz the solver could deliver "
                     "if rendering took zero time. NOT the screen frame rate.")
            self.gui_perf_wall = self.server.gui.add_text("wall tick", initial_value="—")
        with self.server.gui.add_folder("Notes"):
            self.server.gui.add_markdown(
                "**6-DOF rigid body solver** (Solver6DOF). Each box has full "
                "SE(3) state: position, quaternion, linear + angular velocity, "
                "body-local inertia tensor. Per-body local 6×6 SPD solve via "
                "Schur-complement on 3×3 blocks.\n\n"
                "**Broadphase** is now Warp LBVH/SAH (AVBD Alg 1 line 1) + "
                "parallel 15-axis OBB SAT (replaces the Python O(N²) brute "
                "force). Face clipping (up to 4 contacts/pair) stays in "
                "Python on confirmed pairs only.\n\n"
                "**Friction** follows AVBD Sec 3.3 static/dynamic switch: "
                "μ_s applies when the previous step's ||λ_tb|| ≤ μ_s·|λ_n| "
                "(stiction); if the cone clamp fires, the contact switches "
                "back to μ_d for the rest of the step. Adjust both sliders "
                "live in the Simulation panel.\n\n"
                "**Approximate Hessian** uses the AVBD Eq 17 column-norm "
                "diagonal of G_ij for every constraint type (FLOOR, "
                "BOX_BOX, PIN, TANGENT) — guarantees the angular block "
                "stays SPD under high spin (Sec 3.5 SPD theorem).\n\n"
                "The solver substeps internally for stiff stacks "
                "(`--substeps`, default 8 — AVBD paper Fig. 6 uses 5).\n\n"
                "Drag-mode handles translate only; rotation gizmo is locked.")

        self.gui_drop.on_click(lambda _: self._drop_box())
        self.gui_kick.on_click(lambda _: self._kick_all())
        self.gui_snap.on_click(lambda _: self._snap_gizmos())
        self.gui_reset.on_click(lambda _: self._reset())
        self.gui_iters.on_update(self._iters_changed)
        self.gui_gravity.on_update(self._gravity_changed)
        self.gui_drag_mode.on_update(self._drag_mode_changed)
        self.gui_friction.on_update(self._friction_changed)
        self.gui_static_mult.on_update(self._static_mult_changed)

        # drag bookkeeping
        self._drag_targets: dict[int, np.ndarray] = {}
        # Orientation captured at the start of each per-body drag. While a body
        # is being dragged we re-apply this after every solver.step() so contact
        # torques can't spin the box out from under the user's cursor.
        self._drag_orientations: dict[int, np.ndarray] = {}
        # Time of the most recent on_drag event per body. Lets us keep the
        # body pinned across brief cursor pauses (user holding the gizmo
        # without moving the mouse) and clean up ~250 ms after release.
        self._drag_last_event: dict[int, float] = {}
        self._gizmo_suppress = False
        # Single-slot echo trap per body: (last_value, write_time). An incoming
        # on_drag event is treated as an echo of tick's gizmo write only if it
        # matches `last_value` within ~150 ms. Outside that window OR for a
        # different value, the event is a real user drag. This replaces an
        # earlier deque-based history that falsely rejected drag-back-and-forth
        # (any position the body recently visited would be misread as echo).
        self._gizmo_last_write: dict[int, tuple[np.ndarray, float]] = {}
        # self._solver_lock created at top of __init__ before scene setup.

        self._frame = 0
        self._t0 = time.perf_counter()
        self._step_ms_window: list[float] = []
        self._wall_tick_ms_window: list[float] = []
        self._last_tick_t = time.perf_counter()

    # ----- helpers --------------------------------------------------------
    def _add_box_primitive(self, vb: ViewerBox, name: str):
        with self._solver_lock:
            self.solver._flush()
            pos = tuple(self.solver.positions()[vb.body.index])
            q = self.solver.orientations()[vb.body.index]
        wxyz = warp_q_to_viser_wxyz(q)
        ex = vb.body.half_extents
        vb.handle = self.server.scene.add_box(
            name, dimensions=(2 * ex[0], 2 * ex[1], 2 * ex[2]),
            color=vb.color, position=pos, wxyz=wxyz,
        )

    def _gizmo_scale_for(self, body: RigidBody) -> float:
        """Per-body gizmo size: at least 2.5× the cube's largest half-extent
        so the axis arrows extend well past every face and the clickable
        arrow-tip targets are easy to grab. Without this the gizmo arms
        could vanish inside larger cubes."""
        h_max = max(body.half_extents)
        return max(float(self.args.gizmo_scale), h_max * 2.5)

    def _wire_drag(self, vb: ViewerBox):
        body_idx = vb.body.index

        @vb.tc.on_update
        def _on_drag(_evt):  # noqa: ANN001
            if self._gizmo_suppress:
                return
            new_p = np.asarray(vb.tc.position, dtype=np.float32)
            # Echo trap: tick writes the gizmo position every frame to follow
            # the body, and viser may echo those writes back as on_update
            # events 1–2 frames later. We only reject an event if it matches
            # the LAST server-side write within a short time window — older
            # writes are forgotten so the user can drag back to any position
            # the body previously occupied.
            last = self._gizmo_last_write.get(body_idx)
            if last is not None:
                ref_val, ref_t = last
                if (time.perf_counter() - ref_t) < 0.15:
                    if np.linalg.norm(new_p - ref_val) < 5.0e-4:  # 0.5 mm
                        return  # echo
            self._drag_targets[body_idx] = new_p
            self._drag_last_event[body_idx] = time.perf_counter()
            # Immediate visual feedback: render the cube at the new drag
            # position right away instead of waiting for the next tick.
            # Without this, the cube lags behind the gizmo by ~tick_dt
            # (16 ms at 60 Hz, more if solver.step() is expensive — at
            # substeps=8 × iters=25 the step easily takes 10–20 ms).
            # Tick skips writing handle.position/wxyz for any body in
            # _drag_targets, so this write isn't immediately clobbered.
            if vb.handle is not None:
                try:
                    vb.handle.position = (float(new_p[0]), float(new_p[1]), float(new_p[2]))
                    # Also lock the rendered orientation to the captured one
                    # so the cube doesn't visibly spin while being dragged.
                    cap_q = self._drag_orientations.get(body_idx)
                    if cap_q is not None:
                        vb.handle.wxyz = warp_q_to_viser_wxyz(cap_q)
                except (RuntimeError, AttributeError):
                    pass

    def _expire_stale_drags(self):
        """Release any per-body drag pin whose last on_drag event is older
        than 250 ms. Without this the body would stay godmoded forever after
        the user releases the gizmo, because viser's on_update fires on pose
        change (not on mouse release). 250 ms tolerates brief cursor pauses
        mid-drag while still resuming physics quickly after release."""
        EXPIRE_S = 0.25
        now = time.perf_counter()
        for body_idx in [k for k, t in self._drag_last_event.items()
                         if (now - t) > EXPIRE_S]:
            self._drag_targets.pop(body_idx, None)
            self._drag_orientations.pop(body_idx, None)
            self._drag_last_event.pop(body_idx, None)

    def boxes_by_idx(self, body_idx: int) -> ViewerBox:
        for vb in self.boxes:
            if vb.body.index == body_idx:
                return vb
        raise KeyError(body_idx)

    # ----- GUI callbacks --------------------------------------------------
    def _snap_gizmos(self):
        with self._solver_lock:
            pos = self.solver.positions()
        self._gizmo_suppress = True
        now = time.perf_counter()
        try:
            with self.server.atomic():
                for vb in self.boxes:
                    if vb.tc is not None:
                        p = pos[vb.body.index]
                        p_t = (float(p[0]), float(p[1]), float(p[2]))
                        vb.tc.position = p_t
                        self._gizmo_last_write[vb.body.index] = (
                            np.asarray(p_t, dtype=np.float32), now,
                        )
        finally:
            self._gizmo_suppress = False

    def _kick_all(self):
        rng = np.random.default_rng()
        with self._solver_lock:
            for vb in self.boxes[1:]:
                dv = tuple(float(rng.uniform(-2.5, 2.5)) for _ in range(3))
                dw = tuple(float(rng.uniform(-3.0, 3.0)) for _ in range(3))
                v_now = self.solver.velocities()[vb.body.index]
                w_now = self.solver.angular_velocities()[vb.body.index]
                self.solver.set_velocity(vb.body,
                                         (float(v_now[0] + dv[0]),
                                          float(v_now[1] + dv[1]),
                                          float(v_now[2] + dv[2])))
                self.solver.set_angular_velocity(vb.body,
                                                 (float(w_now[0] + dw[0]),
                                                  float(w_now[1] + dw[1]),
                                                  float(w_now[2] + dw[2])))

    def _drop_box(self):
        rng = np.random.default_rng()
        h = float(rng.uniform(0.12, 0.18))
        pos = (float(rng.uniform(-1.5, 1.5)),
               self.args.top_y + 0.6,
               float(rng.uniform(-1.5, 1.5)))
        q = random_orientation(rng)
        omega = tuple(float(rng.uniform(-2.0, 2.0)) for _ in range(3))
        mu = float(self.gui_friction.value)
        color = (float(rng.uniform(0.3, 0.95)),
                 float(rng.uniform(0.3, 0.95)),
                 float(rng.uniform(0.3, 0.95)))
        with self._solver_lock:
            b = self.solver.add_box(pos, (h, h, h), mass=1.0,
                                    orientation=q, angular_velocity=omega,
                                    friction=mu)
            self.solver.add_floor_contact_box(b, friction=mu)
            self.solver._flush()
            vb = ViewerBox(body=b, handle=None, tc=None, color=color)
            idx = len(self.boxes)
            self.boxes.append(vb)
        # Scene/gizmo creation goes through viser but doesn't touch solver
        # state — safe to do outside the lock so we don't block tick().
        self._add_box_primitive(vb, f"/bodies/{idx}")
        tc = self.server.scene.add_transform_controls(
            f"/drag/{idx}", position=pos, scale=self._gizmo_scale_for(vb.body),
            line_width=4.0,
            disable_sliders=False, disable_rotations=True,
            visible=bool(self.gui_drag_mode.value),
        )
        vb.tc = tc
        self._wire_drag(vb)

    def _reset(self):
        for vb in self.boxes:
            if vb.handle is not None:
                try: vb.handle.remove()
                except Exception: pass
            if vb.tc is not None:
                try: vb.tc.remove()
                except Exception: pass
        with self._solver_lock:
            self._drag_targets.clear()
            self._drag_orientations.clear()
            self._drag_last_event.clear()
            self._gizmo_last_write.clear()
            self.solver, self.boxes, self.pin_rows = build_scene(self.args)
            positions_after_build = self.solver.positions().copy()
        for i, vb in enumerate(self.boxes):
            self._add_box_primitive(vb, f"/bodies/{i}")
        for i, vb in enumerate(self.boxes):
            if i == 0:
                continue
            tc = self.server.scene.add_transform_controls(
                f"/drag/{i}",
                position=tuple(positions_after_build[vb.body.index]),
                scale=self._gizmo_scale_for(vb.body),
                line_width=4.0,
                disable_sliders=False, disable_rotations=True,
                visible=bool(self.gui_drag_mode.value),
            )
            vb.tc = tc
            self._wire_drag(vb)
        self._iters_changed(None)
        self._gravity_changed(None)
        self._frame = 0

    def _iters_changed(self, _evt):
        with self._solver_lock:
            self.solver.iterations = int(self.gui_iters.value)

    def _gravity_changed(self, _evt):
        g = float(self.gui_gravity.value)
        with self._solver_lock:
            self.solver.gravity = (0.0, g, 0.0)

    def _friction_changed(self, _evt):
        """Update μ_d (and μ_s = mult·μ_d) for every existing
        CONTACT_TANGENT_6DOF row + the per-body default so subsequent
        add_floor_contact_box calls inherit it."""
        import warp as wp
        mu = float(self.gui_friction.value)
        mu_s = mu * float(self.gui_static_mult.value)
        with self._solver_lock:
            for k in range(len(self.solver._friction)):
                self.solver._friction[k] = mu
            fric = self.solver.c_friction.numpy().copy()
            fric_s = self.solver.c_friction_static.numpy().copy()
            for i, r in enumerate(self.solver._rows):
                if r.type == 1:  # CONTACT_TANGENT_6DOF
                    r.friction = mu
                    r.friction_static = mu_s
                    fric[i] = mu
                    fric_s[i] = mu_s
            self.solver.c_friction = wp.array(
                fric, dtype=float, device=self.solver.device)
            self.solver.c_friction_static = wp.array(
                fric_s, dtype=float, device=self.solver.device)

    def _static_mult_changed(self, _evt):
        """μ_s/μ_d ratio change — recompute μ_s from current μ_d slider."""
        import warp as wp
        mu = float(self.gui_friction.value)
        mu_s = mu * float(self.gui_static_mult.value)
        with self._solver_lock:
            self.solver.friction_static_mult = float(self.gui_static_mult.value)
            fric_s = self.solver.c_friction_static.numpy().copy()
            for i, r in enumerate(self.solver._rows):
                if r.type == 1:
                    r.friction_static = mu_s
                    fric_s[i] = mu_s
            self.solver.c_friction_static = wp.array(
                fric_s, dtype=float, device=self.solver.device)

    def _drag_mode_changed(self, _evt):
        show = bool(self.gui_drag_mode.value)
        if show:
            self._snap_gizmos()
        with self.server.atomic():
            for vb in self.boxes:
                if vb.tc is not None:
                    vb.tc.visible = show

    # ----- main tick ------------------------------------------------------
    def tick(self):
        if self.gui_pause.value:
            return

        with self._solver_lock:
            # ----- pre-step: pin any actively-dragged body to its target -----
            # set_position teleports; we also capture the body's orientation on
            # the FIRST frame of each drag so we can lock it (contact torques
            # would otherwise spin the cube while the user is dragging it).
            for body_idx, target in list(self._drag_targets.items()):
                vb = self.boxes_by_idx(body_idx)
                if body_idx not in self._drag_orientations:
                    self._drag_orientations[body_idx] = \
                        self.solver.orientations()[body_idx].copy()
                cap_q = self._drag_orientations[body_idx]
                self.solver.set_position(vb.body, tuple(target))
                self.solver.set_orientation(vb.body,
                                            (float(cap_q[0]), float(cap_q[1]),
                                             float(cap_q[2]), float(cap_q[3])))
                self.solver.set_velocity(vb.body, (0.0, 0.0, 0.0))
                self.solver.set_angular_velocity(vb.body, (0.0, 0.0, 0.0))

            t0 = time.perf_counter()
            self.solver.step()
            dt = time.perf_counter() - t0

            # ----- post-step: re-pin dragged bodies ---------------------------
            # The solver's constraint pass may have nudged the body off-target
            # (floor pushes up, contact pushes sideways, friction torques the
            # orientation). Override its final state so the user sees their
            # drag honored exactly. Position+orientation are godmoded during
            # drag; physics resumes the frame after release.
            for body_idx, target in list(self._drag_targets.items()):
                vb = self.boxes_by_idx(body_idx)
                cap_q = self._drag_orientations.get(body_idx)
                self.solver.set_position(vb.body, tuple(target))
                if cap_q is not None:
                    self.solver.set_orientation(
                        vb.body, (float(cap_q[0]), float(cap_q[1]),
                                  float(cap_q[2]), float(cap_q[3])))
                self.solver.set_velocity(vb.body, (0.0, 0.0, 0.0))
                self.solver.set_angular_velocity(vb.body, (0.0, 0.0, 0.0))

            pos = self.solver.positions()
            qs = self.solver.orientations()
            w = self.solver.angular_velocities()
            lam = self.solver.lambdas()
            act = self.solver.active()
            n_rows_total = len(self.solver._rows)
            n_colors = int(self.solver.num_colors)
            bp_ms = float(self.solver.broadphase_ms)
            # Static-friction occupancy — c_was_static is per-row but only
            # meaningful on NORMAL contact rows (FLOOR / BOX_BOX). Counting
            # those gives "how many contacts are currently sticking".
            was_static = (self.solver.c_was_static.numpy()
                          if self.solver.c_was_static is not None
                          else None)
            n_static = 0
            if was_static is not None and len(was_static):
                # FLOOR_CONTACT_6DOF = 0, BOX_BOX_CONTACT_6DOF = 3
                c_type = self.solver.c_type.numpy()
                mask = (c_type == 0) | (c_type == 3)
                n_static = int((was_static[mask] != 0).sum())
        # End of lock — pos/qs/w/lam/act are now plain numpy/lists owned by
        # this thread. Scene writes and GUI text updates don't need the lock.
        drag_mode = bool(self.gui_drag_mode.value)
        self._gizmo_suppress = True
        now = time.perf_counter()
        try:
            with self.server.atomic():
                for vb in self.boxes:
                    i = vb.body.index
                    p_solver = pos[i]
                    p_render = (float(p_solver[0]), float(p_solver[1]), float(p_solver[2]))
                    wxyz = warp_q_to_viser_wxyz(qs[i])
                    # While a body is being dragged, the on_drag callback owns
                    # the rendered position + orientation. Skip our writes so
                    # we don't ping-pong against the user's drag.
                    being_dragged = i in self._drag_targets
                    if vb.handle is not None and not being_dragged:
                        try:
                            vb.handle.position = p_render
                            vb.handle.wxyz = wxyz
                        except RuntimeError:
                            vb.handle = None
                    if drag_mode and vb.tc is not None and not being_dragged:
                        try:
                            vb.tc.position = p_render
                            self._gizmo_last_write[i] = (
                                np.asarray(p_render, dtype=np.float32), now,
                            )
                        except RuntimeError:
                            vb.tc = None
        finally:
            self._gizmo_suppress = False
        # NOTE: do NOT clear _drag_targets or _drag_orientations here. They are
        # cleared lazily: each on_drag event overwrites the target; an entry
        # only "ends" when on_drag has been silent for ~150 ms (drag released).
        # See _expire_stale_drags below.
        self._expire_stale_drags()

        # HUD
        self._frame += 1
        self.gui_frame.value = str(self._frame)
        self.gui_time.value = f"{self._frame * self.solver.dt:.2f}"
        max_w = float(np.linalg.norm(w, axis=1).max()) if len(w) else 0.0
        self.gui_maxw.value = f"{max_w:.2f}"
        max_lam = float(np.abs(lam).max()) if len(lam) else 0.0
        self.gui_maxlam.value = f"{max_lam:.1f}"

        # Performance
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
        self.gui_perf_bodies.value = str(len(self.boxes))
        n_active = int(act.sum()) if len(act) else 0
        self.gui_perf_constraints.value = f"{n_active}/{n_rows_total} active"
        per_color = (len(self.boxes) / max(n_colors, 1)) if n_colors else 0.0
        self.gui_perf_colors.value = f"{n_colors}  (~{per_color:.1f} bodies/color)"
        self.gui_perf_broadphase.value = (
            f"{bp_ms:.2f} ms  ({100*bp_ms/max(step_ms,1e-3):.0f}% of step)")
        self.gui_perf_static_n.value = f"{n_static} sticking"

    def run(self):
        target_dt = self.solver.dt
        print("\nviser server running. open the URL above in a browser to interact.\n")
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-y", type=float, default=2.5,
                   help="height of the pinned anchor box's world pin")
    p.add_argument("--friction", type=float, default=0.5,
                   help="kinetic / dynamic friction coefficient μ_d "
                        "(AVBD Sec 3.3). Static μ_s = μ_d × static-mult.")
    p.add_argument("--static-mult", type=float, default=1.5,
                   help="μ_s / μ_d ratio. 1.5 ≈ dry steel; 1.0 disables "
                        "static-vs-kinetic switching.")
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--gizmo-scale", type=float, default=0.35,
                   help="minimum size (m) of the drag-handle axis arrows. "
                        "Per-body actual scale is max(this, 2.5·half_extent) "
                        "so larger cubes get larger gizmos automatically.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu",
                   help="Warp device: 'cpu' (default; only option on Apple "
                        "Silicon since Warp's CUDA backend isn't built for "
                        "macOS) or 'cuda:0' on an NVIDIA host.")
    p.add_argument("--substeps", type=int, default=8,
                   help="Number of inner sub-steps per solver.step(). Stiff "
                        "stacking needs ≥8 to converge without bouncing "
                        "(AVBD paper Fig. 6 uses 5).")
    args = p.parse_args()
    Viewer(args).run()


if __name__ == "__main__":
    main()
