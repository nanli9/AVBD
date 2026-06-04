"""Interactive 3D AVBD viewer — 6-DOF rigid body edition (viser, browser-based).

Drives `Solver6DOF`. Each body is a full rigid box with orientation; the floor
is the only collision target right now (body-body OBB-OBB contact is the next
milestone). The pinned box hangs from a single body-local corner.

Run:

    uv run python examples/viewer.py
    # open http://localhost:8181 in a browser (URL also printed on stdout)
    # (override with --port if 8181 is also taken on your machine)

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

from avbd3d import Solver, Solver6DOF, RigidBody, make_bunny


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


# Above this many bodies (e.g. a --stress block pile) we skip the per-body
# drag gizmos: one transform-control per body would create hundreds of viser
# scene nodes and make the GUI sluggish. Rendering + sim are unaffected.
_MAX_DRAG_GIZMOS = 120

# Unit cube (side 1, centered at origin) for INSTANCED rendering. Every body is
# one instance of this mesh; per-instance position/orientation/scale/color are
# pushed as arrays in a SINGLE viser message per frame (add_batched_meshes_
# simple), instead of one add_box node + two transform messages per body. On a
# 414-body pile the per-node path cost ~27 ms/frame (65% of the tick — see
# AVBD perf notes); the batched path is ~0.05 ms. side='double' so face winding
# never culls a box. Scale is anisotropic (2·half_extents) → handles non-cube
# bodies (e.g. dominoes).
_CUBE_V = np.array([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
], dtype=np.float32)
_CUBE_F = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
], dtype=np.uint32)


def build_scene(args) -> tuple[Solver6DOF, list[ViewerBox], list[int]]:
    """Build a 6-DOF scene: a grid of cube towers + a row of standing
    domino slabs (the pinned anchor box was removed — see git history).
    Stresses the OBB-OBB contact + persistent augmented-Lagrangian
    warm-start that AVBD relies on for stable stacks.

    Pass ``--stress`` to scale up into a paper-style block pile (many more,
    taller towers + a longer domino wall) for a solver stress test; tune
    with ``--stress-grid`` / ``--stress-height`` / ``--stress-dominoes``.
    Returns solver, list of viewer boxes, list of pin-row indices."""
    s = Solver6DOF(
        dt=1.0 / 60.0,
        iterations=int(args.iterations),
        gravity=(0.0, -9.81, 0.0),
        post_stabilize=True,
        device=args.device,
        substeps=int(args.substeps),
        friction_static_mult=float(args.static_mult),
        coloring_mode=str(getattr(args, "coloring", "jacobi")),
        unsafe_fixed_capacity=bool(getattr(args, "fixed_capacity", False)),
        # Fallbacks MATCH the CLI defaults below, so an ad-hoc Namespace that
        # omits these (scripts/tests calling build_scene directly) still gets
        # the current fast path rather than silently reverting to the serial one.
        primal_group_size=int(getattr(args, "primal_group", 16)),
        primal_shuffle=bool(getattr(args, "primal_shuffle", True)),
        primal_fused=bool(getattr(args, "primal_fused", True)),
        gpu_resident=bool(getattr(args, "gpu_resident", True)),
        recolor_every_substep=bool(getattr(args, "recolor_every_substep",
                                            False)),
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

    # --- 1. Pinned anchor box: removed for now (see git history to restore).
    # It was a pin-constrained box off to the side; it never touched the
    # towers, so dropping it doesn't change tower behaviour. (--top-y, which
    # set its pin height, is now unused.)

    # --- 2. Grid of cube towers ---------------------------------------------
    # grid_n × grid_n layout, each tower `tower_height` cubes tall. Cube
    # half-extent h=0.12 → full cube 24 cm. Tower spacing 0.55 m gives ~31 cm
    # gap between towers, enough to keep them from crosstalking on the first
    # substep but tight enough to look dense.
    #   --stress scales this up into a paper-style block pile (Fig. 1/6/10):
    #   many more, taller towers. Default 3×3×3 = 27 cubes; --stress defaults
    #   to 8×8×6 = 384, tunable via --stress-grid / --stress-height.
    stress = bool(getattr(args, "stress", False))
    h_cube = 0.12
    tower_spacing = 0.55
    tower_height = int(getattr(args, "stress_height", 6)) if stress else 3
    grid_n = int(getattr(args, "stress_grid", 8)) if stress else 3
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
    domino_count = int(getattr(args, "stress_dominoes", 30)) if stress else 6
    # Keep the wall just past the +z edge of the (now larger) tower grid in
    # stress mode (the grid is symmetric, so its +z edge is -grid_origin).
    domino_z = (-grid_origin + tower_spacing) if stress else 1.8
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

        # Cap on total bodies (initial + dropped). With --gpu-resident the
        # per-substep overflow readback is gone, so the pre-sized contact pools
        # must stay sufficient — bounding the body count keeps them safe (the
        # pool sizing scales with body count, with generous per-body headroom).
        # 0 → auto: current bodies + 256 headroom. The 'drop a fresh box'
        # button greys out at the cap.
        mb = int(getattr(args, "max_bodies", 0))
        self._max_bodies = mb if mb > 0 else len(self.boxes) + 256

        # ground plane. Grid sits 1mm above the box top so the two
        # surfaces don't Z-fight under camera motion (the ground box's
        # top face is at y=0 and the grid defaults to y=0 too).
        self.server.scene.add_box(
            "/ground",
            dimensions=(8.0, 0.05, 8.0),
            position=(0.0, -0.025, 0.0),
            color=(0.85, 0.85, 0.85),
        )
        self.server.scene.add_grid(
            "/grid", width=8.0, height=8.0, cell_size=0.5, plane="xz",
            position=(0.0, 0.001, 0.0),
        )

        # body primitives are one instanced mesh for all bodies (see
        # _build_batched_boxes) — built below, after the render state is
        # initialized (it needs self._batched / self._rendered_pose).

        # transform controls — hidden by default (toggle via "drag mode")
        # (Every box is draggable now that the pinned anchor is gone.)
        # Skipped for very large scenes (--stress) — see _MAX_DRAG_GIZMOS.
        if len(self.boxes) > _MAX_DRAG_GIZMOS:
            print(f"[viewer] {len(self.boxes)} bodies > {_MAX_DRAG_GIZMOS}: "
                  "per-body drag gizmos disabled for this stress scene.")
        else:
            for i, vb in enumerate(self.boxes):
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
            self.gui_substeps = self.server.gui.add_slider(
                "substeps", 1, 20, step=1, initial_value=int(args.substeps),
                hint="AVBD substeps per visual frame. The augmented-Lagrangian "
                     "penalty grows once per substep, so more substeps "
                     "stabilise stiff stacks better than more iterations "
                     "(paper Fig. 6 uses 5). Higher = stabler but slower.")
            self.gui_coloring = self.server.gui.add_dropdown(
                "coloring", ("jacobi", "jones_plassmann"),
                initial_value=self.solver.coloring_mode,
                hint="Graph-coloring algorithm for the per-color primal "
                     "updates. 'jacobi' (parallel-Jacobi greedy) is the "
                     "default and matches the AVBD paper (§4); it usually "
                     "packs fewer colors → shorter serialization chain. "
                     "'jones_plassmann' is an off-paper alternative. Physics "
                     "is identical; switch live to compare colors/speed.")
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
        with self.server.gui.add_folder("GPU solver"):
            self.gui_resident = self.server.gui.add_checkbox(
                "GPU resident (0 readbacks)",
                initial_value=bool(self.solver.gpu_resident),
                hint="Paper §4 fully-GPU-resident hot loop: ZERO per-substep "
                     "host readbacks. Double-buffers the primal so a stale "
                     "coloring is safe (same-color pairs go Jacobi), recolors "
                     "only on a body-set change, and trusts the pre-sized "
                     "pools. Default ON; toggle live and watch 'step time'. "
                     "Effective on CUDA with primal lanes > 1.")
            self.gui_primal_group = self.server.gui.add_dropdown(
                "primal lanes (G)", ("1", "8", "16", "32"),
                initial_value=str(self.solver.primal_group_size),
                hint="Warp-per-body: GPU lanes cooperating on each body's "
                     "primal update. 1 = serial one-thread-per-body; >1 splits "
                     "the constraint sum across G lanes (warp-shuffle reduce + "
                     "lane-0 Schur solve). G=16 is the robust default (32 best "
                     "for small scenes, 8 for >2000 bodies). Same AVBD math.")
            self.gui_recolor_every = self.server.gui.add_checkbox(
                "recolor every substep",
                initial_value=bool(self.solver.recolor_every_substep),
                hint="With GPU resident: recolor every substep like the paper "
                     "(Alg 1 step 2) instead of only on a body-set change. "
                     "Still readback-free, but ~3x slower on Warp (no indirect "
                     "dispatch) for negligible fidelity gain — advanced/compare "
                     "only.")
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
            self.gui_perf_render = self.server.gui.add_text(
                "render push", initial_value="—",
                hint="Wall-time spent pushing body transforms to the browser "
                     "each frame, and how many of N bodies actually moved "
                     "enough to re-send (a delta filter skips resting bodies). "
                     "For big piles this — not solver.step() — is the viser "
                     "bottleneck; a settled pile should push ~0.")
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
        self.gui_substeps.on_update(self._substeps_changed)
        self.gui_coloring.on_update(self._coloring_changed)
        self.gui_gravity.on_update(self._gravity_changed)
        self.gui_drag_mode.on_update(self._drag_mode_changed)
        self.gui_friction.on_update(self._friction_changed)
        self.gui_static_mult.on_update(self._static_mult_changed)
        self.gui_resident.on_update(self._resident_changed)
        self.gui_primal_group.on_update(self._primal_group_changed)
        self.gui_recolor_every.on_update(self._recolor_every_changed)
        self._refresh_drop_button()

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
        # B1: per-row HUD diagnostics (max λ, active/total, sticking) are
        # read back only every Nth tick — they feed text, not rendering, so
        # ~10 Hz is plenty. Between refreshes the cached strings are reused,
        # dropping 2 of the 3 per-tick stream syncs on the common path.
        self._hud_row_interval = 6
        self._hud_maxlam = "0.0"
        self._hud_constraints = "0/0 active"
        self._hud_static = "0 sticking"
        # Render-push delta filter: viser sends one websocket message per
        # transform write, and the browser re-applies all of them every frame —
        # for a few-hundred-body pile that, not the solver, is the bottleneck.
        # We skip pushing any body whose pose barely changed since its last
        # push (resting bodies → ~0 messages). Keyed by body index; reset on
        # scene rebuild. Thresholds: ~0.1 mm position, ~tiny quaternion.
        self._rendered_pose: dict[int, tuple] = {}
        self._render_pos_eps = 1.0e-4
        self._render_quat_eps = 1.0e-4
        self._render_push_ms_window: list[float] = []
        self._hud_pushed = "0/0"
        # Instanced-render handle (one BatchedMesh for all bodies) + the
        # body-index order of its instances. Built by _build_batched_boxes.
        self._batched = None
        self._batched_body_idx = np.zeros(0, dtype=np.int64)
        # Now that the render state exists, build the instanced body mesh.
        self._build_batched_boxes()

    # ----- helpers --------------------------------------------------------
    def _build_batched_boxes(self):
        """(Re)build the single instanced-cube mesh that renders every body.
        Called at startup and whenever the body set changes (drop / reset).
        One viser node for the whole scene; per-frame updates in tick() set
        batched_positions/batched_wxyzs as arrays (one message)."""
        if self._batched is not None:
            try:
                self._batched.remove()
            except Exception:
                pass
            self._batched = None
        n = len(self.boxes)
        if n == 0:
            self._batched_body_idx = np.zeros(0, dtype=np.int64)
            return
        with self._solver_lock:
            self.solver._flush()
            pos = self.solver.positions().copy()
            qs = self.solver.orientations().copy()
        idx = np.array([vb.body.index for vb in self.boxes], dtype=np.int64)
        scales = np.empty((n, 3), np.float32)
        colors = np.empty((n, 3), np.uint8)
        for k, vb in enumerate(self.boxes):
            ex = vb.body.half_extents
            scales[k] = (2.0 * ex[0], 2.0 * ex[1], 2.0 * ex[2])
            c = vb.color
            colors[k] = (int(np.clip(c[0], 0, 1) * 255),
                         int(np.clip(c[1], 0, 1) * 255),
                         int(np.clip(c[2], 0, 1) * 255))
        bp = pos[idx].astype(np.float32)
        bw = qs[idx][:, [3, 0, 1, 2]].astype(np.float32)  # xyzw → wxyz
        self._batched = self.server.scene.add_batched_meshes_simple(
            "/bodies_batched", _CUBE_V, _CUBE_F,
            batched_wxyzs=bw, batched_positions=bp,
            batched_scales=scales, batched_colors=colors,
            flat_shading=True, side="double")
        self._batched_body_idx = idx
        self._rendered_pose.clear()

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

    def _refresh_drop_button(self):
        """Grey out 'drop a fresh box' at the body cap (keeps --gpu-resident's
        pre-sized pools safe) and show the live count on the label."""
        if not hasattr(self, "gui_drop"):
            return
        n, cap = len(self.boxes), self._max_bodies
        at_cap = n >= cap
        try:
            self.gui_drop.disabled = at_cap
            self.gui_drop.label = (f"drop a fresh box ({n}/{cap})"
                                   if not at_cap
                                   else f"at body cap ({n}/{cap})")
        except Exception:
            pass

    def _drop_box(self):
        if len(self.boxes) >= self._max_bodies:
            self._refresh_drop_button()
            return
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
        # Scene creation goes through viser but doesn't touch solver state —
        # safe outside the lock. Rebuild the instanced mesh with the new body
        # (batched meshes are fixed-size, so a drop recreates the one node).
        self._build_batched_boxes()
        # Per-body drag gizmo (only meaningful below the gizmo cap).
        if len(self.boxes) <= _MAX_DRAG_GIZMOS:
            tc = self.server.scene.add_transform_controls(
                f"/drag/{idx}", position=pos,
                scale=self._gizmo_scale_for(vb.body), line_width=4.0,
                disable_sliders=False, disable_rotations=True,
                visible=bool(self.gui_drag_mode.value),
            )
            vb.tc = tc
            self._wire_drag(vb)
        self._refresh_drop_button()

    def _reset(self):
        for vb in self.boxes:
            if vb.tc is not None:
                try: vb.tc.remove()
                except Exception: pass
        with self._solver_lock:
            self._drag_targets.clear()
            self._drag_orientations.clear()
            self._drag_last_event.clear()
            self._gizmo_last_write.clear()
            self._rendered_pose.clear()
            self.solver, self.boxes, self.pin_rows = build_scene(self.args)
            positions_after_build = self.solver.positions().copy()
        # Rebuild the single instanced mesh for the fresh body set.
        self._build_batched_boxes()
        if len(self.boxes) > _MAX_DRAG_GIZMOS:
            print(f"[viewer] {len(self.boxes)} bodies > {_MAX_DRAG_GIZMOS}: "
                  "per-body drag gizmos disabled for this stress scene.")
        else:
            for i, vb in enumerate(self.boxes):
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
        self._substeps_changed(None)
        self._gravity_changed(None)
        self._refresh_drop_button()
        self._frame = 0

    def _iters_changed(self, _evt):
        with self._solver_lock:
            self.solver.iterations = int(self.gui_iters.value)

    def _substeps_changed(self, _evt):
        with self._solver_lock:
            # Plain int attribute; the next step() re-derives sub_dt and the
            # graph signature picks up the new dt, forcing a clean recapture.
            self.solver.substeps = max(1, int(self.gui_substeps.value))

    def _coloring_changed(self, _evt):
        with self._solver_lock:
            self.solver.coloring_mode = str(self.gui_coloring.value)

    def _resident_changed(self, _evt):
        # In the graph signature → the next step() recaptures with the new
        # mode. Double-buffer arrays are always allocated, so toggling either
        # way is safe mid-sim.
        with self._solver_lock:
            self.solver.gpu_resident = bool(self.gui_resident.value)

    def _primal_group_changed(self, _evt):
        with self._solver_lock:
            self.solver.primal_group_size = int(self.gui_primal_group.value)

    def _recolor_every_changed(self, _evt):
        with self._solver_lock:
            self.solver.recolor_every_substep = bool(self.gui_recolor_every.value)

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

            # Batched readback — see Solver6DOF.read_state_batched.
            # AVBD_PERFORMANCE_GAP §6: this cuts the per-tick stream syncs
            # from ~7 to 2 by packing everything the HUD + scene update
            # needs into two pre-allocated Warp buffers.
            # B1: only pull the HUD-only per-row diagnostics every Nth tick.
            want_rows = (self._frame % self._hud_row_interval) == 0
            state = self.solver.read_state_batched(include_rows=want_rows)
            pos = state["positions"]
            qs = state["orientations"]
            w = state["angular_velocities"]
            lam = state["lambdas"]
            act = state["active"]
            was_static = state["was_static"]
            c_type = state["c_type"]
            n_rows_total = int(state.get("n_rows", len(lam)))
            n_colors = int(self.solver.num_colors)
            bp_ms = float(self.solver.broadphase_ms)
            # Static-friction occupancy — c_was_static is per-row but only
            # meaningful on NORMAL contact rows (FLOOR / BOX_BOX). Counting
            # those gives "how many contacts are currently sticking".
            if want_rows:
                n_static = 0
                if was_static is not None and len(was_static):
                    # FLOOR_CONTACT_6DOF = 0, BOX_BOX_CONTACT_6DOF = 3
                    mask = (c_type == 0) | (c_type == 3)
                    n_static = int((was_static[mask] != 0).sum())
                self._hud_static = f"{n_static} sticking"
            # Snapshot the boxes list under the lock — bodies whose index
            # is past `len(pos)` would IndexError below. `_drop_box`
            # appends to self.boxes inside the solver lock, so taking
            # this snapshot here pins the visible-body set to the same
            # point in time as `pos` / `qs` / `w`.
            boxes_snapshot = list(self.boxes)
        # End of lock — pos/qs/w/lam/act are now plain numpy/lists owned by
        # this thread. Scene writes and GUI text updates don't need the lock.
        drag_mode = bool(self.gui_drag_mode.value)
        self._gizmo_suppress = True
        now = time.perf_counter()
        push_t0 = now
        n_inst = 0
        try:
            # ---- instanced render: ALL bodies in one batched update ----
            # Two array writes (positions + orientations) → one viser message
            # + one instanced draw call, regardless of body count. This is the
            # fix for the per-node bottleneck (was ~27 ms/frame at 414 bodies).
            idx = self._batched_body_idx
            if (self._batched is not None and len(idx)
                    and int(idx.max()) < len(pos)):
                bp = pos[idx].astype(np.float32)
                bw = qs[idx][:, [3, 0, 1, 2]].astype(np.float32)  # xyzw→wxyz
                self._batched.batched_positions = bp
                self._batched.batched_wxyzs = bw
                n_inst = len(idx)
            # ---- drag gizmos (small scenes only; capped at _MAX_DRAG_GIZMOS) --
            if drag_mode:
                with self.server.atomic():
                    for vb in boxes_snapshot:
                        i = vb.body.index
                        if i >= len(pos) or vb.tc is None or i in self._drag_targets:
                            continue
                        p = pos[i]
                        p_render = (float(p[0]), float(p[1]), float(p[2]))
                        try:
                            vb.tc.position = p_render
                            self._gizmo_last_write[i] = (
                                np.asarray(p_render, dtype=np.float32), now)
                        except RuntimeError:
                            vb.tc = None
        finally:
            self._gizmo_suppress = False
        self._render_push_ms_window.append(
            (time.perf_counter() - push_t0) * 1000.0)
        if len(self._render_push_ms_window) > 30:
            self._render_push_ms_window.pop(0)
        self._hud_pushed = f"{n_inst} instanced"
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
        if want_rows:
            max_lam = float(np.abs(lam).max()) if len(lam) else 0.0
            self._hud_maxlam = f"{max_lam:.1f}"
        self.gui_maxlam.value = self._hud_maxlam

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
        render_ms = (float(np.mean(self._render_push_ms_window))
                     if self._render_push_ms_window else 0.0)
        self.gui_perf_render.value = f"{render_ms:.2f} ms  ({self._hud_pushed} moved)"
        self.gui_perf_bodies.value = str(len(self.boxes))
        if want_rows:
            n_active = int(act.sum()) if len(act) else 0
            self._hud_constraints = f"{n_active}/{n_rows_total} active"
        self.gui_perf_constraints.value = self._hud_constraints
        n_active_colors = int(self.solver.num_active_colors)
        per_color = (len(self.boxes) / max(n_active_colors, 1)) \
            if n_active_colors else 0.0
        self.gui_perf_colors.value = (
            f"{n_active_colors} active / {n_colors} cap  "
            f"(~{per_color:.1f} bodies/color, {self.solver.coloring_mode})")
        self.gui_perf_broadphase.value = (
            f"{bp_ms:.2f} ms  ({100*bp_ms/max(step_ms,1e-3):.0f}% of step)")
        self.gui_perf_static_n.value = self._hud_static

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


# =============================================================================
# Deformable bunny viewer
# =============================================================================
# Drives the 3-DOF particle `Solver` (not Solver6DOF) with one particle per
# tet vertex + one DISTANCE constraint per tet edge. This is a v1 mass-spring
# stand-in for proper co-rotated FEM (which AVBD/VBD use for deformables).
# Per-vertex rendering shows the actual simulation DOFs; the deformed surface
# mesh is the tet-boundary triangulation re-emitted each frame.
# =============================================================================
class DeformableViewer:
    def __init__(self, args):
        self.args = args
        self._solver_lock = threading.RLock()

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        try:
            self.server.scene.set_up_direction("+y")
        except Exception:
            pass

        # Bunny mode follows the AVBD paper budget: iterations=4 per
        # substep + multiple substeps per visual frame. The shared
        # --iterations arg now defaults to 4 (the paper budget), so the
        # bunny just uses it directly; pass --iterations N to override.
        # The shared --substeps arg controls how many AVBD substeps run
        # per tick. At 4 iters × 8 substeps the effective work per frame
        # matches the old 25-iters single-step setup but resolves plate
        # contact penetration much better when squeezing hard.
        bunny_iters = int(args.iterations)
        self._bunny_substeps = max(1, int(args.substeps))
        # Gravity off by default in the bunny scene — the canonical demo is
        # the two-plate squash, which reads cleaner with the bunny at rest
        # between the plates rather than sitting on the floor under gravity.
        # The slider stays exposed so the user can re-enable it.
        self.solver = Solver(
            dt=1.0 / 60.0,
            iterations=bunny_iters,
            gravity=(0.0, 0.0, 0.0),
            post_stabilize=True,
            device=args.device,
        )
        print(f"[viewer] building tet bunny at resolution={args.bunny_resolution} ...")
        t0 = time.perf_counter()
        # Squash demo defaults to a much softer volume preservation than the
        # global --volume-stiffness=1e4. At 1e4 the bunny is effectively
        # incompressible and the plates can't push verts past ~30 % of the
        # bunny's natural x-extent without plate-contact penetration; at
        # 1e3 the bunny visibly compresses to ~10 cm wide with only a few
        # mm of plate penetration. The user can override with
        # --volume-stiffness or via the live GUI slider.
        bunny_vol_k = (1.0e3 if float(args.volume_stiffness) == 1.0e4
                       else float(args.volume_stiffness))
        self._bunny_vol_k = bunny_vol_k
        self.deform = make_bunny(
            self.solver,
            resolution=int(args.bunny_resolution),
            scale=float(args.bunny_scale),
            center=(0.0, float(args.bunny_drop_y), 0.0),
            edge_stiffness=float(args.edge_stiffness),
            volume_stiffness=bunny_vol_k,
            friction=float(args.friction),
            floor_y=0.0,
            surface_collide=False,
        )
        # Anchor pin: when the plates close quickly the right plate can hit
        # one side of the bunny before the left plate is fully resisting,
        # and the resulting impulse pushes the whole body sideways past the
        # other plate (AVBD only resolves a few iters per substep, so big
        # constraint violations created by a single slider drag aren't
        # fully snapped back in one frame). Anchor the tet vertex closest
        # to the bunny's rest centre on all three axes via a soft pin —
        # the surface verts are still free to compress in toward this
        # anchor (and bulge in y/z) under the plate constraints, but the
        # body as a whole can't drift off centre.
        rest_positions = self.solver.positions().copy()
        tet_rest = rest_positions[self.deform.indices]
        bunny_center = np.array(
            [0.0, float(args.bunny_drop_y), 0.0], dtype=np.float32)
        center_tet_id = int(np.argmin(
            np.linalg.norm(tet_rest - bunny_center, axis=1)))
        self._anchor_pin = self.solver.add_pin_axes(
            self.deform.bodies[center_tet_id],
            world_point=tuple(map(float, tet_rest[center_tet_id])),
            axes=(True, True, True),
            stiffness=1.0e3,
        )

        # Cache initial state for reset.
        self._initial_positions = self.solver.positions().copy()
        self._initial_velocities = self.solver.velocities().copy()
        print(f"[viewer] bunny built in {time.perf_counter()-t0:.2f}s  "
              f"verts={len(self.deform.bodies)}  "
              f"edges={len(self.deform.edge_constraints)}  "
              f"surface_tris={len(self.deform.tet.surface_tris)}")

        # Static scene: ground + grid. Grid sits 1mm above the box top
        # to avoid Z-fighting (both surfaces are at y=0 otherwise).
        self.server.scene.add_box(
            "/ground",
            dimensions=(8.0, 0.05, 8.0),
            position=(0.0, -0.025, 0.0),
            color=(0.85, 0.85, 0.85),
        )
        self.server.scene.add_grid(
            "/grid", width=8.0, height=8.0, cell_size=0.5, plane="xz",
            position=(0.0, 0.001, 0.0),
        )

        # ---- Two transparent glass plates that squash the bunny ----------
        # Each plate is a thin axis-aligned box with normal ±x, spanning
        # the plate-height in y and plate-depth in z. Per surface vertex of
        # the tet skin we register one PLANE_CONTACT row pointed inward;
        # the inner-face separation is driven by a GUI slider (plate
        # distance), so the user controls the squash directly instead of a
        # canned animation loop.
        self._plate_thickness = 0.04
        # Plate height (y span) and depth (z span). Bunny in this scene
        # uses scale=1.2 by default and its bbox is roughly that tall, so
        # we make the plate substantially taller so it reads as a tall
        # piece of glass even when the bunny stretches under squash.
        self._plate_half_y = float(args.plate_height) * 0.5
        self._plate_half_z = float(args.bunny_scale) * 1.5
        # Allowed slider range, in metres (full distance between the
        # inner faces — so 0 is "the plates touch"). The slider's max sits
        # well outside the bunny so the starting state is fully open.
        self._plate_min_distance = float(args.plate_min_distance)
        self._plate_max_distance = float(args.bunny_scale) * 2.2
        self._plate_initial_distance = self._plate_max_distance
        # Plate vertical centre: keep it at the bunny's centre height so
        # the squash band lines up with the body even after we make the
        # plates much taller than before.
        self._plate_center_y = float(args.bunny_drop_y)

        # Bind one PLANE_CONTACT per surface vertex per plate. The
        # surface_verts list is the tet-mesh boundary; deform.bodies is
        # indexed the same way as the tet vertex array.
        self._plate_left_handles: list = []
        self._plate_right_handles: list = []
        start_inner_x = 0.5 * self._plate_initial_distance
        for sv in self.deform.tet.surface_verts:
            body = self.deform.bodies[int(sv)]
            self._plate_left_handles.append(
                self.solver.add_plane_contact(
                    body, normal=(1.0, 0.0, 0.0),
                    offset=-start_inner_x,
                )
            )
            self._plate_right_handles.append(
                self.solver.add_plane_contact(
                    body, normal=(-1.0, 0.0, 0.0),
                    offset=-start_inner_x,
                )
            )
        # Spawn the glass visuals. viser's add_box accepts opacity for
        # standard-shader transparency.
        half_t = 0.5 * self._plate_thickness
        self._plate_left_handle_vis = self.server.scene.add_box(
            "/plate_left",
            dimensions=(self._plate_thickness,
                        2.0 * self._plate_half_y,
                        2.0 * self._plate_half_z),
            position=(-(start_inner_x + half_t), self._plate_center_y, 0.0),
            color=(170, 220, 255),
            opacity=0.35,
        )
        self._plate_right_handle_vis = self.server.scene.add_box(
            "/plate_right",
            dimensions=(self._plate_thickness,
                        2.0 * self._plate_half_y,
                        2.0 * self._plate_half_z),
            position=(+(start_inner_x + half_t), self._plate_center_y, 0.0),
            color=(170, 220, 255),
            opacity=0.35,
        )

        # First render: surface mesh + vertex point cloud
        self._mesh_handle = None
        self._points_handle = None
        self._refresh_mesh()

        # GUI panel
        with self.server.gui.add_folder("Simulation"):
            self.gui_pause = self.server.gui.add_checkbox(
                "pause", initial_value=False)
            self.gui_iters = self.server.gui.add_slider(
                "iterations", 1, 40, step=1,
                initial_value=bunny_iters,
                hint="AVBD primal/dual iterations per substep. Paper "
                     "default is 4. Combined with the substep count "
                     "(--substeps, default 8) this gives 32 effective "
                     "work iters per visual frame.")
            self.gui_gravity = self.server.gui.add_slider(
                "gravity (m/s²)", -30.0, 0.0, step=0.5,
                initial_value=0.0,
                hint="0 by default in the bunny demo so the plate squash "
                     "reads cleanly. Slide down to re-enable gravity.")
            self.gui_edge_k = self.server.gui.add_slider(
                "edge stiffness (×1e4)", 0.1, 50.0, step=0.1,
                initial_value=float(args.edge_stiffness) / 1e4,
                hint="Per-tet-edge DISTANCE constraint stiffness. AVBD "
                     "clamps each constraint's penalty to this ceiling. "
                     "Higher = stiffer body, more iterations needed to "
                     "converge.")
            self.gui_vol_k = self.server.gui.add_slider(
                "volume stiffness (×1e3)", 0.05, 30.0, step=0.05,
                initial_value=self._bunny_vol_k / 1e3,
                hint="Per-tet TET_VOLUME constraint stiffness — how hard "
                     "the bunny resists compression. Lower this if you "
                     "see plate-contact penetration during a tight "
                     "squash; raise it for a stiffer rubbery body.")
            self.gui_plate_distance = self.server.gui.add_slider(
                "plate distance (m)",
                self._plate_min_distance,
                self._plate_max_distance,
                step=0.01,
                initial_value=self._plate_initial_distance,
                hint="Inner-face gap between the two glass plates. Slide "
                     "down to squash the bunny; slide back up to release. "
                     "0 = plates touch at the centre.")
        with self.server.gui.add_folder("Rendering"):
            self.gui_show_mesh = self.server.gui.add_checkbox(
                "show bunny mesh (high-res, skinned)",
                initial_value=True)
            self.gui_show_points = self.server.gui.add_checkbox(
                "show tet vertices (sim DOFs)",
                initial_value=False,
                hint="The actual AVBD 3-DOF blocks: red on the surface, "
                     "blue interior. The high-res bunny is barycentrically "
                     "skinned to these.")
            self.gui_wireframe = self.server.gui.add_checkbox(
                "wireframe", initial_value=False)
            self.gui_point_size = self.server.gui.add_slider(
                "tet point size (mm)", 1.0, 25.0, step=0.5,
                initial_value=6.0)
        with self.server.gui.add_folder("Actions"):
            self.gui_shake = self.server.gui.add_button("shake (random impulse)")
            self.gui_squish = self.server.gui.add_button("squish (-y push, all verts)")
            self.gui_lift = self.server.gui.add_button("lift (+y push, all verts)")
            self.gui_reset = self.server.gui.add_button("reset bunny")
        with self.server.gui.add_folder("Status"):
            self.gui_frame = self.server.gui.add_text("frame", initial_value="0")
            self.gui_time = self.server.gui.add_text("t (s)", initial_value="0.0")
            self.gui_step_ms = self.server.gui.add_text("step time", initial_value="—")
            self.gui_capacity = self.server.gui.add_text(
                "solver capacity", initial_value="—",
                hint="1 / step_time. Max sustained Hz the solver could deliver "
                     "if rendering took zero time.")
            self.gui_verts_text = self.server.gui.add_text(
                "vertices (= 3-DOF blocks)",
                initial_value=str(len(self.deform.bodies)))
            self.gui_edges_text = self.server.gui.add_text(
                "tet edges (= DISTANCE rows)",
                initial_value=str(len(self.deform.edge_constraints)))
            self.gui_tets_text = self.server.gui.add_text(
                "tets",
                initial_value=str(len(self.deform.tet.tets)))
            self.gui_max_disp = self.server.gui.add_text(
                "max |Δx| (mm)", initial_value="0.0",
                hint="Max per-vertex displacement from the initial rest pose, "
                     "in millimetres. A useful proxy for how deformed the bunny is.")
        with self.server.gui.add_folder("Notes"):
            self.server.gui.add_markdown(
                "**Deformable bunny** — 3-DOF particle Solver. Each tet "
                "vertex is one particle. Each tet edge is one AVBD "
                "`DISTANCE` constraint; each tet is one `TET_VOLUME` "
                "constraint (`C = V/V₀ − 1`) for soft volume "
                "preservation.\n\n"
                "**What you see:** the high-res Stanford bunny surface "
                "(~35k verts, 69k tris) **skinned** to the tet pool via "
                "per-vertex barycentric coordinates pre-computed at "
                "startup. Each frame: `render_vert = Σ b_k · "
                "tet_vert[v_k]` for the 4 vertices of the containing "
                "tet. This is why it looks like a bunny instead of a "
                "pile of voxels — the rendered surface is decoupled "
                "from the (coarse) sim mesh.\n\n"
                "Self-collision is OFF (`surface_collide=False`): zero "
                "SPHERE_CONTACT rows exist in the pool. Any visible "
                "surface 'collision' is just the high-res skin "
                "following its underlying tets through deformation.\n\n"
                "**Next milestone:** per-tet co-rotated linear FEM "
                "(Neo-Hookean / StVK) for shear stiffness — currently "
                "edges + volume preservation only.")

        self.gui_shake.on_click(lambda _: self._shake())
        self.gui_squish.on_click(lambda _: self._impulse_all((0.0, -2.0, 0.0)))
        self.gui_lift.on_click(lambda _: self._impulse_all((0.0, 3.5, 0.0)))
        self.gui_reset.on_click(lambda _: self._reset())
        self.gui_iters.on_update(self._iters_changed)
        self.gui_gravity.on_update(self._gravity_changed)
        self.gui_edge_k.on_update(self._edge_k_changed)
        self.gui_vol_k.on_update(self._vol_k_changed)
        self.gui_plate_distance.on_update(self._plate_distance_changed)

        self._frame = 0
        self._step_ms_window: list[float] = []

    # ----- rendering ------------------------------------------------------
    def _refresh_mesh(self):
        """Tear down + rebuild the SKINNED bunny mesh (high-res Stanford
        surface bound to the tet pool via barycentric weights) plus an
        optional vertex point cloud showing where the actual AVBD 3-DOF
        blocks sit. Wrapped in `server.atomic()` so the client never sees
        a mid-update empty state.
        """
        with self._solver_lock:
            pos = self.solver.positions().copy()
        tet_pos = pos[self.deform.indices].astype(np.float32)
        # Apply the barycentric skin to get the deformed high-res bunny.
        # ~3 ms per call for the 35k-vert Stanford bunny.
        render_verts = self.deform.skin_positions(tet_pos)
        render_tris = self.deform.skin.render_tris

        wireframe = (getattr(self, "gui_wireframe", None) is not None
                     and self.gui_wireframe.value)
        show_mesh = (getattr(self, "gui_show_mesh", None) is None
                     or self.gui_show_mesh.value)
        show_points = (getattr(self, "gui_show_points", None) is None
                       or self.gui_show_points.value)

        # Point cloud colors for the TET (sim) vertices — distinct from the
        # render mesh so it's clear what's physics and what's pure render.
        n_pts = len(tet_pos)
        pt_colors = np.empty((n_pts, 3), dtype=np.uint8)
        pt_colors[:] = (40, 90, 220)  # deeper blue for interior tet verts
        surf = self.deform.tet.surface_verts
        pt_colors[surf] = (255, 90, 60)  # warm red-orange for surface tet verts

        with self.server.atomic():
            if self._mesh_handle is not None:
                try:
                    self._mesh_handle.remove()
                except Exception:
                    pass
                self._mesh_handle = None
            if self._points_handle is not None:
                try:
                    self._points_handle.remove()
                except Exception:
                    pass
                self._points_handle = None

            if show_mesh:
                # The Stanford bunny is rendered as a high-res
                # (~35k vert, 69k tri) skinned mesh — looks like an actual
                # bunny, no voxel blocks. viser rejects
                # (wireframe=True, flat_shading=True); set only one.
                mesh_kwargs = dict(
                    name="/bunny/surface",
                    vertices=render_verts,
                    faces=render_tris,
                    color=(220, 195, 175),  # warm parchment
                    # The Stanford bunny has an open bottom (the original
                    # scan didn't capture the base). With front-only culling
                    # the camera ray enters the front and exits through that
                    # hole onto the background, making the lower body look
                    # transparent. Double-sided rendering draws the inside
                    # of the front face too, so the silhouette stays closed.
                    side="double",
                )
                if wireframe:
                    mesh_kwargs["wireframe"] = True
                else:
                    mesh_kwargs["material"] = "standard"
                self._mesh_handle = self.server.scene.add_mesh_simple(**mesh_kwargs)

            if show_points:
                # The point cloud still shows the actual SIMULATION DOFs
                # (the tet vertices, not the render verts) so the user
                # can see where AVBD is doing its 3-DOF block solves.
                pt_size_m = (getattr(self, "gui_point_size", None).value / 1000.0
                             if getattr(self, "gui_point_size", None) is not None
                             else 0.006)
                self._points_handle = self.server.scene.add_point_cloud(
                    "/bunny/vertices",
                    points=tet_pos,
                    colors=pt_colors,
                    point_size=float(pt_size_m),
                    point_shape="circle",
                    point_shading="gradient",
                )

    # ----- actions --------------------------------------------------------
    def _shake(self):
        rng = np.random.default_rng()
        with self._solver_lock:
            v = self.solver.velocities().copy()
            n = len(self.deform.bodies)
            v[self.deform.indices] += rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)
            self.solver.v = self._wp_array_vec3(v)

    def _impulse_all(self, dv: tuple[float, float, float]):
        with self._solver_lock:
            v = self.solver.velocities().copy()
            for idx in self.deform.indices:
                v[int(idx)] += np.asarray(dv, dtype=np.float32)
            self.solver.v = self._wp_array_vec3(v)

    def _wp_array_vec3(self, arr: np.ndarray):
        import warp as wp
        return wp.array(arr.astype(np.float32), dtype=wp.vec3, device=self.solver.device)

    def _reset(self):
        with self._solver_lock:
            self.solver.x = self._wp_array_vec3(self._initial_positions.copy())
            self.solver.v = self._wp_array_vec3(np.zeros_like(self._initial_velocities))
            self.solver.prev_v = self._wp_array_vec3(np.zeros_like(self._initial_velocities))
            # Reset λ and penalty caches so the bunny re-settles cleanly.
            n_c = len(self.solver._constraints)
            self.solver.c_lambda = self._wp_scalar(np.zeros(n_c, dtype=np.float32))
            self.solver.c_penalty = self._wp_scalar(np.ones(n_c, dtype=np.float32))
            n = len(self.solver._constraints)
            self.solver.c_active = self._wp_int(np.ones(n, dtype=np.int32))
            # Snap plates back to fully open and reset the slider so the
            # bunny re-settles without being trapped by a half-closed
            # plate from the previous run.
            self.gui_plate_distance.value = self._plate_initial_distance
            self._apply_plate_distance(self._plate_initial_distance)
        self._frame = 0

    def _wp_scalar(self, arr: np.ndarray):
        import warp as wp
        return wp.array(arr.astype(np.float32), dtype=float, device=self.solver.device)

    def _wp_int(self, arr: np.ndarray):
        import warp as wp
        return wp.array(arr.astype(np.int32), dtype=int, device=self.solver.device)

    # ----- GUI handlers ---------------------------------------------------
    def _iters_changed(self, _evt):
        with self._solver_lock:
            self.solver.iterations = int(self.gui_iters.value)

    def _gravity_changed(self, _evt):
        with self._solver_lock:
            self.solver.gravity = (0.0, float(self.gui_gravity.value), 0.0)

    def _edge_k_changed(self, _evt):
        new_k = float(self.gui_edge_k.value) * 1e4
        with self._solver_lock:
            for h in self.deform.edge_constraints:
                # ConstraintHandle stores `index` (first row); DISTANCE has 1 row.
                self.solver._constraints[h.index].stiffness = new_k
            self.solver._dirty = True

    def _vol_k_changed(self, _evt):
        """Live-update the per-tet volume preservation stiffness. Writes
        directly to the GPU `tet_stiffness` array so the slider responds
        without a _flush rebuild. AVBD clamps each tet's penalty to this
        ceiling at the next warmstart, so lowering this softens the body
        within one frame.
        """
        new_k = float(self.gui_vol_k.value) * 1e3
        with self._solver_lock:
            self.solver._flush()
            if self.solver.tet_stiffness is None:
                return
            n_t = int(self.solver.tet_stiffness.shape[0])
            import warp as wp
            self.solver.tet_stiffness = wp.array(
                np.full(n_t, new_k, dtype=np.float32),
                dtype=float, device=self.solver.device)

    # ----- plate squash (manual slider) ----------------------------------
    def _apply_plate_distance(self, distance: float) -> None:
        """Push the slider-set inner-face separation into both the
        constraint pool (so the bunny gets pushed) and the viser visuals
        (so the glass tracks). Distance is the total gap between the two
        inner faces — half of it is the per-side x offset.
        """
        x_inner = 0.5 * max(float(distance), 0.0)
        # Left plate normal=(+1,0,0): C = x − (−x_inner) ≥ 0 → x ≥ −x_inner.
        # Right plate normal=(−1,0,0): C = −x − (−x_inner) ≥ 0 → x ≤ +x_inner.
        self.solver.set_plane_contact_bulk(
            self._plate_left_handles, offset=-x_inner)
        self.solver.set_plane_contact_bulk(
            self._plate_right_handles, offset=-x_inner)
        # Box centre sits half-thickness outside the inner face so the
        # visible glass surface aligns with the constraint plane.
        half_t = 0.5 * self._plate_thickness
        self._plate_left_handle_vis.position = (
            -(x_inner + half_t), self._plate_center_y, 0.0)
        self._plate_right_handle_vis.position = (
            +(x_inner + half_t), self._plate_center_y, 0.0)

    def _plate_distance_changed(self, _evt):
        with self._solver_lock:
            self._apply_plate_distance(float(self.gui_plate_distance.value))

    # ----- tick -----------------------------------------------------------
    def tick(self):
        if self.gui_pause.value:
            return
        with self._solver_lock:
            # Manually substep the 3-DOF solver: each visual frame breaks
            # into N AVBD substeps of dt/N. Plate-contact penetration at
            # iterations=4 disappears with 8 substeps because the penalty
            # has 8× as many chances to grow per visual frame.
            full_dt = self.solver.dt
            sub_dt = full_dt / float(self._bunny_substeps)
            t0 = time.perf_counter()
            self.solver.dt = sub_dt
            try:
                for _ in range(self._bunny_substeps):
                    self.solver.step()
            finally:
                self.solver.dt = full_dt
            dt = time.perf_counter() - t0
            pos = self.solver.positions().copy()
        # Refresh mesh + points
        self._refresh_mesh()
        # HUD
        self._frame += 1
        self.gui_frame.value = str(self._frame)
        self.gui_time.value = f"{self._frame * self.solver.dt:.2f}"
        self._step_ms_window.append(dt * 1000.0)
        if len(self._step_ms_window) > 30:
            self._step_ms_window.pop(0)
        step_ms = float(np.mean(self._step_ms_window))
        self.gui_step_ms.value = f"{step_ms:.2f} ms"
        self.gui_capacity.value = f"{1000.0/max(step_ms, 1e-3):.0f} Hz"
        # Max displacement from rest
        verts_now = pos[self.deform.indices]
        rest = self._initial_positions[self.deform.indices]
        disp = np.linalg.norm(verts_now - rest, axis=1).max() * 1000.0
        self.gui_max_disp.value = f"{disp:.1f}"

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
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--gizmo-scale", type=float, default=0.35,
                   help="minimum size (m) of the drag-handle axis arrows. "
                        "Per-body actual scale is max(this, 2.5·half_extent) "
                        "so larger cubes get larger gizmos automatically.")
    p.add_argument("--port", type=int, default=8181,
                   help="HTTP/WS port for the viser server. 8080 (viser "
                        "default) commonly clashes with MCP / Tomcat / dev "
                        "servers; 8090 clashes with Unity. 8181 is a safer "
                        "default — override with --port if it's also taken.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu",
                   help="Warp device: 'cpu' (default; only option on Apple "
                        "Silicon since Warp's CUDA backend isn't built for "
                        "macOS) or 'cuda:0' on an NVIDIA host.")
    p.add_argument("--substeps", type=int, default=8,
                   help="Number of inner sub-steps per solver.step(). Stiff "
                        "stacking needs ≥8 to converge without bouncing "
                        "(AVBD paper Fig. 6 uses 5).")
    p.add_argument("--coloring", type=str, default="jacobi",
                   choices=("jacobi", "jones_plassmann"),
                   help="Graph-coloring algorithm used to parallelize the "
                        "per-color primal updates. 'jacobi' (parallel-Jacobi "
                        "greedy) is the default and matches the AVBD paper "
                        "(§4); it packs colors tighter than Jones–Plassmann, "
                        "shrinking the serialization chain. Switchable live "
                        "in the GUI; the AVBD solve is identical either way.")
    # ---- Stress-test scene (paper-style block pile) -------------------------
    p.add_argument("--stress", action="store_true",
                   help="Scale the rigid scene up into a paper-style block "
                        "pile (Fig. 1/6/10): many more, taller cube towers + "
                        "a longer domino wall, to stress the solver. Default "
                        "3×3×3=27 cubes becomes 8×8×6=384. Use a GPU "
                        "(--device cuda:0) for interactive rates.")
    p.add_argument("--stress-grid", type=int, default=8,
                   help="With --stress: towers per side (grid is N×N). "
                        "8 → 64 towers.")
    p.add_argument("--stress-height", type=int, default=6,
                   help="With --stress: cubes stacked per tower.")
    p.add_argument("--stress-dominoes", type=int, default=30,
                   help="With --stress: number of standing domino slabs in "
                        "the wall.")
    p.add_argument("--primal-group", type=int, default=16,
                   help="Perf (warp-per-body primal): GPU lanes cooperating on "
                        "each body's primal update. 1 = serial one-thread-per-"
                        "body. >1 splits each body's constraint sum across G "
                        "lanes to raise occupancy — measured 2.8-5.1x on RTX "
                        "3060. DEFAULT 16 (robust all-rounder; 32 is best for "
                        "small/medium, 8 for >1000-body scenes). Same AVBD "
                        "math (reduction reorder below the atomic noise "
                        "floor). CUDA only (ignored on --device cpu).")
    p.add_argument("--primal-shuffle", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="With --primal-group >1: join the per-body lane "
                        "partials with a warp-shuffle (__shfl_down_sync) "
                        "register reduction instead of atomic_add. "
                        "Contention-free and faster everywhere — ON by "
                        "default. Use --no-primal-shuffle for the atomic "
                        "join. Same math.")
    p.add_argument("--primal-fused", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="With --primal-group >1 and --primal-shuffle: fuse the "
                        "cooperative reduction and the per-body Schur solve into "
                        "ONE kernel (lane 0 solves in registers) instead of "
                        "writing per-body sums to global scratch and launching a "
                        "separate low-occupancy solve. ON by default; "
                        "--no-primal-fused restores the two-kernel split. Same "
                        "math.")
    p.add_argument("--gpu-resident", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Paper §4 fully-GPU-resident hot loop (CUDA, fused "
                        "shuffle primal): ZERO per-substep host readbacks. "
                        "Double-buffers the primal (so a stale coloring is safe "
                        "— same-color pairs go Jacobi, per the paper), recolors "
                        "only when the body set changes, and trusts the "
                        "pre-sized pools. ON by default; --no-gpu-resident "
                        "restores the per-substep overflow/conflict readbacks. "
                        "Capacity is kept safe by the droppable-box cap "
                        "(--max-bodies).")
    p.add_argument("--recolor-every-substep", action="store_true",
                   help="With --gpu-resident: recolour every substep like the "
                        "paper (Alg 1 step 2) instead of only on a body-set "
                        "change. Still readback-free (fixed colouring rounds + "
                        "fixed MAX_COLORS primal loop). Fresher colouring tracks "
                        "the serial reference more tightly on dynamic scenes, at "
                        "the cost of per-substep recolour compute + empty-colour "
                        "launches.")
    p.add_argument("--max-bodies", type=int, default=0,
                   help="Cap on total bodies (initial + dropped). 0 = auto "
                        "(initial count + 256 headroom). Bounds the pre-sized "
                        "contact pools so --gpu-resident never overflows; the "
                        "viewer greys out 'drop a fresh box' at the cap.")
    p.add_argument("--fixed-capacity", action="store_true",
                   help="Perf (A3): drop the two per-substep host syncs that "
                        "only detect contact-pool overflow, trusting the "
                        "pre-sized pools. ~1.1–1.3 ms/frame faster, physics "
                        "identical — but contacts are silently dropped if the "
                        "pool is exceeded. Use only with a bounded object "
                        "count; a once-per-frame check warns on overflow.")
    # ---- Deformable-bunny mode flags ----------------------------------------
    p.add_argument("--deformable-bunny", action="store_true",
                   help="Replace the rigid-body scene with a deformable "
                        "Stanford bunny. Uses the 3-DOF particle Solver, "
                        "one DISTANCE constraint per tet edge (mass-spring "
                        "approximation — proper co-rotated FEM is the "
                        "next milestone). Vertex point cloud + deformed "
                        "surface are both rendered each frame.")
    p.add_argument("--bunny-resolution", type=int, default=10,
                   help="Voxel grid resolution along the longest bunny bbox "
                        "axis. 10 → ~350 verts / ~1700 edges (sane on CPU). "
                        "16 → ~1500 verts / ~10k edges (slow on CPU, fine "
                        "on CUDA).")
    p.add_argument("--bunny-scale", type=float, default=1.2,
                   help="World-space scale of the bunny (unit-bbox before "
                        "scaling).")
    p.add_argument("--plate-height", type=float, default=4.0,
                   help="Total height (y-span) of each glass plate, in "
                        "metres. Default 4.0 so the plates clearly extend "
                        "above and below the bunny even during the squash.")
    p.add_argument("--plate-min-distance", type=float, default=0.0,
                   help="Smallest inner-face gap reachable by the GUI "
                        "slider, in metres. 0 lets the plates fully meet "
                        "at the centre; raise this if you want to cap how "
                        "hard the bunny can be squashed.")
    p.add_argument("--bunny-drop-y", type=float, default=1.5,
                   help="Initial y-centre of the bunny. The bunny falls "
                        "from here onto the floor at y=0.")
    p.add_argument("--edge-stiffness", type=float, default=5.0e4,
                   help="Per-tet-edge DISTANCE constraint stiffness "
                        "(AVBD penalty clamp ceiling).")
    p.add_argument("--volume-stiffness", type=float, default=1.0e4,
                   help="Per-tet TET_VOLUME constraint material stiffness "
                        "(AVBD penalty clamp ceiling, in N·m). Soft volume "
                        "preservation — keeps the tet network from "
                        "pancaking under floor contact. Too high → contact "
                        "instability; 1e4 is a reasonable default for the "
                        "rubber-bunny look. Set to 0 to disable.")
    p.add_argument("mode", nargs="?", default="interactive",
                   choices=("interactive", "headless"),
                   help="`interactive` (default) opens the viser browser viewer; "
                        "`headless` skips rendering and prints a benchmark "
                        "summary (avg step time, broadphase ms, body / row "
                        "counts, color count). Used for parity + perf checks "
                        "in CI and when comparing against the AVBD paper.")
    p.add_argument("--headless-warmup", type=int, default=5,
                   help="Headless-mode warmup frames (excluded from timing).")
    p.add_argument("--headless-frames", type=int, default=30,
                   help="Headless-mode timed frames.")
    args = p.parse_args()
    if args.mode == "headless":
        run_headless(args)
        return
    if args.deformable_bunny:
        DeformableViewer(args).run()
    else:
        Viewer(args).run()


def run_headless(args) -> None:
    """Benchmark the default rigid scene with no rendering or server. Mirrors
    the build_scene() output used by the interactive viewer so numbers are
    directly comparable. See AVBD_PERFORMANCE_GAP.md for context."""
    import time

    solver, boxes, _ = build_scene(args)
    n_b = len(boxes)
    print(f"bodies:           {n_b}")
    print(f"static rows:      {len(solver._rows)}")
    print(f"device:           {solver.device}")
    print(f"substeps:         {solver.substeps}")
    print(f"iterations:       {solver.iterations}")
    print(f"post_stabilize:   {solver.post_stabilize}")
    print(f"coloring mode:    {solver.coloring_mode}")

    # Trigger the one-time _flush + JIT compile inside the warmup window.
    for _ in range(args.headless_warmup):
        solver.step()
    print(f"colors (cap):     {solver.num_colors}")
    print(f"colors (active):  {solver.num_active_colors}")
    print(f"total rows (cap): {solver._gpu_pool_n_capacity}")

    n = max(1, int(args.headless_frames))
    step_times = []
    bp_times = []
    t0_all = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        solver.step()
        step_times.append((time.perf_counter() - t0) * 1000.0)
        bp_times.append(solver.broadphase_ms)
    total_ms = (time.perf_counter() - t0_all) * 1000.0

    step_times.sort()
    avg = sum(step_times) / len(step_times)
    p50 = step_times[len(step_times) // 2]
    p95 = step_times[min(len(step_times) - 1, int(0.95 * len(step_times)))]
    bp_avg = sum(bp_times) / len(bp_times)

    n_active = int(solver.n_active_rows.numpy()[0])
    print(f"active rows:      {n_active}")
    print(f"avg step time:    {avg:.2f} ms")
    print(f"p50 step time:    {p50:.2f} ms")
    print(f"p95 step time:    {p95:.2f} ms")
    print(f"avg broadphase:   {bp_avg:.2f} ms")
    print(f"total {n} frames: {total_ms:.1f} ms")


if __name__ == "__main__":
    main()
