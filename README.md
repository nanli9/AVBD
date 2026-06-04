# avbd3d — 3D Augmented Vertex Block Descent in NVIDIA Warp

A 3D port of [AVBD (SIGGRAPH 2025)](https://graphics.cs.utah.edu/research/projects/avbd/)
implemented as Python kernels on NVIDIA Warp. Built as the simulation core for
the real-time brittle fracture project — the AVBD-native dual variable `λ` is
intended to double as the cohesive-interface traction estimate (see
`../formal_model_to_energy_accounting.md` and `../project_roadmap.md`).

The repo ships **two solvers**, mirroring the structure of the 2D reference at
[savant117/avbd-demo2d](https://github.com/savant117/avbd-demo2d):

- **`Solver`** — 3-DOF point particles + scalar constraints (CPU broad phase,
  AABB box collision). Original port; useful for chains, particle scenes,
  fracture-graph prototyping.
- **`Solver6DOF`** — full SE(3) rigid bodies with quaternion orientation,
  rotated inertia, GPU LBVH broad phase, full 15-axis OBB SAT narrow phase,
  static/dynamic friction switch, and per-substep stepping.

## What's implemented (AVBD paper coverage)

| Paper requirement | 3-DOF `Solver` | 6-DOF `Solver6DOF` |
|---|---|---|
| Predict + inertial target (Eq. 2) | ✅ `predict_inertial` (with adaptive gravity-weighted guess) | ✅ `predict_inertial_6dof` (linear + `exp_q(ω·dt)`) |
| Warm-start λ, penalty (Eq. 19) | ✅ `warmstart_duals` | ✅ fused in `substep_prelude_6dof` |
| C̃ = C − α·C₀ stabilization (Eq. 18) | ✅ `cache_alpha_C0` | ✅ `cache_alpha_C0_6dof` |
| Per-body primal solve (Eqs. 4, 13, 17) | ✅ 3×3 SPD inverse | ✅ 6×6 via 3×3 Schur complement |
| Dual update (Eqs. 11, 16) | ✅ `dual_update` | ✅ `dual_update_6dof` |
| Diagonal geometric stiffness G̃ (Sec. 3.5) | ⚠️ set to 0 (paper-noncompliant for DISTANCE/SPHERE/BOX rows — stable but a deviation) | ✅ `geom_stiffness_diag` |
| Eq. 14 Hessian rescaling on clamp saturation | ❌ uses raw k | ✅ `k̃ = \|bound − λ⁺\|/\|C\|` for LHS only |
| Proper edge-edge OBB contact (Ericson §5.1.9) | n/a (AABB only) | ✅ closest-segment-pair, was deepest-vertex fallback |
| `TET_VOLUME` soft volume preservation for deformables | ✅ new tet pool + adjacency | n/a |
| Post-stabilization extra pass (α → 0) | ✅ | ✅ |
| BDF1 velocity finalize | ✅ | ✅ (world-frame ω via `quat_to_rotvec`) |
| Greedy graph coloring (Sec. 4) | ✅ Welsh-Powell (CPU) | ✅ GPU parallel-Jacobi greedy (`jacobi`, the paper §4 default) or Jones-Plassmann (`jones_plassmann`), switchable at runtime |
| LBVH broad phase (Sec. 4) | ❌ brute O(N²) on CPU | ✅ `wp.Bvh` + GPU SAT kernel |
| Eq. 15 contact form `[t̂ b̂ n̂]ᵀ(r_a − r_b)` | ✅ | ✅ |
| Square-cone friction `|λ_t| ≤ μ\|λ_n\|` (Sec. 3.3) | ✅ per-row | ✅ + **isotropic disk static-stick + μ_s/μ_d switch** (beyond paper) |
| Quaternion rigid update (Eqs. 20–21), always quasi-Newton for rigids | n/a | ✅ |
| Persistent λ across frames (manifold caching) | ✅ `_contact_state_cache` | ✅ `_contact_cache` (5 mm quantized key) |
| Breakable hard constraint by max-force (paper Fig. 13) | ✅ `\|λ\| ≥ fracture` | ✅ same |
| Substepping | ❌ | ✅ |
| Joint constraints (ball-socket, hinge, motor, angular spring) shown in paper Figs. 5/7/9 | ❌ only PIN + DISTANCE | ❌ only PIN |

### Constraint type catalog

`Solver` (3-DOF particles) — codes in `solver.py`:
`PIN_X/Y/Z`, `DISTANCE`, `FLOOR_CONTACT`, `SPHERE_CONTACT`,
`CONTACT_TANGENT`, `SPHERE_BOX_CONTACT` (analytical closest-point-on-AABB),
`BOX_BOX_CONTACT` (AABB SAT + face clip ≤4 contacts).

`Solver6DOF` (rigid bodies) — codes in `solver_6dof.py`:
`FLOOR_CONTACT_6DOF` (per-corner), `CONTACT_TANGENT_6DOF` (with static-stick
flag, μ_s and μ_d), `PIN_6DOF` (3-axis), `BOX_BOX_CONTACT_6DOF` (full
15-axis OBB SAT + Sutherland-Hodgman face clip ≤4 contacts; edge-edge cases
fall back to a single deepest-vertex contact).

### Fracture roadmap (NOT in the paper)

These items belong to the brittle-fracture project, not the AVBD paper.
AVBD's only failure mechanism (Fig. 13 "wall break") is the |λ|-threshold
hard-constraint break, which is implemented.

| Item | Status |
|---|---|
| `\|λ\| ≥ threshold` breakable constraint | ✅ (matches paper Fig. 13) |
| Cohesive interface (CZM) constraint | ❌ |
| Griffith energy criterion `E_i ≥ G_c · ΔA_i` | ❌ |
| I3D-2018 Δv impulse transfer on break | ❌ |
| Voronoi prefracture → candidate graph | ❌ |

## Layout

```
src/avbd3d/
├── __init__.py
├── solver.py          # 3-DOF particle Solver (CPU broadphase, AABB box collision)
├── kernels.py         # Warp kernels for the 3-DOF solver
├── solver_6dof.py     # 6-DOF rigid Solver6DOF (GPU LBVH, OBB SAT)
├── kernels_6dof.py    # Warp kernels for the 6-DOF solver
├── coloring.py        # greedy body-graph coloring (Welsh-Powell)
├── deformable.py      # Stanford bunny → tet lattice → particles + DISTANCE rows
└── scene.py           # Body / ConstraintHandle / Shape dataclasses

examples/
├── smoke_test.py            # single particle free fall → confirms BDF1
├── pin_only.py              # pinned particle, λ_y → −m·g
├── hanging_chain.py         # N-link chain (--plot for PNG)
├── chain_break.py           # |λ| ≥ threshold breakage
├── swinging_chain_anim.py   # 3D GIF: swinging + breaking + λ panel
├── spinning_box_6dof.py     # 6-DOF: spinning cube falls + lands, friction brakes spin
├── interactive_demo.py      # matplotlib live window
├── viewer.py                # viser browser viewer for Solver6DOF stacks
└── viewer_particles.py      # viser browser viewer for the 3-DOF Solver

tests/
├── test_solver.py          # 24 tests — BDF1, pin, distance, chain, fracture,
│                           #   coloring, sphere/box/box-box contact, friction
│                           #   sliding, pillar stack, broken-pair re-collide
├── test_solver_6dof.py     # 20 tests — free-fall/spin, OBB SAT, cube stacks,
│                           #   corner pin, static stick, kinetic slip, BVH
│                           #   counts, geom_stiffness_diag, Eq.14 resting
│                           #   contact stability, edge-edge OBB contact
└── test_deformable.py      # 10 tests — tet-lattice topology, resolution
                            #   scaling, edge rest = init length, floor only
                            #   on surface verts, mass distribution, TET_VOLUME
                            #   constraint emission + on/off behaviour,
                            #   volume preservation prevents tet inversion
```

## Quickstart

```bash
# install (editable) + dev deps
uv sync
uv pip install -e .

# run tests (both solvers + deformable pipeline)
uv run pytest tests/ -v

# 3-DOF demos (particles + scalar constraints)
uv run python examples/smoke_test.py
uv run python examples/hanging_chain.py --n 6 --frames 600 --plot
uv run python examples/chain_break.py --n 5 --threshold 30
uv run python examples/swinging_chain_anim.py --n 8 --heavy-mass 15 --threshold 80
uv run python examples/interactive_demo.py

# 6-DOF demo (spinning rigid box + floor friction)
uv run python examples/spinning_box_6dof.py

# Browser viewers (viser)
uv run python examples/viewer.py                       # 6-DOF rigid stacks + dominoes
uv run python examples/viewer.py --deformable-bunny    # deformable Stanford bunny (tet mass-spring)
uv run python examples/viewer_particles.py             # 3-DOF particle viewer
# open the printed URL (default http://localhost:8080) in a browser

# Large paper-style block pile on the GPU (414 bodies). The warp-per-body
# primal solve (warp-shuffle reduction) is ON by default on CUDA — ~5x faster
# than the serial kernel on this scene (see "Warp-per-body primal" below):
uv run python examples/viewer.py --device cuda:0 --stress --iterations 25
```

### Performance: warp-per-body primal (CUDA)

Colored Gauss-Seidel launches only ~`n_bodies / n_colors` threads per color, so
the one-thread-per-body primal kernel leaves the GPU mostly idle on big scenes
(e.g. a 414-body pile fills ~0.5% of an RTX 3060's threads) while each thread
serially walks ~32 incident constraints. The solver instead gives each body a
**group of `G` lanes** that cooperatively accumulate its constraint Hessian /
gradient, then join the partials — by default with a `wp.func_native`
**warp-shuffle** (`__shfl_down_sync`) register reduction (contention-free; an
`atomic_add` join is also available). Lane 0 runs the 6×6 Schur solve.

This is **implementation only — the AVBD math is unchanged**; the reduction
just reorders a sum (divergence vs the serial kernel = the GPU-atomic noise
floor, ~0.02 mm). It is **on by default on CUDA** (`G=16`, shuffle) and a no-op
on `--device cpu` (the serial kernel is used). Measured RTX 3060, 25 iters ×
8 substeps:

| scene | bodies | serial | warp-per-body (best) | speedup |
|-------|-------:|-------:|---------------------:|--------:|
| small (3×3×3) | 33 | 46 ms | 12 ms | 4.0× |
| `--stress` (8×8×6) | 414 | 116 ms | 22 ms | 5.1× |
| 12×12×8 | 1212 | 133 ms | 36 ms | 3.6× |
| 16×16×8 | 2128 | 149 ms | 52 ms | 2.8× |

Flags (`viewer.py`, or `Solver6DOF(primal_group_size=…, primal_shuffle=…)`):

```bash
--primal-group N      # lanes per body; 1 = serial, default 16
                      #   (32 best for small/medium, 8 for >1000-body scenes)
--no-primal-shuffle   # use the atomic-add join instead of warp-shuffle
```

### Deformable bunny mode

`viewer.py --deformable-bunny` switches the scene to a soft Stanford bunny
driven by the 3-DOF particle `Solver`:

- one particle per tet vertex (the actual AVBD 3-DOF block)
- one `DISTANCE` constraint per unique tet edge (length-spring network)
- one `TET_VOLUME` constraint per tet (soft volume preservation,
  `C = V/V₀ − 1`) — this is what stops the bunny pancaking under floor
  contact when many surface verts simultaneously hit the floor
- the deformed surface (tet-boundary triangulation) AND the vertex point
  cloud are both re-emitted to viser each frame

The bunny OBJ is fetched once from `alecjacobson/common-3d-test-models`
and cached under `~/.cache/avbd3d/`. A coarse axis-aligned voxel grid is
intersected with the surface via trimesh's inside-test, then each kept
voxel is split into 6 tets via the Kuhn diagonal subdivision. Default
`--bunny-resolution 10` gives ~350 vertices, ~1700 edges, ~1700 tets.

Useful flags:
- `--bunny-resolution N` — voxel grid resolution along the longest bbox
  axis. 8 ≈ 200 verts, 12 ≈ 700 verts, 16 ≈ 1500 verts (slow on CPU).
- `--bunny-scale s` — world-space size (unit bbox before scaling).
- `--bunny-drop-y h` — initial centre height; bunny falls onto floor at y=0.
- `--edge-stiffness k` — AVBD penalty clamp ceiling per edge constraint
  (default 5e4).
- `--volume-stiffness k` — AVBD penalty clamp ceiling per `TET_VOLUME`
  row (default 1e4). Set to 0 to disable volume preservation entirely;
  too high → contact instability.

GUI panel exposes: iterations slider, gravity slider, edge stiffness
slider, toggles for mesh/wireframe/point-cloud, point size, and **shake
/ squish / lift / reset** action buttons.

> **Next milestone:** the constraint pool currently combines edge springs
> (`DISTANCE`) with volume preservation (`TET_VOLUME`). Paper-faithful
> AVBD/VBD deformables also include per-element strain energy
> (co-rotated linear FEM / Neo-Hookean / StVK) with `wp.svd3` polar
> decomposition for shear stiffness. Edge + volume is enough to keep the
> bunny shaped under contact; shear stiffness is what would let it
> resist twisting like a real rubber object.

## Math ↔ kernel mapping

Every equation cited is from the AVBD SIGGRAPH 2025 paper unless noted.

| Equation | 3-DOF kernel | 6-DOF kernel |
|---|---|---|
| Eq. 1 (objective) | implicit in `primal_update` LHS/RHS | implicit in `primal_update_6dof` |
| Eq. 2 (inertial target `y`) | `predict_inertial` | `predict_inertial_6dof` |
| Eq. 4 (per-body local solve) | `primal_update`: `dx = inv(lhs)·rhs` | `primal_update_6dof`: Schur complement on 3×3 blocks |
| Eq. 8 (constraint energy form) | `primal_update`: `f = clamp(k·C + λ, fmin, fmax)` | same |
| Eq. 11 (dual clamp) | `dual_update` | `dual_update_6dof` |
| Eqs. 12/16 (penalty growth + clamp) | `dual_update`: `k ← min(k + β\|C\|, k*)` only when λ in-bounds | same |
| Eq. 15 (contact form) | `SPHERE_CONTACT` + `CONTACT_TANGENT` rows | `BOX_BOX_CONTACT_6DOF` + `CONTACT_TANGENT_6DOF` |
| Eq. 17 (Hessian assembly) | `primal_update`: `lhs += k·outer(J, J)` (no G term) | `primal_update_6dof`: `lhs += k·outer(J,J) + diag(g)` |
| Eq. 18 (C̃ = C − α·C₀) | `cache_alpha_C0` | `cache_alpha_C0_6dof` |
| Eq. 19 (warm start) | `warmstart_duals` | fused in `substep_prelude_6dof` |
| Eqs. 20–21 (quaternion rigid update) | n/a | `quat_from_rotvec`, `quat_to_rotvec` + `primal_update_6dof` |

Reference 2D implementation that drove the port:
[savant117/avbd-demo2d `solver.cpp`](https://github.com/savant117/avbd-demo2d/blob/main/source/solver.cpp).

## Known issues & sharp edges

- **Apple Silicon ⇒ CPU-only Warp.** All demos run on CPU; switch
  `device="cuda:0"` when on an NVIDIA machine. Kernels are GPU-clean already.
  The 6-DOF solver's LBVH falls back to a SAH BVH on CPU.
- **3-DOF box collision is AABB**, not OBB — the 3-DOF `Solver` was written
  for particles + scalar constraints, so cubes don't rotate there. The full
  OBB SAT lives in `Solver6DOF`.
- **3-DOF G term in the Hessian is zero.** Paper Sec. 3.5 says to use a
  diagonal G̃ from `λ⁺·∂²C/∂x²`. For DISTANCE/SPHERE_CONTACT/SPHERE_BOX/
  BOX_BOX rows this is paper-noncompliant. Stable on the demos but a real
  deviation. The 6-DOF solver implements G̃ via `geom_stiffness_diag`.
- **Eq. 14 saturated-Hessian rescaling** is implemented in `Solver6DOF`
  (`primal_update_6dof` rescales `k` in the LHS only to
  `|bound − λ⁺| / |C|` when force clamps). The 3-DOF `primal_update`
  still uses raw `c_penalty[j]`.
- **Edge-edge OBB contact** in `Solver6DOF` now does the proper
  closest-segment-pair on the two edges that produced the SAT axis
  (Ericson §5.1.9, `solver_6dof._emit_obb_edge_edge`). The 3-DOF
  AABB-only box path stays single-feature.
- **Integration is BDF1.** Adds modest numerical damping; expected for
  AVBD. The hanging-chain demo's un-damped oscillation around steady state
  is correct (no extra viscous term wired in).
- **3-DOF constraints are scalar-per-row.** A pin is 3 scalar
  constraints. The 2D reference groups rows in a `MAX_ROWS=4` struct; this
  port trades layout simplicity for more constraints.
- **No revolute / spherical / motor joints**, no angular springs.
  Paper showcases these (Figs. 5/7/9) using the same Eq. 8 hard-constraint
  energy with a bespoke `C` — straightforward to add but not done.

## Next steps for the fracture project

Implementation roadmap is in `../project_roadmap.md`. The next milestones:

1. **Cohesive interface constraint** (`CohesiveInterface`): a 3-row
   constraint between two rigid fragments (1 normal + 2 tangential).
   Stiffness from `k_n = E·A/h` calibrated against AVBD's penalty clamp
   (Gap M1.1 in the audit).
2. **Griffith energy criterion** alongside the |λ|-threshold check:
   evaluate `E_i = ½k_n δ_n² + ½k_t‖δ_t‖²` and break when
   `E_i ≥ G_c·ΔA_i`.
3. **I3D-2018 Δv impulse**: on break, apply
   `μ = √(2α·E_release·m_eff)` along the interface normal with
   Newton-Euler distribution to both fragments.
4. **Offline candidate-graph generation** (Voronoi prefracture → ΔA, n_i,
   t_i, adjacency).

And separately, to close the remaining AVBD-paper gaps:

5. Implement **Eq. 14 Hessian rescaling** for saturated constraints in
   both solvers.
6. Wire **diagonal G̃** into the 3-DOF `primal_update` for DISTANCE /
   SPHERE / BOX rows.
7. Replace 3-DOF brute O(N²) broad phase with `wp.Bvh`.
8. Add revolute / ball-socket joints if the fracture demo ever needs
   articulated bodies.
