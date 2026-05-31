# avbd3d — 3D Augmented Vertex Block Descent in NVIDIA Warp

A 3D port of [AVBD (SIGGRAPH 2025)](https://graphics.cs.utah.edu/research/projects/avbd/)
implemented as Python kernels on NVIDIA Warp. Built as the simulation core for
the real-time brittle fracture project — the AVBD-native dual variable `λ` is
intended to double as the cohesive-interface traction estimate (see
`../formal_model_to_energy_accounting.md` and `../project_roadmap.md`).

The structure mirrors the 2D reference at
[savant117/avbd-demo2d](https://github.com/savant117/avbd-demo2d) (rigid bodies
in 2D), extended to **3D point particles + scalar constraints**. Rigid-body
6-DOF and cohesive-interface constraints are queued for the next iteration.

## What's implemented

| Piece | Status | Notes |
|---|---|---|
| Predict + adaptive warm-start (Eq. 2, VBD §4.2) | ✅ | `predict_inertial` kernel |
| Warm-started λ, penalty (Eq. 19) | ✅ | `warmstart_duals` kernel |
| α·C₀ stabilization caching (Eq. 18) | ✅ | `cache_alpha_C0` kernel |
| Per-body primal solve (Eqs. 4, 13, 17) | ✅ | `primal_update` kernel, 3×3 SPD inverse |
| Dual update (Eqs. 11, 16) | ✅ | `dual_update` kernel |
| Velocity finalise (BDF1) | ✅ | `finalize_velocity` kernel |
| Post-stabilization extra pass | ✅ | matches 2D reference default |
| Greedy graph coloring (Welsh-Powell) | ✅ | `coloring.py` — uses (max_deg + 1) colors |
| Fracture via |λ| ≥ threshold | ✅ | AVBD-native impulse criterion (matches 2D ref `force->fracture`) |
| PIN (3-scalar) and DISTANCE constraints | ✅ | other constraint types added by following the same pattern |
| 6-DOF rigid bodies (quaternions, inertia tensor) | ❌ | next iteration |
| Cohesive interface (CZM) constraint for fracture | ❌ | next iteration |
| Griffith energy break criterion `E_i ≥ G_c · ΔA_i` | ❌ | next iteration |
| I3D-2018 Δv impulse transfer on break | ❌ | next iteration |
| Box-box / particle-plane collision | ❌ | next iteration |

## Layout

```
src/avbd3d/
├── __init__.py
├── solver.py            # Solver class + step() loop
├── kernels.py           # Warp kernels (every equation cited)
├── coloring.py          # greedy body-graph coloring
└── scene.py             # Body / ConstraintHandle dataclasses

examples/
├── smoke_test.py            # single particle free fall → confirms BDF1
├── pin_only.py              # particle pinned at world point → λ_y → −m·g
├── hanging_chain.py         # N-link chain (use --plot for matplotlib PNG)
├── chain_break.py           # chain with |λ| ≥ threshold breakage (lands on floor)
├── swinging_chain_anim.py   # 3D animated GIF: swinging + breaking + λ panel
├── interactive_demo.py      # LIVE matplotlib window with keyboard control
└── viewer.py                # browser-based 3D viewer (viser) — DRAGGABLE bodies,
                             #   sphere / cube / pillar primitives, ground plane,
                             #   GUI controls (kick, threshold, gravity, reset, drop)

tests/
└── test_solver.py           # 11 pytest tests (BDF1, pin, distance, chain, warm-start,
                             #                   fracture, coloring, no-NaN)
```

## Quickstart

```bash
# install (editable) + dev deps
uv sync
uv pip install -e .

# run tests
uv run pytest tests/ -v          # 11 passed in ~1.3 s

# headless examples (text + PNG/GIF output)
uv run python examples/smoke_test.py
uv run python examples/hanging_chain.py --n 6 --frames 600 --plot
uv run python examples/chain_break.py --n 5 --threshold 30
uv run python examples/swinging_chain_anim.py --n 8 --heavy-mass 15 --threshold 80

# live window with keyboard controls (matplotlib)
uv run python examples/interactive_demo.py

# REAL interactive 3D viewer with mouse drag + 3D primitives + ground (viser)
uv run python examples/viewer.py
# then open the printed URL (default http://localhost:8080) in a browser:
#   - DRAG body gizmos to move bodies around in 3D
#   - SLIDERS: iterations, gravity, fracture threshold
#   - BUTTONS: pause, reset, kick all, drop a cube
#   SPACE  random sideways kick on every body
#   K      hard kick on a single body
#   B / H  lower / raise the fracture threshold of every link
#   R      reset
#   + / -  add / remove a solver iteration per step
#   ESC    quit
```

## Tests

`uv run pytest tests/ -v` covers:

| Test | What it pins |
|---|---|
| `test_free_fall_matches_bdf1` | unconstrained integration = closed-form BDF1 |
| `test_static_body_stays_put` | `mass=0` ⇒ kinematic |
| `test_pin_holds_against_gravity` | pinned particle stays put; λ → −m·g |
| `test_distance_constraint_settles` | two-body distance error < 2 mm |
| `test_chain_distance_errors_small` | 5-link chain avg error < 5% link |
| `test_warmstart_reduces_iteration_load` | λ is non-trivial across frames |
| `test_fracture_breaks_top_link_first` | highest tension breaks first |
| `test_no_fracture_when_threshold_is_inf` | default fracture=∞ never breaks |
| `test_coloring_is_valid_for_chain` | adjacent bodies have distinct colors; path graph ⇒ 2 colors |
| `test_coloring_complete_graph` | K_n needs n colors |
| `test_no_nans_in_chain_simulation` | no NaN/inf in positions/velocities/λ over 300 steps |

## Math <-> kernel mapping

Every equation cited is from the AVBD SIGGRAPH 2025 paper unless noted.

| Equation | Where |
|---|---|
| Eq. 1 (objective) | implicit in `primal_update` LHS/RHS assembly |
| Eq. 2 (inertial target `y`) | `predict_inertial` |
| Eq. 4 (per-body local solve) | `primal_update`: `dx = inv(lhs) * rhs; x -= dx` |
| Eq. 8 (constraint energy form) | `primal_update`: `f = clamp(k*C + λ, fmin, fmax)` |
| Eq. 11 (dual update) | `dual_update`: `λ ← clamp(k*C + λ, fmin, fmax)` |
| Eq. 13 (force accumulation) | `primal_update`: `rhs += J * f` |
| Eq. 16 (penalty growth + clamp) | `dual_update`: `k += β·|C|`, clamped to material |
| Eq. 17 (Hessian assembly) | `primal_update`: `lhs += outer(J, J) * k` |
| Eq. 18 (C̃ = C − α·C₀) | `cache_alpha_C0` + use inside `primal_update` |
| Eq. 19 (warm start) | `warmstart_duals` |

Reference 2D implementation lines that drove this port:
[savant117/avbd-demo2d `solver.cpp` step()](https://github.com/savant117/avbd-demo2d/blob/main/source/solver.cpp).

## Known issues & sharp edges

- **Apple Silicon ⇒ CPU-only Warp.** All demos run on CPU; switch `device="cuda:0"`
  when on an NVIDIA machine. Kernels are GPU-clean already.
- **Coloring is greedy Welsh-Powell.** Fine for static graphs; for dynamic
  topology (e.g. after a break) we re-flush and re-color on the next step.
  Gaia's GPU coloring (incremental, partition-based) is the eventual target
  if we need to re-color every frame at scale.
- **Integration is BDF1.** Adds modest numerical damping; expected for AVBD.
  No extra viscous term is wired in yet — the hanging-chain plot shows
  un-damped oscillation around steady state, which is correct (not a bug).
- **Constraints are scalar-per-row.** A pin is 3 separate scalar
  constraints. This is what makes the Warp arrays flat. The 2D reference
  groups rows in a `MAX_ROWS=4` struct; the trade-off here is more
  constraints but simpler memory layout.

## Next iteration

Implementation roadmap is in `../project_roadmap.md`. The next milestones for
this codebase specifically are:

1. **6-DOF rigid bodies.** Add per-body quaternion, world-frame ω, inverse
   inertia tensor. Per-body block becomes 6×6. Generalize all constraint
   Jacobians to map from twist (Δp, Δθ) ∈ ℝ⁶ to scalar C.
2. **Cohesive interface constraint** (`CohesiveInterface`): a 2-row
   constraint between two rigid fragments. Row 0 = normal separation δ_n,
   row 1+2 = tangential δ_t. Stiffness from `k_n = E·A/h` calibrated against
   AVBD's penalty clamp (Gap M1.1 in the audit).
3. **Energy criterion** in the dual update path: in addition to the
   |λ| ≥ fracture impulse check, also evaluate `E_i = ½k_n δ_n² + ½k_t‖δ_t‖²`
   and break when `E_i ≥ G_c·ΔA_i`.
4. **I3D-2018 Δv impulse**: on break, apply `μ = √(2α·E_release·m_eff)` along
   the interface normal with Newton-Euler distribution to both fragments.
5. **Offline candidate-graph generation** (Voronoi prefracture → ΔA, n_i,
   t_i, adjacency).
