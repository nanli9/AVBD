# AVBD Implementation Gap Review

Date: 2026-06-01

Scope: review of the current `avbd3d` Warp implementation against the local
paper reference `../reference/Augmented_VBD-SIGGRAPH25.pdf` and the public
SIGGRAPH 2025 Real-Time Live AVBD demo claims.

No code changes were made for this review.

## Executive Summary

The current implementation has a meaningful subset of AVBD math, especially in
`Solver6DOF`: augmented Lagrangian dual variables, penalty warm-starting,
constraint-error stabilization, per-body primal solves, dual updates,
Eq. 14 clamp-aware Hessian rescaling, geometric stiffness diagonalization,
OBB SAT contacts, contact friction, and post-stabilization.

The gap to the SIGGRAPH paper/live demo is primarily architectural, not just
one missing formula:

- this repo is a Python/Warp prototype around GPU kernels;
- the paper/live demo numbers come from a large-batch parallel GPU engine;
- dynamic constraints are still Python `_Row` objects rebuilt every substep;
- contact generation still synchronizes back to CPU before row emission;
- graph coloring and array reconstruction happen on the CPU;
- the default viewer is a tiny 34-body workload using 8 substeps and 25
  iterations, while the headline demo is 110,000 blocks using 4 iterations on
  an RTX 4090;
- articulated joints, motors/angular springs, and paper-level deformable FEM
  are not implemented.

Bottom line: the current implementation is useful as a correctness/prototyping
port, but it is not architecturally comparable to the optimized real-time
SIGGRAPH AVBD demo.

## Local Benchmark Observation

The requested command does not currently exist as written:

```bash
uv run python examples/viewer.py headless
```

`examples/viewer.py` has no positional `headless` argument, so argparse exits
with:

```text
viewer.py: error: unrecognized arguments: headless
```

I measured the same default rigid-body scene by importing `build_scene()` and
stepping `Solver6DOF` directly without starting the viser server.

Environment:

```text
Warp 1.13.0 initialized
CUDA not enabled in this build
Device: cpu / arm
```

Measured default rigid scene:

```text
bodies:              34
constraint rows:     ~1035
graph colors:        2
substeps:            8
iterations:          25
device:              cpu
avg step time:       104.6 ms
p50 step time:       99.4 ms
avg broadphase time: 3.25 ms
```

This is consistent with the existing local observation that the default viewer
is small, launch-heavy, and not representative of the paper's large-batch GPU
benchmark.

## Paper And Live Demo Target

From the AVBD SIGGRAPH 2025 paper and public Real-Time Live abstract:

- Figure 1 / RTL example: 110,000 blocks smashed by a sphere.
- Solver iterations: 4.
- GPU: NVIDIA RTX 4090.
- Reported simulation time: about 3.5 ms.
- Reported time including collision detection: about 9.8 ms.
- The paper also claims a parallel GPU implementation with stable low-iteration
  performance for large rigid-body scenes, articulated bodies, contacts,
  springs, and soft-body interactions.

The current default viewer is therefore not an apples-to-apples comparison:

```text
paper/RTL:      110,000 bodies, 4 iterations, RTX 4090, optimized GPU engine
current viewer: 34 bodies, 25 iterations, 8 substeps, Python/Warp prototype
```

## Main Gaps

### 1. Dynamic Constraint Rows Are Still Python Objects

Current code strips and rebuilds dynamic contact rows every substep. The rows
are Python `_Row` objects, then `_flush()` rebuilds Warp arrays from Python
lists.

Relevant path:

- `src/avbd3d/solver_6dof.py::_step_one`
- `src/avbd3d/solver_6dof.py::_rebuild_contact_pool`
- `src/avbd3d/solver_6dof.py::_flush`

Why this matters:

- It forces CPU participation in the hot path.
- It reallocates/rebuilds device arrays when contact topology changes.
- It prevents a fixed GPU-resident constraint pool.
- It makes CUDA graph capture difficult because topology and buffers are not
  stable.

Paper/live-demo expectation:

- Contacts and constraints should live in GPU-resident buffers or fixed dynamic
  regions.
- Contact append/compaction should be GPU-side or at least avoid full Python
  row reconstruction per substep.

### 2. Contact Manifold Generation Is Only Partially GPU-Resident

The implementation has an opt-in Warp face-clip path:

```python
Solver6DOF(..., use_warp_face_clip=True)
```

This moves Sutherland-Hodgman face clipping and edge-edge closest-segment math
into `obb_contact_manifold_6dof`.

However, the output still comes back to Python, and Python appends `_Row`
objects for normal/tangent constraints.

Relevant path:

- `src/avbd3d/solver_6dof.py::_warp_emit_contacts_kernel_path`
- `src/avbd3d/kernels_6dof.py::obb_contact_manifold_6dof`

Remaining gap:

- The geometry math is closer to GPU-resident.
- The dynamic constraint storage and row emission are still CPU-side.

### 3. Too Many Small Kernel Launches Per Step

The main solve loop launches:

- broadphase kernels;
- contact SAT/manifold kernels;
- cache restore/collect kernels;
- prelude kernel;
- one primal kernel per color per iteration;
- one dual kernel per iteration;
- post-stabilization cache pass;
- finalize/cap kernel.

With default viewer settings:

```text
substeps = 8
iterations = 25
post_stabilize = True
colors = 2
```

That becomes hundreds of small launches per visual `solver.step()`.

Relevant path:

- `src/avbd3d/solver_6dof.py::step`
- `src/avbd3d/solver_6dof.py::_step_one`

Paper/live-demo expectation:

- Low iteration count.
- Large per-launch workloads.
- Minimal launch overhead.
- Likely CUDA graph capture or equivalent low-overhead scheduling in the demo
  engine.

### 4. Default Workload Is Far Too Small For GPU Speedup

The default viewer scene creates:

- 1 pinned anchor box;
- 3 x 3 towers, 3 cubes high = 27 cubes;
- 6 domino slabs.

Total: 34 bodies.

With 2 graph colors, each primal launch does useful work on roughly half the
bodies. This is too small to amortize GPU launch overhead.

Relevant path:

- `examples/viewer.py::build_scene`

Paper/live-demo expectation:

- Tens of thousands to hundreds of thousands of bodies per frame.
- Enough parallel work to saturate the GPU.

### 5. Graph Coloring And Adjacency Are CPU-Side

`_flush()` builds body adjacency and performs greedy graph coloring on the CPU.
This happens after row reconstruction.

Relevant path:

- `src/avbd3d/solver_6dof.py::_flush`
- `src/avbd3d/coloring.py`

Why this matters:

- Recoloring dynamic contact graphs every substep is expensive and sync-heavy.
- Paper-scale demos need either precomputed conservative coloring, GPU-side
  coloring/partitioning, or a stable contact graph strategy.

### 6. Broadphase Is Better Than Brute Force, But Still Not Paper-Scale

`Solver6DOF` uses Warp AABB generation, `wp.Bvh`, broadphase pair generation,
and parallel 15-axis OBB SAT.

Relevant path:

- `src/avbd3d/solver_6dof.py::_warp_broadphase_emit_contacts`
- `src/avbd3d/kernels_6dof.py::compute_body_aabb_6dof`
- `src/avbd3d/kernels_6dof.py::bvh_broadphase_pairs`
- `src/avbd3d/kernels_6dof.py::obb_sat_pairs`

Remaining gap:

- On CPU this uses SAH BVH, not CUDA LBVH.
- Pair count is read back to Python.
- Contact rows are emitted on Python side.
- There is no fully GPU-resident narrowphase-to-solver pipeline.

### 7. 6-DOF Rigid Math Coverage Is Good, But Feature Coverage Is Narrow

Implemented in `Solver6DOF`:

- full SE(3) rigid bodies;
- quaternion orientation;
- rotated inertia;
- floor contact;
- OBB box-box contact;
- tangent friction rows;
- static/dynamic friction switch;
- pin constraints;
- breakable constraints through force thresholds.

Missing relative to paper/demo examples:

- ball-socket joints;
- hinge/revolute joints;
- limited-DOF joints;
- motors;
- angular springs;
- articulated chains as first-class constraints;
- large mixed articulated scenes.

The README currently notes this gap: 6-DOF has only `PIN`, not the joint
families shown in paper figures 5/7/9.

### 8. Deformables Are A Mass-Spring/Tet-Volume Approximation

The deformable bunny mode uses the 3-DOF particle solver:

- one particle per tet vertex;
- one `DISTANCE` constraint per tet edge;
- one `TET_VOLUME` constraint per tet.

This is useful for a soft-body prototype, but it is not paper-faithful
deformable mechanics.

Missing:

- per-element strain energy;
- co-rotated FEM;
- Neo-Hookean or StVK materials;
- proper shear stiffness;
- unified high-performance rigid/deformable contact at paper scale.

Relevant path:

- `src/avbd3d/deformable.py`
- `src/avbd3d/solver.py`
- `src/avbd3d/kernels.py`

### 9. 3-DOF Solver Deviates More From The Paper

The 3-DOF particle solver is useful for chains, particles, and early fracture
experiments, but it has known deviations:

- broadphase is CPU brute force;
- box collision is AABB-oriented;
- geometric stiffness `G` is not implemented for 3-DOF rows;
- Eq. 14 saturated Hessian rescaling is not implemented there;
- constraints are scalar rows rather than grouped constraints.

Relevant path:

- `src/avbd3d/solver.py`
- `src/avbd3d/kernels.py`

### 10. Viewer And Benchmarking Are Not Set Up For Paper Comparison

Current viewer defaults prioritize stability and interactive inspection:

```text
iterations = 25
substeps = 8
post_stabilize = True
```

The paper headline result uses 4 iterations.

Missing benchmark support:

- no `headless` mode in `examples/viewer.py`;
- no paper-style preset such as `substeps=1`, `iterations=4`;
- no large procedural body pile benchmark;
- no solver-only benchmark command that avoids rendering/server overhead;
- no CUDA benchmark path validated on an NVIDIA GPU from this machine.

## What Is Already In Good Shape

The current 6-DOF solver has several important AVBD pieces implemented:

- Eq. 2 inertial prediction for translation and rotation;
- Eq. 8 augmented Lagrangian force form;
- Eq. 11 dual update;
- Eq. 12/16 penalty growth;
- Eq. 14 clamp-aware LHS stiffness rescaling;
- Eq. 17 Hessian assembly with geometric stiffness diagonal;
- Eq. 18 `C - alpha*C0` stabilization;
- Eq. 19 warm-starting;
- BDF1 velocity finalization;
- persistent contact cache;
- OBB SAT narrowphase;
- edge-edge OBB contact handling;
- static/dynamic friction switching beyond the basic square-cone friction
  described in the paper.

So the implementation is not "missing AVBD." It is missing the engine-level
architecture needed to make AVBD run like the SIGGRAPH live demo.

## Prioritized Next Steps

1. Add a real headless benchmark entry point.

   It should build the default scene, run warmup frames, run timed frames, and
   print bodies, rows, colors, substeps, iterations, broadphase time, and solver
   step time. This prevents confusion around `viewer.py headless`.

2. Add a paper-style benchmark preset.

   Start with:

   ```text
   substeps = 1
   iterations = 4
   post_stabilize = configurable
   rendering = off
   ```

   This will not match paper performance by itself, but it makes comparisons
   honest.

3. Move dynamic contact rows into GPU-resident buffers.

   Use a fixed-capacity dynamic contact region in the existing `c_*` arrays or
   a separate dynamic row pool. Contact manifold kernels should write row data
   directly into this pool.

4. Avoid full `_flush()` rebuilds when only dynamic contacts change.

   Preserve static scene arrays. Update only dynamic row counts/regions and
   avoid reconstructing every Warp array from Python lists.

5. Replace per-substep CPU recoloring with a scalable strategy.

   Likely options:

   - conservative precomputed body/proximity coloring;
   - fixed spatial coloring for pile-like scenes;
   - GPU-side adjacency/coloring if dynamic exact coloring is required.

6. Add CUDA graph capture or equivalent launch batching.

   This should happen after the solver has stable buffer shapes. Otherwise the
   graph will recapture too often to help.

7. Add missing joint constraint families.

   Implement ball-socket, hinge/revolute, angular spring, motor, and joint
   limits as first-class 6-DOF constraints.

8. Replace deformable edge+volume approximation with real FEM energies.

   Add co-rotated FEM / Neo-Hookean / StVK-style element energies for the
   3-DOF deformable path.

9. Validate on NVIDIA hardware.

   This machine cannot validate CUDA behavior. A real benchmark pass needs at
   least one NVIDIA GPU run, ideally with both a midrange card and an RTX 4090
   class card if comparing against the RTL claim.

## Sources

- Local paper: `../reference/Augmented_VBD-SIGGRAPH25.pdf`
- Local implementation: `src/avbd3d/solver_6dof.py`,
  `src/avbd3d/kernels_6dof.py`, `src/avbd3d/solver.py`,
  `src/avbd3d/kernels.py`, `examples/viewer.py`
- Public AVBD project page:
  https://graphics.cs.utah.edu/research/projects/avbd/
- Public SIGGRAPH 2025 Real-Time Live listing:
  https://s2025.siggraph.org/program/real-time-live/
- Public Real-Time Live abstract PDF:
  https://www.cemyuksel.com/research/papers/Augmented_VBD-SIGGRAPH25_RTL.pdf

