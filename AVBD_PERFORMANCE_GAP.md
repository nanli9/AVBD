# AVBD Performance Gap Notes

> **Status update (2026-06-01).** Two of the gaps below have been closed.
> The rest are deferred with implementation plans in
> [PERFORMANCE_PROGRESS.md](PERFORMANCE_PROGRESS.md). Constraint: this
> machine has no NVIDIA GPU, so changes had to be architectural shifts
> validatable on the Warp CPU backend — no GPU perf numbers reportable.
>
> **Done**
> - §5 *Contact warm-start cache crosses CPU/GPU* — replaced 6 full-array
>   `.numpy()` syncs + 3 `wp.array` rebuilds per substep with 2 Warp
>   kernels (`cache_restore_6dof`, `cache_collect_6dof`) operating on
>   preallocated GPU-resident scratch buffers. Host staging arrays are
>   reused via `wp.array.assign(prefix)` so the device side no longer
>   reallocates per substep.
> - §6 *Viewer readbacks every frame* — added
>   `Solver6DOF.read_state_batched()` backed by two pack kernels
>   (`viewer_pack_bodies_6dof`, `viewer_pack_rows_6dof`). Per-frame
>   stream syncs in `examples/viewer.py:tick` drop from ~7 to 2.
> - §3 *Dynamic contact manifold not GPU-resident* (geometry hot loop)
>   — added `obb_contact_manifold_6dof` kernel: handles both face-face
>   (Sutherland-Hodgman, up to 4 contacts) and edge-edge
>   (closest-segment-pair, 1 contact) cases. Opt-in via
>   `Solver6DOF(..., use_warp_face_clip=True)`. Parity vs the Python
>   emitter is verified by `test_kernel_face_clip_matches_python_stack`
>   (3-cube tower, 360 × 8 = 2880 manifold passes, max y divergence
>   ~6 µm). Row-append still touches the Python `_Row` list — that's
>   gap §4 (deferred).
>
> **Not done (deferred)**
> - §4 *Constraint rows are Python objects rebuilt every substep* —
>   needs GPU-resident dynamic row buffers + the body adjacency /
>   coloring rebuild problem solved. Detailed plan in
>   PERFORMANCE_PROGRESS.md §4.
> - §2 *Too many launches per visual frame* — CUDA-graph capture path.
>   `wp.ScopedCapture` is CUDA-only, no CPU validation possible.
> - §1 *Tiny GPU work per launch* and §8 *Hardware difference* —
>   intrinsic to the scene size + local GPU, not addressable in code.
> - §7 *Renderer not built for many bodies* — out of scope (rendering,
>   not solver).
>
> See PERFORMANCE_PROGRESS.md for the full change list and the deferred
> work plans.

## Summary

The GPU is visible and Warp is launching kernels on `cuda:0`, but the current
viewer/solver path is not architecturally comparable to the AVBD paper's
high-performance GPU demo.

The current repo implements AVBD-style math in Python/Warp, but it still has:

- very small active GPU workloads per kernel launch,
- hundreds of Python-driven Warp launches per rendered frame,
- CPU/GPU synchronization in the hot path,
- Python-side dynamic contact row generation,
- viewer readbacks every frame.

The main gap is therefore not "CUDA is missing"; it is:

> AVBD equations in a Python/Warp prototype vs. a paper-grade GPU engine with
> large batches, GPU-resident contacts/constraints, low iteration counts, and
> minimal CPU synchronization.

## Paper/Demo Claim Being Compared

The AVBD project page describes the paper results as coming from a parallel GPU
implementation for large rigid-body scenes. The Real-Time Live material reports
roughly:

- `110,000` blocks,
- `4` AVBD iterations,
- RTX 4090,
- about `3.5 ms` for simulation only,
- about `9.8 ms` including collision detection.

These numbers are not equivalent to running this repo's interactive viewer with
`--substeps 8 --iterations 25` on a small scene.

Sources:

- AVBD project page: https://graphics.cs.utah.edu/research/projects/avbd/
- Real-Time Live PDF: https://graphics.cs.utah.edu/research/projects/avbd/Augmented_VBD-SIGGRAPH25_RTL.pdf
- Official open-source 3D demo README: https://raw.githubusercontent.com/savant117/avbd-demo3d/main/README.md

The official open-source `avbd-demo3d` README explicitly says that repository is
"not intended to be a super optimized implementation", but an easy-to-understand
demonstration. This local repo is also a prototype-style Python/Warp
implementation rather than the paper's optimized production GPU pipeline.

## Local Observation

Command discussed:

```bash
uv run python examples/viewer.py --device cuda:0
```

Warp output confirms CUDA is available:

```text
Warp 1.13.0 initialized:
   CUDA Toolkit 12.9, Driver 13.2
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA GeForce RTX 3060 Laptop GPU" (6 GiB, sm_86, mempool enabled)
```

So the issue is not that Warp cannot see the GPU.

## Default Viewer Workload

The default 6-DOF viewer scene in `examples/viewer.py` creates a small number of
bodies:

- one pinned anchor box,
- a 3 x 3 grid of cube towers, 3 cubes high,
- 6 domino slabs.

The measured default scene had:

- `34` bodies,
- about `1035` constraint rows after dynamic contacts,
- `2` graph colors.

This is too small to saturate the GPU. In practice, each primal kernel launch is
only doing useful work on about half the bodies per color, roughly `17` bodies
per launch in the default scene.

## Benchmark Results Observed

Initial headless benchmark on the default viewer scene reproduced the reported
problem:

```text
cpu    bodies 34 rows 1035 colors 2 step_ms_avg ~108.1 ms
cuda:0 bodies 34 rows 1035 colors 2 step_ms_avg ~113.5 ms
```

After removing some redundant per-substep uploads, the same default scene was
roughly parity:

```text
cpu    bodies 34 rows 1035 colors 2 step_ms_avg ~106.6 ms
cuda:0 bodies 34 rows 1035 colors 2 step_ms_avg ~106.1 ms
```

This still is not a meaningful GPU speedup. It shows the GPU is active, but the
implementation is dominated by launch overhead, synchronization, and Python-side
work.

A small parameter sweep also showed the same pattern:

```text
case substeps=1 iterations=25
cpu    ~11.4 ms
cuda:0 ~14.8 ms

case substeps=2 iterations=25
cpu    ~25.1 ms
cuda:0 ~28.5 ms

case substeps=8 iterations=5
cpu    ~54.4 ms
cuda:0 ~69.2 ms

case substeps=8 iterations=25
cpu    ~99.9 ms
cuda:0 ~106.8 ms
```

The exact numbers vary run to run, but the pattern is stable: the default
problem size is too small and too synchronization-heavy for CUDA to win.

## Main Gaps

### 1. Tiny GPU Work Per Launch

**Status: STILL OPEN. Intrinsic to scene size; not addressable in
code without a larger benchmark scene.**

Relevant code:

- `src/avbd3d/solver_6dof.py`, primal loop around the per-color launch.
- `src/avbd3d/kernels_6dof.py`, `primal_update_6dof`.

The solver launches the primal update once per graph color per iteration:

```python
for color_id in range(self.num_colors):
    wp.launch(K.primal_update_6dof, dim=n_b, ...)
```

The default scene has only `34` bodies and `2` colors, so each launch has very
little useful work. GPU launch overhead is large compared with the compute.

The paper claim uses enormous scenes, where each launch has tens or hundreds of
thousands of independent bodies to process.

### 2. Too Many Launches Per Visual Frame

**Status: STILL OPEN. Needs CUDA-only `wp.ScopedCapture` to validate.**

Default viewer settings:

```text
--substeps 8
--iterations 25
post_stabilize=True
```

Each substep does broadphase/contact work, warm-start/prelude, many primal
launches, many dual launches, and finalize work.

With `8` substeps, `25` iterations, and `2` colors, the frame has hundreds of
small Warp launches.

Paper highlighted numbers use about `4` iterations. They are not directly
comparable to the current viewer's stability-heavy settings.

What closing this needs: CUDA-graph capture of the substep kernel
sequence (predict_inertial → substep_prelude → (primal × colors +
dual) × iters → finalize) with re-capture on topology change. The
graph replays as one launch per substep on CUDA. `wp.ScopedCapture`
is CUDA-only — there is no CPU code path to validate against, so
this stays deferred until a GPU is available. Implementation plan
in `PERFORMANCE_PROGRESS.md` §5.

### 3. Dynamic Contact Generation Is Not Fully GPU-Resident

**Status: geometry hot loop CLOSED (opt-in). Row append still Python.**

The Sutherland-Hodgman face-clip + closest-segment-pair + tangent-basis
+ body-local-offset math now lives in
`kernels_6dof.py::obb_contact_manifold_6dof`. Enable with:

```python
Solver6DOF(..., use_warp_face_clip=True)
```

The kernel handles both face-face (up to 4 contacts per pair via SH
clipping) and edge-edge (1 contact via closest-segment-pair) inside
one launch. Output is written to fixed-size buffers (4 contact slots
per pair) and read back once per substep.

What is still CPU-side: the row-append into `self._rows` (the Python
`_Row` list). That's gap §4 below — the contact pipeline math is now
on GPU, but the dynamic constraint storage isn't yet.

Relevant code:

- `kernels_6dof.py::obb_contact_manifold_6dof` (kernel).
- `solver_6dof.py::_warp_emit_contacts_kernel_path` (integration).
- `solver_6dof.py::_warp_broadphase_emit_contacts` (the branch on the flag).

Parity vs the Python emitter is guarded by
`tests/test_kernel_face_clip.py::test_kernel_face_clip_matches_python_stack`
(3-cube tower, 360 frames × 8 substeps = 2880 manifold passes; max y
divergence between paths ~6 µm on the CPU backend).

The flag stays default-False until benchmarked on real CUDA hardware.

### 4. Constraint Rows Are Python Objects Rebuilt Every Substep

**Status: STILL OPEN.** This is now the largest remaining
software-side gap.

The §3 kernel computed each contact's geometry on GPU but still
appends the resulting Box-Box + tangent rows to `self._rows` (the
Python `_Row` list). `_flush()` then rebuilds the full `c_*` Warp
arrays + the body-adjacency CSR + the graph coloring on every substep.

What closing this needs:

1. Preallocate a fixed-cap "dynamic constraint region" at the tail of
   the `c_*` Warp arrays. Have `obb_contact_manifold_6dof` (or a
   sibling kernel) atomic-append directly into that region instead
   of writing to the per-pair contact slots that Python then reads.
2. Skip `_flush()` per substep when only the dynamic region changes.
3. Solve the coloring problem — either a conservative proximity-graph
   coloring computed once per scene (large N pays back the
   over-coloring cost) or atomic-primal updates (loses Gauss-Seidel
   ordering; AVBD paper avoids this).

The official `avbd-demo3d` GPU pipeline uses option (3a) — fixed
coloring + GPU-resident constraint buffers + atomic-append from the
contact kernel. That's the target architecture.

### 5. Contact Warm-Start Cache Crosses CPU/GPU

**Status: CLOSED.**

The per-substep cache restore + persist now run as Warp kernels
(`cache_restore_6dof`, `cache_collect_6dof`) against preallocated
GPU-resident scratch buffers. The previous code did 6 full-array
`.numpy()` reads + 3 `wp.array(...)` rebuilds per substep (≈ 48
stream syncs per frame at `substeps=8`); the new code does 1 small
upload + 1 launch (restore) + 1 launch + 1 small readback (persist).

Host-side staging arrays are reused via `wp.array.assign(prefix)`
so the device side does not reallocate per substep.

Relevant code:

- `kernels_6dof.py::cache_restore_6dof`,
  `kernels_6dof.py::cache_collect_6dof`.
- `solver_6dof.py::_restore_cache_from_pool`,
  `solver_6dof.py::_persist_cache_from_pool`,
  `solver_6dof.py::_ensure_pool_buffers`.

Validated by the existing `test_two_cubes_stack_axis_aligned` and
`test_three_cubes_stack` cases — both stress the cache restore/persist
path across substep boundaries.

### 6. Viewer Readbacks Happen Every Frame

**Status: CLOSED.**

`Solver6DOF.read_state_batched()` packs all seven per-frame fields the
viewer reads (positions, orientations, angular velocities, lambdas,
active, was_static, c_type) into two staging buffers via the
`viewer_pack_bodies_6dof` + `viewer_pack_rows_6dof` kernels. The HUD +
scene update now does 2 `.numpy()` calls per rendered frame instead of
7. The remaining residual cost — individual viser scene-handle updates
per body — is gap §7 (renderer, out of scope for the solver).

Relevant code:

- `kernels_6dof.py::viewer_pack_bodies_6dof`,
  `kernels_6dof.py::viewer_pack_rows_6dof`.
- `solver_6dof.py::Solver6DOF.read_state_batched`.
- `examples/viewer.py::Viewer.tick` (now reads `state["..."]` keys).

A direct comparison vs. the per-array readers (`positions()`,
`orientations()`, …) produces identical float values and exact int
matches for `active` / `was_static` / `c_type`.

### 7. Renderer/View Layer Is Not Built for Millions of Bodies

**Status: STILL OPEN. Out of scope for the solver.**

`viser` is convenient for debugging and interaction, but the current viewer
updates individual scene handles from Python. That is not comparable to an
instanced renderer or GPU-driven visualization.

Even if the solver were faster, the viewer architecture would become a
bottleneck long before millions of bodies. Closing this is a renderer
change (instanced draw via a different visualization stack), not a
solver change.

### 8. Hardware Difference

**Status: STILL OPEN. Not addressable in code.**

The paper number cited above uses an RTX 4090. The local machine reported:

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
```

The 3060 Laptop GPU is much slower than a 4090. This does not explain a 30x gap
by itself, but it matters once the implementation is otherwise optimized.

## What Is Implemented Correctly vs. What Is Missing

Implemented in this repo:

- AVBD-style primal/dual iteration.
- 6-DOF rigid-body local solve.
- Graph coloring for parallel body updates (rebuilt per substep — see §4).
- Warp kernels for core solver operations.
- Warp broadphase / SAT / contact manifold (manifold opt-in via
  `use_warp_face_clip=True`, see §3).
- GPU-resident contact warm-start cache (was §5, now closed).
- Batched viewer readback (was §6, now closed).
- Persistent lambda/penalty warm-starting for contacts.
- Static/dynamic friction logic.
- Interactive Viser viewer.

Still missing for paper-like performance:

- GPU-resident dynamic constraint buffers (§4 — biggest open item).
- Fewer Python-driven kernel launches (§2 — needs CUDA-graph capture).
- Larger batched workloads (§1 — scene-size bound, not addressable in code).
- Lower iteration-count benchmark mode (cheap to add; gated on GPU).
- GPU-friendly rendering / instancing (§7 — out of scope for the solver).

## Practical Interpretation

The current repo is useful for validating AVBD concepts and debugging behavior,
but it is not yet a reproduction of the paper's optimized performance path.

After this session's changes:

- Cache restore/persist no longer hops through CPU per substep (§5).
- Viewer no longer issues 7 separate stream syncs per rendered frame (§6).
- The contact-manifold geometry hot loop runs on GPU when the opt-in
  flag is on (§3).

The slow result is still expected on small scenes:

- CPU does relatively well because the scene is small and Python/Warp launch
  overhead dominates.
- GPU does not win because it is doing very little work per launch (§1) and
  Python still drives hundreds of launches per frame (§2).
- Increasing bodies will help GPU occupancy, but the per-substep `_flush()`
  rebuild (§4) will then be the next bottleneck — that is the next milestone.

## Most Important Code Hotspots

After this session's changes, the remaining hotspots are:

- `src/avbd3d/solver_6dof.py`
  - `_flush` / constraint upload path: still rebuilds the full `c_*`
    Warp arrays + body adjacency CSR + graph coloring on every substep
    when dynamic contacts mutate `self._rows`. This is gap §4 and is
    now the single largest software-side cost on CPU. Closing it
    requires GPU-resident dynamic row buffers.
  - primal loop: one launch per color per iteration. With small scenes
    each launch is mostly overhead (§1). With large scenes the launches
    add up (§2) and would benefit from CUDA-graph capture.
  - `_warp_emit_contacts_kernel_path`: still has one batched readback
    of the kernel's per-pair contact output (count + ref_is_a + n/t/b
    + off_ref + off_inc — 8 small arrays). One sync per substep, vs. the
    previous Python path that read back broadphase results AND did the
    face-clip math in Python. Eliminating this last readback is part
    of closing §4 (rows would be on GPU; no Python row append needed).

- `examples/viewer.py`
  - Per-frame Viser scene-handle updates (per-box `.position` /
    `.wxyz`). Solver-side state is now batched (§6 closed) but the
    rendering layer still iterates Python objects (§7, out of scope).

- `src/avbd3d/kernels_6dof.py`
  - `primal_update_6dof` / `dual_update_6dof`: still launched per-color
    per-iter. Mostly bound by §1/§2, not by the kernel math itself.

## Bottom Line

The gap is not that AVBD is slow. The gap is that this implementation
does not yet have the GPU-resident, large-batch, low-synchronization
architecture needed to reproduce the paper's performance claims.

Progress on the six-milestone roadmap:

1. ☑ Keep dynamic contact generation and contact manifolds on GPU.
   (§3 — `obb_contact_manifold_6dof` kernel, opt-in.)
2. ☐ Store constraints in preallocated GPU buffers rather than Python
   `_Row` objects. (§4 — open; now the biggest single item.)
3. ☑ Keep the contact warm-start cache on GPU. (§5 — closed.)
4. ◐ Reduce `.numpy()` calls in the solver/viewer hot path. (§5 + §6
   closed; the solver still has one batched readback per substep for
   the §3 kernel output, which closes when §4 closes.)
5. ☐ Benchmark with paper-like iteration counts and much larger scenes.
   (Gated on GPU access.)
6. ☐ Use a renderer path that can instance many bodies without Python
   per-object updates. (§7 — out of scope for the solver.)

Six-milestone score: 2.5 done, 0 + 2 + 1 deferred (the latter gated on
either GPU access or scope-expansion into the renderer).

