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

### 3. Dynamic Contact Generation Is Not Fully GPU-Resident

Relevant code:

- `src/avbd3d/solver_6dof.py`, `_warp_broadphase_emit_contacts`.
- `src/avbd3d/solver_6dof.py`, `_emit_obb_pair_with_sat`.
- `src/avbd3d/solver_6dof.py`, `_rebuild_contact_pool`.

The current code does use Warp for AABB generation, BVH broadphase, and SAT, but
then reads data back to CPU:

```python
n_pairs = int(self._bp_pair_count.numpy()[0])
a_np = self._bp_pair_a.numpy()[:n_pairs]
b_np = self._bp_pair_b.numpy()[:n_pairs]
ov_np = self._bp_pair_overlap.numpy()[:n_pairs]
si_np = self._bp_pair_sat_idx.numpy()[:n_pairs]
nh_np = self._bp_pair_n_hat.numpy().reshape(-1, 3)[:n_pairs]
```

Then contact face clipping and row emission happen in Python/NumPy. This creates
CPU/GPU synchronization and prevents the contact pipeline from scaling like a
fully GPU-resident implementation.

Paper-level performance requires the collision/contact/constraint pipeline to
stay on GPU or to use carefully batched GPU buffers with minimal readback.

### 4. Constraint Rows Are Python Objects Rebuilt Every Substep

Dynamic contacts mutate `self._rows`, a Python list of `_Row` objects. After
contacts are rebuilt, the solver has to upload arrays again for:

- row types,
- body indices,
- anchors,
- offsets,
- stiffness,
- lambda,
- penalty,
- friction,
- active flags,
- adjacency,
- body colors.

That is good for clarity and debugging, but it is not how the paper-scale
implementation would represent constraints.

A high-performance implementation would usually use preallocated GPU buffers,
active counts, compaction, persistent contact keys, and on-device cache updates.

### 5. Contact Warm-Start Cache Crosses CPU/GPU

Relevant code:

- `src/avbd3d/solver_6dof.py`, contact cache restore before the solve.
- `src/avbd3d/solver_6dof.py`, contact cache persist after the solve.

The current implementation reads lambda, penalty, active flags, and static
friction state back to CPU to preserve contact-pool warm-start data.

This helps physical stability, but it adds synchronization and CPU-side work.

For paper-like throughput, this cache would need to be GPU-resident.

### 6. Viewer Readbacks Happen Every Frame

Relevant code:

- `examples/viewer.py`, `tick`.

The viewer reads from the solver every frame:

- positions,
- orientations,
- angular velocities,
- lambdas,
- active flags,
- static friction flags,
- row types.

Each `.numpy()` on CUDA synchronizes or transfers data to CPU. This is fine for a
small interactive debug viewer, but not for benchmark-style GPU throughput.

The paper demo's performance should be considered simulation/rendering pipeline
performance, not Python GUI object update performance.

### 7. Renderer/View Layer Is Not Built for Millions of Bodies

`viser` is convenient for debugging and interaction, but the current viewer
updates individual scene handles from Python. That is not comparable to an
instanced renderer or GPU-driven visualization.

Even if the solver were faster, the viewer architecture would become a
bottleneck long before millions of bodies.

### 8. Hardware Difference

The paper number cited above uses an RTX 4090. The local machine reported:

```text
NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
```

The 3060 Laptop GPU is much slower than a 4090. This does not explain a 30x gap
by itself, but it matters once the implementation is otherwise optimized.

## What Is Implemented Correctly vs. What Is Missing

Implemented or partially implemented in this repo:

- AVBD-style primal/dual iteration.
- 6-DOF rigid-body local solve.
- Graph coloring for parallel body updates.
- Warp kernels for core solver operations.
- Warp broadphase/SAT pieces.
- Persistent lambda/penalty warm-starting for contacts.
- Static/dynamic friction logic.
- Interactive Viser viewer.

Missing for paper-like performance:

- fully GPU-resident contact manifold generation,
- GPU-resident dynamic constraint buffers,
- GPU-resident contact warm-start cache,
- fewer CPU `.numpy()` readbacks,
- fewer Python-driven kernel launches,
- larger batched workloads,
- lower iteration-count benchmark mode,
- GPU-friendly rendering/instancing,
- likely CUDA graph capture or fused kernels for stable parts of the step.

## Practical Interpretation

The current repo is useful for validating AVBD concepts and debugging behavior,
but it is not a reproduction of the paper's optimized performance path.

The slow result is therefore expected:

- CPU does relatively well because the scene is small and Python/Warp launch
  overhead dominates.
- GPU does not win because it is doing very little work per launch and is forced
  to synchronize with CPU repeatedly.
- Increasing bodies substantially would help GPU occupancy, but the current
  Python contact/viewer pipeline will then become the next bottleneck.

## Most Important Code Hotspots

These are the areas responsible for the performance gap:

- `examples/viewer.py`
  - Per-frame `.numpy()` readbacks and individual Viser object updates.

- `src/avbd3d/solver_6dof.py`
  - `_step_one`: substep structure, contact rebuild, cache restore/persist.
  - `_warp_broadphase_emit_contacts`: GPU broadphase/SAT followed by CPU
    readback and Python face clipping.
  - `_emit_obb_pair_with_sat`: Python/NumPy contact manifold generation.
  - `_flush` / constraint upload path: Python rows converted into Warp arrays.
  - primal loop: one launch per color per iteration.

- `src/avbd3d/kernels_6dof.py`
  - `primal_update_6dof`: does the local solve on GPU, but the default scene
    launches it with too few active bodies.
  - `dual_update_6dof`: also launched many times over relatively small arrays.

## Bottom Line

The gap is not that AVBD is slow. The gap is that this implementation does not
yet have the GPU-resident, large-batch, low-synchronization architecture needed
to reproduce the paper's performance claims.

To move toward the paper result, the first major milestone would be:

1. keep dynamic contact generation and contact manifolds on GPU,
2. store constraints in preallocated GPU buffers rather than Python `_Row`
   objects,
3. keep the contact warm-start cache on GPU,
4. reduce `.numpy()` calls in the solver/viewer hot path,
5. benchmark with paper-like iteration counts and much larger scenes,
6. use a renderer path that can instance many bodies without Python per-object
   updates.

