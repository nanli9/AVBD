# AVBD GPU Residency — Progress Notes

This document tracks work done against the gaps identified in
`AVBD_PERFORMANCE_GAP.md`. It is the companion to that file: gap doc says
what is wrong, this doc says what got fixed and what is queued.

## Constraint

This session was done on a machine **without an NVIDIA GPU**. The Warp CUDA
backend is unavailable here, so:

- Every change had to be validated on the Warp CPU backend.
- Real GPU performance numbers cannot be reported.
- Changes were limited to architectural shifts whose correctness is
  testable on CPU but whose performance only materializes on CUDA.

The existing pytest suite is the correctness gate. All 20 tests in
`tests/test_solver_6dof.py` pass before and after each change.

---

## Done

### 1. GPU-resident contact warm-start cache (closes GAP §5)

**What changed.** The per-substep contact cache restore + persist used to
do six full-array `.numpy()` transfers + three `wp.array(...)` re-creations
per substep. With `substeps=8` that was 48 stream syncs / frame just to
move λ, k, and `was_static` across the CPU↔GPU boundary.

The new code:

- adds two Warp kernels in `src/avbd3d/kernels_6dof.py`:
  - `cache_restore_6dof` — sparse-writes λ/k/was_static into the live
    constraint arrays from a packed input buffer using per-pair row
    indices.
  - `cache_collect_6dof` — reads the live λ/k/active/was_static for every
    pool-pair row into a packed output buffer (8 floats per pair).
- adds preallocated GPU scratch buffers on `Solver6DOF`
  (`_pool_idx_n/_pool_idx_t/_pool_idx_b`, `_pool_in_packed`,
  `_pool_out_packed`) that grow in 256-pair chunks and persist across
  substeps.
- adds preallocated host-side numpy staging buffers that are reused via
  `wp.array.assign(np_slice)` so the per-substep upload pattern is no
  longer "allocate temporary wp.array → wp.copy → free".
- rewrites the restore + persist code paths in `Solver6DOF._step_one`
  into two helper methods (`_restore_cache_from_pool` and
  `_persist_cache_from_pool`) that issue exactly one small upload +
  one launch (restore) and one launch + one small readback (persist)
  per substep.

**Effect.** From `6 full-array transfers + 3 full-array uploads + Python
dict-lookup loop + 4 full-array reads` per substep down to
`1 small upload + 1 small readback + 2 kernel launches + Python dict
loop`. The reduction in CUDA stream syncs is the win — even when the
transfers themselves are small, each `.numpy()` call serializes the
stream.

**Validated.** `test_two_cubes_stack_axis_aligned` and
`test_three_cubes_stack` exercise the cache restore/persist path
(`enable_self_collision(True)` plus a multi-substep solve that has to
preserve λ across substep boundaries to settle). Both pass.

### 2. Batched viewer readback (closes GAP §6)

**What changed.** The interactive viewer did 7 separate `.numpy()` calls
per rendered frame (positions, orientations, angular velocities,
lambdas, active, c_was_static, c_type). On CUDA each one is a stream sync.

The new code:

- adds two Warp kernels:
  - `viewer_pack_bodies_6dof` packs (x, q, ω) into 10 contiguous floats
    per body.
  - `viewer_pack_rows_6dof` packs (λ, active, was_static·16 + c_type)
    into 3 contiguous floats per row — the small bit-pack keeps it to
    one float since both fields fit in a single byte.
- adds `Solver6DOF.read_state_batched()` which launches both kernels
  and returns a dict with all seven fields after two `.numpy()` calls.
- rewires `examples/viewer.py:tick` to use the batched call.

**Effect.** Per-frame stream syncs in the viewer drop from ~7 to 2,
with one shared scratch allocation that is reused frame-to-frame.

**Validated.** A direct comparison of `read_state_batched()` against the
per-array readers on a stacked-box scene produces identical values
within float tolerance (positions/orientations/ω/λ) and exact integer
match (active/was_static/c_type).

---

## Deferred (with implementation plans)

### 3. Warp face-clip + contact-emit kernel (GAP §3 — biggest remaining)

**Why deferred.** The Python face-clip in `_emit_obb_pair_with_sat` /
`_emit_obb_edge_edge` is the largest single Python hot-spot per substep
for OBB stacks. Porting it to Warp would close the §3 gap. But the
math is dense (Sutherland-Hodgman on a variable-length polygon,
reference vs incident axis selection, edge-edge closest-segment-pair,
body-local offset extraction, tangent basis), the output is variable
per pair (0–4 contacts), and the kernel has no CPU baseline to diff
against — any geometry bug would only surface under real OBB stacks
on CUDA. Without a GPU to validate, the risk of a quiet correctness
regression is too high.

**Implementation plan when GPU access exists.**

1. Add a `poly_scratch` Warp array of shape `(max_pairs * 16,)` with
   dtype `wp.vec3`. Each pair owns 16 slots: 8 for the current
   polygon, 8 for the SH double-buffer.
2. New kernel `obb_face_clip_emit_6dof`:
   - inputs: `x`, `q`, `half_extents`, the existing
     `pair_a/b/overlap/sat_idx/n_hat` buffers from `obb_sat_pairs`,
     `n_pairs`, `margin`, `poly_scratch`.
   - outputs (one block per pair × up to 4 contacts):
     `contact_valid`, `contact_ref_is_a`, `contact_off_ref`,
     `contact_off_inc`, `contact_n_hat`, `contact_t_hat`,
     `contact_b_hat`, `contact_depth`.
   - per pair:
     - skip if `pair_overlap[p] == 0`.
     - if `sat_idx >= 6`: leave all `contact_valid` slots = 0
       (Python edge-edge handler runs for these pairs only).
     - otherwise: identify ref/inc, compute ref-face center/normal/4
       side planes, compute incident-face vertices, run SH clip in
       the scratch buffer (double-buffer with two 8-vec3 chunks),
       filter to `depth < margin`, top-4 by insertion sort, compute
       body-local offsets via `R^T · (p − c)`, build tangent basis
       (Duff 2017, identical to `_orthonormal_basis`).
3. Add a Solver6DOF flag `use_warp_face_clip: bool = False` (default
   off until a GPU validation pass lands). When on,
   `_warp_broadphase_emit_contacts` skips the Python face-clip and
   reads back the kernel's contact arrays instead.
4. Edge-edge pairs still go through the existing Python
   `_emit_obb_edge_edge` code — the kernel marks them
   `contact_valid = 0` and Python re-runs the closest-segment-pair
   math.
5. Bench plan: an RTX 3060 Laptop run of `examples/viewer.py` with
   `--bodies 200` and `--substeps 5 --iterations 4` (paper-like
   iteration count) before and after the flag flip, looking for
   `step_ms_avg` and `bp_ms` shifts.

**Risk.** The SH-clip floating-point edge cases (vertex exactly on
plane, polygon degenerates to a point) are the most likely source of
bugs. The Python version has been hardened by the existing stacking
tests; the kernel version needs an equivalent test pass under CUDA.

### 4. GPU-resident dynamic constraint rows (GAP §4)

**Why deferred.** Closing this gap requires also closing §3 — the
contact manifold needs to land in GPU-resident buffers, not get
appended to a Python `_Row` list. Once §3 lands, §4 becomes:
preallocate a fixed-cap "dynamic constraint region" in the existing
`c_*` Warp arrays, have the face-clip kernel atomic-append into it,
and skip the full `_flush()` rebuild when only the dynamic region
changes.

This also forces the body adjacency / coloring problem. Today
`_build_adjacency` and `greedy_color` run on CPU and use the row
list. Options for paper-scale:

- Pre-compute a conservative proximity-graph coloring (one per body,
  fixed for a scene) and accept that some same-color body pairs may
  later acquire a contact constraint — at large N, the over-coloring
  cost is negligible vs the per-substep recolor.
- Lock-free atomic primal updates and skip coloring (well-defined
  Gauss-Seidel sub-iteration semantics get lost; the paper uses
  graph coloring for exactly this reason, so probably a no-go).

The official `avbd-demo3d` GPU pipeline uses the conservative
proximity-graph approach.

### 5. CUDA graph capture for the substep (GAP §2)

**Why deferred.** `wp.ScopedCapture` is CUDA-only. There is no CPU
test path. The win is real (hundreds of small kernel launches replaced
by one graph replay) but is limited to scenarios where the topology
is stable across substeps — which means either `self_collide=False`
or §4 already landed (dynamic rows in fixed slots, kernels still see
the same buffer shapes).

**Implementation plan when GPU access exists.**

1. Capture happens after `_flush()` and before the per-iteration loop.
   The captured graph holds the `predict_inertial → substep_prelude →
   (cache_alpha_C0?) → (primal × colors + dual) × iters → finalize`
   sequence.
2. Invalidate + recapture when:
   - `n_b`, `n_c`, `num_colors` changes (topology change), or
   - `self_collide=True` and the broadphase report contains any new
     pair this substep (requires a GPU "new pairs" counter).
3. Replay via `wp.capture_launch(graph)` per substep; the broadphase
   and contact-emit live outside the captured region.
4. Skip on CPU backend — `wp.ScopedCapture` raises on non-CUDA, so
   the integration is `if device.startswith("cuda"): ... else: ...`.

### 6. Iteration-count benchmark mode (GAP § header — "paper reports 4 iters")

A small change: expose a `Solver6DOF` benchmark-friendly preset that
matches the paper's `substeps=1, iterations=4`, and add a headless
benchmark script `examples/bench_6dof.py` that prints `step_ms_avg`
over a fixed warm-up + measurement window. This is GPU-agnostic — it
just removes the "viewer settings make the numbers look bad" issue
from future GPU runs. Cheap to add but not done in this session
since the relevant gating is GPU access.

---

## Net architectural state

After this session:

- Contact warm-start cache: **GPU-resident** (was CPU-mediated).
- Viewer per-frame state: **batched** (was 7 separate syncs).
- Per-substep upload pattern: **reuses preallocated buffers via
  `assign()`** (was allocating temporary `wp.array`s).
- Contact manifold generation: still mixed (GPU broadphase + SAT,
  CPU face-clip + edge-edge).
- Dynamic constraint rows: still Python `_Row` list, rebuilt every
  substep via `_flush()`.
- CUDA graph capture: not yet present.

The remaining gaps are the ones the original document calls out as
needing paper-scale architecture. They are queued with implementation
plans above and gated on GPU access for validation.
