"""Robustness tests for the 6-DOF solver's GPU-resident contact pool.

Targets the regressions found in the codex code review (plan file
~/.claude/plans/findings-high-zippy-robin.md):

  - Dense overlapping clusters used to SIGSEGV when the broadphase
    pair count exceeded `16·n_b` because the overflow branch only
    bumped the cap scalar without reallocating the pair buffers, and
    `gpu_pool_emit_rows` had no row-pool capacity guard. Fixed by the
    retry-loop in `_gpu_emit_dynamic_contacts` plus the kernel-side
    `n_cap_total` guard plus the host-side `_grow_row_pool`.

  - Body coloring used `spatial_8color` which assigned the same color
    to two bodies sharing a grid cell. Fixed by reverting to
    `greedy_color` over a body-adjacency graph seeded by static rows
    and an inflated-AABB overlap predicate.
"""

import numpy as np
import pytest
import warp as wp

from avbd3d.solver_6dof import Solver6DOF


def test_dense_cluster_no_crash():
    """60 unit boxes packed inside the span of a single cell. Previously
    SIGSEGV'd (exit 139) inside `gpu_pool_emit_rows` because the row pool
    overflowed `n_static + n_dyn_capacity`. Now the kernel guard plus
    host-side regrow keep n_active_rows ≤ capacity after the regrow."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    for i in range(60):
        s.add_box(position=(0.01 * i, 2.0, 0.0),
                  half_extents=(0.5, 0.5, 0.5),
                  mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s._flush()
    # First substep triggers the overflow → regrow → next-step capacity OK.
    with pytest.warns(RuntimeWarning, match="row pool overflow"):
        s.step()
    # Subsequent steps must not crash and must respect the grown capacity.
    for _ in range(4):
        s.step()
    n_active = int(s.n_active_rows.numpy()[0])
    assert n_active <= s._gpu_pool_n_capacity, (
        f"n_active_rows={n_active} exceeds capacity={s._gpu_pool_n_capacity}")


def test_coloring_safety_for_touching_bodies():
    """Two unit cubes overlapping in x. After round-2's runtime recolor
    the coloring reflects the *live* contact graph — step once so the
    broadphase has populated the adjacency, then check distinctness."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    s.add_box(position=(0.0, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.add_box(position=(0.6, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s.step()
    colors = s.body_color.numpy()
    assert colors[0] != colors[1], (
        f"touching bodies share color {colors[0]} — coloring would race")


def test_coloring_safety_for_stack():
    """A 5-tall stack of unit cubes — adjacent bodies must have
    different colors after the runtime recolor has run."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    for i in range(5):
        s.add_box(position=(0.0, 0.5 + i * 1.0, 0.0),
                  half_extents=(0.5, 0.5, 0.5), mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s.step()
    colors = s.body_color.numpy()
    for i in range(4):
        assert colors[i] != colors[i + 1], (
            f"stack bodies {i} and {i+1} share color {colors[i]}")


@pytest.mark.parametrize("mode", ["jones_plassmann", "jacobi"])
def test_coloring_mode_conflict_free(mode):
    """Both coloring algorithms (Jones–Plassmann and the speculative
    'jacobi' greedy, A2) must produce a conflict-free partition — no two
    bodies sharing an active constraint may share a color, else
    primal_update would race. A dense overlapping stack forces several
    colors so the check is meaningful."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4, coloring_mode=mode)
    for i in range(6):
        s.add_box(position=(0.0, 0.5 + i * 0.85, 0.0),
                  half_extents=(0.5, 0.5, 0.5), mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s.step()
    assert s.coloring_mode == mode
    assert s.count_color_conflicts() == 0, (
        f"{mode} coloring produced adjacent same-color bodies")
    assert 1 <= s.num_active_colors <= s._max_colors


def test_coloring_mode_switch_takes_effect_at_runtime():
    """Switching `coloring_mode` mid-sim must actually re-partition the
    bodies, not just relabel the attribute. Regression for the A4 gate
    bug: the setter left `_color_dirty` False, so `_step_one` reused the
    prior (still conflict-free) coloring and the new method never ran
    until a contact change happened to force a recolor.

    Uses the dense overlapping stack where Jones–Plassmann needs 3 colors
    but the speculative 'jacobi' greedy packs into 2 — so a switch that
    truly takes effect is observable as a drop in `num_active_colors`."""
    def make(mode):
        s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4, coloring_mode=mode)
        for i in range(6):
            s.add_box(position=(0.0, 0.5 + i * 0.85, 0.0),
                      half_extents=(0.5, 0.5, 0.5), mass=1.0, friction=0.3)
        s.enable_self_collision(True)
        return s

    s = make("jones_plassmann")
    s.step()
    jp_colors = s.num_active_colors
    # Steady state: the gate has settled and would skip a recolor.
    assert s._color_dirty is False

    # The precise regression guard: the setter must arm a recolor.
    s.coloring_mode = "jacobi"
    assert s._color_dirty is True, (
        "switching coloring_mode did not mark the coloring dirty — the "
        "live switch would be a no-op until contacts change")

    s.step()
    assert s.coloring_mode == "jacobi"
    assert s.count_color_conflicts() == 0
    # The jacobi partition actually ran: it packs no worse than JP did.
    assert s.num_active_colors <= jp_colors

    # Setting the same mode again is a no-op — must NOT force a needless
    # recolor on an already-settled stack.
    s.step()
    assert s._color_dirty is False
    s.coloring_mode = "jacobi"
    assert s._color_dirty is False


def test_jacobi_coloring_packs_no_worse_than_jp():
    """The speculative ('jacobi') coloring is first-fit, so on a chain it
    should use no more colors than Jones–Plassmann (typically fewer)."""
    def colors_for(mode):
        s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4, coloring_mode=mode)
        for i in range(6):
            s.add_box(position=(0.0, 0.5 + i * 0.85, 0.0),
                      half_extents=(0.5, 0.5, 0.5), mass=1.0, friction=0.3)
        s.enable_self_collision(True)
        s.step()
        return s.num_active_colors
    assert colors_for("jacobi") <= colors_for("jones_plassmann")


def test_coloring_after_dynamic_contact():
    """Two bodies start far enough apart that their initial AABBs
    (inflated or not) do not overlap, then move into contact. Before
    round-2's runtime recolor this case shared a color and raced; after
    it the colors must differ once broadphase has reported the pair.

    Codex repro: 'colors stayed [0, 0], then pairs=1 and active=4 when
    they collided'."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4, dt=1 / 240)
    s.enable_self_collision(True)
    a = s.add_box(position=(0.0, 2.0, 0.0), half_extents=(0.2, 0.2, 0.2),
                  mass=1.0, friction=0.3)
    b = s.add_box(position=(5.0, 2.0, 0.0), half_extents=(0.2, 0.2, 0.2),
                  mass=1.0, friction=0.3)
    # Give body b a strong velocity toward body a; with gravity off-axis
    # they meet within a few steps. (We're testing the *coloring* response
    # to dynamic contact, not the contact response itself.)
    s.set_velocity(b, (-40.0, 0.0, 0.0))
    # Step until broadphase reports a pair — at most ~60 substeps at dt=1/240.
    for _ in range(80):
        s.step()
        if int(s._bp_pair_count.numpy()[0]) > 0:
            break
    assert int(s._bp_pair_count.numpy()[0]) > 0, (
        "test setup: bodies never reached contact")
    colors = s.body_color.numpy()
    assert colors[a.index] != colors[b.index], (
        f"bodies in dynamic contact share color {colors[a.index]} — "
        "Gauss-Seidel race")


def test_graph_invalidates_on_iterations():
    """Mutating solver.iterations after a step must invalidate the
    cached graph signature. Meaningful even on the CPU fallback path
    because the signature is computed and compared regardless of
    whether capture is active."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    s.add_box(position=(0.0, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.add_box(position=(0.6, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s.step()
    sig_before = s._graph_signature
    assert sig_before is not None
    assert sig_before[0] == 4
    s.iterations = 9
    s.step()
    sig_after = s._graph_signature
    assert sig_after is not None
    assert sig_after[0] == 9, (
        f"signature did not pick up iterations change: {sig_after}")
    assert sig_before != sig_after


def test_primal_group_size_invalidates_graph():
    """primal_group_size (warp-per-body lane count) is part of the captured-
    graph signature: changing it must force a recapture, because the inner
    loop dispatches the serial vs accumulate+solve kernels based on it.
    CPU-safe — the signature is computed regardless of capture."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    s.add_box(position=(0.0, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5), mass=1.0)
    s.add_box(position=(0.6, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5), mass=1.0)
    s.enable_self_collision(True)
    s.step()
    sig_before = s._graph_signature
    s.primal_group_size = 8
    assert s.primal_group_size == 8
    s.step()
    assert s._graph_signature != sig_before, (
        "graph signature did not pick up the primal_group_size change")


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="warp-per-body primal path is CUDA-only")
@pytest.mark.parametrize("shuffle", [False, True], ids=["atomic", "shuffle"])
def test_warp_per_body_matches_serial_cuda(shuffle):
    """Both warp-per-body join flavours (group_size > 1) are math-preserving:
    they parallelize only the per-body constraint reduction (atomic_add, or a
    func_native __shfl_down_sync register reduction), so a small stable stack
    must follow the serial one-thread-per-body kernel to within the GPU-atomic
    noise floor (sub-mm). A systematic kernel error blows past the 1 mm bar."""
    def build(group, shuf=False):
        s = Solver6DOF(gravity=(0, -9.81, 0), iterations=20, substeps=4,
                       dt=1 / 60, device="cuda:0", primal_group_size=group,
                       primal_shuffle=shuf)
        s.enable_self_collision(True, default_friction=0.5)
        for k in range(4):
            b = s.add_box(position=(0.0, 0.13 + 0.24 * k, 0.0),
                          half_extents=(0.12, 0.12, 0.12), mass=1.0,
                          friction=0.5)
            s.add_floor_contact_box(b, friction=0.5)
        return s

    serial = build(1)
    parallel = build(8, shuf=shuffle)
    for _ in range(40):
        serial.step()
        parallel.step()
    wp.synchronize_device("cuda:0")
    diff = float(np.abs(serial.positions() - parallel.positions()).max())
    assert diff < 1.0e-3, (
        f"warp-per-body ({'shuffle' if shuffle else 'atomic'}) diverged from "
        f"serial by {diff*1e3:.3f} mm — expected sub-mm (reduction reorder only)")


def _independent_boxes(**kw):
    """Four boxes resting on the floor, spaced apart so there are NEVER any
    body-body contacts. The body graph stays edge-free, so the flush-time
    coloring (all color 0) never goes stale — this isolates the double-buffer
    MATH from the recolor-at-flush convergence trade."""
    s = Solver6DOF(gravity=(0, -9.81, 0), iterations=20, substeps=4,
                   dt=1 / 60, device="cuda:0", **kw)
    s.enable_self_collision(True, default_friction=0.5)
    for k in range(4):
        b = s.add_box(position=(2.0 * k, 0.13, 0.0),
                      half_extents=(0.12, 0.12, 0.12), mass=1.0, friction=0.5)
        s.add_floor_contact_box(b, friction=0.5)
    return s


def _stable_stack(**kw):
    s = Solver6DOF(gravity=(0, -9.81, 0), iterations=20, substeps=4,
                   dt=1 / 60, device="cuda:0", **kw)
    s.enable_self_collision(True, default_friction=0.5)
    for k in range(4):
        b = s.add_box(position=(0.0, 0.13 + 0.24 * k, 0.0),
                      half_extents=(0.12, 0.12, 0.12), mass=1.0, friction=0.5)
        s.add_floor_contact_box(b, friction=0.5)
    return s


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="gpu_resident path is CUDA-only")
@pytest.mark.parametrize("group", [8, 16, 32])
def test_gpu_resident_matches_serial_cuda(group):
    """The fully-GPU-resident path (double-buffered fused primal, paper §4)
    only changes WHEN data crosses PCIe, not the AVBD math. On a scene whose
    coloring never goes stale (independent boxes), the double buffer must
    reproduce the serial reference essentially bit-for-bit — the per-color
    Jacobi-vs-Gauss-Seidel choice is irrelevant when bodies share no
    constraints. (Stale-coloring convergence on contacting scenes is covered
    by test_gpu_resident_stack_stable.)"""
    serial = _independent_boxes(primal_group_size=1, gpu_resident=False)
    resident = _independent_boxes(primal_group_size=group, primal_shuffle=True,
                                  primal_fused=True, gpu_resident=True)
    for _ in range(40):
        serial.step()
        resident.step()
    wp.synchronize_device("cuda:0")
    diff = float(np.abs(serial.positions() - resident.positions()).max())
    assert diff < 1.0e-4, (
        f"gpu_resident (G={group}) diverged from serial by {diff*1e3:.4f} mm "
        "on a stale-coloring-free scene — the double-buffer math must match")


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="gpu_resident path is CUDA-only")
def test_gpu_resident_stack_stable():
    """On a contacting stack the resident coloring goes stale between flushes,
    so same-color pairs fall back to Jacobi (paper's documented behaviour). We
    don't require bit-parity with serial (the Jacobi fallback loosens
    convergence), but the result MUST stay finite and physically bounded — the
    4-box stack settles in place, it must not explode or tunnel through the
    floor."""
    resident = _stable_stack(primal_group_size=16, primal_shuffle=True,
                             primal_fused=True, gpu_resident=True)
    for _ in range(60):
        resident.step()
    wp.synchronize_device("cuda:0")
    xr = resident.positions()
    assert np.isfinite(xr).all(), "gpu_resident produced non-finite positions"
    # 4 stacked 0.24 m boxes: bodies must stay above the floor and near the
    # original ~1 m column, not blow up or sink through.
    assert xr[:, 1].min() > -0.05, "a body tunnelled through the floor"
    assert np.abs(xr).max() < 3.0, "the stack exploded (position out of bounds)"


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="gpu_resident path is CUDA-only")
def test_gpu_resident_no_per_substep_readback():
    """gpu_resident's contract: ZERO host readbacks in the per-substep hot
    loop. Count .numpy() calls across a steady-state frame — only the
    once-per-frame overflow safety check may fire (row + pair pool = 2 reads),
    versus the 3-per-substep (=> ~24/frame) the readback path issues."""
    s = _stable_stack(primal_group_size=16, primal_shuffle=True,
                      primal_fused=True, gpu_resident=True)
    for _ in range(8):                      # flush, first recolor, capture
        s.step()
    wp.synchronize_device("cuda:0")

    calls = {"n": 0}
    orig = wp.array.numpy

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    wp.array.numpy = counting
    try:
        s.step()
    finally:
        wp.array.numpy = orig
    wp.synchronize_device("cuda:0")
    assert calls["n"] <= 2, (
        f"gpu_resident issued {calls['n']} host readbacks in one frame; the "
        "hot loop must be readback-free (only the once-per-frame row+pair "
        "overflow check is allowed)")


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="gpu_resident path is CUDA-only")
def test_gpu_resident_recolor_every_substep_no_readback():
    """recolor_every_substep recolours each substep but must STILL be
    readback-free: fixed colouring rounds (no convergence sync) + a fixed
    MAX_COLORS primal loop (achieved count never read). Same <=2/frame budget
    as the flush-recolor path."""
    s = _stable_stack(primal_group_size=16, primal_shuffle=True,
                      primal_fused=True, gpu_resident=True,
                      recolor_every_substep=True)
    for _ in range(8):
        s.step()
    wp.synchronize_device("cuda:0")
    calls = {"n": 0}
    orig = wp.array.numpy

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    wp.array.numpy = counting
    try:
        s.step()
    finally:
        wp.array.numpy = orig
    wp.synchronize_device("cuda:0")
    assert calls["n"] <= 2, (
        f"recolor_every_substep issued {calls['n']} readbacks/frame; it must "
        "stay readback-free (fixed rounds + fixed color loop)")


@pytest.mark.skipif(not wp.is_cuda_available(),
                    reason="gpu_resident path is CUDA-only")
@pytest.mark.parametrize("recolor_every", [False, True],
                         ids=["recolor-flush", "recolor-substep"])
def test_gpu_resident_stack_quality_vs_nonresident(recolor_every):
    """Quality (not just non-explosion): a settled 4-box stack under the
    resident path must match the non-resident path's settle — small contact
    penetration and a near-identical rest column. The resident coloring may go
    stale (more Jacobi), but a *settled* stack is an attractor, so both reach
    the same ~0.24 m-spaced rest config. Guards against the Jacobi fallback
    quietly degrading contact resolution (the thing recolor_every_substep is
    meant to tighten)."""
    def settle(**kw):
        s = _stable_stack(**kw)
        for _ in range(150):                # long enough to reach rest
            s.step()
        wp.synchronize_device("cuda:0")
        return np.sort(s.positions()[:, 1])  # ascending stack heights

    ref = settle(primal_group_size=1, gpu_resident=False)
    res = settle(primal_group_size=16, primal_shuffle=True, primal_fused=True,
                 gpu_resident=True, recolor_every_substep=recolor_every)

    # Adjacent-box gaps: full box height is 0.24 m. A healthy stack keeps gaps
    # near 0.24; deep penetration would collapse them. Allow a few mm of soft
    # contact penetration, but not a collapse.
    ref_gaps = np.diff(ref)
    res_gaps = np.diff(res)
    assert res_gaps.min() > 0.20, (
        f"resident stack penetrated: min adjacent gap {res_gaps.min()*1e3:.1f} "
        f"mm << 240 mm box height (nonresident min {ref_gaps.min()*1e3:.1f})")
    # Rest column should match the reference settle within ~3 cm.
    assert float(np.abs(res - ref).max()) < 0.03, (
        "resident settled column drifted >30 mm from the non-resident settle")
