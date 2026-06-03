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
