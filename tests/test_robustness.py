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
    """Two unit cubes overlapping in x: previously got the same color
    because they share the parity-cell at cell_size=2·he. With greedy
    coloring over the inflated-AABB adjacency they must differ."""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    s.add_box(position=(0.0, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.add_box(position=(0.6, 2.0, 0.0), half_extents=(0.5, 0.5, 0.5),
              mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s._flush()
    colors = s.body_color.numpy()
    assert colors[0] != colors[1], (
        f"touching bodies share color {colors[0]} — coloring would race")


def test_coloring_safety_for_stack():
    """A 5-tall stack of unit cubes — adjacent bodies must have
    different colors. (Non-adjacent bodies in the stack may share, that's
    fine — they have no constraint.)"""
    s = Solver6DOF(gravity=(0, -9.8, 0), iterations=4)
    for i in range(5):
        s.add_box(position=(0.0, 0.5 + i * 1.0, 0.0),
                  half_extents=(0.5, 0.5, 0.5), mass=1.0, friction=0.3)
    s.enable_self_collision(True)
    s._flush()
    colors = s.body_color.numpy()
    for i in range(4):
        assert colors[i] != colors[i + 1], (
            f"stack bodies {i} and {i+1} share color {colors[i]}")
