"""GPU-resident contact manifold settling test.

The kernel face-clip path (`obb_contact_manifold_6dof` + `gpu_pool_emit_rows`)
is the only contact-manifold code path after AVBD_PERFORMANCE_GAP §1/§2 were
closed — Sutherland-Hodgman clipping, edge-edge closest-segment math, and
the BOX_BOX / tangent row writes all live on the GPU now and never round-trip
through Python during a substep.

This test guards that the GPU manifold + emit path settles a 3-cube tower to
the canonical stacking heights across 360 frames × 8 substeps = 2880 contact
manifold passes. It used to compare against a Python reference path; that
path was removed when the row pool moved to GPU.
"""

import numpy as np

from avbd3d import Solver6DOF


def _settle_3_cubes():
    h = 0.20
    s = Solver6DOF(dt=1.0 / 60.0, iterations=15, substeps=8)
    s.enable_self_collision(True, default_friction=0.5)
    b0 = s.add_box((0., 0.20, 0.), (h, h, h), mass=1.0, friction=0.5)
    b1 = s.add_box((0., 0.62, 0.), (h, h, h), mass=1.0, friction=0.5)
    b2 = s.add_box((0., 1.04, 0.), (h, h, h), mass=1.0, friction=0.5)
    for b in (b0, b1, b2):
        s.add_floor_contact_box(b, friction=0.5)
    for _ in range(360):
        s.step()
    return s.positions()[:, 1], s.velocities()[:, 1]


def test_kernel_face_clip_settles_3_cube_stack():
    ys, vs = _settle_3_cubes()
    expected = np.array([0.20, 0.60, 1.00], dtype=np.float32)
    for i, e in enumerate(expected):
        assert abs(float(ys[i]) - e) < 0.03, \
            f"cube {i}: y={ys[i]} expected ~{e}"
    # Stack should be quiescent — no residual sliding or bouncing.
    assert float(np.abs(vs).max()) < 0.05, \
        f"residual |v|: {float(np.abs(vs).max())}"
