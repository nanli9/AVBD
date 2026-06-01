"""Parity test for the opt-in Warp face-clip path.

`Solver6DOF(use_warp_face_clip=True)` swaps the Python Sutherland-Hodgman
face-clip + edge-edge closest-segment-pair for the
`obb_contact_manifold_6dof` Warp kernel. This test runs a 3-cube tower
under both paths and confirms they settle to the canonical stacking
heights AND agree with each other to within tight bounds across 360
frames × 8 substeps = 2880 contact-manifold passes.

The kernel path is opt-in (default off) until benchmarked on real CUDA
hardware. This test guards the correctness side on the CPU backend.
"""

import numpy as np

from avbd3d import Solver6DOF


def _settle_3_cubes(use_kernel: bool):
    h = 0.20
    s = Solver6DOF(dt=1.0 / 60.0, iterations=15, substeps=8,
                   use_warp_face_clip=use_kernel)
    s.enable_self_collision(True, default_friction=0.5)
    b0 = s.add_box((0., 0.20, 0.), (h, h, h), mass=1.0, friction=0.5)
    b1 = s.add_box((0., 0.62, 0.), (h, h, h), mass=1.0, friction=0.5)
    b2 = s.add_box((0., 1.04, 0.), (h, h, h), mass=1.0, friction=0.5)
    for b in (b0, b1, b2):
        s.add_floor_contact_box(b, friction=0.5)
    for _ in range(360):
        s.step()
    return s.positions()[:, 1], s.velocities()[:, 1]


def test_kernel_face_clip_matches_python_stack():
    py_y, py_v = _settle_3_cubes(False)
    k_y, k_v = _settle_3_cubes(True)
    # Both paths must settle at the canonical stacking heights.
    expected = np.array([0.20, 0.60, 1.00], dtype=np.float32)
    for label, ys in (("python", py_y), ("kernel", k_y)):
        for i, e in enumerate(expected):
            assert abs(float(ys[i]) - e) < 0.03, \
                f"{label} cube {i}: y={ys[i]} expected ~{e}"
    # Trajectory parity — cumulative float order may diverge slightly
    # over 360 steps but stay well inside the band the stack test
    # already accepts.
    dy = float(np.abs(py_y - k_y).max())
    dv = float(np.abs(py_v - k_v).max())
    assert dy < 0.005, f"y mismatch between paths: {dy*1000:.2f} mm"
    assert dv < 0.1, f"v mismatch between paths: {dv} m/s"
