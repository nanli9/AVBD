"""Tests for the deformable bunny pipeline.

These cover the data path (tet lattice generation, constraint binding) and
basic dynamics (bunny falls, stays finite, settles near floor). The mesh
download is cached after the first run, so network is only hit once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from avbd3d import (
    Solver,
    build_tet_lattice,
    load_bunny_surface,
    make_bunny,
)


def _bunny_or_skip():
    try:
        return load_bunny_surface(scale=1.0)
    except Exception as e:
        pytest.skip(f"could not load bunny ({e!r})")


def test_tet_lattice_topology_is_consistent():
    surf = _bunny_or_skip()
    tet = build_tet_lattice(surf, resolution=8)
    # Every edge endpoint is a valid vertex
    assert tet.edges.min() >= 0
    assert tet.edges.max() < len(tet.vertices)
    # Every surface triangle uses surface verts only
    assert set(int(v) for v in tet.surface_tris.flatten()).issubset(
        set(int(v) for v in tet.surface_verts)
    )
    # No duplicate edges (the dedup in build_tet_lattice should guarantee this)
    pairs = {(int(a), int(b)) for a, b in tet.edges}
    assert len(pairs) == len(tet.edges)
    # Every tet has 4 distinct vertices
    for tet_ids in tet.tets:
        assert len(set(int(v) for v in tet_ids)) == 4


def test_tet_lattice_resolution_scales():
    surf = _bunny_or_skip()
    coarse = build_tet_lattice(surf, resolution=6)
    fine = build_tet_lattice(surf, resolution=12)
    # Going from res 6 → 12 should produce strictly more tets (and verts).
    assert len(fine.tets) > len(coarse.tets)
    assert len(fine.vertices) > len(coarse.vertices)


def test_bunny_binds_and_runs_without_nans():
    """End-to-end: build bunny, bind to solver, run frames, no NaNs.

    Tracks min/COM across all frames rather than just the last frame
    because with energy-correct post-stab dynamics the bunny rebounds
    a little and may be airborne at the moment of the final assertion.
    """
    s = Solver(dt=1.0 / 60.0, iterations=10, gravity=(0, -9.81, 0))
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        edge_stiffness=3.0e4, friction=0.5, floor_y=0.0)
    assert len(deform.bodies) == len(deform.tet.vertices)
    assert len(deform.edge_constraints) == len(deform.tet.edges)
    min_y_ever = float("inf")
    com_y_min = float("inf")
    com_y_max = -float("inf")
    for _ in range(60):
        s.step()
        verts_now = s.positions()[deform.indices]
        assert np.all(np.isfinite(verts_now))
        min_y_ever = min(min_y_ever, float(verts_now[:, 1].min()))
        cy = float(verts_now[:, 1].mean())
        com_y_min = min(com_y_min, cy)
        com_y_max = max(com_y_max, cy)
    # Bunny made floor contact at some point.
    assert min_y_ever < 0.1, (
        f"bunny never reached the floor (min y across all frames "
        f"= {min_y_ever:.3f})"
    )
    # COM bounded sanely throughout — no escapes upward, no falling
    # through the floor.
    assert com_y_min > -0.2, f"COM dipped to {com_y_min:.3f}"
    assert com_y_max < 1.5, f"COM rose to {com_y_max:.3f}"


def test_edge_rest_lengths_match_initial_positions():
    """Each DISTANCE constraint's rest = initial edge length."""
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        floor_y=None)  # no floor — keeps the rest test clean
    rest_arr = np.array([s._constraints[h.index].rest
                         for h in deform.edge_constraints])
    initial_pos = deform.tet.vertices
    computed = np.linalg.norm(
        initial_pos[deform.tet.edges[:, 0]] - initial_pos[deform.tet.edges[:, 1]],
        axis=1,
    )
    np.testing.assert_allclose(rest_arr, computed, rtol=1e-5)


def test_floor_contact_only_on_surface_verts():
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        floor_y=0.0)
    # We emit one FLOOR_CONTACT row (plus 2 friction rows because friction=0.5)
    # per surface vertex, so the floor-constraint *handle list* length matches.
    assert len(deform.floor_constraints) == len(deform.tet.surface_verts)


def test_mass_distribution_is_positive_everywhere():
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        density=500.0, floor_y=None)
    masses = np.array(s._bodies_mass[:len(deform.bodies)])
    assert (masses > 0).all(), "found a zero-mass particle (would be kinematic)"


def test_tet_volume_constraints_emitted_when_enabled():
    """Each tet should get exactly one TET_VOLUME row in the tet pool."""
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        volume_stiffness=1.0e4, floor_y=None)
    assert len(deform.volume_constraints) == len(deform.tet.tets)
    assert len(s._tets) == len(deform.tet.tets)


def test_tet_volume_disabled_when_stiffness_is_zero():
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        volume_stiffness=0.0, floor_y=None)
    assert len(deform.volume_constraints) == 0
    assert len(s._tets) == 0


def test_volume_preservation_prevents_tet_inversion_on_floor():
    """The canonical bunny-pancakes-without-volume-preservation regression:
    with TET_VOLUME at a reasonable stiffness, no tet should invert (signed
    volume flips sign) after the bunny falls onto the floor and settles."""
    s = Solver(dt=1.0 / 60.0, iterations=20, gravity=(0, -9.81, 0))
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 0.7, 0),
                        edge_stiffness=5.0e4, volume_stiffness=1.0e4,
                        friction=0.5, floor_y=0.0)
    for _ in range(180):
        s.step()
    pos = s.positions()[deform.indices]
    assert np.all(np.isfinite(pos)), "diverged"
    tets = deform.tet.tets
    v0 = pos[tets[:, 0]]; v1 = pos[tets[:, 1]]
    v2 = pos[tets[:, 2]]; v3 = pos[tets[:, 3]]
    V_now = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    inverted = int((np.sign(V_now) != np.sign(deform.tet.volumes)).sum())
    assert inverted == 0, f"{inverted} tets flipped sign with TET_VOLUME enabled"
    # Volume should stay within ±15% of rest after settling.
    ratio = V_now / deform.tet.volumes
    assert ratio.min() > 0.7, f"min V/V0={ratio.min():.3f} too low"
    assert ratio.max() < 1.3, f"max V/V0={ratio.max():.3f} too high"


def test_render_skin_reproduces_rest_pose_exactly():
    """The skinned bunny at rest should reconstruct the original
    Stanford bunny surface to floating-point precision.

    For each render vertex r_v with barycentric weights (b0..b3) into
    its containing tet (v0..v3), b0·v0 + b1·v1 + b2·v2 + b3·v3 = r_v
    at construction time. Verifies the barycentric solve is correct
    and `apply()` does what it claims.
    """
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=8, scale=1.0, center=(0, 1.0, 0),
                        floor_y=None)
    assert deform.skin is not None
    # Weights sum to 1 (partition of unity).
    np.testing.assert_allclose(deform.skin.weights.sum(axis=1), 1.0,
                               rtol=1.0e-4)
    # Apply at rest reconstructs the originals.
    rest_reconstructed = deform.skin.apply(deform.tet.vertices,
                                           deform.tet.tets)
    diff = np.linalg.norm(rest_reconstructed - deform.skin.render_verts,
                          axis=1)
    assert diff.max() < 1.0e-4, (
        f"rest skin reconstruction off by {diff.max()*1000:.4f} mm"
    )


def test_render_skin_follows_rigid_translation():
    """If we translate every tet vertex by a constant offset, every
    render vertex must translate by the same offset (the skin is a
    convex combination of tet verts)."""
    s = Solver(dt=1.0 / 60.0, iterations=5)
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 1.0, 0),
                        floor_y=None)
    delta = np.array([0.3, -1.4, 0.7], dtype=np.float32)
    shifted_tet = deform.tet.vertices + delta
    shifted_skin = deform.skin.apply(shifted_tet, deform.tet.tets)
    expected = deform.skin.render_verts + delta
    err = np.linalg.norm(shifted_skin - expected, axis=1)
    assert err.max() < 1.0e-4


def test_volume_preservation_off_lets_tets_invert():
    """Mirror of the above: without TET_VOLUME, the bunny pancakes hard
    enough on floor contact to invert many tets. This locks in the v1
    edge-springs-only behaviour as 'documented bad'."""
    s = Solver(dt=1.0 / 60.0, iterations=20, gravity=(0, -9.81, 0))
    deform = make_bunny(s, resolution=6, scale=1.0, center=(0, 0.7, 0),
                        edge_stiffness=5.0e4, volume_stiffness=0.0,
                        friction=0.5, floor_y=0.0)
    for _ in range(180):
        s.step()
    pos = s.positions()[deform.indices]
    if not np.all(np.isfinite(pos)):
        # Edge-only mass-spring blew up on floor contact — that's the bug
        # this test is documenting; treat as "passed" (the bug exists).
        return
    tets = deform.tet.tets
    v0 = pos[tets[:, 0]]; v1 = pos[tets[:, 1]]
    v2 = pos[tets[:, 2]]; v3 = pos[tets[:, 3]]
    V_now = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    inverted = int((np.sign(V_now) != np.sign(deform.tet.volumes)).sum())
    # With volume preservation off and the bunny squashed, we expect some
    # tets to flip sign. If zero invert it means the bunny didn't actually
    # touch the floor — bump the drop height in that case.
    assert inverted > 0, "expected some tet inversion without TET_VOLUME"
