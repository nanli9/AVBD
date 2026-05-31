"""Greedy graph coloring for AVBD's parallel per-body update.

Two bodies share an edge iff they share at least one constraint. Bodies in the
same color class touch disjoint constraint sets, so their Gauss-Seidel
primal updates commute and can run in parallel.

For Gaia's GPU coloring algorithm (which is incremental and topology-aware)
see Gaia/Source/.../GraphColoring*. The greedy variant here matches the
Welsh-Powell algorithm and is good enough for the small static graphs we
have during prototyping. Worst-case it uses (max_degree + 1) colors, which
is usually fine for sparse constraint graphs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def build_body_edges(
    n_bodies: int,
    c_body_a: Iterable[int],
    c_body_b: Iterable[int],
) -> list[set[int]]:
    """Adjacency list: adj[i] = set of bodies that share at least one constraint
    with body i (excluding world-anchored constraints where body_b < 0)."""
    adj: list[set[int]] = [set() for _ in range(n_bodies)]
    for a, b in zip(c_body_a, c_body_b):
        if b < 0 or a < 0:
            continue
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def greedy_color(adj: list[set[int]]) -> np.ndarray:
    """Welsh-Powell greedy coloring. Returns int32 array of body → color id.
    Color ids are dense (0..k-1)."""
    n = len(adj)
    # Order by descending degree (Welsh-Powell heuristic)
    order = sorted(range(n), key=lambda i: -len(adj[i]))
    color = np.full(n, -1, dtype=np.int32)
    for i in order:
        used = {color[j] for j in adj[i] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color


def color_summary(color: np.ndarray) -> dict[int, int]:
    """Return {color_id: count} for diagnostics."""
    out: dict[int, int] = {}
    for c in color:
        out[int(c)] = out.get(int(c), 0) + 1
    return out
