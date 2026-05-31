"""Light-weight scene-building handles. The actual state lives as Warp arrays
inside Solver; these objects just remember indices.

`Body.shape` is metadata the viewer reads — the solver still treats every body
as a 3-DOF point mass. Once 6-DOF rigid bodies are added, the same shape will
drive both render and collision."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Shape:
    """Visual primitive attached to a body. Solver still treats body as a
    point mass; this only affects the viewer."""
    kind: Literal["sphere", "cube", "pillar"] = "sphere"
    # sphere: radius. cube: half-extents (length 3). pillar: (radius, half-height)
    size: tuple[float, ...] = (0.1,)
    color: tuple[float, float, float] = (0.4, 0.6, 0.9)


@dataclass
class Body:
    """Handle to a particle in the solver. Index is the row in the body arrays."""
    index: int
    shape: Shape = field(default_factory=Shape)


@dataclass
class ConstraintHandle:
    """Handle to a constraint in the solver."""
    index: int
    rows: int
