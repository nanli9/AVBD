"""3D Augmented Vertex Block Descent rigid-body solver.

Ported from the 2D reference at https://github.com/savant117/avbd-demo2d
and generalized to 3D following Giles et al., SIGGRAPH 2025
(Augmented Vertex Block Descent).
"""

from .solver import Solver
from .scene import Body, ConstraintHandle, Shape

__all__ = ["Solver", "Body", "ConstraintHandle", "Shape"]
