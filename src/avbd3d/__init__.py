"""3D Augmented Vertex Block Descent rigid-body solver.

Ported from the 2D reference at https://github.com/savant117/avbd-demo2d
and generalized to 3D following Giles et al., SIGGRAPH 2025
(Augmented Vertex Block Descent).
"""

from .solver import Solver
from .solver_6dof import Solver6DOF, RigidBody, box_inertia_local, box_inv_inertia_local
from .scene import Body, ConstraintHandle, Shape
from .deformable import (
    DeformableBody, TetMesh, RenderSkin,
    build_tet_lattice, bind_tet_mesh_to_solver, load_bunny_surface, make_bunny,
    build_render_skin,
)

__all__ = [
    "Solver", "Body", "ConstraintHandle", "Shape",
    "Solver6DOF", "RigidBody", "box_inertia_local", "box_inv_inertia_local",
    "DeformableBody", "TetMesh", "RenderSkin",
    "build_tet_lattice", "bind_tet_mesh_to_solver", "load_bunny_surface",
    "make_bunny", "build_render_skin",
]
