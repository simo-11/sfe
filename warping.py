# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026
@author: simon with help of Copilot, reading the code
and knowledge on the correct solution.
"""
import numpy as np
import skfem.visuals.matplotlib as sfplot
import skfem as sf
from skfem.models.poisson import laplace
# Simple rectangular cross-section for demonstration.
mesh = sf.MeshTri.init_tensor(
    np.linspace(-0.05, 0.05, 20),
    np.linspace(-0.05, 0.05, 20)
)
# Quadratic triangular elements (P2) are a good choice for torsion.
basis = sf.Basis(mesh, sf.ElementTriP2())
# Stiffness matrix: ∫ grad(v)·grad(u) dA
A = sf.asm(laplace, basis)
# boundary condition
def bc(v, w):
    x, y = w.x[0], w.x[1]
    nx, ny = w.n[0], w.n[1]
    g = x * ny - y * nx
    return g * v
b = sf.asm(bc, basis.boundary())
# Fix the constant mode (Neumann nullspace)
boundary_facets = mesh.facets_satisfying(
    lambda x: np.ones(x.shape[1], dtype=bool)
)
D = basis.get_dofs(facets=boundary_facets)
A, b = sf.enforce(A, b,D=D.nodal['u'][[0]])
S = sf.solve(A, b)
ax=sfplot.plot3(basis, S, shading='gouraud')
