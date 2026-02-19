# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026

@author: simon with help of Copilot
Initial version with Copilot produced zero solution and
interpolation failure.
"""
import numpy as np
import matplotlib.pyplot as plt
import skfem as sf
from skfem.models.poisson import laplace


# ============================================================
# 1. Mesh generation
# ============================================================

# Simple rectangular cross-section for demonstration.
# You can later replace this with a pygmsh-generated mesh
# to include rounded corners or arbitrary shapes.
mesh = sf.MeshTri.init_tensor(
    np.linspace(-1.0, 1.0, 40),
    np.linspace(-0.5, 0.5, 20)
)


# ============================================================
# 2. Finite element basis
# ============================================================

# Quadratic triangular elements (P2) are a good choice for torsion.
basis = sf.Basis(mesh, sf.ElementTriP2())


# ============================================================
# 3. Weak form assembly
# ============================================================

# Stiffness matrix: ∫ grad(v)·grad(u) dA
A = sf.asm(laplace, basis)

# RHS: ∫ (-2) v dA
def rhs_load(v, w):
    return -2.0 * v

b = sf.asm(rhs_load, basis)


# ============================================================
# 4. Neumann boundary condition
# ============================================================

# Neumann term: ∫ g v ds, where g = x*n_y - y*n_x
def neumann_term(v, w):
    x, y = w.x[0], w.x[1]
    nx, ny = w.n[0], w.n[1]
    g = x * ny - y * nx
    return g * v

# Add Neumann contribution on the boundary
b += sf.asm(neumann_term, basis.boundary())


# ============================================================
# 5. Fix the constant mode (Neumann nullspace)
# ============================================================

# Select all boundary facets
boundary_facets = mesh.facets_satisfying(
    lambda x: np.ones(x.shape[1], dtype=bool)
)

# Get DOFs on these facets
D = basis.get_dofs(facets=boundary_facets)

# Apply constraint
A, b = sf.enforce(A, b,D=D)

# ============================================================
# 6. Solve the linear system
# ============================================================

S = sf.solve(A, b)


# ============================================================
# 7. Visualization
# ============================================================

# Interpolate solution to a grid for plotting
X, Y = np.meshgrid(
    np.linspace(-1, 1, 200),
    np.linspace(-0.5, 0.5, 100)
)

# coordinates in shape (2, N)
XY = np.vstack([X.ravel(), Y.ravel()])
# Build interpolator
interp = basis.interpolator(XY)
# Evaluate solution on grid
Sgrid = interp(S).reshape(X.shape)
plt.figure(figsize=(8, 3))
plt.contourf(X, Y, Sgrid, 40, cmap='coolwarm')
plt.colorbar(label="Warping function S(x,y)")
plt.title("Saint-Venant torsion: warping function")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()