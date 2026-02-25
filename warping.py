# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026
@author: simon with help of Copilot, reading the code
and knowledge on the correct solution.
"""
import numpy as np
import skfem as sf
from skfem.models.poisson import laplace
import matplotlib.pyplot as pyplot
import math
import types
import pyvista as pv

def solve(mesh: sf.mesh.Mesh, elem: sf.element.Element):
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
    return (S,basis)
mesh = sf.MeshTri.init_tensor(
    np.linspace(-0.05, 0.05, 20),
    np.linspace(-0.05, 0.05, 20)
)
ucs=[
     types.SimpleNamespace(elem=sf.ElementTriP2()),
     types.SimpleNamespace(elem=sf.ElementTriP1()),
     ]
names=[]
for uc in ucs:
    uc.name=type(uc.elem).__name__.split('Element')[-1]
    names.append(uc.name)
rows = math.ceil(len(names) / 2)
mosaic = []
idx = 0
for r in range(rows):
    row = []
    for c in range(2):
        row.append(names[idx] if idx < len(names) else ".")
        idx += 1
    mosaic.append(row)
fig = pyplot.figure(num='warping using skfem',clear=True)
axes = {}
for r, row in enumerate(mosaic):
    for c, key in enumerate(row):
        if key == ".":
            continue
        ax = fig.add_subplot(rows, 2, r * 2 + c + 1, projection="3d")
        axes[key] = ax
for uc in ucs:
    (uc.S,uc.basis)=solve(mesh,uc.elem)
    ax=axes[uc.name]
    ax.set_title(uc.name)
    ax.axis("equal")
    # using larger nref will create smoother plot
    # sfplot.plot3(basis, S, nref=1,shading='gouraud')
    (m,z)=uc.basis.refinterp(uc.S,nrefs=1)
    x=m.p[0]
    y=m.p[1]
    triangles=m.t.T
    ax.plot_trisurf(x, y, z,
                         triangles=triangles,
                         cmap='coolwarm')
    pyplot.pause(0.01)
    points = np.column_stack([x, y, z])
    faces = np.hstack(
        [np.c_[np.full(len(triangles), 3), triangles]]).astype(np.int32)
    mesh = pv.PolyData(points,faces)
    mesh = mesh.delaunay_2d()
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, cmap="coolwarm")
    plotter.show()
pyplot.tight_layout()
pyplot.show()
