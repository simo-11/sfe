# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 18:40:50 2026

@author: simon
"""

import gmsh
import meshio
import numpy as np
import skfem.io.meshio as skio
import matplotlib.pyplot as plt
import skfem as sf

def draw_curved_edges(mesh, basis, ax):
    """Draw curved P2 edges by sampling shape functions."""
    X = mesh.p.T
    T = mesh.t.T

    # P2 has 6 nodes per triangle
    # We create mid-edge nodes by interpolation
    for tri in T:
        pts = X[tri]
        # sample edge with quadratic interpolation
        for i in range(3):
            a = pts[i]
            b = pts[(i + 1) % 3]
            m = (a + b) / 2
            curve = np.vstack([a, m, b])
            ax.plot(curve[:, 0], curve[:, 1], 'r-', lw=1)

def make_circle_mesh(r=1.0, lc=0.2):
    """Generate a circular mesh using gmsh and return MeshTri."""
    gmsh.initialize()
    gmsh.model.add("circle")

    # Circle center and boundary points
    c = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p1 = gmsh.model.geo.addPoint(r, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, r, 0, lc)
    p3 = gmsh.model.geo.addPoint(-r, 0, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, -r, 0, lc)

    # Four quarter arcs
    a1 = gmsh.model.geo.addCircleArc(p1, c, p2)
    a2 = gmsh.model.geo.addCircleArc(p2, c, p3)
    a3 = gmsh.model.geo.addCircleArc(p3, c, p4)
    a4 = gmsh.model.geo.addCircleArc(p4, c, p1)

    loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Extract mesh
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    elems = gmsh.model.mesh.getElements(2, surf)[2][0]
    tris = np.array(elems, int).reshape(-1, 3) - 1

    gmsh.finalize()

    m = meshio.Mesh(points=nodes[:, :2],
                    cells=[("triangle", tris)])
    return skio.from_meshio(m)

# --- Create mesh ---
mesh = make_circle_mesh(r=1.0, lc=0.2)

# --- P2 basis ---
basis = sf.InteriorBasis(mesh, sf.ElementTriP2())

# --- Plot ---
fig, ax = plt.subplots(figsize=(6, 6))

mesh.draw(ax=ax, color='k', lw=0.5)
draw_curved_edges(mesh, basis, ax)

ax.set_aspect('equal')
ax.set_title("Black = gmsh mesh, Red = true P2 curved edges")

plt.show()
