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
import pyvistaqt
import pyvista as pv


def gmsh_to_pyvista():
    """Convert current gmsh model to a PyVista PolyData mesh."""
    # --- Nodes ---
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)

    # --- Elements ---
    elem_types, elem_tags, elem_node_tags = \
        gmsh.model.mesh.getElements()

    tri3 = None    # etype 2
    tri6 = None    # etype 9
    quad4 = None   # etype 3
    quad9 = None   # etype 10
    quad8 = None   # etype 16

    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 2:      # 3-node triangle
            tri3 = np.array(enodes, int).reshape(-1, 3) - 1
        elif etype == 9:    # 6-node triangle
            tri6 = np.array(enodes, int).reshape(-1, 6) - 1
        elif etype == 3:    # 4-node quad
            quad4 = np.array(enodes, int).reshape(-1, 4) - 1
        elif etype == 10:   # 9-node quad
            quad9 = np.array(enodes, int).reshape(-1, 9) - 1
        elif etype == 16:   # 8-node serendipity quad
            quad8 = np.array(enodes, int).reshape(-1, 8) - 1

    if all(x is None for x in [tri3, tri6, quad4, quad9, quad8]):
        raise ValueError("No supported elements (2,3,9,10,16).")

    # --- Build VTK cell array ---
    cell_list = []

    if tri3 is not None:
        for t in tri3:
            cell_list.append(3)
            cell_list.extend(t)

    if tri6 is not None:
        for t in tri6:
            cell_list.append(6)
            cell_list.extend(t)

    if quad4 is not None:
        for q in quad4:
            cell_list.append(4)
            cell_list.extend(q)

    if quad8 is not None:
        for q in quad8:
            cell_list.append(8)
            cell_list.extend(q)

    if quad9 is not None:
        for q in quad9:
            cell_list.append(9)
            cell_list.extend(q)

    cells = np.array(cell_list, dtype=np.int64)

    # --- Create PyVista mesh ---
    pv_mesh = pv.PolyData(nodes, cells)

    return pv_mesh
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

def start_mp(rows=1):
    global mp
    if not "mp" in globals():
        mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=1)
    if mp._nrows != rows:
        mp.close()
        mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=1)
    return mp[0,0]


def make_circle_mesh(r=1.0, lc=0.2, order=1,
                     quad=False, serendipity=False,
                     pv_plot=False):
    """Generate a circular mesh using gmsh with adjustable radius,
    element size, element order, quad/tri mode and serendipity/full."""

    gmsh.initialize()
    gmsh.model.add("circle")

    # --- Element order ---
    gmsh.option.setNumber("Mesh.ElementOrder", order)

    # --- Points ---
    c  = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p1 = gmsh.model.geo.addPoint(r, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, r, 0, lc)
    p3 = gmsh.model.geo.addPoint(-r, 0, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, -r, 0, lc)

    # --- Arcs ---
    a1 = gmsh.model.geo.addCircleArc(p1, c, p2)
    a2 = gmsh.model.geo.addCircleArc(p2, c, p3)
    a3 = gmsh.model.geo.addCircleArc(p3, c, p4)
    a4 = gmsh.model.geo.addCircleArc(p4, c, p1)

    loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # --- Quad mode ---
    if quad:
        gmsh.model.mesh.setRecombine(2, surf)

        # Full 9-node quad (etype 10)
        if not serendipity:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        # Serendipity 8-node quad (etype 16)
        else:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)

    # --- Generate mesh ---
    gmsh.model.mesh.generate(2)

    # --- Optional PyVista plot ---
    if pv_plot:
        pv_mesh = gmsh_to_pyvista()
        plotter.add_mesh(pv_mesh, show_edges=True)
        plotter.show()

    # --- Extract mesh ---
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    elem_types, elem_tags, elem_node_tags = \
        gmsh.model.mesh.getElements(2, surf)

    gmsh.finalize()
    return nodes, elem_types, elem_node_tags
    # Extract mesh
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    elems = gmsh.model.mesh.getElements(2, surf)[2][0]
    tris = np.array(elems, int).reshape(-1, 3) - 1

    gmsh.finalize()

    m = meshio.Mesh(points=nodes[:, :2],
                    cells=[("triangle", tris)])
    return skio.from_meshio(m)

do_plot=False
pv_plot=True
plotter = start_mp()
# --- Create mesh ---
mesh = make_circle_mesh(r=1.0, lc=0.8)

# --- P2 basis ---
basis = sf.InteriorBasis(mesh, sf.ElementTriP2())

if do_plot:
    # --- Plot ---
    fig = plt.figure(num='circle_mesh',clear=True)
    ax = fig.add_subplot()
    mesh.draw(ax=ax, color='k', lw=0.5)
    #draw_curved_edges(mesh, basis, ax)
    ax.set_aspect('equal')
    ax.set_title("Black = gmsh mesh, Red = true P2 curved edges")
    plt.show()
