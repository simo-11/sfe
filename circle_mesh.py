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
import vtk

def gmsh_to_pyvista():
    """
    Convert current gmsh model to a PyVista UnstructuredGrid.
    Supports curved quadratic elements (Tri6, Quad8, Quad9).
    """
    # --- Nodes ---
    # Get node tags and coordinates (N, 3)
    node_tags, nodes_flat, _ = gmsh.model.mesh.getNodes()
    nodes = nodes_flat.reshape(-1, 3)

    # Map Gmsh tags to 0-based indices for VTK connectivity
    max_tag = int(np.max(node_tags))
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
    tag_to_idx[node_tags.astype(int)] = np.arange(len(node_tags))

    # --- Elements ---
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()

    cells = []
    cell_types = []
    gmsh_gamma = []      # To store quality
    gmsh_distortion = [] # To store distortion
    # Map Gmsh element types to VTK cell types
    # 2: Tri3, 9: Tri6, 3: Quad4, 16: Quad8, 10: Quad9
    mapping = {
        2:  (3, vtk.VTK_TRIANGLE),
        9:  (6, vtk.VTK_QUADRATIC_TRIANGLE),
        3:  (4, vtk.VTK_QUAD),
        16: (8, vtk.VTK_QUADRATIC_QUAD),
        10: (9, vtk.VTK_LAGRANGE_QUADRILATERAL)
    }

    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype not in mapping:
            continue
        # Get Gmsh internal quality metrics for these specific tags
        # 'gamma' (shape) and 'distortion' (Jacobian determinant)
        q_gamma = gmsh.model.mesh.getElementQualities(etags, "gamma")
        q_dist = gmsh.model.mesh.getElementQualities(etags, "minSJ")
        num_nodes, vtk_type = mapping[etype]
        # Convert all tags in this group to local indices
        node_indices = tag_to_idx[enodes.astype(int)].reshape(-1, num_nodes)

        for i, cell_nodes in enumerate(node_indices):
            cells.append(num_nodes)
            cells.extend(cell_nodes)
            cell_types.append(vtk_type)
            # Store the metrics corresponding to this cell
            gmsh_gamma.append(q_gamma[i])
            gmsh_distortion.append(q_dist[i])

    if not cells:
        raise ValueError("No supported elements found.")

    # --- Create UnstructuredGrid ---
    # PolyData is for linear faces; UnstructuredGrid handles quadratic cells
    pv_mesh = pv.UnstructuredGrid(np.array(cells),
                                 np.array(cell_types, dtype=np.uint8),
                                 nodes)

    # Attach Gmsh internal quality as Cell Data
    pv_mesh.cell_data["Gmsh_gamma"] = np.array(gmsh_gamma)
    pv_mesh.cell_data["Gmsh_minSJ"] = np.array(gmsh_distortion)

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
    make_new=False
    if not "mp" in globals():
        make_new=True
    if mp._nrows != rows:
        mp.close()
        make_new=True
    if make_new:
        mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=2)
        mp._window.setWindowTitle("circle_mesh")
    return mp[0,0]

def get_curve_info(curve_tag):
    """Return a dictionary with detailed information about a curve entity."""
    info = {}
    b = gmsh.model.getBoundary([(1, curve_tag)])
    start_pt = b[0][1]
    end_pt = b[1][1]
    info["center"] = b[0][0]
    info["start_point"] = start_pt
    info["end_point"] = end_pt
    return info

def get_surface_info(surface_tag):
    """Return detailed information about a surface entity (Gmsh 4.12+)."""
    info = {}
    # --- 2) Boundary curves ---
    _upward, curves = gmsh.model.getAdjacencies(2, surface_tag)
    info["boundary_curves"] = curves.tolist()
    return info

def list_entities():
    """Print all Gmsh entities and their types in a clean table."""
    ents = gmsh.model.getEntities()

    print(f"{'Dim':<4} {'Tag':<6} {'Type':<20} {'Extra info'}")
    print("-" * 60)

    for dim, tag in ents:
        etype = gmsh.model.getType(dim, tag)

        # Extra info depending on dimension
        extra = ""

        # --- Points ---
        if dim == 0:
            x, y, z = gmsh.model.getValue(0, tag, np.array([]))
            extra = f"({x:.3f}, {y:.3f}, {z:.3f})"

        # --- Curves ---
        elif dim == 1:
            extra=get_curve_info(tag)
        # --- Surfaces ---
        elif dim == 2:
            extra=get_surface_info(tag)
        # --- Volumes ---
        elif dim == 3:
            try:
                vol = gmsh.model.getMeasure(3, [tag])
                extra = f"volume={vol:.3f}"
            except:
                extra = ""

        print(f"{dim:<4} {tag:<6} {etype:<20} {extra}")


def make_circle_mesh(r=1.0, lc=0.2, order=1,
                     quad=False, serendipity=False):
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
    # --- Extract mesh ---
    nodes = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
    elem_types, elem_tags, elem_node_tags = \
        gmsh.model.mesh.getElements(2, surf)
    if do_list_entities:
        list_entities()
    # --- Optional PyVista plot ---
    if pv_plot:
        amp=mp[0,order-1]
        amp.clear()
        amp.add_text(f'gmsh mesh using order of {order}'
                     ,font_size=12)
        pv_mesh = gmsh_to_pyvista()
        smooth_mesh = pv_mesh.tessellate()
        # 3. Add the filled surface (The "Skin")
        # We set show_edges=False here to hide the tiny triangles
        amp.add_mesh(smooth_mesh
                    ,scalars="Gmsh_gamma", clim=[0, 1], cmap="RdYlGn"
                    ,show_edges=False
                    ,opacity=0.8)
        amp.add_points(pv_mesh.points,
                      color="red",
                      point_size=8,
                      render_points_as_spheres=True)
        plotter.show()
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

do_list_entities=False
do_plot=False
pv_plot=True
to_sf=False
plotter = start_mp()
# --- Create mesh ---
for order in range(1,3):
    elem=sf.ElementTriP2()
    mesh = make_circle_mesh(r=1.0, lc=0.8,order=order)
    do_list_entities=False
    if to_sf:
        basis = sf.InteriorBasis(mesh, elem)
    if do_plot:
        # --- Plot ---
        fig = plt.figure(num='circle_mesh',clear=True)
        ax = fig.add_subplot()
        mesh.draw(ax=ax, color='k', lw=0.5)
        #draw_curved_edges(mesh, basis, ax)
        ax.set_aspect('equal')
        ax.set_title("Black = gmsh mesh, Red = true P2 curved edges")
        plt.show()
