# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026
@author: Simo
Assisted by Microsoft Copilot

Provides framework for testing various elements and processes for
solution of warping function and cross section properties.

TODO:
    scale to shear center
    calculate section properties
    support for at least U and RSH with rounded corners

For a ready solution see https://sectionproperties.readthedocs.io/
"""
import numpy as np
import skfem as sf
from skfem.models.poisson import laplace
import matplotlib.pyplot as pyplot
import math
import enum
import types
import logging
import pyvista as pv
import pyvistaqt
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

def tsplot(uc,m: sf.mesh.Mesh,z:np.ndarray):
    ax=uc.ax
    ax.set_title(uc.name)
    ax.axis("equal")
    x=m.p[0]
    y=m.p[1]
    triangles=m.t.T
    ax.plot_trisurf(x, y, z,
                         triangles=triangles,
                         cmap='coolwarm')
    pyplot.pause(0.01)

def qtplot(uc, m: sf.mesh.Mesh,z:np.ndarray,
           scale=None, **kwargs):
    # Apply scaled displacement to mesh
    if scale==None or scale<0:
        x_range = max(m.p[0])-min(m.p[0])
        y_range = max(m.p[1])-min(m.p[1])
        max_dim = max(x_range, y_range)
        max_disp = max(z.max(),-z.min())
        if scale==None:
            scaler=0.1
        else:
            scaler=-scale
        target_disp = scaler * max_dim
        scale = target_disp / max_disp if max_disp > 0 else 1.0
    # Create a diverging colormap centered at zero
    cmap = plt.get_cmap("coolwarm")  # or "seismic", "RdBu", "PiYG", etc.
    vmin=z.min()
    vmax=z.max()
    if vmin<0 and vmax>0:
        # Normalize so that zero is white
        norm = plt_colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = plt_colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(z))[:, :3]  # Drop alpha channel
    triangles=m.t.T
    x=m.p[0]
    y=m.p[1]
    sz=scale*z
    points = np.column_stack([x, y, sz])
    faces = np.hstack(
        [np.c_[np.full(len(triangles), 3), triangles]]).astype(np.int32)
    mesh = pv.PolyData(points,faces)
    try:
        uc.mp.clear()
    except AttributeError as e:
        logging.debug(f"First run {e}")
    uc.mp.add_mesh(mesh,
                     scalars=colors,rgb=True,
                     scalar_bar_args={"title": f"Warping for {uc.name}"},
                     show_edges=True)
    uc.mp.show_bounds(
        grid='back',
        location='outer',
        ticks='both',
        xtitle='X',
        ytitle='Y',
        ztitle='warping'
    )
    uc.mp.add_text(f"""{uc.name} {mesh.n_points} points
scale={scale:.3g}, max warping={max_disp:.3G}
""", font_size=12)
    uc.mp.render()
    return (scale)

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
class Model(enum.Enum):
    SQUARE=1
    RECTANGLE=2
model=Model.RECTANGLE
do_tsplot=False
do_qtplot=False
mesh_scale=100
if not do_tsplot:
    plt.close('all')
if not do_qtplot:
    if "mp" in globals():
        mp=globals()["mp"]
        mp.close()
        del(mp)
for nc in range(3,14,2):
    match model:
        case Model.SQUARE:
            x_nodes=nc
            y_nodes=nc
            qtplot_scale=-0.3
            mesh = sf.MeshTri.init_tensor(
                mesh_scale*np.linspace(-0.05, 0.05, x_nodes),
                mesh_scale*np.linspace(-0.05, 0.05, y_nodes)
            )
        case Model.RECTANGLE:
            x_nodes=nc
            y_nodes=nc
            qtplot_scale=-0.1
            mesh_scale=1000
            mesh = sf.MeshTri.init_tensor(
                mesh_scale*np.linspace(-0.05, 0.05, x_nodes),
                mesh_scale*np.linspace(-0.005, 0.005, y_nodes)
            )
        case _:
            raise ValueError(f"model {model} is not supported")
    print(f'Model={model}, nvertices={mesh.nvertices}')
    ucs=[
    #     types.SimpleNamespace(elem=sf.ElementTriN1()),
    #     types.SimpleNamespace(elem=sf.ElementTriN2()),
    #     types.SimpleNamespace(elem=sf.ElementTriN3()),
         types.SimpleNamespace(elem=sf.ElementTriP0()),
    #     types.SimpleNamespace(elem=sf.ElementTriP1()),
    #     types.SimpleNamespace(elem=sf.ElementTriP1B()),
    #    types.SimpleNamespace(elem=sf.ElementTriP1G()),
         types.SimpleNamespace(elem=sf.ElementTriP2()),
    #     types.SimpleNamespace(elem=sf.ElementTriP2B()),
    #     types.SimpleNamespace(elem=sf.ElementTriP2G()),
    #     types.SimpleNamespace(elem=sf.ElementTriP3()),
    #    types.SimpleNamespace(elem=sf.ElementTriP4()),
         ]
    rows = math.ceil(len(ucs) / 2)
    if do_qtplot:
        if not "mp" in globals():
            mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=2)
        if mp._nrows != rows:
            mp.close()
            mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=2)
    if do_tsplot:
        fig = pyplot.figure(num='warping using skfem',clear=True)
        pyplot.tight_layout()
    r=0
    c=0
    for uc in ucs:
        uc.name=type(uc.elem).__name__.split('Element')[-1]
        if do_tsplot:
            uc.ax=fig.add_subplot(rows, 2, r * 2 + c + 1, projection="3d")
        if do_qtplot:
            uc.mp=mp[r,c]
        if c==1:
            c=0
            r=r+1
        else:
            c=1
        (uc.S,uc.basis)=solve(mesh,uc.elem)
        (m,z)=uc.basis.refinterp(uc.S,nrefs=3)
        print(f'{uc.name}: max_warping = {max(z.max(),-z.min()):.3G}')
        if do_tsplot:
            tsplot(uc,m,z)
        if do_qtplot:
            qtplot(uc,m,z,scale=qtplot_scale)
