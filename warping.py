# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026
@author: Simo
Assisted by Microsoft Copilot

Provides framework for testing various elements and processes for
solution of warping function and cross section properties.

TODO:
    support for at least U and RSH with rounded corners

For a ready solution see https://sectionproperties.readthedocs.io/
"""
import numpy as np
import skfem as sf
import skfem.io.meshio as skio
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
import gmsh
import meshio
import vtk

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
    gmsh.model.geo.synchronize()
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


def u_mesh(h=0.1, b=0.05, t=0.004, ri=0.004, order=1):
    """
    Generate a thin-walled U-section with inner corner radius.
    Returns scikit-fem MeshTri.
    """
    gmsh.initialize()
    gmsh.model.add("u_section")
# --- Local mesh sizes ---
    lc_radius = ri / 1.0
    lc_web = h / 5.0
    lc_flange = b / 5.0
    # --- Helper to add a point with region-based lc ---
    def P(x, y, region):
        if region == "radius":
            lc = lc_radius
        elif region == "web":
            lc = lc_web
        else:
            lc = lc_flange
        return gmsh.model.geo.addPoint(x, y, 0.0, lc)

    # --- Outer radius ---
    ro = ri + t
    p1 = P(ro, 0, "radius")
    p2 = P(b, 0, "flange")
    p3 = P(b, t, "flange")
    p4 = P(ro, t, "radius")
    p5 = P(t, ro, "radius")
    p6 = P(t, h/2, "web")
    p7 = P(0, h/2, "web")
    p8 = P(0, ro, "radius")
    # --- straight lines ---
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p5, p6)
    l5 = gmsh.model.geo.addLine(p6, p7)
    l6 = gmsh.model.geo.addLine(p7, p8)
    # --- Arc centers ---
    c1 = P(t + ri, t + ri, "radius")
    a1 = gmsh.model.geo.addCircleArc(p4, c1, p5)
    a2 = gmsh.model.geo.addCircleArc(p8, c1, p1)
    if do_list_entities:
        list_entities()

    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, a1, l4, l5, l6, a2])

    # --- Surface (outer minus inner) ---
    surf = gmsh.model.geo.addPlaneSurface([loop])
    gmsh.model.geo.synchronize()
    original_entities = gmsh.model.getEntities(dim=1)
    copy_tags = gmsh.model.occ.copy(original_entities)
    # Mirror through the YZ-plane (x=0)
    # Parameters: copy_tags, point_on_plane(x,y,z), normal_of_plane(nx,ny,nz)
    gmsh.model.occ.mirror(copy_tags, 0, h/2, 0, 0, 1, 0)
    # 4. Remove duplicate nodes at the symmetry line
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.addPhysicalGroup(2, [surf], 1)

    # --- Mesh generation ---
    gmsh.model.mesh.generate(2)

    # --- Extract mesh ---
    nodes = gmsh.model.mesh.getNodes()
    node_coords = nodes[1].reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = \
        gmsh.model.mesh.getElements(2, surf)

    tri_cells = None
    for etype, enodes in zip(elem_types, elem_node_tags):
        if len(enodes) % 3 == 0:
            tri_cells = np.array(enodes, int).reshape(-1, 3) - 1
            break

    if do_list_entities:
        list_entities()
     # --- Optional PyVista plot ---
    if gmsh_plot:
        amp=mp[0,order-1]
        amp.clear()
        amp.add_text(f'gmsh mesh for U using order of {order}'
                      ,font_size=12)
        pv_mesh = gmsh_to_pyvista()
        smooth_mesh = pv_mesh.tessellate()
        amp.add_mesh(smooth_mesh
                     ,scalars="Gmsh_gamma", clim=[0, 1], cmap="RdYlGn"
                     ,show_edges=False
                     ,opacity=0.8)
        if nodes[0].shape[0]<200:
            amp.add_points(pv_mesh.points,
                       color="red",
                       point_size=8,
                       render_points_as_spheres=True)
        mp.show()
    gmsh.finalize()
    meshio_mesh = meshio.Mesh(
        points=node_coords[:, :2],
        cells=[("triangle", tri_cells)]
    )

    return skio.from_meshio(meshio_mesh)

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

def start_mp(rows=1):
    global mp
    if not "mp" in globals():
        mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=2)
    if mp._nrows != rows:
        mp.close()
        mp = pyvistaqt.MultiPlotter(nrows=rows, ncols=2)
    return mp[0,0]

def mplot(mesh: sf.mesh.Mesh, **fields):
    """
    mesh: skfem.Mesh
    fields: named arrays, e.g. omega=omega, ux=ux, sigma=sigma
    """
    # skfem stores triangles as shape (3, nelems)
    pts = mesh.p.T
    cells = mesh.t.T
    # PyVista wants a flat array: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    cell_data = np.hstack(
        [np.insert(c, 0, 3) for c in cells]
    )
    pv_mesh = pv.UnstructuredGrid(
        cell_data,
        np.full(len(cells), pv.CellType.TRIANGLE),
        pts
    )
    for name, arr in fields.items():
        pv_mesh.point_data[name] = arr
    plotter = start_mp()
    plotter.add_mesh(pv_mesh, scalars=list(fields.keys())[0])
    plotter.show()
    return pv_mesh

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
        xtitle='X[mm]',
        ytitle='Y[mm]',
        fmt="%.0f",
        ztitle='warping'
    )
    uc.mp.add_text(f"""{uc.name} {mesh.n_points} points
scale={scale:.4G}, max warping={max_disp:.4G}
""", font_size=12)
    uc.mp.render()
    return (scale)

def solve(uc):
    uc.t_basis=sf.Basis(uc.t_mesh, uc.elem)
    # Stiffness matrix: ∫ grad(v)·grad(u) dA
    A = sf.asm(laplace, uc.t_basis)
    # boundary condition
    def bc(v, w):
        x, y = w.x[0], w.x[1]
        nx, ny = w.n[0], w.n[1]
        g = x * ny - y * nx
        return g * v
    b = sf.asm(bc, uc.t_basis.boundary())
    # Fix the constant at random point, scale later
    D = uc.t_basis.split_indices()[0][[0]]
    A, b = sf.enforce(A, b,D=D)
    uc.S = sf.solve(A, b)

def sp(uc):
    """
    Parameters
    ----------
    uc : types.SimpleNamespace
        use case, this routine fills dict sp using
        section properties with similar namings as in
        https://sectionproperties.readthedocs.io/
    """
    sp={}
    @sf.Functional
    def i_area(w):
      return 1
    sp["area"]=area=i_area.assemble(uc.basis)
    @sf.Functional
    def i_cx(w):
      return w['x'][0]
    cx=i_cx.assemble(uc.basis)/area
    @sf.Functional
    def i_cy(w):
      return w['x'][1]
    cy=i_cy.assemble(uc.basis)/area
    @sf.Functional
    def i_xx(w):
      return (w['x'][1]-cy)**2
    ixx=i_xx.assemble(uc.basis)
    @sf.Functional
    def i_yy(w):
      return (w['x'][0]-cx)**2
    iyy=i_yy.assemble(uc.basis)
    @sf.Functional
    def i_xy(w):
      return (w['x'][0]-cx)*(w['x'][1]-cy)
    ixy=i_xy.assemble(uc.basis)
    sp["c"]=[cx,cy]
    sp["ic"]=[ixx,iyy,ixy]
    p = mesh.p.copy()
    t = mesh.t.copy()
    p = p + np.array([[-cx], [-cy]])
    uc.t_mesh=sf.MeshTri(p,t)
    solve(uc)
    @sf.Functional
    def i_xw(w):
       return w['uh']*w['x'][1]
    xw=i_xw.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    @sf.Functional
    def i_yw(w):
       return w['uh']*w['x'][0]
    yw=i_yw.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    scx = yw/iyy
    scy = xw/ixx
    @sf.Functional
    def i_w(w):
       return w['uh']
    iw=i_w.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    @sf.Functional
    def i_ww(w):
       return w['uh']**2
    iww=i_ww.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    gamma=iww-iw*iw/area-scy*xw+scx*yw
    sp["gamma"]=gamma
    #sp["j"]=j
    sp["sc"]=[cx+scx,cy+scy]
    uc.sp=sp

class Model(enum.Enum):
    SQUARE=1
    RECTANGLE=2
    U=3
models=list(Model)
models=(Model.U,)
do_tsplot=True
do_qtplot=False
do_sp=True
do_list_entities=True
gmsh_plot=True
mesh_scale=1000
if not do_tsplot:
    plt.close('all')
if not do_qtplot and not gmsh_plot:
    if "mp" in globals():
        mp=globals()["mp"]
        mp.close()
        del(mp)
r=0
c=0
for model in models:
    for nc in (10,):
        match model:
            case Model.SQUARE:
                x_nodes=nc
                y_nodes=nc
                qtplot_scale=-0.3
                mesh = sf.MeshTri.init_tensor(
                    mesh_scale*np.linspace(0, 0.1, x_nodes),
                    mesh_scale*np.linspace(0, 0.1, y_nodes)
                )
            case Model.RECTANGLE:
                x_nodes=nc
                y_nodes=nc
                qtplot_scale=-0.1
                mesh = sf.MeshTri.init_tensor(
                    mesh_scale*np.linspace(0, 0.1, x_nodes),
                    mesh_scale*np.linspace(0, 0.01, y_nodes)
                )
            case Model.U:
                x_nodes=nc
                y_nodes=nc
                qtplot_scale=-0.1
                if gmsh_plot:
                    start_mp(rows=1)
                mesh = u_mesh(0.1,0.05,0.004,0.004,order=1)
            case _:
                raise ValueError(f"model {model} is not supported")
        print(f'Model={model}, nc={nc}, nvertices={mesh.nvertices}')
        ucs=[
            #types.SimpleNamespace(elem=sf.ElementTriN1()), # c_einsum fails
            #types.SimpleNamespace(elem=sf.ElementTriN2()), # c_einsum fails
            #types.SimpleNamespace(elem=sf.ElementTriN3()), # c_einsum fails
            #types.SimpleNamespace(elem=sf.ElementTriP0()),  # Solve fails
            #types.SimpleNamespace(elem=sf.ElementTriP1()),
            #types.SimpleNamespace(elem=sf.ElementTriP1B()),
            #types.SimpleNamespace(elem=sf.ElementTriP1G()),
            types.SimpleNamespace(elem=sf.ElementTriP2()),
            #types.SimpleNamespace(elem=sf.ElementTriP2B()),
            #types.SimpleNamespace(elem=sf.ElementTriP2G()),
            #types.SimpleNamespace(elem=sf.ElementTriP3()),
            #types.SimpleNamespace(elem=sf.ElementTriP4()),
             ]
        rows = math.ceil(len(ucs) * len(models)/ 2)
        if do_qtplot:
            start_mp(rows=rows)
        if do_tsplot:
            fig = pyplot.figure(num='warping using skfem',clear=True)
            pyplot.tight_layout()
        for uc in ucs:
            try:
                uc.name=type(uc.elem).__name__.split('Element')[-1]
                uc.basis=sf.Basis(mesh, uc.elem)
                uc.scale=mesh_scale
                if do_tsplot:
                    uc.ax=fig.add_subplot(rows,
                                          2, r * 2 + c + 1,
                                          projection="3d")
                if do_qtplot:
                    uc.mp=mp[r,c]
                if c==1:
                    c=0
                    r=r+1
                else:
                    c=1
                if do_sp or do_qtplot or do_tsplot:
                    sp(uc)
                    if do_sp:
                        m4=mesh_scale**4
                        m6=mesh_scale**6
                        print(f'''Section properties
  area={uc.sp['area']/(mesh_scale**2):.4G}
  c=[{uc.sp['c'][0]/mesh_scale:.4G},{uc.sp['c'][1]/mesh_scale:.4G}]
  ic=[{uc.sp['ic'][0]/m4:.4G}, \
{uc.sp['ic'][1]/m4:.4G}, \
{uc.sp['ic'][2]/m4:.4G}]
  sc=[{uc.sp['sc'][0]/mesh_scale:.4G},{uc.sp['sc'][1]/mesh_scale:.4G}]
  gamma={uc.sp['gamma']/m6:.4G}
''')
                if do_qtplot or do_tsplot:
                    (m,z)=uc.basis.refinterp(uc.S,nrefs=1)
                    if np.isnan(z).any():
                        print(f'Solution failed for {uc.name}')
                        continue
                    print((f'{uc.name}: max_warping ='
                           f' {(z.max()-z.min())/2:.4G}'))
                    if do_tsplot:
                        tsplot(uc,m,z)
                    if do_qtplot:
                        qtplot(uc,m,z,scale=qtplot_scale)
            except Exception:
                import traceback
                print(f'Solution failed for {uc.name}')
                traceback.print_exc()
# %%  Saint‑Venant
# warping (ω): Laplace = 0
# \nabla ^2\omega =0,\qquad \frac{\partial \omega }{\partial n}=yn_x-xn_y.
'''
import numpy as np
import skfem

# Mesh
m = skfem.MeshTri().init_symmetric().refined(3)

# Basis
e = skfem.ElementTriP1()
basis = skfem.Basis(m, e)

# Bilinear form: ∫ ∇u · ∇v
@sf.BilinearForm
def bilinf(u, v, _):
    return skfem.helpers.dot(u.grad, v.grad)
A = skfem.asm(bilinf, basis)
b = np.zeros(A.shape[0])
# Natural BC: ∂ω/∂n = y n_x − x n_y
@sf.LinearForm
def neumann(v, w):
    nx, ny = w.n
    x, y = w.x
    return (y * nx - x * ny) * v
b += skfem.asm(neumann, basis.boundary())
# Solve
omega = skfem.solve(A, b)
@sf.Functional
def den_integral(w):
   return w['uh']
den=den_integral.assemble(basis, uh=basis.interpolate(omega))
@sf.Functional
def ex_integral(w):
   return w['uh']*w['x'][1]
num_ex=ex_integral.assemble(basis, uh=basis.interpolate(omega))
@sf.Functional
def ey_integral(w):
   return w['uh']*w['x'][0]
num_ey=ey_integral.assemble(basis, uh=basis.interpolate(omega))
ex = num_ex / den
ey = -num_ey / den
print("Shear center (ω):", ex, ey)
Cw = basis.integrate(lambda w: w.u**2, omega)
print("Warping constant Cw =", Cw)
interp = basis.interpolate(omega)
Cw_grad = basis.integrate(lambda w: skfem.helpers.dot(w.grad, w.grad), omega)
print("Cw (gradient form) =", Cw_grad)
mplot(m, omega=omega)
@sf.Integral
def gradx(w):
    return w.grad(w.u)[0]

@sf.Integral
def grady(w):
    return w.grad(w.u)[1]

# Evaluate gradient at nodes
omega_grad = basis.interpolate(omega)
omega_x = omega_grad.grad[0]
omega_y = omega_grad.grad[1]

theta = 1.0  # unit twist

ux = theta * omega_x
uy = theta * omega_y
uz = np.zeros_like(ux)

mplot(m, ux=ux, uy=uy, uz=uz, omega=omega)
E = 210e9      # Young's modulus (Pa)
theta = 1.0    # unit twist

sigma_w = E * theta * omega
@sf.Integral
def bimoment_integrand(w):
    return w.u**2

B = E * skfem.asm(bimoment_integrand, basis, omega)

print("Bimoment B =", B)

mplot(m, omega=omega, sigma_w=sigma_w)
# %%  Prandtl
# stress function (φ): Laplace = –2
# \nabla ^2\\phi =-2.
import numpy as np
import skfem

# Mesh
m = skfem.MeshTri().init_symmetric().refined(3)

# Basis
e = skfem.ElementTriP2()
basis = skfem.Basis(m, e)

# Bilinear form
@sf.BilinearForm
def bilinf(u, v, w):
    return w.grad(u) @ w.grad(v)

A = skfem.asm(bilinf, basis)

# RHS: ∫ 2 v dA
@sf.LinearForm
def rhs(v, w):
    return 2.0 * v

b = skfem.asm(rhs, basis)

# Dirichlet BC: φ = 0 on boundary
D = basis.get_dofs().all_boundary()
A, b = skfem.enforce(A, b, D=D)

# Solve
phi = skfem.solve(A, b)

@sf.Integral
def tau_xz(w):
    return w.grad(w.u)[1]   # dφ/dy

@sf.Integral
def tau_yz(w):
    return -w.grad(w.u)[0]  # -dφ/dx

qx = skfem.asm(tau_xz, basis, phi)
qy = skfem.asm(tau_yz, basis, phi)
@sf.Integral
def Vx_integrand(w):
    return w.grad(w.u)[1]   # qx

@sf.Integral
def Vy_integrand(w):
    return -w.grad(w.u)[0]  # qy

Vx = skfem.asm(Vx_integrand, basis, phi)
Vy = skfem.asm(Vy_integrand, basis, phi)
@sf.Integral
def Mx_integrand(w):
    x, y = w.x
    return y * (-w.grad(w.u)[0])   # y * qy

@sf.Integral
def My_integrand(w):
    x, y = w.x
    return x * (w.grad(w.u)[1])    # x * qx

Mx = skfem.asm(Mx_integrand, basis, phi)
My = skfem.asm(My_integrand, basis, phi)

ex = Mx / Vy
ey = -My / Vx

print("Shear center coordinates:")
print("e_x =", ex)
print("e_y =", ey)

# Compute torsion constant J = 2 ∫ φ dA
@sf.Integral
def integrand(w):
    return 2.0 * w.u

J = skfem.asm(integrand, basis, phi)

print("Torsion constant J =", J)

# Save
m.save('phi.vtk', phi=phi)
# %% shear‑flow
# (q): H(div) → Raviart–Thomas RT0
# \nabla \\cdot \\mathbf{q}=2.
import numpy as np
import skfem

# Mesh
m = skfem.MeshTri().init_symmetric().refined(3)

# RT0 element
e = skfem.ElementTriRT0()
basis = skfem.Basis(m, e)

# Bilinear form: mass matrix (q, v)
@sf.BilinearForm
def mass(u, v, w):
    return u @ v

A = skfem.asm(mass, basis)

# RHS: ∫ 2 div(v) dA
@sf.LinearForm
def rhs(v, w):
    return 2.0 * w.div(v)

b = skfem.asm(rhs, basis)

# Solve
q = skfem.solve(A, b)

# Save vector field
qx = q[0::2]
qy = q[1::2]
m.save('q.vtk', qx=qx, qy=qy)
'''