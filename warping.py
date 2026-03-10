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
    def i_den(w):
       return w['uh']
    den=i_den.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    @sf.Functional
    def i_nex(w):
       return w['uh']*w['x'][1]
    nex=i_nex.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    @sf.Functional
    def i_ney(w):
       return w['uh']*w['x'][0]
    ney=i_ney.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    scx = nex / den
    scy = -ney / den
    @sf.Functional
    def i_cw(w):
       return w['uh']**2
    gamma=i_cw.assemble(uc.t_basis, uh=uc.t_basis.interpolate(uc.S))
    sp["gamma"]=gamma
    #sp["j"]=j
    sp["sc"]=[scx,scy]
    uc.sp=sp

class Model(enum.Enum):
    SQUARE=1
    RECTANGLE=2
models=list(Model)
do_tsplot=False
do_qtplot=True
do_sp=True
mesh_scale=1
if not do_tsplot:
    plt.close('all')
if not do_qtplot:
    if "mp" in globals():
        mp=globals()["mp"]
        mp.close()
        del(mp)
r=0
c=0
for model in models:
    for nc in (11,):
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
# \nabla ^2\phi =-2.
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
# \nabla \cdot \mathbf{q}=2.
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