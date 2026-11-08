# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 19:49:39 2026
@author: Simo
Assisted by Microsoft Copilot and Google Search AI

Provides framework for testing various elements and processes for
solution of warping function and cross section properties.
Supported elements are linear and quadratic triangles.
Supported sections are square, rectangle, RSH and U.
Main target is SHS.

Most routines use parameter uc which is types.SimpleNamespace
containing e.g.
model enum
profile text description of profile
elem scikit-element -> basis.elem
name short str based on element e.g. TriP2
basis sf.assembly.basis.CellBasis
  N Number of DOFs
  doflocs - coordinates of nodes for scikit-solution, (2,N) in this context
  element_dofs - element topology, int32:s refer to doflocs
  mesh sf.mesh.mesh.Mesh
  elem sf.element.Element
t_basis copy of basis transferred to centroid
S solution of warping function
sp dict of section properties - note use of scale
scale scaling of dimensions for calculation and plots

TODO:
  - Create RHS and U profiles as scikit-fem Meshes using class Profile
  - plot shear stress distribution using probes
    warping values at given points P(2,N) can be obtained
    uc.basis.probes(P)@uc.S
    values and gradients using probe_value_and_grad

For a ready and tested solution for section properties
see https://sectionproperties.readthedocs.io/
"""
import copy
import numpy as np
import skfem as sf
import types
import logging
import pyvista as pv
import sys
import PySide6
import matplotlib
import gmsh
import vtk
import vedo
import scipy
import cc

def reset_logging():
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()
    root.setLevel(logging.NOTSET)
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            for h in logger.handlers[:]:
                logger.removeHandler(h)
                h.close()
            logger.setLevel(logging.NOTSET)

reset_logging()

levels = {
    "DEBUG": [],#"sfe"],
    "INFO": ["sfe"],
    "WARNING": ["skfem.assembly.basis",
                "skfem.assembly.form.form",
                "skfem.assembly",
                "skfem.utils"
            ]
}
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "short": {
            "format": "%(levelname)s: %(message)s"
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "sfe.log",
            "encoding": "utf-8",
        },
    },

    "loggers": {
        **{
            name: {
                "level": level,
                "handlers": ["console", "file"],
                "propagate": False,
            }
            for level, names in levels.items()
            for name in names
        }
    },

    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
logging.config.dictConfig(LOGGING)
logger = logging.getLogger("sfe")
logger.info(f"Using python {sys.version.split()[0]}"
            f", numpy {np.__version__}"
            f", scipy {scipy.__version__}"
            f", skfem {sf.__version__}"
            f", Qt {PySide6.QtCore.__version__}"
            f", PySide6 {PySide6.__version__}"
            f", vtk {vtk.__version__}"
            f", pyvista {pv.__version__}"
            f", vedo {vedo.__version__}"
            f", gmsh {gmsh.__version__}"
            f", matplotlib {matplotlib.__version__}"
            )

#%% test_elements
"""
ucs=cc.test_elements()
"""
#%% test profile.rhs
"""
rhs0 = Profile.rhs(b=0.15, h=0.15, t=0.008, ri=0.008,n_arc=4, draw=True)
"""
#%% test u
#%% test init circle
def test_init_circle():
    ucs=[#types.SimpleNamespace(elem=sf.ElementTriP2()),
         types.SimpleNamespace(elem=sf.ElementTriP3())
         ]
    nref_range_stop=2
    for nrefs in range(1,nref_range_stop):
        ucs.append(copy.deepcopy(ucs[0]))
    cc.start_mp(nrows=len(ucs))
    mp_global = globals().get("mp")
    for row,uc in enumerate(ucs):
        uc.vtk_tessellate=0
        uc.pick_cells=True
        uc.show_cell_ids=True
        # For P2
        # 0 -> 13
        # 1 -> 41
        # 2 -> 145
        # 3 -> 545
        uc.basis = sf.Basis(sf.MeshTri2.init_circle(row),
                        uc.elem)
        uc.mp=mp_global[row,0]
        uc.profile='Unit circle'
        cc.qtplot(uc)
        cc.sp(uc)
        uc.mp=mp_global[row,1]
        cc.qtplot(uc)
        cc.report_sp(uc)
    return ucs
"""
ccs=test_init_circle()
qtplot(ccs[0])
qtplot(ccs[1])
"""
#%% manual circle
def get_mesh_data_for_circle(elem:sf.ElementTri,
                             n_elem=None,
                             r=1,
                             t_order=1):
    match type(elem):
        case sf.ElementTriP1:
            if n_elem==None:
                n_elem=4*1
            doflocs=np.zeros((2,1+n_elem))
            doflocs[:,1:]=cc.Profile.arc(0,0,r,0,2*np.pi,n_elem)
            t=np.zeros((3,n_elem),dtype=np.int32)
            t[1,:]=np.r_[1:n_elem+1]
            t[2,:]=np.r_[2:n_elem+1,1]
        case sf.ElementTriP2:
            if n_elem==None:
                n_elem=4*1
            doflocs=np.zeros((2,1+n_elem*3))
            doflocs[:,1:1+n_elem]=cc.Profile.arc(0,0,r,0,2*np.pi,n_elem)
            rs1=1+n_elem
            doflocs[:,rs1:rs1+n_elem]=cc.Profile.arc(0,0,r/2,0,2*np.pi,n_elem)
            rs2=rs1+n_elem
            d_theta=np.pi/n_elem
            doflocs[:,rs2:rs2+n_elem]=cc.Profile.arc(0,0,r,d_theta
                                                  ,2*np.pi+d_theta,n_elem)
            nodes_in_elem=6
            t=np.zeros((nodes_in_elem,n_elem),dtype=np.int32)
            t[1,:]=np.r_[1:n_elem+1]
            t[2,:]=np.r_[2:n_elem+1,1]
            t[3,:]=np.r_[rs1:rs2]
            t[4,:]=np.r_[rs2:rs2+n_elem]
            t[5,:]=np.r_[rs1+1:rs2,rs1]
        case sf.ElementTriP3:
            if n_elem==None:
                n_elem=4*1
            doflocs=np.zeros((2,1+n_elem*6))
            doflocs[:,1:1+n_elem]=cc.Profile.arc(0,0,r,0,2*np.pi,n_elem)
            rs1=1+n_elem
            doflocs[:,rs1:rs1+n_elem]=cc.Profile.arc(0,0,r/3,0,2*np.pi,n_elem)
            rs2=rs1+n_elem
            doflocs[:,rs2:rs2+n_elem]=cc.Profile.arc(0,0,2*r/3,0
                                                     ,2*np.pi,n_elem)
            rs3=rs2+n_elem
            d_theta=2/3*np.pi/n_elem
            doflocs[:,rs3:rs3+n_elem]=cc.Profile.arc(0,0,r,d_theta
                                                  ,2*np.pi+d_theta,n_elem)
            rs4=rs3+n_elem
            d_theta=4/3*np.pi/n_elem
            doflocs[:,rs4:rs4+n_elem]=cc.Profile.arc(0,0,r,d_theta
                                                  ,2*np.pi+d_theta,n_elem)
            rs5=rs4+n_elem
            d_theta=np.pi/n_elem
            r_in=0.64395*r# from gmsh using gmsh_annulus
            doflocs[:,rs5:rs5+n_elem]=cc.Profile.arc(0,0,r_in,d_theta
                                                  ,2*np.pi+d_theta,n_elem)
            nodes_in_elem=10
            t=np.zeros((nodes_in_elem,n_elem),dtype=np.int32)
            match t_order:
                case 0: # rotate ccw
                    t[1,:]=np.r_[1:n_elem+1]
                    t[2,:]=np.r_[2:n_elem+1,1]
                    t[3,:]=np.r_[rs1:rs2]
                    t[4,:]=np.r_[rs2:rs3]
                    t[5,:]=np.r_[rs3:rs4]
                    t[6,:]=np.r_[rs4:rs5]
                    t[7,:]=np.r_[rs1+1:rs2,rs1]
                    t[8,:]=np.r_[rs2+1:rs3,rs2]
                case 1: # last element cw
                    t[1,:]=np.r_[1:n_elem,1]
                    t[2,:]=np.r_[2:n_elem+1,n_elem]
                    t[3,:]=np.r_[rs1:rs2-1,rs1]
                    t[4,:]=np.r_[rs2:rs3-1,rs2]
                    t[5,:]=np.r_[rs3:rs4-1,rs5-1]
                    t[6,:]=np.r_[rs4:rs5-1,rs4-1]
                    t[7,:]=np.r_[rs1+1:rs2,rs2-1]
                    t[8,:]=np.r_[rs2+1:rs3,rs3-1]
                    t[9,:]=np.r_[rs5:rs5+n_elem]
        case _: raise ValueError((f'Element {type(elem)}'
                                 'not supported'))
    return (doflocs,t)
def test_manual_circle():
    write_json=True
    elem_classes = [sf.ElementTriP1,sf.ElementTriP2,sf.ElementTriP3]
    ucs=[types.SimpleNamespace() for _ in range(len(elem_classes))]
    mp_global=cc.start_mp(nrows=len(elem_classes),ncols=2)
    for row, uc in enumerate(ucs):
        uc.profile='Unit circle'
        uc.elem=elem_classes[row]()
        match elem_classes[row]:
            case sf.ElementTriP1:
                mc=sf.MeshTri
            case _:
                mc=sf.MeshTri2
        (doflocs,t)=get_mesh_data_for_circle(uc.elem)
        mesh=mc(doflocs=doflocs,t=t,elem=uc.elem)
        cc.vedo_plot_mesh(mesh,logging.DEBUG)
        match elem_classes[row]:
            case sf.ElementTriP1:
                mapping=sf.MappingAffine(mesh)
            case sf.ElementTriP2:
                mapping=sf.MappingIsoparametric(
                    mesh,
                    uc.elem,
                    bndelem=sf.ElementLineP2())
            case sf.ElementTriP3:
                mapping=sf.MappingIsoparametric(
                    mesh,
                    uc.elem,
                    bndelem=cc.ElementLineP3())
        for intorder in [None]:#range(1,10,2):# default is 2 * elem.maxdeg
            uc.basis = sf.Basis(mesh,uc.elem,
                                mapping=mapping,intorder=intorder)
            if write_json:
                cc.sf_mesh_to_json(uc)
            uc.mp=mp_global[row,0]
            if hasattr(uc,'name'):
                del uc.name
            cc.qtplot(uc)
            try:
                cc.sp(uc)
                uc.mp=mp_global[row,1]
                cc.qtplot(uc)
                cc.report_sp(uc)
            except Exception:
                logger.exception("Solve failed")
    return ucs
"""
mcs=test_manual_circle()
vedo_plot_mesh(mcs[0].basis.mesh)
vedo_plot_mesh(mcs[0].basis.mesh,logging.INFO)
qtplot(mcs[0])
qtplot(mcs[1])
def f1(w):
    return 1
o=mcs[0]
fb=o.basis.boundary()
f1.assemble(o.basis)
f1.assemble(fb)
"""
#%% circle_area
def circle_area(r=1.0, ntri=32):
   """Approximate circle area using ntri straight-edged triangles."""
   area = 0.0
   dtheta = 2 * np.pi / ntri
   for i in range(ntri):
        th0 = i * dtheta
        th1 = (i + 1) * dtheta
        # Corner nodes on the circle
        p0 = np.array([r * np.cos(th0), r * np.sin(th0)])
        p1 = np.array([r * np.cos(th1), r * np.sin(th1)])
        # 2D cross product magnitude = |x0*y1 - x1*y0|
        cross2d = p0[0] * p1[1] - p0[1] * p1[0]
        area += 0.5 * abs(cross2d)
   return area
def run_circle_areas():
    r=1.0
    exact=np.pi*r ** 2
    for n in [4, 8, 16, 32, 64, 128]:
        approx = circle_area(r, n)
        print(f"{n:3d} {approx:10.5g} {100*(approx - exact)/exact:8.3g} %")
"""
run_circle_areas()
"""
#%% various elements, meshes and mesh refinements for circle
"""
Note that element type does not have affect on calculation of area
"""
@sf.Functional
def i_area(w):
    return 1
def test_circle_areas():
    elem_classes = [sf.ElementTriP1,sf.ElementTriP2,sf.ElementTriP3]
    mesh_classes = [sf.MeshTri,sf.MeshTri2,cc.MeshTri3]
    ucs=[[types.SimpleNamespace() for _ in range(len(mesh_classes))]
         for _ in range(len(elem_classes))]
    cc.start_mp(nrows=len(elem_classes),ncols=len(mesh_classes))
    write_json=False
    mp_global = globals().get("mp")
    exact=np.pi
    print(f"Exact: {exact:.6g}")
    for row, row_items in enumerate(ucs):
        for col, uc in enumerate(row_items):
            uc.vtk_tessellate=4
            mapping=None
            match row:
                case 0:
                    nrefs=2-col
                case 1:
                    nrefs=max(1-col,0)
                case _:
                    nrefs=0
            uc.elem=elem_classes[row]()
            mesh=mesh_classes[col].init_circle(nrefs)
            uc.basis = sf.Basis(mesh,uc.elem,mapping=mapping)
            if write_json:
                cc.sf_mesh_to_json(uc)
            area=i_area.assemble(uc.basis)
            uc.mp=mp_global[row,col]
            if hasattr(uc,'name'):
                del uc.name
            cc.qtplot(uc)
            epc=100*(area-exact)/exact
            print((f"{uc.name}, nrefs={nrefs}, nnodes={uc.basis.N:2d}:"
                   f" area={area:7.6g},"
                   f" error={epc: 5.4g} %"))
    return ucs
"""
qcs=test_circle_areas()
"""