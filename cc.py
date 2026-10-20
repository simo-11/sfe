# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 20:33:08 2026

Common code as module so that reloading can take place
@author: simon
"""
import dataclasses
import numpy as np
import skfem as sf
import typing
import pyvistaqt
import enum
import PySide6.QtWidgets as qtw

class Model(enum.Enum):
    SQUARE=1
    RECTANGLE=2
    U=3
    RHS=4
    CIRCLE=5

@dataclasses.dataclass(repr=False)
class MeshTri3(sf.MeshTri2):
    elem: typing.Type[sf.Element] = sf.ElementTriP3
    @classmethod
    def init_circle(cls: typing.Type,
                    nrefs: int = 3,
                    smoothed: bool = False) -> 'MeshTri3':
        m = sf.MeshTri2.init_circle(nrefs=nrefs, smoothed=smoothed)
        M = cls.from_mesh(m)
        D = M.dofs.get_facet_dofs(M.boundary_facets()).flatten()
        doflocs = M.doflocs.copy()
        doflocs[:, D] /= np.linalg.norm(doflocs[:, D], axis=0)
        return dataclasses.replace(M, doflocs=doflocs)

class MappingCubicLine(sf.Mapping):
    """Isoparametric cubic mapping for 1D MeshLine."""

    def __init__(self, mesh: sf.MeshLine, elem: sf.Element):
        self.mesh = mesh
        self.elem = elem
        self.dim = 1

        # global node coordinates (1, ndofs)
        self.p = self.mesh.doflocs
        # element -> dofs connectivity (ndofs_per_elem, nelems)
        self.t = self.mesh.dofs.element_dofs

    def F(self, X, tind=None):
        """
        Physical coordinates x(ξ) = Σ x_i φ_i(ξ).

        X: shape (1, npts)
        """
        X = np.atleast_2d(X)
        npts = X.shape[1]

        if tind is None:
            nelems = self.t.shape[1]
            out = np.zeros((self.dim, nelems, npts))
            elems = np.arange(nelems)
        else:
            elems = np.asarray(tind)
            nelems = elems.size
            out = np.zeros((self.dim, nelems, npts))

        for i_local in range(self.t.shape[0]):
            phi, _ = self.elem.lbasis(X, i_local)
            g = self.t[i_local, elems]
            out[0, :, :] += self.p[0, g][:, None] * phi

        return out

    def J(self, X, tind=None):
        """
        Jacobian dx/dξ, sama shape kuin F.
        """
        X = np.atleast_2d(X)
        npts = X.shape[1]

        if tind is None:
            nelems = self.t.shape[1]
            out = np.zeros((self.dim, nelems, npts))
            elems = np.arange(nelems)
        else:
            elems = np.asarray(tind)
            nelems = elems.size
            out = np.zeros((self.dim, nelems, npts))

        for i_local in range(self.t.shape[0]):
            _, dphi = self.elem.lbasis(X, i_local)
            g = self.t[i_local, elems]
            out[0, :, :] += self.p[0, g][:, None] * dphi[0]

        return out

    def detJ(self, X, tind=None):
        J = self.J(X, tind)
        return np.abs(J[0, :, :])

    def invJ(self, X, tind=None):
        J = self.J(X, tind)
        out = np.zeros_like(J)
        out[0, :, :] = 1.0 / J[0, :, :]
        return out


class ElementLineP3(sf.ElementH1):
    """Piecewise cubic line element."""

    nodal_dofs = 1
    interior_dofs = 2
    maxdeg = 3
    dofnames = ['u', 'u', 'u', 'u']
    doflocs = np.array([[0.],
                        [1.],
                        [1/3],
                        [2/3]])
    refdom = sf.refdom.RefLine

    def lbasis(self, X, i):
        x = X[0]

        # Node locations
        xs = [0.0, 1.0, 1.0/3.0, 2.0/3.0]

        # Lagrange basis φ_i and derivative dφ_i
        def phi_and_dphi(x, i):
            xi = xs[i]
            num = 1.0
            dnum = 0.0
            for j in range(4):
                if j == i:
                    continue
                xj = xs[j]
                num *= (x - xj)
            # derivative: sum over k of product excluding k
            for k in range(4):
                if k == i:
                    continue
                term = 1.0
                for j in range(4):
                    if j == i or j == k:
                        continue
                    term *= (x - xs[j])
                dnum += term
            den = 1.0
            for j in range(4):
                if j == i:
                    continue
                den *= (xi - xs[j])
            return num / den, np.array([dnum / den])

        return phi_and_dphi(x, i)


class Profile:
    """Mesh generator for RHS and U profiles."""

    @staticmethod
    def arc(cx, cy, r, a0, a1, n):
        # Generate arc points
        th = np.linspace(a0, a1, n, endpoint=False)
        x = cx + r * np.cos(th)
        y = cy + r * np.sin(th)
        x = np.round(x, 14)
        y = np.round(y, 14)
        return np.vstack((x, y))

    @staticmethod
    def rectangle_with_arcs(b, h, r, n_arc, nx, ny):
        # Straight edges
        xe=b-r
        ye=h-r
        x0 = np.linspace(r, xe, nx, endpoint=False)
        x1 = np.linspace(xe, r, nx, endpoint=False)
        y0 = np.linspace(r, ye, ny, endpoint=False)
        y1 = np.linspace(ye, r, ny, endpoint=False)
        # Straight segments between arcs
        s1 = np.vstack((x0, np.zeros_like(x0)))
        s2 = np.vstack((np.full_like(y0, b), y0))
        s3 = np.vstack((x1, np.full_like(x1, h)))
        s4 = np.vstack((np.zeros_like(y1), y1))
        # Concatenate CCW
        if n_arc>0:
            # Four arcs
            a1 = Profile.arc(r, r, r, np.pi, 1.5*np.pi, n_arc)
            a2 = Profile.arc(b - r, r, r, 1.5*np.pi, 2*np.pi, n_arc)
            a3 = Profile.arc(b - r, h - r, r, 0, 0.5*np.pi, n_arc)
            a4 = Profile.arc(r, h - r, r, 0.5*np.pi, np.pi, n_arc)
            pts = np.hstack((a1, s1, a2, s2, a3, s3, a4, s4))
        else:
            pts=np.hstack((s1,s2,s3,s4))
        return pts

    @classmethod
    def rhs(cls, h, b, t,
            ri=0, n_arc=0, nx=1, ny=1,
            element: sf.Element=sf.ElementTriP1,
            draw=False):
        """Rectangular hollow section mesh.
        Using higher order elements is under study.
        Creating mesh with higher order nodes is still fuzzy.
        Mesh.__post_init has code that is for importing
        Check if Mesh.from_mesh() and
        Mesh.morphed could be used on
        """
        if n_arc>0:
            ro = ri + t
        else:
            ro=0
            ri=0
        # Outer and inner contours
        outer = cls.rectangle_with_arcs(b, h, ro, n_arc, nx,ny)
        inner = cls.rectangle_with_arcs(b - 2*t, h - 2*t,
                                        ri, n_arc, nx,ny)
        inner[0, :] += t
        inner[1, :] += t
        n1 = outer.shape[1]
        facets = []
        match element:
            case sf.ElementTriP1:
                pts = np.hstack((outer, inner))
                mesh_class=sf.MeshTri
                for i in range(n1):
                    ipn1=i+n1
                    r1=(i + 1)%n1
                    facets.append([i, r1, ipn1])
                    facets.append([r1, r1 + n1, ipn1])
            case sf.ElementTriP2:
                pts = np.hstack((outer, inner))
                mesh_class=sf.MeshTri2
                for i in range(n1):
                    ipn1=i+n1
                    r1=(i + 1)%n1
                    facets.append([i, r1, ipn1])
                    facets.append([r1, r1 + n1, ipn1])
            case sf.ElementTriP3:
                pts = np.hstack((outer, inner))
                mesh_class=MeshTri3
                for i in range(n1):
                    ipn1=i+n1
                    r1=(i + 1)%n1
                    facets.append([i, r1, ipn1])
                    facets.append([r1, r1 + n1, ipn1])
            case sf.ElementTriP8:
                middle=(inner+outer)/2
                n2=2*n1
                pts = np.hstack((outer, inner, middle))
                mesh_class=sf.MeshTri2
                for i in range(0,n1,2):
                    ipn1=i+n1
                    ipn2=i+2*n1
                    r1=(i + 2)%n1
                    r2=n2+(i+2)%n1
                    facets.append([i, r1, ipn1,
                                   i+1, ipn2+1, ipn2])
                    facets.append([r1, r1 + n1, ipn1,
                                   r2, ipn1+1, ipn2+1])
            case _:
                raise TypeError(f"Non-supported element: {element!r}")
        facets = np.array(facets).T
        #mesh=mesh_class(pts, facets)
        mesh=sf.MeshTri(pts,facets)
        check_mesh(mesh)
        match mesh_class:
            case sf.MeshTri:
                pass
            case _:
                mesh=mesh_class.from_mesh(mesh)
        check_mesh(mesh)
        uc=types.SimpleNamespace(elem=element())
        uc.profile=f'RHS {h}x{b}x{t} ri={ri} n_arc={n_arc} nx={nx} ny={ny}'
        uc.model=Model.RHS
        uc.basis=sf.Basis(mesh,uc.elem)
        fill_uc_defaults(uc)
        sp(uc)
        report_sp(uc)
        if draw:
            vedo_plot_mesh(mesh,logging.DEBUG)
            start_mp(ncols=1)
            uc.mp=mp[0,0]
            qtplot(uc,scale=-0.1)
        return uc

class MultiViewport(qtw.QWidget):
    def __init__(self, nrows=1, ncols=2):
        super().__init__()
        self.layout = qtw.QGridLayout(self)
        self.views = []
        self.ensure_item(nrows-1,ncols-1)

    def ensure_item(self,rn: int,cn: int):
        if rn < 0:
            raise ValueError(f"rn {rn!r} must >= 0")
        if cn < 0:
            raise ValueError(f"cn {cn!r} must >= 0")
        for r in range(len(self.views),rn+1):
            self.views.append([])
        row=self.views[rn]
        for c in range(len(row),cn+1):
            view = pyvistaqt.QtInteractor(self)
            view.add_camera_orientation_widget()
            view.view_xy()
            view.camera.up = (0, 1, 0)
            view.reset_camera()
            self.layout.addWidget(view, rn, c)
            row.append(view)

    # tuple indexing: win[r,c]
    def __getitem__(self, idx):
        r, c = idx
        self.ensure_item(r,c)
        return self.views[r][c]


